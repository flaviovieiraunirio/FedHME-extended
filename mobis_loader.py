"""
mobis_loader.py
---------------
Loads the MOBIS / MOBIS-Covid Dataset (ETH Zurich / IVT).

Dataset: ~4,598 participants, 6.27M legs, Switzerland, Sep/2019-Jan/2023
Access: requires an agreement with IVT ETH Zurich (https://ivtmobis.ethz.ch)
Reference: Molloy et al. (2021), "Observed impacts of the Covid-19 first
  lockdown on mode use and travel patterns in Switzerland"

Expected structure under MOBIS_ROOT:
  tracking/legs.csv              -- 6.27M legs with mode + geometry
  tracking/activities.csv        -- activities with imputed_purpose
  enrichments/covid_externalities.csv -- external CO2/health/congestion cost

Coordinates: EPSG:2056 (Swiss LV95) -> converted to WGS84 internally.
GPS points: 3 per leg (start, mid, end) -- sliding window over consecutive
LEGS instead of dense GPS points.

Supported tasks:
  task='eta'     -- predicts [dist_meters, seconds] until the next mode change
                   (comparable to geolife_loader.py)
  task='purpose' -- predicts the imputed_purpose of the next activity
                   (8-class classification)

Interface: identical to geolife_loader.py.
  load_mobis(n_clients, seed, task) ->
    (samples, profiles, val_samples, val_profiles)

FedHME estimator: operates on (velocity, mode, cell_id) -- task-agnostic.
The co2/health/congestion fields are kept in the samples as extra fields
for potential future use; they are not consumed by this pipeline.

Note on WINDOW_SIZE: here WINDOW_SIZE refers to windows of consecutive
LEGS (not GPS points). Each leg is a complete trip segment.
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Literal

from pyproj import Transformer
from geo_utils import haversine, normalize_position
import config

# -- Constants ----------------------------------------------------

MOBIS_ROOT = getattr(config, "MOBIS_ROOT", "./data/mobis")

# Switzerland bounding box in WGS84 (covers the whole territory + margin)
MOBIS_BBOX = getattr(config, "MOBIS_BBOX", {
    "lat_min": 45.80, "lat_max": 47.80,
    "lon_min":  5.90, "lon_max": 10.65,
})

# Maximum number of participants to consider from the full pool
MOBIS_MAX_PARTICIPANTS = getattr(config, "MOBIS_MAX_PARTICIPANTS", 4500)

# MOBIS_WINDOW_SIZE and MOBIS_MIN_LEGS are intentionally NOT frozen as
# module-level constants at import time -- every call site reads
# config.MOBIS_WINDOW_SIZE directly instead. Reason: as plain module
# constants, a --mobis-window-size CLI override (applied in main() after
# argument parsing) would arrive too late -- the import would already have
# run and captured the old default.
#
# Background: W=10 was never calibrated for MOBIS -- it was inherited from
# GeoLife's global WINDOW_SIZE (a decision to keep both datasets at the
# same value). GeoLife's W=10 GPS points (~2-5s apart each) amounts to
# ~20-50s of context; MOBIS's W=10 legs (average duration 16.8min, not
# necessarily contiguous in time) amounts to ~1.7-2.5 DAYS of mobility --
# mixing contexts from different days/occasions in the same input vector,
# quite different from the original semantics (capturing the immediate
# kinematic pattern right before a transition). A plausible factor behind
# the training stagnation observed on MOBIS (train_loss/mae_dist_global
# without sustained improvement over 80 rounds). Client viability is not a
# limiting factor even at a smaller W (median legs/participant is well
# above the minimum even at W=3), so the choice of W can be guided purely
# by the task's semantics.

# Leg quality filters
MIN_LEG_DURATION_S = 30      # discard legs shorter than 30s
MIN_LEG_LENGTH_M   = 50      # discard legs shorter than 50m
MAX_LEG_DURATION_S = 7200    # discard legs longer than 2h (outlier)
MAX_LEG_LENGTH_M   = 200_000 # discard legs longer than 200km

# ETA filters (task='eta')
MIN_ETA_DIST_M   = 100       # minimum distance until a mode change
MIN_ETA_TIME_S   = 30        # minimum time until a mode change
MAX_ETA_DIST_M   = 15_000    # 15km -- calibrated for MAE/median ~ GeoLife (0.96 vs 0.82)
MAX_ETA_TIME_S   = 3_600     # 60min -- keeps 90% of samples, MAE/median=0.89 ~ GeoLife

# Versioned cache
MOBIS_SAMPLES_CACHE = getattr(
    config, "MOBIS_SAMPLES_CACHE",
    "./data/mobis_samples_v1_{task}_seed{seed}.pkl"
)

# -- Geographic (region) filter ----------------------------------
# Bounding box in LV95 (EPSG:2056, MOBIS's native coordinates, meters).
# Criterion: the leg's destination (end_x/end_y) falls inside the bbox.
ZH_LV95_E_MIN, ZH_LV95_E_MAX = 2_680_000, 2_695_000
ZH_LV95_N_MIN, ZH_LV95_N_MAX = 1_242_000, 1_254_000

REGION_FILTERS = {
    "zurich": {
        "e_min": ZH_LV95_E_MIN, "e_max": ZH_LV95_E_MAX,
        "n_min": ZH_LV95_N_MIN, "n_max": ZH_LV95_N_MAX,
    },
}

# -- Period cuts (covid/pre-covid/post-covid) -------------------
# Dates per Molloy et al. (2021) and the dataset's data-discovery slides
# (experiment phases, documented modal shift).
# period=None (default) loads the whole dataset, unfiltered -- same
# behavior as before this option existed.
MOBIS_COVID_START = getattr(config, "MOBIS_COVID_START", "2020-03-02")
# Start of the post-covid cut: 2022-01-01, matching the grouping used in
# the dataset's data-discovery materials (period: pre-covid / covid / post-covid).
MOBIS_POST_COVID_START = getattr(config, "MOBIS_POST_COVID_START", "2022-01-01")

MOBIS_PERIODS = {
    "pre_covid":  (None, MOBIS_COVID_START),
    "covid":      (MOBIS_COVID_START, MOBIS_POST_COVID_START),
    "post_covid": (MOBIS_POST_COVID_START, None),
    # "all" applies no filter -- included for symmetry, same result as period=None
    "all":        (None, None),
}


# EPSG:2056 -> WGS84 converter (reusable global instance)
_LV95_TO_WGS84 = Transformer.from_crs("EPSG:2056", "EPSG:4326",
                                        always_xy=True)

# Mapping from MOBIS modes to the project's standard modes.
# IMPORTANT: this is the CLOSED set of modes recognized by the pipeline
# (walk/bike/bus/car/subway/train/moto/moto_scooter/scooter -- the same
# ones as config.TRANSPORT_MODES). Modes outside this mapping are
# FILTERED OUT (their legs dropped), not folded into a fallback. Each
# mode with a distinct kinematic profile is kept separate to preserve
# the dataset's real heterogeneity (Motorbike != Car, MotorbikeScooter
# != Motorbike, Scooter/Etrottinett != Bike). TaxiUber/CarsharingMobility
# -> car (indistinguishable in speed/distance from the LSTM model's
# point of view).
#
# Dropped, with an explicit rationale:
#   Mode::Ski      -> seasonal, outside the urban mobility regime
#   Mode::Airplane -> 800+ km/h, entirely outside ITS scope
#   Mode::Boat     -> future extension (Swiss lake ferries)
#   Mode::Cablecar -> future extension (alpine cable cars)
_MOBIS_MODE_MAP = {
    # Modos originais
    "Mode::Walk":             "walk",
    "Mode::Bicycle":          "bike",
    "Mode::Ebicycle":         "bike",
    "Mode::Car":              "car",
    "Mode::Bus":              "bus",
    "Mode::Tram":             "bus",
    "Mode::LightRail":        "train",
    "Mode::Train":            "train",
    "Mode::RegionalTrain":    "train",
    "Mode::Subway":           "subway",
    "Mode::Ferry":            "bus",
    # Expanded modes -- distinct kinematic profiles, kept separate
    "Mode::Motorbike":        "moto",         # ~50-80 km/h, aggressive acceleration
    "Mode::MotorbikeScooter": "moto_scooter", # ~25-45 km/h, between bike and moto
    "Mode::Scooter":          "scooter",      # motorized scooter ~15-25 km/h
    "Mode::Etrottinett":      "scooter",      # electric scooter ~15-20 km/h
    # Merged into car -- indistinguishable in speed/distance
    "Mode::TaxiUber":           "car",
    "Mode::CarsharingMobility": "car",
}
# Modes outside this set are dropped -- see the filter right after
# legs_df is read, before viability is computed.


# Mapping from purpose to a numeric index
PURPOSE_CLASSES = [
    "Home", "Work", "Leisure", "Shopping",
    "Other", "Errand", "Assistance", "Education",
]
PURPOSE2IDX = {p: i for i, p in enumerate(PURPOSE_CLASSES)}
N_PURPOSE_CLASSES = len(PURPOSE_CLASSES)


# -- Main loading -----------------------------------------------

def load_mobis(
    n_clients: int = config.N_CLIENTS,
    seed:      int = config.EXPERIMENT_SEEDS[0],
    task:      Literal["eta", "purpose"] = "eta",
    period:    Literal["pre_covid", "covid", "post_covid", "all", None] = None,
    stratify_by_mode: bool = False,
    min_per_mode: int = 3,
    purpose_filter: list[str] | None = None,
    region_filter: Literal["zurich", None] = None,
    region_frac_min: float = 0.50,
) -> tuple[list[dict], dict, list[dict], dict]:
    """
    Loads MOBIS and returns:
      samples, profiles, val_samples, val_profiles

    Interface identical to load_geolife(), except for the `period`,
    `stratify_by_mode` and `min_per_mode` parameters (optional).

    task='eta'     -> target = [dist_to_transition_m, time_to_transition_s]
    task='purpose' -> target = [purpose_idx]  (integer, 0-7)

    period: cuts legs.csv by started_at before computing participant
    viability and selecting the n_clients sample. None (default) and
    "all" apply no filter -- same behavior as before this option
    existed. "pre_covid"/"covid"/"post_covid" use the cuts in
    MOBIS_PERIODS (see the constants above). The filter is applied
    BEFORE the MOBIS_MIN_LEGS/viability computation -- a participant
    only counts as viable if they have enough legs WITHIN the
    requested period, not in the whole dataset (avoids selecting a
    participant "viable" over the full 3-year span but with too few
    legs in the specific cut).

    stratify_by_mode: when True, the n_clients selection reserves
    min_per_mode slots for participants whose predominant mode (mode
    of legs.mode, mapped via _MOBIS_MODE_MAP) is each of the 6 standard
    modes (walk/bike/bus/car/subway/train), before filling the rest by
    random shuffle. Does not force the presence of ALL modes if the
    viable population doesn't have enough participants with that
    predominant mode (the reservation is best-effort, logged, and does
    not raise an error). Goal: measure whether the model generalizes
    across all modes, without favoring any particular estimator.
    Default False -- same behavior as before this option existed when
    unused.

    min_per_mode: minimum number of participants per predominant mode
    to reserve when stratify_by_mode=True. Default 3 -- the same
    minimum threshold used by calibrate_mode_rates() in estimator.py to
    move past DEFAULT_MODE_RATES (consistency between the two
    mechanisms).

    purpose_filter: list of MOBIS purposes to filter participants by
    their predominant destination activity. When provided, only
    participants whose most frequent imputed_purpose in activities is
    in the list are considered viable. Default None (no filter -- same
    behavior as before). Valid purposes:
    Home, Work, Leisure, Shopping, Other, Errand, Assistance, Education.
    Examples:
      purpose_filter=["Work", "Errand", "Education"]  -> local mobility
      purpose_filter=["Leisure", "Shopping"]           -> mixed/long-distance mobility
    The filter is applied AFTER computing viable_pids (MOBIS_MIN_LEGS)
    and BEFORE selecting n_clients -- it shrinks the candidate pool, not
    the legs of an already-selected participant (whole clients are kept
    or dropped, never partially filtered). Cache is versioned by the
    sorted list of purposes to avoid collisions between runs with
    different filters.

    region_filter: geographic cut by named region (currently only
    "zurich"). A viable participant is kept if the fraction of legs
    whose DESTINATION (end_x/end_y, LV95) falls inside the region's
    bounding box is >= region_frac_min (default 0.50). Default None (no
    filter -- same behavior as before). Applied at the same pipeline
    point as purpose_filter (after viability/shuffling, before
    selecting n_clients) -- it shrinks the candidate pool, not the legs
    of an already-selected participant (see REGION_FILTERS above).
    """
    _period_key = period if period is not None else "all"
    cache_path = MOBIS_SAMPLES_CACHE.format(task=task, seed=seed)
    if _period_key != "all":
        cache_path = cache_path.replace(".pkl", f"_{_period_key}.pkl")
    if stratify_by_mode:
        cache_path = cache_path.replace(".pkl", f"_stratmode{min_per_mode}.pkl")
    if purpose_filter:
        _pf_key = "_".join(sorted(purpose_filter))
        cache_path = cache_path.replace(".pkl", f"_purp{_pf_key}.pkl")
    if region_filter:
        cache_path = cache_path.replace(
            ".pkl", f"_region{region_filter}{region_frac_min:.2f}.pkl")

    if os.path.exists(cache_path):
        print(f"[MOBIS/{task}] Loading cache...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        samples      = data["samples"]
        profiles     = data["profiles"]
        val_samples  = data.get("val_samples", [])
        val_profiles = data.get("val_profiles", {})
        # Validate position_seq
        n_rounds = config.N_ROUNDS
        valid = all(
            len(p.get("position_seq", [])) >= n_rounds
            for p in profiles.values()
        )
        if valid:
            print(f"  {len(samples):,} training samples, "
                  f"{len(profiles)} profiles, "
                  f"{len(val_samples):,} val samples")
            return samples, profiles, val_samples, val_profiles
        print("  [Invalid cache] rebuilding...")
        os.remove(cache_path)

    # -- Load CSVs -----------------------------------------------
    legs_path  = _find_file("tracking/legs.csv")
    acts_path  = _find_file("tracking/activities.csv")
    ext_path   = _find_file("enrichments/covid_externalities.csv",
                             required=False)

    print(f"[MOBIS/{task}] Reading legs.csv...")
    # usecols reduces memory: 15 of ~30 columns -> ~0.8 GB vs ~1.5 GB
    _LEGS_COLS = [
        "participant_id", "leg_id", "trip_id",
        "started_at", "finished_at",
        "length", "duration", "mode",
        "start_x", "start_y", "mid_x", "mid_y", "end_x", "end_y",
        "implausible",
    ]
    legs_df = pd.read_csv(
        legs_path,
        parse_dates=["started_at", "finished_at"],
        low_memory=False,
        usecols=_LEGS_COLS,
    )
    # Normalize column names
    legs_df.columns = legs_df.columns.str.strip()

    print(f"[MOBIS/{task}] Reading activities.csv...")
    acts_df = pd.read_csv(
        acts_path,
        parse_dates=["started_at", "finished_at"],
        low_memory=False,
        usecols=["participant_id", "activity_id", "started_at",
                 "finished_at", "imputed_purpose", "x", "y"],
    )
    acts_df.columns = acts_df.columns.str.strip()

    # -- Load externalities (optional -- for extra fields) ------
    ext_by_trip: dict[int, dict] = {}
    if ext_path and os.path.exists(ext_path):
        print(f"[MOBIS/{task}] Reading covid_externalities.csv...")
        ext_df = pd.read_csv(ext_path, low_memory=False,
                             usecols=["user_id", "trip_id",
                                      "health", "co2", "congestion", "total"])
        ext_df.columns = ext_df.columns.str.strip()
        for row in ext_df.itertuples(index=False):
            ext_by_trip[int(row.trip_id)] = {
                "co2":        float(row.co2),
                "health":     float(row.health),
                "congestion": float(row.congestion),
                "ext_total":  float(row.total),
            }
        print(f"  {len(ext_by_trip):,} trips with external cost")

    # -- Period filter (pre-covid/covid/post-covid) ----------------
    # Applied BEFORE the viability computation (MOBIS_MIN_LEGS) -- a
    # participant is only considered viable if they have enough legs
    # WITHIN the requested cut, not in the whole dataset. legs_df and
    # acts_df are filtered by the same cut so activities outside the
    # period never mix with legs from the period (avoids leaking
    # mobility patterns across periods).
    if _period_key != "all":
        _start, _end = MOBIS_PERIODS[_period_key]
        _n_legs_before = len(legs_df)
        if _start is not None:
            legs_df = legs_df[legs_df["started_at"] >= _start]
            acts_df = acts_df[acts_df["started_at"] >= _start]
        if _end is not None:
            legs_df = legs_df[legs_df["started_at"] < _end]
            acts_df = acts_df[acts_df["started_at"] < _end]
        print(f"[MOBIS/{task}] Period filter '{_period_key}' "
              f"({_start or '...'} to {_end or '...'}): "
              f"{_n_legs_before:,} -> {len(legs_df):,} legs")

    # -- Unknown-mode filter (ALWAYS active, not optional) ---------
    # legs.csv has 25 detected modes, but _MOBIS_MODE_MAP only
    # recognizes 6 (walk/bike/bus/car/subway/train) plus a few expanded
    # ones. Modes outside the mapping (Airplane, Boat, Ski, Cablecar,
    # Other, ...) are DROPPED here -- they are never folded into "car"
    # via a fallback, since that would artificially inflate the "car"
    # count with incompatible kinematic profiles (flight, cable car)
    # and contaminate any reading of real modal representativeness,
    # including predominant-mode stratification (stratify_by_mode) and
    # delta_m calibration in estimator.py (calibrate_mode_rates).
    # Applied BEFORE viability -- a participant only counts recognized
    # legs.
    _n_legs_before_mode = len(legs_df)
    legs_df = legs_df[legs_df["mode"].isin(_MOBIS_MODE_MAP.keys())]
    _n_dropped_mode = _n_legs_before_mode - len(legs_df)
    # Read from config on every call (not a module-level constant frozen
    # at import time) -- allows overriding via --mobis-window-size.
    _mobis_min_legs = config.MOBIS_WINDOW_SIZE + 64
    if _n_dropped_mode > 0:
        _pct = 100 * _n_dropped_mode / _n_legs_before_mode
        print(f"[MOBIS/{task}] Unknown-mode filter: "
              f"{_n_legs_before_mode:,} -> {len(legs_df):,} legs "
              f"(-{_n_dropped_mode:,}, -{_pct:.2f}%, outside the "
              f"{len(_MOBIS_MODE_MAP)} standard modes)")
        print(f"[MOBIS/{task}] NOTE: this filter runs BEFORE the "
              f"viability computation (MOBIS_MIN_LEGS={_mobis_min_legs}) -- a "
              f"participant who had enough legs only by counting "
              f"Airplane/Boat/etc. can now fall below the minimum and "
              f"become unviable. This is expected, intentional behavior.")

    # -- Select participants ---------------------------------------
    all_pids   = legs_df["participant_id"].unique().tolist()
    # Count legs per participant to filter viable ones
    leg_counts = legs_df.groupby("participant_id").size()
    viable_pids = [p for p in all_pids
                   if leg_counts.get(p, 0) >= _mobis_min_legs]
    print(f"[MOBIS/{task}] {len(all_pids):,} total participants, "
          f"{len(viable_pids):,} viable (>= {_mobis_min_legs} legs)")

    rng = np.random.default_rng(seed)
    viable_arr = np.array(viable_pids)
    rng.shuffle(viable_arr)
    viable_pids = viable_arr.tolist()

    n_val  = getattr(config, "N_VAL_CLIENTS", 5)
    needed = n_clients + n_val

    # -- Predominant-purpose filter (whole clients kept or dropped) --
    # Applied AFTER viability and shuffling, BEFORE selecting n_clients.
    # Computes the most frequent imputed_purpose in each viable
    # participant's activities. Only participants whose dominant
    # purpose is in purpose_filter are kept in the pool.
    # Note: the filter uses the full acts_df (not period-cut) when
    # period=None; it uses the period-cut acts_df when period!=None --
    # the dominant purpose is evaluated over the same time window as
    # the legs.
    if purpose_filter:
        _valid_purposes = set(PURPOSE_CLASSES)
        _invalid = [p for p in purpose_filter if p not in _valid_purposes]
        if _invalid:
            raise ValueError(
                f"[MOBIS] purpose_filter contains invalid purposes: "
                f"{_invalid}. Valid: {PURPOSE_CLASSES}"
            )
        _viable_set = set(viable_pids)
        _dominant_purpose = (
            acts_df[acts_df["participant_id"].isin(_viable_set)]
            .groupby("participant_id")["imputed_purpose"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        )
        _n_before = len(viable_pids)
        viable_pids = [
            p for p in viable_pids
            if _dominant_purpose.get(p) in set(purpose_filter)
        ]
        print(f"[MOBIS/{task}] Purpose filter {purpose_filter}: "
              f"{_n_before:,} -> {len(viable_pids):,} viable participants "
              f"({_n_before - len(viable_pids):,} excluded for a "
              f"predominant purpose outside the filter)")

    # -- Geographic filter by region (e.g. Zurich) -------------------
    # Applied at the same point as purpose_filter: after
    # viability/shuffling, before selecting n_clients. Criterion:
    # fraction of the participant's legs whose DESTINATION (end_x/end_y,
    # LV95) falls inside the region's bbox is >= region_frac_min. Uses
    # raw LV95 coordinates (not the WGS84-converted version from
    # _parse_legs) -- simpler, and avoids depending on that conversion
    # at this point in the pipeline.
    if region_filter:
        if region_filter not in REGION_FILTERS:
            raise ValueError(
                f"[MOBIS] unknown region_filter: {region_filter!r}. "
                f"Valid: {list(REGION_FILTERS.keys())}"
            )
        _bbox = REGION_FILTERS[region_filter]
        _viable_set = set(viable_pids)
        _legs_viable = legs_df[legs_df["participant_id"].isin(_viable_set)]
        _in_region = (
            (_legs_viable["end_x"] >= _bbox["e_min"]) &
            (_legs_viable["end_x"] <= _bbox["e_max"]) &
            (_legs_viable["end_y"] >= _bbox["n_min"]) &
            (_legs_viable["end_y"] <= _bbox["n_max"])
        )
        _total_by_pid  = _legs_viable.groupby("participant_id").size()
        _region_by_pid = _legs_viable[_in_region].groupby(
            "participant_id").size()
        _frac_by_pid = (_region_by_pid.reindex(_total_by_pid.index, fill_value=0)
                         / _total_by_pid)
        _n_before = len(viable_pids)
        viable_pids = [
            p for p in viable_pids
            if _frac_by_pid.get(p, 0.0) >= region_frac_min
        ]
        print(f"[MOBIS/{task}] Region filter '{region_filter}' "
              f"(dest_frac>={region_frac_min:.0%}): "
              f"{_n_before:,} -> {len(viable_pids):,} viable participants "
              f"({_n_before - len(viable_pids):,} excluded outside the region)")

    if not stratify_by_mode:
        selected = viable_pids[:needed]
        print(f"[MOBIS/{task}] Using {len(selected)} participants "
              f"({n_clients} train + {n_val} val) -- plain shuffle, "
              f"no mode stratification")
    else:
        # -- Predominant-mode stratification --------------------
        # For each viable participant, computes the predominant mode
        # (mode of legs.mode, mapped via _MOBIS_MODE_MAP). Reserves
        # min_per_mode slots for each of the 6 standard modes (in the
        # order of the already-shuffled list, preserving per-seed
        # determinism), then fills the rest of the `needed` slots by
        # plain shuffle (excluding those already reserved). Best-effort:
        # if a mode doesn't have min_per_mode viable participants
        # available, reserves whatever is available and logs it --
        # does not raise an error (see this function's docstring).
        _mode_of_leg = legs_df["mode"].map(_MOBIS_MODE_MAP)
        _predominant = (
            legs_df.assign(_mode_mapped=_mode_of_leg)
            .groupby("participant_id")["_mode_mapped"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        )
        _viable_set = set(viable_pids)
        _by_mode: dict[str, list] = {m: [] for m in _MOBIS_MODE_MAP.values()}
        for pid in viable_pids:  # already shuffled -- preserves determinism
            m = _predominant.get(pid)
            if m in _by_mode:
                _by_mode[m].append(pid)

        reserved: list = []
        reserved_set: set = set()
        print(f"[MOBIS/{task}] Predominant-mode stratification "
              f"(min_per_mode={min_per_mode}):")
        for m in sorted(set(_MOBIS_MODE_MAP.values())):
            pool = _by_mode.get(m, [])
            take = pool[:min_per_mode]
            reserved.extend(take)
            reserved_set.update(take)
            _status = "OK" if len(take) >= min_per_mode else "INSUFFICIENT"
            print(f"    {m:8s}: {len(pool):4d} viable candidates, "
                  f"{len(take)} reserved [{_status}]")

        remaining = [p for p in viable_pids if p not in reserved_set]
        selected = (reserved + remaining)[:needed]
        print(f"[MOBIS/{task}] Using {len(selected)} participants "
              f"({n_clients} train + {n_val} val) -- "
              f"{len(reserved)} reserved by mode + "
              f"{len(selected) - min(len(reserved), len(selected))} "
              f"by plain shuffle")

    # -- Pre-process activities per participant --------------------
    selected_set = set(selected)   # precompute outside the loop
    acts_by_pid: dict[str, pd.DataFrame] = {}
    for pid, grp in acts_df.groupby("participant_id"):
        if pid in selected_set:
            acts_by_pid[pid] = grp.sort_values("started_at").reset_index(
                drop=True)

    # -- Extract samples per participant -----------------------------
    all_samples  = []
    val_samples  = []
    profiles     = {}
    val_profiles = {}
    cid_seq = val_seq = 0

    for i, pid in enumerate(selected):
        pid_legs = legs_df[legs_df["participant_id"] == pid].sort_values(
            "started_at").reset_index(drop=True)
        pid_acts = acts_by_pid.get(pid, pd.DataFrame())

        print(f"  Participant {i:03d} ({pid}) -- "
              f"{len(pid_legs)} legs...", end=" ")

        # Convert legs to a list of dicts in WGS84
        leg_list = _parse_legs(pid_legs, ext_by_trip)

        if len(leg_list) < _mobis_min_legs:
            print("too few legs, skipping")
            continue

        # Extract samples according to the task
        if task == "eta":
            p_samples = _extract_eta_samples(leg_list, pid)
        elif task == "eta_purpose":
            p_samples = _extract_eta_purpose_samples(leg_list, pid_acts, pid)
        else:  # purpose
            p_samples = _extract_purpose_samples(
                leg_list, pid_acts, pid)

        if not p_samples:
            print("0 samples, skipping")
            continue

        # Build the profile for the estimator
        profile = _build_mobis_profile(leg_list, config.N_ROUNDS, pid)

        if val_seq < n_val:
            val_samples.extend(p_samples)
            val_profiles[val_seq] = profile
            print(f"{len(p_samples):,} samples [val]")
            val_seq += 1
        else:
            all_samples.extend(p_samples)
            profiles[cid_seq] = profile
            print(f"{len(p_samples):,} samples [train]")
            cid_seq += 1

    print(f"\n[MOBIS/{task}] Train : {len(all_samples):,} samples -- "
          f"{len(profiles)} clients")
    print(f"[MOBIS/{task}] Val   : {len(val_samples):,} samples -- "
          f"{len(val_profiles)} clients")

    if task in ("eta", "eta_purpose"):
        _print_eta_stats(all_samples)
    else:
        _print_purpose_stats(all_samples)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)),
                exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({
            "samples":      all_samples,
            "profiles":     profiles,
            "val_samples":  val_samples,
            "val_profiles": val_profiles,
        }, f)

    return all_samples, profiles, val_samples, val_profiles


# -- Legs parser ----------------------------------------------

def _parse_legs(
    legs_df: pd.DataFrame,
    ext_by_trip: dict,
) -> list[dict]:
    """
    Converts a participant's legs DataFrame into a list of dicts.
    Converts EPSG:2056 coordinates to WGS84. Filters outliers.
    """
    leg_list = []
    bbox = MOBIS_BBOX

    for row in legs_df.itertuples(index=False):
        # Basic quality filters
        length   = float(row.length)   if not pd.isna(row.length)   else 0.0
        duration = float(row.duration) if not pd.isna(row.duration) else 0.0

        if (duration < MIN_LEG_DURATION_S
                or duration > MAX_LEG_DURATION_S
                or length   < MIN_LEG_LENGTH_M
                or length   > MAX_LEG_LENGTH_M):
            continue

        # Implausible filter -- robust to both bool and string
        impl = getattr(row, "implausible", False)
        if str(impl).lower() == "true":
            continue

        # Convert MOBIS modes -> standard. At this point legs_df has
        # already been filtered (in load_mobis, before calling
        # _parse_legs) to only contain modes in _MOBIS_MODE_MAP -- this
        # check is defensive: it raises a clear error instead of
        # silently falling back to "car" if the earlier filter is ever
        # removed in a future edit (no fallback to car here, since that
        # would contaminate the real automobile count and any
        # predominant-mode stratification).
        raw_mode = str(getattr(row, "mode", None))
        if raw_mode not in _MOBIS_MODE_MAP:
            raise ValueError(
                f"Mode '{raw_mode}' outside _MOBIS_MODE_MAP reached "
                f"_parse_legs -- the mode filter in load_mobis() should "
                f"have dropped this leg earlier. Do not fall back to "
                f"'car' here (it would contaminate the count)."
            )
        mode     = _MOBIS_MODE_MAP[raw_mode]
        mode_idx = config.MODE2IDX.get(mode, 3)

        # Convert start/end from EPSG:2056 -> WGS84
        sx, sy = float(row.start_x), float(row.start_y)
        ex, ey = float(row.end_x),   float(row.end_y)
        mx, my = float(row.mid_x),   float(row.mid_y)

        s_lon, s_lat = _LV95_TO_WGS84.transform(sx, sy)
        e_lon, e_lat = _LV95_TO_WGS84.transform(ex, ey)
        m_lon, m_lat = _LV95_TO_WGS84.transform(mx, my)

        # Filter out legs outside Switzerland
        if not (bbox["lat_min"] <= s_lat <= bbox["lat_max"]
                and bbox["lon_min"] <= s_lon <= bbox["lon_max"]):
            continue

        # Average speed of the leg (km/h)
        velocity = (length / max(duration, 1.0)) * 3.6

        # Position normalization
        norm_s_lat, norm_s_lon = normalize_position(
            s_lat, s_lon,
            lat_min=bbox["lat_min"], lat_max=bbox["lat_max"],
            lon_min=bbox["lon_min"], lon_max=bbox["lon_max"],
        )
        norm_e_lat, norm_e_lon = normalize_position(
            e_lat, e_lon,
            lat_min=bbox["lat_min"], lat_max=bbox["lat_max"],
            lon_min=bbox["lon_min"], lon_max=bbox["lon_max"],
        )

        # Normalized spatial displacement (start -> end)
        dx = norm_e_lon - norm_s_lon
        dy = norm_e_lat - norm_s_lat

        # Timestamp as a timezone-aware datetime
        ts = row.started_at
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        hour_norm = ts.hour / 23.0
        weekday   = ts.weekday() / 6.0  # 0=Mon, 1=Sun normalized

        # trip_id to join with externalities
        trip_id = int(row.trip_id) if not pd.isna(row.trip_id) else -1
        ext = ext_by_trip.get(trip_id, {
            "co2": 0.0, "health": 0.0, "congestion": 0.0, "ext_total": 0.0
        })

        leg_list.append({
            # Metadata
            "leg_id":    int(row.leg_id),
            "trip_id":   trip_id,
            "timestamp": ts,
            # WGS84 geometry
            "start_lat": s_lat, "start_lon": s_lon,
            "end_lat":   e_lat, "end_lon":   e_lon,
            "mid_lat":   m_lat, "mid_lon":   m_lon,
            # Normalized positions
            "norm_lat":  norm_s_lat,
            "norm_lon":  norm_s_lon,
            # Leg metrics
            "length":    length,
            "duration":  duration,
            "velocity":  velocity,
            "dx":        float(dx),
            "dy":        float(dy),
            "hour_norm": float(hour_norm),
            "weekday":   float(weekday),
            # Mode
            "mode":      mode,
            "mode_idx":  mode_idx,
            # External cost (extra fields, not consumed by this pipeline)
            "co2":       float(ext["co2"]),
            "health":    float(ext["health"]),
            "congestion":float(ext["congestion"]),
            "ext_total": float(ext["ext_total"]),
        })

    return leg_list


# -- ETA sample extraction --------------------------------------

def _extract_eta_samples(
    leg_list: list[dict],
    pid:      str,
) -> list[dict]:
    """
    Sliding window over consecutive legs.
    Target: [dist_to_transition_m, time_to_transition_s]

    A "mode transition" occurs when the next leg's mode differs from
    the window's last leg mode. Accumulates distance and time from the
    window's last leg up to the next mode change.
    """
    W       = config.MOBIS_WINDOW_SIZE  # read on every call, never frozen
    samples = []
    n       = len(leg_list)
    bbox    = MOBIS_BBOX

    for i in range(n - W):
        window = leg_list[i: i + W]
        current_mode = window[-1]["mode"]

        # Find the next mode transition after the window
        eta_dist = 0.0
        eta_time = 0.0
        found    = False

        for j in range(i + W, n):
            leg_j = leg_list[j]
            eta_dist += leg_j["length"]
            eta_time += leg_j["duration"]

            if leg_j["mode"] != current_mode:
                found = True
                break

        if not found:
            continue  # no transition until the end -- skip

        # Target quality filters
        if (eta_dist < MIN_ETA_DIST_M
                or eta_dist > MAX_ETA_DIST_M
                or eta_time < MIN_ETA_TIME_S
                or eta_time > MAX_ETA_TIME_S):
            continue

        # Window feature vector
        features = _window_to_features(window)

        samples.append({
            "features":  features,
            "targets":   np.array([eta_dist, eta_time], dtype=np.float32),
            "mode":      current_mode,
            "mode_to":   leg_list[i + W]["mode"],
            "lat":       window[-1]["start_lat"],
            "lon":       window[-1]["start_lon"],
            "norm_lat":  window[-1]["norm_lat"],
            "norm_lon":  window[-1]["norm_lon"],
            "user_id":   pid,
            "timestamp": window[-1]["timestamp"],
            # Extra fields, not consumed by this pipeline
            "co2":       float(np.mean([l["co2"]       for l in window])),
            "health":    float(np.mean([l["health"]    for l in window])),
            "congestion":float(np.mean([l["congestion"] for l in window])),
        })

    return samples


# -- ETA sample extraction with purpose as a feature ----------

def _extract_eta_purpose_samples(
    leg_list: list[dict],
    acts_df:  pd.DataFrame,
    pid:      str,
) -> list[dict]:
    """
    Variant of _extract_eta_samples that adds purpose_idx as a 10th
    feature on the window's last leg.

    The purpose is that of the activity immediately BEFORE the start
    of the window -- it indicates what the participant was doing
    before starting this sequence of legs.

    Returns samples with features of shape (W, 10).
    """
    W       = config.MOBIS_WINDOW_SIZE  # read on every call, never frozen
    samples = []
    n       = len(leg_list)
    bbox    = MOBIS_BBOX

    if acts_df is None or acts_df.empty:
        # Fallback: no activities available, use the plain _extract_eta_samples
        return _extract_eta_samples(leg_list, pid)

    # Pre-process activities: keep only known purposes
    acts_valid = acts_df[
        acts_df["imputed_purpose"].isin(PURPOSE2IDX)
    ].copy()

    def _ensure_aware(ts):
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts

    if not acts_valid.empty:
        act_times    = [_ensure_aware(r.started_at)
                        for r in acts_valid.itertuples(index=False)]
        act_purposes = [str(r.imputed_purpose)
                        for r in acts_valid.itertuples(index=False)]
    else:
        act_times, act_purposes = [], []

    for i in range(n - W):
        window       = leg_list[i: i + W]
        current_mode = window[-1]["mode"]

        # ETA: distance and time until the next mode transition
        eta_dist = 0.0
        eta_time = 0.0
        found    = False
        for j in range(i + W, n):
            leg_j = leg_list[j]
            eta_dist += leg_j["length"]
            eta_time += leg_j["duration"]
            if leg_j["mode"] != current_mode:
                found = True
                break

        if not found:
            continue

        if (eta_dist < MIN_ETA_DIST_M or eta_dist > MAX_ETA_DIST_M
                or eta_time < MIN_ETA_TIME_S or eta_time > MAX_ETA_TIME_S):
            continue

        # Purpose of the activity immediately BEFORE the window start
        window_start_ts = window[0]["timestamp"]
        purpose_idx = None
        if act_times:
            # Find the most recent activity that ended before the window
            for k in range(len(act_times) - 1, -1, -1):
                if act_times[k] <= window_start_ts:
                    purpose_str = act_purposes[k]
                    purpose_idx = PURPOSE2IDX.get(purpose_str)
                    break

        if purpose_idx is None:
            # No known prior activity -- default to 0 (Home)
            purpose_idx = 0

        features = _window_to_features(window, purpose_idx=purpose_idx)

        samples.append({
            "features":  features,
            "targets":   np.array([eta_dist, eta_time], dtype=np.float32),
            "mode":      current_mode,
            "mode_to":   leg_list[i + W]["mode"],
            "lat":       window[-1]["start_lat"],
            "lon":       window[-1]["start_lon"],
            "norm_lat":  window[-1]["norm_lat"],
            "norm_lon":  window[-1]["norm_lon"],
            "user_id":   pid,
            "timestamp": window[-1]["timestamp"],
            "purpose_idx": purpose_idx,
        })

    return samples


# -- Purpose sample extraction ---------------------------------

def _extract_purpose_samples(
    leg_list: list[dict],
    acts_df:  pd.DataFrame,
    pid:      str,
) -> list[dict]:
    """
    Sliding window over consecutive legs.
    Target: [purpose_idx] -- purpose of the activity immediately after
    the window's last leg.

    The next activity is identified by the closest started_at after
    the window's last leg finished_at.
    """
    W       = config.MOBIS_WINDOW_SIZE  # read on every call, never frozen
    samples = []
    n       = len(leg_list)

    if acts_df.empty:
        return samples

    # Pre-process activities: keep only known purposes
    acts_valid = acts_df[
        acts_df["imputed_purpose"].isin(PURPOSE2IDX)
    ].copy()
    if acts_valid.empty:
        return samples

    # Convert activities' started_at to timezone-aware datetimes
    def _ensure_aware(ts):
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts

    act_times    = [_ensure_aware(r.started_at)
                    for r in acts_valid.itertuples(index=False)]
    act_purposes = [str(r.imputed_purpose)
                    for r in acts_valid.itertuples(index=False)]

    for i in range(n - W):
        window   = leg_list[i: i + W]
        last_leg = window[-1]

        # Timestamp of the last leg's end (timedelta imported at the top)
        last_ts     = last_leg["timestamp"]
        finished_ts = last_ts + timedelta(seconds=last_leg["duration"])

        # Find the closest activity after finished_ts
        purpose_idx = None
        for k, act_ts in enumerate(act_times):
            if act_ts >= finished_ts:
                purpose_str = act_purposes[k]
                purpose_idx = PURPOSE2IDX.get(purpose_str)
                break

        if purpose_idx is None:
            continue

        features = _window_to_features(window)

        samples.append({
            "features":  features,
            "targets":   np.array([purpose_idx], dtype=np.int64),
            "mode":      last_leg["mode"],
            "mode_to":   last_leg["mode"],  # same mode (different task)
            "lat":       last_leg["start_lat"],
            "lon":       last_leg["start_lon"],
            "norm_lat":  last_leg["norm_lat"],
            "norm_lon":  last_leg["norm_lon"],
            "user_id":   pid,
            "timestamp": last_leg["timestamp"],
            # Extra fields
            "co2":       float(np.mean([l["co2"]       for l in window])),
            "health":    float(np.mean([l["health"]    for l in window])),
            "congestion":float(np.mean([l["congestion"] for l in window])),
        })

    return samples


# -- Window feature vector ------------------------------------

MAX_GAP_S = 86400.0  # normalization cap for the gap between legs (24h)


def _window_to_features(
    window:      list[dict],
    purpose_idx: int = None,   # None = no purpose (mobis_eta)
                               # int  = purpose of the last leg (mobis_eta_purpose)
) -> np.ndarray:
    """
    Converts a window of W legs into a feature vector.

    Three layout choices here (not just appending columns at the end)
    were driven by a review of the MOBIS feature engineering:
      (a) mode moved from a normalized ordinal index (1 column, a false
          ordinality between modes with no real order) to one-hot
          (N_MODES columns);
      (b) hour/weekday moved from linear encoding (a false discontinuity
          at the day/week boundary, e.g. 23h->0.9999, 0h->0.0) to
          sin/cos (2 columns each, no discontinuity);
      (c) `gap` (time since the previous leg) was corrected: it used to
          be computed as started_at[i] - started_at[i-1] (start to
          start), which mixes the previous leg's DURATION with the real
          idle time between them -- a 30min leg followed by a 5min wait
          showed up as a "35min gap". It is now computed as
          started_at[i] - finished_at[i-1] (end of the previous leg, =
          started_at + duration), the actual idle gap. It also gained
          `log1p` (the gap distribution is heavy-tailed -- a few
          overnight gaps, capped at MAX_GAP_S -- linear normalization
          crushed the resolution in the common range).

    Standard mode (purpose_idx=None): 3 + N_MODES + 14 features per leg
      [0] velocity_norm         velocity / 200 km/h
      [1] length_norm           length / MAX_LEG_LENGTH_M
      [2] duration_norm         duration / MAX_LEG_DURATION_S
      [3 .. 3+N_MODES-1]        mode one-hot -- 1.0 at position mode_idx,
                                 0.0 elsewhere
      [3+N_MODES+0] hour_sin    sin(2*pi*hour/24)
      [3+N_MODES+1] hour_cos    cos(2*pi*hour/24)
      [3+N_MODES+2] dx          normalized lon displacement [-1,1]
      [3+N_MODES+3] dy          normalized lat displacement [-1,1]
      [3+N_MODES+4] bearing_norm    arctan2(dy, dx) / pi  [-1,1]
      [3+N_MODES+5] dist_accum_norm cumulative distance / MAX_LEG_LENGTH_M
      [3+N_MODES+6] weekday_sin sin(2*pi*weekday/7)
      [3+N_MODES+7] weekday_cos cos(2*pi*weekday/7)
      [3+N_MODES+8] gap_log_norm    log1p(real idle gap, s) / log1p(MAX_GAP_S),
                                 0.0 on the window's first leg
      [3+N_MODES+9] same_trip_prev  1.0 if trip_id == the previous leg's
                                 trip_id in the window, 0.0 on the first
                                 leg or if the trip changed
      [3+N_MODES+10] trip_pos_norm  consecutive legs of the SAME trip
                                 seen in the window (resets on every
                                 trip_id change) / W
      [3+N_MODES+11] mode_run_len_norm     same as [10] but keyed by
                                 `mode` instead of `trip_id`
      [3+N_MODES+12] mode_dist_accum_norm  cumulative distance since the
                                 last mode change visible in the window,
                                 / MAX_LEG_LENGTH_M
      [3+N_MODES+13] mode_dur_accum_norm   cumulative duration since the
                                 last mode change visible in the window,
                                 / MAX_LEG_DURATION_S

    With N_MODES=9 (config.TRANSPORT_MODES): 3+9+14 = 26 standard features.

    Purpose mode (purpose_idx=int): 27 features -- same as standard plus
    the normalized purpose_idx in the last index, only on the window's
    last leg.
    """
    W         = len(window)
    N_MODES   = len(config.MODE2IDX)
    N_BASE    = 3 + N_MODES + 14   # velocity/length/duration + one-hot + 14 remaining
    N_FEAT    = N_BASE + 1 if purpose_idx is not None else N_BASE
    feats     = np.zeros((W, N_FEAT), dtype=np.float32)

    # Indices of the post-one-hot blocks (computed once, outside the loop)
    IDX_HOUR_SIN   = 3 + N_MODES + 0
    IDX_HOUR_COS   = 3 + N_MODES + 1
    IDX_DX         = 3 + N_MODES + 2
    IDX_DY         = 3 + N_MODES + 3
    IDX_BEARING    = 3 + N_MODES + 4
    IDX_DIST_ACCUM = 3 + N_MODES + 5
    IDX_WD_SIN     = 3 + N_MODES + 6
    IDX_WD_COS     = 3 + N_MODES + 7
    IDX_GAP        = 3 + N_MODES + 8
    IDX_SAME_TRIP  = 3 + N_MODES + 9
    IDX_TRIP_POS   = 3 + N_MODES + 10
    IDX_MODE_RUN   = 3 + N_MODES + 11
    IDX_MODE_DIST  = 3 + N_MODES + 12
    IDX_MODE_DUR   = 3 + N_MODES + 13
    IDX_PURPOSE    = N_BASE

    dist_accum      = 0.0
    trip_run_len    = 0    # counter of consecutive legs of the same trip in the window
    mode_run_len    = 0    # same, keyed by mode
    mode_dist_accum = 0.0  # cumulative distance since the last mode change
    mode_dur_accum  = 0.0  # cumulative duration since the last mode change
    for k, leg in enumerate(window):
        dist_accum += leg["length"]

        feats[k, 0] = float(np.clip(leg["velocity"] / 200.0, 0, 1))
        feats[k, 1] = float(np.clip(leg["length"]   / MAX_LEG_LENGTH_M, 0, 1))
        feats[k, 2] = float(np.clip(leg["duration"] / MAX_LEG_DURATION_S, 0, 1))

        # [3 .. 3+N_MODES-1] mode one-hot
        feats[k, 3 + leg["mode_idx"]] = 1.0

        # hour_sin/hour_cos
        hour_angle = 2.0 * np.pi * (leg["hour_norm"] * 23.0 / 24.0)
        feats[k, IDX_HOUR_SIN] = float(np.sin(hour_angle))
        feats[k, IDX_HOUR_COS] = float(np.cos(hour_angle))

        feats[k, IDX_DX] = float(np.clip(leg["dx"], -1, 1))
        feats[k, IDX_DY] = float(np.clip(leg["dy"], -1, 1))
        bearing = np.arctan2(leg["dy"], leg["dx"]) / np.pi
        feats[k, IDX_BEARING] = float(np.clip(bearing, -1, 1))
        feats[k, IDX_DIST_ACCUM] = float(np.clip(dist_accum / MAX_LEG_LENGTH_M, 0, 1))

        # weekday_sin/weekday_cos
        wd_angle = 2.0 * np.pi * (leg.get("weekday", 0.0) * 6.0 / 7.0)
        feats[k, IDX_WD_SIN] = float(np.sin(wd_angle))
        feats[k, IDX_WD_COS] = float(np.cos(wd_angle))

        # gap/same_trip_prev/trip_pos_norm -- depend on the previous leg IN THE WINDOW (k-1)
        if k == 0:
            feats[k, IDX_GAP]       = 0.0   # no previous leg visible in the window
            feats[k, IDX_SAME_TRIP] = 0.0
            trip_run_len = 0
        else:
            prev = window[k - 1]
            # Real idle gap = current leg's start - previous leg's END
            # (not start-to-start, which mixed in the previous leg's
            # duration -- see the docstring). log1p for the gap
            # distribution's heavy tail.
            prev_finished_at = prev["timestamp"] + timedelta(seconds=prev["duration"])
            gap_s = (leg["timestamp"] - prev_finished_at).total_seconds()
            gap_log = np.log1p(max(gap_s, 0.0))
            feats[k, IDX_GAP] = float(np.clip(gap_log / np.log1p(MAX_GAP_S), 0, 1))
            feats[k, IDX_SAME_TRIP] = 1.0 if leg["trip_id"] == prev["trip_id"] else 0.0
            if leg["trip_id"] != prev["trip_id"]:
                trip_run_len = 0
        trip_run_len += 1
        feats[k, IDX_TRIP_POS] = float(np.clip(trip_run_len / W, 0, 1))

        # per-mode run-length -- keyed by `mode` instead of `trip_id`
        if k == 0 or leg["mode"] != window[k - 1]["mode"]:
            mode_run_len    = 0
            mode_dist_accum = 0.0
            mode_dur_accum  = 0.0
        mode_run_len    += 1
        mode_dist_accum += leg["length"]
        mode_dur_accum  += leg["duration"]
        feats[k, IDX_MODE_RUN]  = float(np.clip(mode_run_len / W, 0, 1))
        feats[k, IDX_MODE_DIST] = float(np.clip(mode_dist_accum / MAX_LEG_LENGTH_M, 0, 1))
        feats[k, IDX_MODE_DUR]  = float(np.clip(mode_dur_accum / MAX_LEG_DURATION_S, 0, 1))

        # normalized purpose of the last leg -- mobis_eta_purpose only
        if purpose_idx is not None and k == W - 1:
            feats[k, IDX_PURPOSE] = float(purpose_idx) / max(N_PURPOSE_CLASSES - 1, 1)

    return feats  # shape (W, N_FEAT) -- dataset.py expects (W, F)


# -- Building the profile for the estimator -----------------------

def _build_mobis_profile(
    leg_list: list[dict],
    n_rounds: int,
    pid:      str,
) -> dict:
    """
    Builds a mobility profile for the FedHME estimator.
    Interface identical to geolife_loader.py's _build_profile().

    position_seq: n_rounds entries with {lat, lon, norm_lat, norm_lon, mode}
    velocity_seq: n_rounds speeds in km/h
    """
    if not leg_list:
        bbox    = MOBIS_BBOX
        lat_c   = (bbox["lat_min"] + bbox["lat_max"]) / 2
        lon_c   = (bbox["lon_min"] + bbox["lon_max"]) / 2
        pos_def = {"lat": lat_c, "lon": lon_c,
                   "norm_lat": 0.5, "norm_lon": 0.5,
                   "mode": "car"}
        return {
            "position_seq":   [pos_def] * n_rounds,
            "velocity_seq":   [20.0]    * n_rounds,
            "mean_velocity":  20.0,
            "mobility_group": "media",
            "user_id":        pid,
            "type":           "real_mobis",
        }

    step = max(len(leg_list) // n_rounds, 1)
    position_seq = []
    velocity_seq = []

    for r in range(n_rounds):
        idx = min(r * step, len(leg_list) - 1)
        leg = leg_list[idx]
        position_seq.append({
            "lat":      leg["start_lat"],
            "lon":      leg["start_lon"],
            "norm_lat": leg["norm_lat"],
            "norm_lon": leg["norm_lon"],
            "mode":     leg["mode"],
        })
        velocity_seq.append(float(min(leg["velocity"], 200.0)))

    # Guarantees exactly n_rounds entries
    while len(position_seq) < n_rounds:
        position_seq.append(position_seq[-1])
        velocity_seq.append(velocity_seq[-1])

    mean_v = float(np.mean(velocity_seq))
    return {
        "position_seq":   position_seq,
        "velocity_seq":   velocity_seq,
        "mean_velocity":  mean_v,
        "mobility_group": _classify_mobility(mean_v),
        "user_id":        pid,
        "type":           "real_mobis",
    }


# -- Utilities ---------------------------------------------------

def _find_file(
    relative_path: str,
    required:      bool = True,
) -> str | None:
    """Locates a file inside MOBIS_ROOT."""
    full = os.path.join(MOBIS_ROOT, relative_path)
    if os.path.exists(full):
        return full
    if required:
        raise FileNotFoundError(
            f"[MOBIS] File not found: {full}\n"
            f"  Adjust MOBIS_ROOT in config.py\n"
            f"  Current value: {MOBIS_ROOT}"
        )
    return None


def _classify_mobility(mean_v: float) -> str:
    """Classifies the mobility group by average speed."""
    if mean_v < 5.0:
        return "lenta"       # predominantly walk
    if mean_v < 25.0:
        return "media"       # urban mix
    return "rapida"          # predominantly car/train


def _print_eta_stats(samples: list[dict]) -> None:
    """Prints ETA target statistics."""
    if not samples:
        return
    dists = [s["targets"][0] for s in samples]
    times = [s["targets"][1] for s in samples]
    modes = [s["mode"] for s in samples]
    from collections import Counter
    mode_counts = Counter(modes)
    print(f"[MOBIS/eta] Dist: "
          f"median={np.median(dists)/1000:.1f}km, "
          f"p95={np.percentile(dists,95)/1000:.1f}km")
    print(f"[MOBIS/eta] Time: "
          f"median={np.median(times)/60:.1f}min, "
          f"p95={np.percentile(times,95)/60:.1f}min")
    print(f"[MOBIS/eta] Modes: {dict(mode_counts.most_common())}")


def _print_purpose_stats(samples: list[dict]) -> None:
    """Prints the distribution of purpose classes."""
    if not samples:
        return
    from collections import Counter
    idx_counts = Counter(int(s["targets"][0]) for s in samples)
    idx2purpose = {v: k for k, v in PURPOSE2IDX.items()}
    total = sum(idx_counts.values())
    print(f"[MOBIS/purpose] Class distribution ({total:,} samples):")
    for idx in sorted(idx_counts):
        name = idx2purpose.get(idx, "?")
        pct  = 100 * idx_counts[idx] / total
        print(f"  {idx} {name:<12}: {idx_counts[idx]:7,} ({pct:.1f}%)")
