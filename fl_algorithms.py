"""
fl_algorithms.py
----------------
Implementation of FedProx, SCAFFOLD and FedNova as baselines and as
algorithms with the hybrid estimator plugin.

Each algorithm exists in two versions:
  - Pure: standard behavior from the literature
  - +Hybrid: the server uses the estimator for client selection, epoch
    allocation, and algorithm-specific adjustments:
    - FedProx: dynamic mu proportional to h_hat
    - SCAFFOLD: bypass (plain SGD) when h_hat < threshold
    - FedNova: dynamic epochs proportional to h_hat

EstimatorMixin injects the estimator logic into the strategies without
duplicating code.

References:
  FedProx: Li et al., MLSys 2020
  SCAFFOLD: Karimireddy et al., ICML 2020
  FedNova: Wang et al., NeurIPS 2020
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import Optional

import flwr as fl
import flwr.common as flc

from model    import (create_model, get_parameters,
                       set_parameters, evaluate_model,
                       train_local, compute_task_loss)
from dataset  import (TransitionPredictionDataset,
                       get_local_dataset,
                       get_client_dataset,
                       collate_with_mode)
from estimator import (compute_gradient_divergence,
                        create_estimator)
import config
from fl_utils import (_seed_worker, _pos_to_cfg,
                       _cfg_to_pos, make_client_fn,
                       _load_shared_data)


# ================================================================
#  MobilityClient -- Lightweight client outside Flower/Ray
# ================================================================
#
# Represents each candidate client before it's selected for
# training. Answers the server's mobility queries without
# Ray/Flower overhead.
#
# Design: each MobilityClient owns its own profile -- the server
# never accesses profiles directly. Before configure_fit, the
# server queries each candidate via get_mobility(), which returns
# velocity, mode and cell_id computed from the client's local
# samples.
#
# In a real deployment (UAVs, phones), this would be a lightweight
# message from the client to the server before training -- without
# exposing GPS coordinates or local data. In this simulation it
# runs in the server's own process (without Ray) due to a Flower
# architectural limitation: it doesn't support client->server
# communication between configure_fit and fit().

class MobilityClient:
    """
    Lightweight client for mobility queries from the server.

    Each instance holds a single client's profile and computes
    velocity, mode and cell_id from its local samples every round --
    without exposing the profile to the server.

    Parameters
    ----------
    client_id : int
    dataset   : TransitionPredictionDataset
    profile   : dict  -- client profile (position_seq,
                velocity_seq, mobility_group, etc.)
    """

    def __init__(
        self,
        client_id: int,
        dataset,
        profile: dict,
    ):
        self.cid     = client_id
        self.dataset = dataset
        self.profile = profile

    def get_mobility(self, server_round: int) -> dict:
        """
        Returns mobility metadata for the current round.

        Computes velocity (average of local samples), mode
        (predominant across local samples) and cell_id (predominant
        geographic cell) -- without exposing GPS coordinates.

        Used by the server in configure_fit to compute alpha(v,m)
        and h_hat before deciding skip and allocating epochs.

        Returns
        -------
        dict with keys: velocity, mode, cell_gx, cell_gy,
                       position (for the estimator's update())
        """
        r        = (server_round - 1) % config.N_ROUNDS
        position = self.profile["position_seq"][r]

        # MOBIS: position_seq does not carry user_id (only
        # lat/lon/mode) -- uses the profile's user_id (client level)
        # to fetch the participant's real data via get_client_dataset.
        _uid  = self.profile.get("user_id")
        _type = self.profile.get("type", "")
        if _uid and _type == "real_mobis":
            local_data = get_client_dataset(self.dataset, user_id=_uid)
        else:
            local_data = get_local_dataset(self.dataset, position)

        if len(local_data) == 0:
            # Fallback: use the profile when there are no local samples
            velocity = self.profile["velocity_seq"][r]
            mode     = position.get("mode", "unknown")
            gx = min(int(position.get("norm_lat", 0.0) * config.GRID_SIZE),
                     config.GRID_SIZE - 1)
            gy = min(int(position.get("norm_lon", 0.0) * config.GRID_SIZE),
                     config.GRID_SIZE - 1)
            return {
                "velocity": velocity,
                "mode":     mode,
                "cell_gx":  gx,
                "cell_gy":  gy,
                "position": position,
            }

        # Velocity and mode: identical to the old code path where the
        # server accessed profiles[cid] directly -- guarantees the same result.
        velocity = self.profile["velocity_seq"][r]
        mode     = position.get("mode", "unknown")

        # Geographic cell computed identically to the estimator's
        # _to_grid(position) -- uses norm_lat/norm_lon from position_seq[r],
        # not from local samples, to guarantee the same result as the
        # old code path where the server called _to_grid directly.
        cell_gx = min(int(position.get("norm_lat", 0.0) * config.GRID_SIZE),
                      config.GRID_SIZE - 1)
        cell_gy = min(int(position.get("norm_lon", 0.0) * config.GRID_SIZE),
                      config.GRID_SIZE - 1)

        return {
            "velocity": velocity,
            "mode":     mode,
            "cell_gx":  cell_gx,
            "cell_gy":  cell_gy,
            "position": position,  # for the estimator's update()
        }

    def get_position(self, server_round: int) -> dict:
        """
        Returns the client's position for sample selection in
        evaluate() -- without exposing the profile to the server.
        """
        r = (server_round - 1) % config.N_ROUNDS
        return self.profile["position_seq"][r]

    @property
    def mobility_group(self) -> str:
        return self.profile.get("mobility_group", "unknown")





# ================================================================
#  FEDAVG -- Client
# ================================================================

class FedAvgClient(fl.client.NumPyClient):
    """
    FL client for mode-transition point prediction.

    Each round:
      1. Receives the global model + config from the server
      2. Rebuilds position from the config's scalar fields
      3. Selects local samples by real GPS position
      4. Trains the LSTM locally
      5. Returns the model + divergence + metrics (primitives only)
    """

    def __init__(
        self,
        client_id: int,
        dataset:   TransitionPredictionDataset,
        profile:   dict,
        device:    str = "cpu",
        dataset_name: str = None,
    ):
        self.cid     = client_id
        self.dataset = dataset
        self.profile = profile
        self.device  = device
        self._dataset_name = dataset_name
        torch.manual_seed(config.ACTUAL_SEED)
        self.model   = create_model(self._dataset_name)
        self._global_params = None

    def get_parameters(self, cfg: dict) -> list[np.ndarray]:
        return get_parameters(self.model)

    @staticmethod
    def _extract_mode_weights(cfg: dict) -> dict:
        """Extracts mode_weights from mw_* config fields."""
        weights = {}
        for key, val in cfg.items():
            if key.startswith("mw_"):
                weights[key[3:]] = float(val)
        return weights if weights else None

    def fit(
        self,
        parameters: list[np.ndarray],
        cfg:        dict,
    ) -> tuple[list[np.ndarray], int, dict]:

        round_num    = cfg.get("round", 0)
        local_epochs = cfg.get("local_epochs",
                                config.LOCAL_EPOCHS_BASE)
        _seed_worker(config.ACTUAL_SEED, self.cid, round_num)

        # Rebuild position from the scalar fields
        # (falls back to the profile if the server didn't send it)
        if "pos_lat" in cfg:
            position = _cfg_to_pos(cfg)
        else:
            position = self.profile["position_seq"][
                (round_num - 1) % config.N_ROUNDS
            ]

        set_parameters(self.model, parameters)
        self._global_params = [p.copy() for p in parameters]

        # MOBIS: client_user_id in cfg -> the participant's own dataset
        # (position_seq does not carry user_id per position -- only lat/lon/mode)
        # GeoLife: no client_user_id in cfg -> filter by position (original)
        if cfg.get("client_user_id"):
            local_data = get_client_dataset(
                self.dataset,
                user_id=cfg["client_user_id"],
            )
        else:
            local_data = get_local_dataset(
                self.dataset, position,
                radius_km=cfg.get("radius_km", 3.0),
            )

        if len(local_data) == 0:
            return get_parameters(self.model), 0, {
                "skipped":   True,
                "client_id": self.cid,
                "round":     round_num,
            }

        # Deterministic generator tied to the global seed and the client
        g = torch.Generator()
        g.manual_seed(config.ACTUAL_SEED + int(self.cid))

        loader = DataLoader(
            local_data,
            batch_size  = config.BATCH_SIZE,
            shuffle     = True,
            num_workers = 0,
            collate_fn  = collate_with_mode,
            generator   = g  # <-- determinism fix
        )

        metrics = train_local(
            self.model, loader, local_epochs,
            device=self.device,
            mode_weights=self._extract_mode_weights(cfg),
            dataset_name=self._dataset_name,
        )

        local_p    = get_parameters(self.model)
        divergence = compute_gradient_divergence(
            local_p, self._global_params
        )

        # Average distance to transition of the local samples (raw, meters)
        raw_dists = [
            local_data.dataset.samples[i]["targets"][0]
            for i in local_data.indices
        ]
        mean_dist_to_trans = float(np.mean(raw_dists)) if raw_dists else -1.0

        # Free memory
        self._global_params = None
        self.model.cpu()

        return local_p, len(local_data), {
            "loss":                float(metrics["loss"]),
            "divergence":          float(divergence),
            "n_samples":           len(local_data),
            "gradient_steps":      int(metrics.get("gradient_steps", 0)),
            "client_id":           self.cid,
            "round":               round_num,
            "local_epochs":        local_epochs,
            "mobility_group":      self.profile["mobility_group"],
            "mode":                position.get("mode", "unknown"),
            "velocity":            float(self.profile["velocity_seq"][
                                       (round_num - 1) % config.N_ROUNDS
                                   ]),
            "dist_to_transition":  mean_dist_to_trans,
            # Local gradient stats reported to the server
            "grad_norm_pre_mean":  float(metrics.get("grad_norm_pre_mean",  0.0)),
            "grad_norm_pre_std":   float(metrics.get("grad_norm_pre_std",   0.0)),
            "grad_norm_post_mean": float(metrics.get("grad_norm_post_mean", 0.0)),
            "grad_norm_post_std":  float(metrics.get("grad_norm_post_std",  0.0)),
            "grad_clip_rate":      float(metrics.get("grad_clip_rate",      0.0)),
            # h_rank reported back for the client_log
            "h_rank_norm":         float(cfg.get("h_rank_norm", -1.0)),
        }

    def evaluate(
        self,
        parameters: list[np.ndarray],
        cfg:        dict,
    ) -> tuple[float, int, dict]:

        round_num = cfg.get("round", 0)

        if "pos_lat" in cfg:
            position = _cfg_to_pos(cfg)
        else:
            position = self.profile["position_seq"][
                (round_num - 1) % config.N_ROUNDS
            ]

        set_parameters(self.model, parameters)
        # MOBIS: same reason as fit() -- position_seq doesn't carry user_id
        _uid  = self.profile.get("user_id")
        _type = self.profile.get("type", "")
        if _uid and _type == "real_mobis":
            eval_data = get_client_dataset(self.dataset, user_id=_uid)
        else:
            eval_data = get_local_dataset(
                self.dataset, position, radius_km=3.0
            )

        if len(eval_data) == 0:
            return 0.0, 0, {
                "mae_dist": 0.0, "mae_time": 0.0
            }

        loader  = DataLoader(
            eval_data, batch_size=32,
            shuffle=False, num_workers=0,
            collate_fn=collate_with_mode,
        )
        metrics = evaluate_model(
            self.model, loader, self.dataset, self.device
        )

        extra = {}
        for mode, vals in metrics.get(
                "mae_by_mode", {}).items():
            extra[f"mode_dist_{mode}"] = float(vals["mae_dist"])
            extra[f"mode_time_{mode}"] = float(vals["mae_time"])
            extra[f"mode_n_{mode}"]    = float(vals["n"])
        return float(metrics["loss"]), len(eval_data), {
            "mae_dist": float(metrics["mae_dist"]),
            "mae_time": float(metrics["mae_time"]),
            **extra,
        }


def _log_train_no_data(
    client_log: list,
    client_consecutive: dict,
    configure_fit_state: dict,
    mobility_clients: dict,
    server_round: int,
    no_data_results: list,
) -> None:
    """
    Logs, in client_log, clients that were DISPATCHED to train this
    round (decision != "skip" in configure_fit -- either genuinely or
    via a MIN_SKIP_PARTICIPANTS top-up) but returned num_examples=0
    (no local samples for the client's user_id in this seed's training
    split -- an "ineligible" client, see clients_meta.csv).

    A category distinct from "skip" (the estimator's decision, already
    logged in configure_fit) and from a successful "train" (logged in
    aggregate_fit from the filtered results). Without this, these
    rounds would be entirely absent from client_rank_log -- neither
    skip nor train -- which makes phi_k = trained_rounds/available_rounds
    ambiguous for the few clients with no training data in this seed.

    cid comes from cp.cid (ClientProxy), not from fit_res.metrics -- the
    return value of fit() on the 0-samples path is {"skipped": True}
    (FedProx/SCAFFOLD/FedNova) or includes client_id without
    mode/velocity (FedAvg); cp.cid is the reliable source that's uniform
    across all 4 algorithms.

    Does not mutate client_consecutive (the event is neither skip nor
    train -- the consecutive-streak counters stay frozen at their
    previous value, recorded as-is for this round).

    Pure instrumentation -- called with the subset of results already
    separated OUT before the num_examples>0 filter used for
    aggregation; it does not affect weights, client selection, or
    epoch allocation.
    """
    for cp, _fit_res in no_data_results:
        cid = int(cp.cid)
        _prev = client_consecutive.get(cid, {"skip": 0, "train": 0})
        _cfs  = configure_fit_state.get(cid, {})
        mob   = (mobility_clients[cid].get_mobility(server_round)
                 if cid in mobility_clients else {})
        client_log.append({
            "round":            server_round,
            "client_id":        cid,
            "decision":         "train_no_data",
            "h_rank_value":     _cfs.get("h_rank_value", -1.0),
            "h_rank_pos":       _cfs.get("h_rank_pos",   -1),
            "h_raw":            _cfs.get("h_raw",        -1.0),
            "h_raw_pool_std":   _cfs.get("h_raw_pool_std",   0.0),
            "h_raw_pool_range": _cfs.get("h_raw_pool_range", 0.0),
            "h_raw_pool_mean":  _cfs.get("h_raw_pool_mean",  0.0),
            "mode":             mob.get("mode", "unknown"),
            "divergence":       None,
            "velocity":         mob.get("velocity", None),
            "n_epochs":         0,
            "loss_mean":        None,
            "grad_norm_pre":    None,
            "grad_norm_post":   None,
            "grad_clip_rate":   None,
            "consecutive_skip":  _prev["skip"],
            "consecutive_train": _prev["train"],
        })


# ================================================================
#  FEDAVG -- Strategy
# ================================================================

class FedAvgStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg extended with the hybrid estimator.

    Phases:
      Rounds 1-CALIBRATION_ROUNDS : calibrate delta_m
      Following rounds            : estimator active

    Design note: FedProx, SCAFFOLD and FedNova delegate the estimator
    logic to EstimatorMixin (_plugin_configure_fit,
    _plugin_post_aggregate). FedAvgStrategy implements the same logic
    directly -- without inheriting from the mixin -- for historical
    development reasons (it was the first strategy implemented). The
    behavior is functionally identical: every algorithm uses
    MobilityClient.get_mobility() to get mobility metadata, and the
    server never accesses profiles directly.
    """

    def __init__(
        self,
        profiles:       dict,
        estimator_name: str = "hybrid",
        n_clients:      int = config.N_CLIENTS,
        wandb_run=None,
        **kwargs,
    ):
        # fit_metrics_aggregation_fn: records fit metrics in Flower's
        # history.metrics_distributed_fit, suppressing the "No
        # fit_metrics_aggregation_fn provided" warning.
        def _agg_fit(metrics):
            losses = [m.get("loss",       0.0) for _, m in metrics]
            divs   = [m.get("divergence", 0.0) for _, m in metrics]
            return {
                "loss":       float(np.mean(losses)),
                "divergence": float(np.mean(divs)),
            }

        # Extract dataset before passing **kwargs to Flower (which doesn't expect dataset)
        dataset = kwargs.pop("dataset", None)

        super().__init__(
            fit_metrics_aggregation_fn=_agg_fit,
            **kwargs,
        )
        # MobilityClient: each candidate client owns its own profile.
        # The server never accesses profiles directly -- it queries each
        # MobilityClient via get_mobility() before configure_fit.
        self.mobility_clients: dict[int, MobilityClient] = {
            cid: MobilityClient(cid, dataset, prof)
            for cid, prof in profiles.items()
        }
        self.estimator_name = estimator_name
        # velocity_decay and grid_size are passed explicitly (not just
        # relying on __init__'s default, which is fixed at import time) --
        # needed for the --velocity-decay and --grid-size (Zurich G10)
        # CLI overrides to take effect. See run_experiment.py, main().
        self.estimator      = create_estimator(
            estimator_name, n_clients=n_clients,
            velocity_decay=config.VELOCITY_DECAY,
            grid_size=config.GRID_SIZE,
        )
        self.round_log: list[dict] = []
        # Cumulative cost: total gradient steps (clients x epochs x batches)
        self.total_grad_steps: int = 0
        self._calibrated = False
        # Per-client log to correctly calibrate delta_m.
        # round_log aggregates per round -- no client_id/mode.
        # client_log preserves individual entries used by the
        # estimator's calibrate_mode_rates.
        self.client_log: list[dict] = []
        self._client_consecutive: dict = {}  # cid -> {'skip': N, 'train': N}
        self._configure_fit_state: dict = {}  # cid -> configure_fit state
        # -- tqdm: per-method progress bar --------------------------
        # Created on the first aggregate_fit (not in __init__) to avoid
        # clashing with the Flower/Ray INFO logs still being emitted
        # during initialization.
        self._val_log: dict[int, float] = {}
        self._pbar = None   # lazily initialized in aggregate_fit
        self._wandb_run = wandb_run
        self._val_ds   = None   # set after construction
        self._device   = "cpu"  # set after construction
        self.global_eval_log: list[dict] = []
        self._final_params = None  # stores params from the last aggregate_fit
        # Data for the divergence x transition-proximity relationship
        self.h4_data: list[dict] = []
        # tracks samples used per client to detect recycling
        self._prev_sample_indices: dict[int, set] = {}
        # gradient reliability per client
        self._gradient_reliability: dict[int, float] = {}
        # weighted loss mode weights (only for hybrid)
        self._mode_weights: dict = {}
        # memory and time monitor
        from resource_monitor import ResourceMonitor
        self._monitor = ResourceMonitor()

    def compute_mode_weights(self, dataset) -> dict:
        """
        Computes per-mode weight = 1/sqrt(freq(mode)).
        Normalized so that the average weight = 1.0.
        Only used by the hybrid -- baselines receive {}.
        """
        if (self.estimator_name != "hybrid"
                or not config.WEIGHTED_LOSS_ENABLED):
            return {}
        from collections import Counter
        modes = [s.get("mode", "unknown") for s in dataset.samples]
        counts = Counter(modes)
        if not counts:
            return {}
        raw = {m: 1.0 / np.sqrt(max(c, 1)) for m, c in counts.items()}
        mean_w = np.mean(list(raw.values()))
        if mean_w == 0:
            return {}
        weights = {m: w / mean_w for m, w in raw.items()}
        print(f"  [FedHME] Weighted loss -- per-mode weights:")
        for m in config.TRANSPORT_MODES:
            if m in weights:
                print(f"    {m:8s}: {weights[m]:.3f} "
                      f"(freq={counts.get(m, 0)})")
        return weights

    # -- Configure training ------------------------------------

    def configure_fit(
        self,
        server_round:   int,
        parameters,
        client_manager,
    ):
        import random
        random.seed(config.ACTUAL_SEED + server_round)
    
        # dynamic client selection by h_hat
        # Sample a larger pool and select the most informative ones
        all_clients = client_manager.sample(
            num_clients     = min(config.N_CLIENTS,
                                   client_manager.num_available()),
            min_num_clients = config.MIN_FIT_CLIENTS,
        )

        all_clients = sorted(all_clients, key=lambda c: int(c.cid))

        # Query each MobilityClient to get velocity, mode and cell_id.
        # Each client owns its own profile -- the server only receives
        # lightweight metadata, with no direct access to the internal profile.
        client_estimates = []
        for client in all_clients:
            cid = int(client.cid)
            if cid not in self.mobility_clients:
                continue
            mob  = self.mobility_clients[cid].get_mobility(server_round)
            position = mob["position"]
            velocity = mob["velocity"]
            mode     = mob["mode"]
            cell_id  = (mob["cell_gx"], mob["cell_gy"])
            est = self.estimator.estimate(
                cid, position, velocity, mode, server_round,
                cell_id=cell_id,
            )
            client_estimates.append((client, cid, position, est))

        # Design B: explicit skip over full pool + BUDGET_SAVING
        h_values_pool = [x[3]["estimate"] for x in client_estimates]
        mean_h   = float(np.mean(h_values_pool))   if h_values_pool else 1.0
        median_h = float(np.median(h_values_pool)) if h_values_pool else 1.0
        n_valid  = len(client_estimates)

        # Calibrate delta_m every round after the initial calibration
        if (server_round > config.CALIBRATION_ROUNDS
                and hasattr(self.estimator, "calibrate_mode_rates")):
            self.estimator.client_log = self.client_log
            rates = self.estimator.calibrate_mode_rates(self.round_log)
            if not self._calibrated:
                self._calibrated = True

        # -- h_hat rank normalization ---------------------------------
        # h_norm = rank(h_hat_i) / (N-1)  ->  h_norm in [0.0, 1.0]
        # Makes the skip invariant to scale and to the distribution
        # shape (long-tailed GeoLife, more homogeneous datasets).
        # SKIP_FACTOR becomes directly the fraction of the pool skipped.
        # Applied before decide() in every method.
        # -- Diagnostic log -- captures h_raw BEFORE rank normalization --
        _h_raw_dict       = {x[1]: x[3]["estimate"] for x in client_estimates}
        _h_raw_arr        = np.array(list(_h_raw_dict.values()), dtype=float)
        _h_raw_pool_std   = float(np.std(_h_raw_arr))              if len(_h_raw_arr) > 1 else 0.0
        _h_raw_pool_range = float(_h_raw_arr.max()-_h_raw_arr.min()) if len(_h_raw_arr) > 1 else 0.0
        _h_raw_pool_mean  = float(np.mean(_h_raw_arr))             if len(_h_raw_arr) > 0 else 0.0

        if len(h_values_pool) > 1:
            _h_arr  = np.array(h_values_pool, dtype=float)
            _ranks  = np.argsort(np.argsort(_h_arr)).astype(float)
            _h_norm = _ranks / max(len(_h_arr) - 1, 1)
            _h_pos  = (len(_h_arr) - _ranks).astype(int)  # 1=most heterogeneous
            for _i, (_c, _cid, _pos, _est) in enumerate(client_estimates):
                client_estimates[_i] = (_c, _cid, _pos,
                                        {**_est,
                                         "estimate":   float(_h_norm[_i]),
                                         "h_rank_pos": int(_h_pos[_i]),
                                         "h_raw":      float(_h_raw_dict.get(_cid, 0.0)),
                                        })
            h_values_pool = [x[3]["estimate"] for x in client_estimates]

        # -- Explicit skip over the full pool ----------------------
        participating = []
        n_skipped_this_round = 0

        if self.estimator_name == "random_skip":
            # m(r) is a FIXED, external target (following the
            # Power-of-Choice/Oort pattern -- see BaselineRandomSkip in
            # estimator.py), it does not emerge from independent
            # per-client decisions. That's why selection must be
            # coordinated here (it can't be done from inside decide(),
            # which only sees one client at a time).
            #
            # The consecutive-skip cap (MAX_CONSECUTIVE_SKIPS) stays
            # active and identical to the hybrid: clients at the
            # consecutive-skip cap are force-included, consuming part of
            # the m(r) slots; a uniform draw (self.estimator._rng, a
            # dedicated RNG never shared with SGD/shuffling/fraction_fit)
            # fills the rest -- never looking at h_hat/cell at any point.
            m_target = self.estimator.get_target_n_participating(server_round, n_valid)
            all_cids_list = [x[1] for x in client_estimates]

            forced_cids = [
                cid for cid in all_cids_list
                if self.estimator.consecutive_skips.get(cid, 0)
                >= config.MAX_CONSECUTIVE_SKIPS
            ]
            non_forced_cids = [
                cid for cid in all_cids_list if cid not in forced_cids
            ]

            remaining_slots = max(m_target - len(forced_cids), 0)
            remaining_slots = min(remaining_slots, len(non_forced_cids))

            if remaining_slots > 0:
                drawn_cids = set(
                    self.estimator._rng.choice(
                        np.array(non_forced_cids),
                        size=remaining_slots, replace=False,
                    ).tolist()
                )
            else:
                drawn_cids = set()

            participating_cids = set(forced_cids) | drawn_cids

            for client, cid, position, est in client_estimates:
                if cid in participating_cids:
                    # Trains (drawn or forced by the consecutive-skip
                    # cap) -- resets the streak, same as any training
                    # decision in the hybrid's decide().
                    self.estimator.consecutive_skips[cid] = 0
                    participating.append((client, cid, position, est))
                    continue

                # Confirmed skip
                self.estimator.consecutive_skips[cid] = (
                    self.estimator.consecutive_skips.get(cid, 0) + 1
                )
                n_skipped_this_round += 1
                _prev = self._client_consecutive.get(cid, {"skip": 0, "train": 0})
                _consec_skip  = _prev["skip"] + 1
                _consec_train = 0
                self._client_consecutive[cid] = {"skip": _consec_skip, "train": _consec_train}
                self.client_log.append({
                    "round":            server_round,
                    "client_id":        cid,
                    "decision":         "skip",
                    "h_rank_value":     float(est["estimate"]),
                    "h_rank_pos":       int(est.get("h_rank_pos", -1)),
                    "h_raw":            float(est.get("h_raw", -1.0)),
                    "h_raw_pool_std":   _h_raw_pool_std,
                    "h_raw_pool_range": _h_raw_pool_range,
                    "h_raw_pool_mean":  _h_raw_pool_mean,
                    "mode":             None,
                    "divergence":       None,
                    "velocity":         None,
                    "n_epochs":         0,
                    "loss_mean":        None,
                    "grad_norm_pre":    None,
                    "grad_norm_post":   None,
                    "grad_clip_rate":   None,
                    "consecutive_skip":  _consec_skip,
                    "consecutive_train": _consec_train,
                })
        else:
            for _idx, (client, cid, position, est) in enumerate(client_estimates):
                decision, _ = self.estimator.decide(
                    cid, est["estimate"], mean_h=mean_h, median_h=median_h
                )
                if decision == "skip":
                    n_skipped_this_round += 1
                    # Streak -- how many consecutive rounds skipped
                    _prev = self._client_consecutive.get(cid, {"skip": 0, "train": 0})
                    _consec_skip  = _prev["skip"] + 1
                    _consec_train = 0
                    self._client_consecutive[cid] = {"skip": _consec_skip, "train": _consec_train}
                    self.client_log.append({
                        "round":            server_round,
                        "client_id":        cid,
                        "decision":         "skip",
                        "h_rank_value":     float(est["estimate"]),
                        "h_rank_pos":       int(est.get("h_rank_pos", -1)),
                        "h_raw":            float(est.get("h_raw", -1.0)),
                        "h_raw_pool_std":   _h_raw_pool_std,
                        "h_raw_pool_range": _h_raw_pool_range,
                        "h_raw_pool_mean":  _h_raw_pool_mean,
                        "mode":             None,
                        "divergence":       None,
                        "velocity":         None,
                        "n_epochs":         0,
                        "loss_mean":        None,
                        "grad_norm_pre":    None,
                        "grad_norm_post":   None,
                        "grad_clip_rate":   None,
                        "consecutive_skip":  _consec_skip,
                        "consecutive_train": _consec_train,
                    })
                    continue
                participating.append((client, cid, position, est))

        # Estimator metrics for round_log (filled in after allocate_epochs)
        self._last_estimator_metrics = {
            "n_skipped":       n_skipped_this_round,
            "n_participating": len(participating),
            "skip_rate":       n_skipped_this_round / max(len(participating) + n_skipped_this_round, 1),
            "h_mean":          float(np.mean(h_values_pool)) if h_values_pool else 0.0,
            "h_std":           float(np.std(h_values_pool))  if h_values_pool else 0.0,
            "mean_h_pool":     mean_h,
            "median_h_pool":   median_h,
            "epochs_mean":     0.0,  # updated after allocate_epochs
        }

        # Enforce the minimum
        if len(participating) < config.MIN_SKIP_PARTICIPANTS:
            client_estimates_sorted = sorted(
                client_estimates,
                key=lambda x: x[3]["estimate"], reverse=True
            )
            for client, cid, position, est in client_estimates_sorted:
                if not any(c == cid for _, c, _, _ in participating):
                    participating.append((client, cid, position, est))
                if len(participating) >= config.MIN_SKIP_PARTICIPANTS:
                    break

        self._n_skipped_last_round = n_skipped_this_round

        # Store configure_fit state for use in aggregate_fit
        # (h_rank_pos, h_raw, n_epochs per client)
        self._configure_fit_state = {
            cid: {
                "h_rank_value": float(est["estimate"]),
                "h_rank_pos":   int(est.get("h_rank_pos", -1)),
                "h_raw":        float(est.get("h_raw", -1.0)),
                "h_raw_pool_std":   _h_raw_pool_std,
                "h_raw_pool_range": _h_raw_pool_range,
                "h_raw_pool_mean":  _h_raw_pool_mean,
            }
            for _, cid, _, est in participating
        }

        # -- Proportional allocation with BUDGET_SAVING ----------------
        # E_total = n_valid x base x (1 - BUDGET_SAVING)
        # Skipped clients' epochs + the BUDGET_SAVING savings are redistributed
        if self.estimator_name == "random_skip":
            # Deterministic uniform allocation -- only selection is random.
            # Not also randomizing the allocation weights avoids mixing two
            # axes of randomness (selection vs. allocation) in the same
            # experiment. Equal weights make allocate_epochs_proportional
            # degenerate to E_k(r) = E_total(r) / n_valid(r) for every k.
            estimates = {cid: 1.0 for _, cid, _, _ in participating}
        else:
            estimates = {cid: est["estimate"]
                         for _, cid, _, est in participating}

        if self.estimator_name == "no_estimate":
            # Fixed pure baseline: BaselineNoEstimate.estimate() returns the
            # SAME raw value for every client this round, but the rank
            # normalization above (unconditional, lines ~620) still turns
            # that constant into an artificial 0..1 spread via argsort
            # tie-breaking by cid. Feeding that spread into
            # allocate_epochs_proportional() redistributes epochs
            # near-linearly across clients (not a small uniform bump) instead
            # of the intended fixed E_base for everyone. Bypass it here,
            # matching the "modo puro" shortcut already used by the other
            # four algorithms (EstimatorMixin._plugin_configure_fit).
            epoch_alloc = {cid: config.LOCAL_EPOCHS_BASE for cid in estimates}
        else:
            epoch_alloc = self.estimator.allocate_epochs_proportional(
                estimates,
                base_epochs     = config.LOCAL_EPOCHS_BASE,
                n_total_clients = n_valid,
                apply_budget    = (self.estimator_name != "no_estimate"),
            )

        # Update epochs_mean and h_hat components in the estimator metrics
        if epoch_alloc and hasattr(self, '_last_estimator_metrics'):
            self._last_estimator_metrics["epochs_mean"] = float(np.mean(list(epoch_alloc.values())))
        if hasattr(self, '_last_estimator_metrics') and participating:
            _ests = [e for _, _, _, e in participating]
            self._last_estimator_metrics.update({
                "temporal_mean":          float(np.mean([e.get("temporal", 0.0) for e in _ests])),
                "spatial_mean":           float(np.mean([e.get("spatial",  0.0) for e in _ests])),
                "alpha_mean":             float(np.mean([e.get("alpha",    0.5) for e in _ests])),
                "delta_m_mean":           float(np.mean([e.get("delta_m",  0.0) for e in _ests])),
                "h_temporal_contrib_mean":float(np.mean([e.get("h_temporal_contrib", 0.0) for e in _ests])),
                "h_spatial_contrib_mean": float(np.mean([e.get("h_spatial_contrib",  0.0) for e in _ests])),
                "geo_map_val_mean":       float(np.mean([e.get("geo_map_val",  0.0) for e in _ests])),
                "client_ema_val_mean":    float(np.mean([e.get("client_ema_val",0.0) for e in _ests])),
                "budget_saving":          config.BUDGET_SAVING,
                # map coverage per round
                "map_coverage_pct": float(
                    (self.estimator.geo_count > 0).sum()
                    / self.estimator.geo_count.size * 100
                ) if hasattr(self.estimator, "geo_count") else 0.0,
            })

        # -- Build FitIns -----------------------------------------
        self._round_estimator_decisions = []
        fit_ins = []

        for client, cid, position, est in participating:
            n_epochs = epoch_alloc.get(cid, config.LOCAL_EPOCHS_BASE)

            # log decision and estimator components
            self._round_estimator_decisions.append({
                "round":        server_round,
                "client_id":    cid,
                "h_estimate":   float(est["estimate"]),
                "temporal":     float(est.get("temporal", 0.0)),
                "spatial":      float(est.get("spatial",  0.0)),
                "alpha":        float(est.get("alpha",    0.5)),
                "delta_m":      float(est.get("delta_m",  0.0)),
                "h_temporal_contrib": float(est.get("h_temporal_contrib", 0.0)),
                "h_spatial_contrib":  float(est.get("h_spatial_contrib",  0.0)),
                "geo_map_val":        float(est.get("geo_map_val",        0.0)),
                "client_ema_val":     float(est.get("client_ema_val",     0.0)),
                "decision":     "train",
                "epochs_allocated": n_epochs,
                "mode":         position.get("mode", "unknown"),
                "velocity":     float(
                    self.mobility_clients[cid].get_mobility(server_round)["velocity"]
                    if cid in self.mobility_clients else 0.0
                ),
                "mean_h":       mean_h,
                "median_h":     median_h,
                "skip_threshold": max(config.SKIP_FACTOR * mean_h,
                                      config.SKIP_FLOOR),
            })

            cfg = {
                "round":        server_round,
                "local_epochs": n_epochs,
                **_pos_to_cfg(position),
                "radius_km":    3.0,
                # MOBIS: the real participant's user_id -- position_seq
                # does not carry user_id per position (only lat/lon/mode),
                # so the client needs the profile's user_id to fetch its
                # real data via get_client_dataset(). Empty for GeoLife
                # (uses the geographic filter by position, original behavior).
                "client_user_id": str(
                    self.mobility_clients[cid].profile.get("user_id", "")
                ) if (cid in self.mobility_clients and
                      self.mobility_clients[cid].profile.get("type", "")
                      == "real_mobis") else "",
                "h_estimate":   float(est["estimate"]),
                "h_rank_norm":  float(est["estimate"]),  # rank norm [0,1] reported back for client_log
                "alpha":        float(est.get("alpha", 0.5)),
                "velocity":     float(
                    self.mobility_clients[cid].get_mobility(server_round)["velocity"]
                    if cid in self.mobility_clients else 0.0
                ),
            }
            # pass per-mode weights (hybrid only)
            if self._mode_weights:
                for m, w in self._mode_weights.items():
                    cfg[f"mw_{m}"] = float(w)

            fit_ins.append((client, flc.FitIns(parameters, cfg)))

        return fit_ins

    # -- Aggregate and update the estimator ---------------------

    def aggregate_fit(self, server_round, results, failures):
        self._monitor.start_round()  # round timing
        # Create the progress bar on the first round -- after Ray has
        # already initialized and Flower's INFO logs have stopped appearing.
        if self._pbar is None:
            try:
                from tqdm import tqdm as _tqdm
                self._pbar = _tqdm(
                    total        = config.N_ROUNDS,
                    desc         = f"  {self.estimator_name:<14s}",
                    unit         = "rd",
                    dynamic_ncols= True,
                    colour       = "cyan",
                    postfix      = {"loss": "—", "val": "—",
                                    "div": "—", "MAE": "—"},
                )
            except ImportError:
                pass

        if failures:
            print(f"[DEBUG ERROR] {len(failures)} failures occurred "
                  f"in round {server_round}.")
            if failures:
                print(f"  1st error: {failures[0]}")

        if not results:
            return None, {}

        # Filter out results with num_examples == 0 before passing to Flower.
        # Avoids a ZeroDivisionError in aggregate_inplace when every
        # selected client returns an empty Subset (user_id isolation).
        #
        # Clients discarded here were dispatched to train (not skipped)
        # but had no local samples -- logged in client_log via
        # _log_train_no_data before being discarded, so they aren't
        # absent from both logs.
        _no_data_results = [(cp, fit_res) for cp, fit_res in results
                             if fit_res.num_examples == 0]
        results = [(cp, fit_res) for cp, fit_res in results
                   if fit_res.num_examples > 0]
        _log_train_no_data(
            self.client_log, self._client_consecutive,
            self._configure_fit_state, self.mobility_clients,
            server_round, _no_data_results,
        )

        if not results:
            print(f"[WARN] Round {server_round}: every client "
                  f"returned 0 samples -- round skipped.")
            return None, {}

        # Deterministic order -- Ray doesn't guarantee the return order of
        # clients; without this, FedAvg's weighted sum (not associative in
        # floating point) diverges across runs even with the same seed.
        results = sorted(results, key=lambda x: int(x[0].cid))

        agg_params, _ = super().aggregate_fit(
            server_round, results, failures
        )
        if agg_params is None:
            return None, {}

        round_metrics = defaultdict(list)

        # Sort results by Client ID (CID) -- neutralizes the random
        # finishing order of the Ray workers
        results = sorted(results, key=lambda x: int(x[0].cid))

        for cp, fit_res in results:
            cid      = int(cp.cid)
            m        = fit_res.metrics
            position = self.mobility_clients[cid].get_position(server_round)
            div      = m.get("divergence", 0.0)

            if fit_res.num_examples == 0:
                continue

            # compute gradient reliability
            # Clients with few samples have noisy gradients.
            # gradient_steps=1 -> low reliability, 10+ -> high
            grad_steps = m.get("gradient_steps", 1)
            reliability = min(1.0, grad_steps / 10.0)  # saturates at 10
            self._gradient_reliability[cid] = reliability

            # Update the estimator with the client's real divergence and cell_id
            _cell_gx = m.get("cell_gx", None)
            _cell_gy = m.get("cell_gy", None)
            _cell_id = (int(_cell_gx), int(_cell_gy))                        if (_cell_gx is not None and _cell_gy is not None) else None
            self.estimator.update(cid, div, position, cell_id=_cell_id)

            # Record estimation error
            self.estimator.record_true_divergence(
                server_round, cid, div
            )

            group = m.get("mobility_group", "?")
            mode  = m.get("mode", "?")
            round_metrics[f"acc_{group}"].append(
                m.get("loss", 0.0))
            round_metrics["loss_all"].append(m.get("loss", 0.0))
            round_metrics["div_all"].append(div)
            round_metrics["modes"].append(mode)
            # Aggregate global gradient stats per round
            round_metrics["grad_norm_pre_mean"].append(m.get("grad_norm_pre_mean",  0.0))
            round_metrics["grad_norm_pre_std"].append( m.get("grad_norm_pre_std",   0.0))
            round_metrics["grad_norm_post_mean"].append(m.get("grad_norm_post_mean",0.0))
            round_metrics["grad_clip_rate"].append(    m.get("grad_clip_rate",      0.0))

            # Collect data on the divergence x transition-proximity relationship
            dist_trans = m.get("dist_to_transition", -1.0)
            if dist_trans > 0:
                self.h4_data.append({
                    "round":      server_round,
                    "client_id":  cid,
                    "divergence": div,
                    "dist_to_transition": dist_trans,
                    "mode":       mode,
                    "velocity":   m.get("velocity", 0.0),
                })

            # Preserve the individual entry for delta_m calibration.
            # Always collected (not just during calibration) to support
            # periodic recalibration as new modes appear later on.
            # Streak counters
            _prev_c = self._client_consecutive.get(cid, {"skip": 0, "train": 0})
            _consec_train = _prev_c["train"] + 1
            _consec_skip  = 0
            self._client_consecutive[cid] = {"skip": _consec_skip, "train": _consec_train}

            # This client's configure_fit state
            _cfs = self._configure_fit_state.get(cid, {})

            self.client_log.append({
                "round":            server_round,
                "client_id":        cid,
                "decision":         "train",
                "h_rank_value":     _cfs.get("h_rank_value",     float(m.get("h_rank_norm", -1.0))),
                "h_rank_pos":       _cfs.get("h_rank_pos",       -1),
                "h_raw":            _cfs.get("h_raw",            -1.0),
                "h_raw_pool_std":   _cfs.get("h_raw_pool_std",   0.0),
                "h_raw_pool_range": _cfs.get("h_raw_pool_range", 0.0),
                "h_raw_pool_mean":  _cfs.get("h_raw_pool_mean",  0.0),
                "mode":             mode,
                "divergence":       div,
                "velocity":         float(m.get("velocity", 0.0)),
                "n_epochs":         int(m.get("local_epochs", 0)),
                "loss_mean":        float(m.get("loss", 0.0)),
                "grad_norm_pre":    float(m.get("grad_norm_pre_mean",  0.0)),
                "grad_norm_post":   float(m.get("grad_norm_post_mean", 0.0)),
                "grad_clip_rate":   float(m.get("grad_clip_rate",      0.0)),
                "consecutive_skip":  _consec_skip,
                "consecutive_train": _consec_train,
            })

        # This round's cost: gradient steps = sum_clients(epochs x batches)
        # Captures all 3 dimensions: n_clients x n_epochs x n_batches
        round_grad_steps = sum(
            m.get("gradient_steps", 0)
            for _, fit_res in results
            for m in [fit_res.metrics]
            if fit_res.num_examples > 0
        )
        # Fallback if gradient_steps wasn't reported (older baselines)
        if round_grad_steps == 0:
            round_grad_steps = sum(
                m.get("local_epochs", config.LOCAL_EPOCHS_BASE)
                for _, fit_res in results
                for m in [fit_res.metrics]
                if fit_res.num_examples > 0
            )
        self.total_grad_steps += round_grad_steps

        n_active = sum(1 for _, fr in results if fr.num_examples > 0)

        # Estimator metrics (from this round's configure_fit)
        _em = getattr(self, "_last_estimator_metrics", {})

        log = {
            "round":            server_round,
            "mean_loss":        float(np.mean(
                                    round_metrics["loss_all"])),
            "mean_div":         float(np.mean(
                                    round_metrics["div_all"])),
            "n_clients":        len(results),
            "n_active_clients": n_active,
            "round_grad_steps": round_grad_steps,
            "total_grad_steps": self.total_grad_steps,
            "modes":            round_metrics["modes"],
            "calibrated":       self._calibrated,
            # Estimator
            "n_skipped":        _em.get("n_skipped",       0),
            "n_participating":  _em.get("n_participating", len(results)),
            "skip_rate":        _em.get("skip_rate",       0.0),
            "h_mean":           _em.get("h_mean",          0.0),
            "h_std":            _em.get("h_std",           0.0),
            "epochs_mean":      _em.get("epochs_mean",     0.0),
            "temporal_mean":           _em.get("temporal_mean",           0.0),
            "spatial_mean":            _em.get("spatial_mean",            0.0),
            "alpha_mean":              _em.get("alpha_mean",              0.5),
            "delta_m_mean":            _em.get("delta_m_mean",            0.0),
            "h_temporal_contrib_mean": _em.get("h_temporal_contrib_mean", 0.0),
            "h_spatial_contrib_mean":  _em.get("h_spatial_contrib_mean",  0.0),
            "geo_map_val_mean":        _em.get("geo_map_val_mean",        0.0),
            "client_ema_val_mean":     _em.get("client_ema_val_mean",     0.0),
            "budget_saving":           _em.get("budget_saving", config.BUDGET_SAVING),
            "map_coverage_pct":        _em.get("map_coverage_pct",        0.0),
            # Global gradient stats aggregated per round
            # Extracted directly from results for compatibility with every algorithm
            "grad_norm_pre_mean":  float(np.mean([r.metrics.get("grad_norm_pre_mean",  0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
            "grad_norm_pre_std":   float(np.mean([r.metrics.get("grad_norm_pre_std",   0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
            "grad_norm_post_mean": float(np.mean([r.metrics.get("grad_norm_post_mean", 0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
            "grad_clip_rate":      float(np.mean([r.metrics.get("grad_clip_rate",      0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
        }
        self.round_log.append(log)
        # -- Unified validation every VAL_EVERY rounds ------------
        if (self._val_ds is not None
                and server_round % config.VAL_EVERY == 0):
            from run_experiment import _global_eval_callback
            ge = _global_eval_callback(
                server_round, agg_params,
                self._val_ds, self._device,
                self._wandb_run,
                method_name=self.estimator_name,
            )
            if ge:
                self.global_eval_log.append(ge)
        # -- wandb: real-time per-round logging --------------
        if self._wandb_run is not None:
            try:
                log_dict = {
                    "train/loss":           log["mean_loss"],
                    "train/div":            log["mean_div"],
                    "train/n_clients":      log["n_clients"],
                    "cost/round_grad_steps": round_grad_steps,
                    "cost/total_grad_steps": self.total_grad_steps,
                    "cost/n_active_clients": n_active,
                }
                # estimator metrics
                decisions = getattr(self, '_round_estimator_decisions', [])
                if decisions:
                    h_vals = [d["h_estimate"] for d in decisions]
                    epochs_vals = [d["epochs_allocated"] for d in decisions]
                    n_skipped = getattr(self, '_n_skipped_last_round', 0)
                    # median_h and skip_threshold from the full pool
                    _mean_h   = decisions[0].get("mean_h", 0.0)
                    _median_h = decisions[0].get("median_h", 0.0)
                    _skip_thr = decisions[0].get("skip_threshold",
                                    max(config.SKIP_FACTOR * _mean_h, _median_h))
                    temporal_vals = [d.get("temporal", 0.0) for d in decisions]
                    spatial_vals  = [d.get("spatial",  0.0) for d in decisions]
                    alpha_vals    = [d.get("alpha",    0.5) for d in decisions]
                    delta_m_vals  = [d.get("delta_m",  0.0) for d in decisions]
                    h_t_contrib   = [d.get("h_temporal_contrib", 0.0) for d in decisions]
                    h_s_contrib   = [d.get("h_spatial_contrib",  0.0) for d in decisions]
                    log_dict.update({
                        "estimator/h_mean":    float(np.mean(h_vals)),
                        "estimator/h_min":     float(np.min(h_vals)),
                        "estimator/h_max":     float(np.max(h_vals)),
                        "estimator/h_std":     float(np.std(h_vals)),
                        "estimator/epochs_mean": float(np.mean(epochs_vals)),
                        "estimator/epochs_min":  float(np.min(epochs_vals)),
                        "estimator/epochs_max":  float(np.max(epochs_vals)),
                        "estimator/n_participating": len(decisions),
                        "estimator/n_skipped":   n_skipped,
                        "estimator/mean_h_pool":    _mean_h,
                        "estimator/median_h_pool":  _median_h,
                        "estimator/skip_threshold": _skip_thr,
                        "estimator/skip_rate": n_skipped / max(len(decisions) + n_skipped, 1),
                        # h_hat components
                        "estimator/temporal_mean": float(np.mean(temporal_vals)),
                        "estimator/spatial_mean":  float(np.mean(spatial_vals)),
                        "estimator/alpha_mean":    float(np.mean(alpha_vals)),
                        "estimator/delta_m_mean":  float(np.mean(delta_m_vals)),
                        "estimator/h_temporal_contrib_mean": float(np.mean(h_t_contrib)),
                        "estimator/h_spatial_contrib_mean":  float(np.mean(h_s_contrib)),
                        "estimator/geo_map_val_mean": float(np.mean(
                            [d.get("geo_map_val", 0.0) for d in decisions])),
                        "estimator/client_ema_val_mean": float(np.mean(
                            [d.get("client_ema_val", 0.0) for d in decisions])),
                        "estimator/budget_saving": config.BUDGET_SAVING,
                    })
                # memory and time metrics
                log_dict.update(self._monitor.end_round())
                self._wandb_run.log(log_dict, step=server_round)
            except Exception:
                pass
        if self._pbar is not None:
            val_str = (
                f"{self._val_log[server_round - 1]:.4f}"
                if (server_round - 1) in self._val_log
                else "—"
            )
            self._pbar.set_postfix(
                loss=f"{log['mean_loss']:.4f}",
                val =val_str,
                div =f"{log['mean_div']:.4f}",
                MAE ="—",
                refresh=False,
            )
            self._pbar.update(1)
        else:
            print(f"  [R{server_round:03d}] "
                  f"loss={log['mean_loss']:.4f} "
                  f"div={log['mean_div']:.4f} "
                  f"n={log['n_clients']}")

        self._final_params = agg_params
        return agg_params, log

    # -- Evaluation ------------------------------------------------

    def configure_evaluate(self, server_round, parameters,
                            client_manager):
        # Per-local-client evaluation only every VAL_EVERY rounds.
        # The main validation is done by the _global_eval_callback
        # called in aggregate_fit -- this evaluation is complementary.
        if server_round % config.VAL_EVERY != 0:
            return []
        clients = client_manager.sample(
            num_clients     = config.MIN_EVAL_CLIENTS,
            min_num_clients = config.MIN_EVAL_CLIENTS,
        )
        return [(c, flc.EvaluateIns(parameters, {
            "round": server_round,
            **_pos_to_cfg(
                self.mobility_clients[int(c.cid)].get_position(server_round)
            ),
        })) for c in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        mae_dists = [r.metrics.get("mae_dist", 0)
                     for _, r in results]
        mae_times = [r.metrics.get("mae_time", 0)
                     for _, r in results]
        val_loss  = float(np.mean([r.loss for _, r in results]))
        # Store val_loss for display in the next round's progress bar
        self._val_log[server_round] = val_loss
        # -- Aggregate MAE per mode -----------------------------
        from collections import defaultdict as _dd
        mode_dist: dict = _dd(list)
        mode_time: dict = _dd(list)

        # Same deterministic ordering for evaluation
        results = sorted(results, key=lambda x: int(x[0].cid))
        
        for _, r in results:
            for key, val in r.metrics.items():
                if key.startswith("mode_dist_"):
                    mode_dist[key[10:]].append(val)
                elif key.startswith("mode_time_"):
                    mode_time[key[10:]].append(val)
        mae_by_mode = {
            m: {"mae_dist": float(np.mean(mode_dist[m])),
                "mae_time": float(np.mean(mode_time[m]))}
            for m in mode_dist
        }
        # -- wandb: real-time val_loss + per-mode MAE logging -
        if self._wandb_run is not None:
            try:
                log_dict = {
                    "eval/val_loss": val_loss,
                    "eval/mae_dist": float(np.mean(mae_dists)),
                    "eval/mae_time": float(np.mean(mae_times)),
                }
                for mode, vals in mae_by_mode.items():
                    log_dict[f"eval/mae_dist_{mode}"] = vals["mae_dist"]
                    log_dict[f"eval/mae_time_{mode}"] = vals["mae_time"]
                self._wandb_run.log(log_dict, step=server_round)
            except Exception:
                pass
        # Update the progress bar's postfix with MAE and val_loss
        if self._pbar is not None:
            self._pbar.set_postfix(
                loss=self._pbar.postfix.get("loss", "—"),
                val =f"{val_loss:.4f}",
                div =self._pbar.postfix.get("div",  "—"),
                MAE =f"{np.mean(mae_dists):.0f}m",
                refresh=True,
            )
        else:
            print(f"  [Eval R{server_round:03d}] "
                  f"val_loss={val_loss:.4f}  "
                  f"MAE dist={np.mean(mae_dists):.1f}m "
                  f"MAE time={np.mean(mae_times):.1f}s")
        return (val_loss,
                {"mae_dist": float(np.mean(mae_dists)),
                 "mae_time": float(np.mean(mae_times)),
                 **{f"mae_dist_{m}": v["mae_dist"]
                    for m, v in mae_by_mode.items()},
                 **{f"mae_time_{m}": v["mae_time"]
                    for m, v in mae_by_mode.items()}})

    # -- Analysis -------------------------------------------------

    def get_analysis(self) -> dict:
        # Convert mobility_clients to the format expected by the estimator
        _profiles_for_analysis = {
            cid: mc.profile
            for cid, mc in self.mobility_clients.items()
        }
        return {
            "by_group":    self.estimator.get_errors_by_group(
                               _profiles_for_analysis),
            "by_mode":     self.estimator.get_errors_by_mode(),
            "convergence": self.estimator.get_convergence_curve(),
            "estimator":   self.estimator_name,
            "calibrated":  self._calibrated,
            "mode_rates":  getattr(
                               self.estimator,
                               "mode_rates", {}),
        }


# ================================================================
#  FEDAVG -- Factory
# ================================================================

def make_fedavg_client_fn(
    dataset,
    profiles: dict,
    device: str = "cpu",
    dataset_name: str = None,
):
    """Factory for FedAvgClient -- uses make_client_fn from fl_utils."""
    return make_client_fn(dataset, profiles, FedAvgClient, device,
                           dataset_name=dataset_name)




class EstimatorMixin:
    """
    Mixin that injects the hybrid estimator as a plugin into any FL
    strategy. Provides:
    - Client selection by h_hat (top-N most informative)
    - Epoch allocation proportional to h_hat
    - Detailed estimator logging
    - Round-by-round estimator calibration

    A strategy using this mixin must call:
    - _plugin_init() in __init__
    - _plugin_configure_fit() in configure_fit
    - _plugin_post_aggregate() in aggregate_fit
    """

    def _plugin_init(self, profiles, use_estimator=False, dataset=None):
        """Initializes the estimator plugin with MobilityClients."""
        self._use_estimator = use_estimator
        self._estimator = None
        self._calibrated = False
        self._estimator_log: list[dict] = []
        self.client_log: list[dict] = []
        self._client_consecutive: dict = {}  # cid -> {'skip': N, 'train': N}
        self._configure_fit_state: dict = {}  # cid -> configure_fit state

        # Each candidate client owns its own profile via MobilityClient
        self.mobility_clients: dict[int, MobilityClient] = {
            cid: MobilityClient(cid, dataset, prof)
            for cid, prof in profiles.items()
        } if dataset is not None else {
            cid: MobilityClient(cid, None, prof)
            for cid, prof in profiles.items()
        }

        if use_estimator:
            # velocity_decay and grid_size passed explicitly here too --
            # same reason as the other call site (EstimatorMixin is used
            # by FedProx/SCAFFOLD/FedNova).
            self._estimator = create_estimator(
                "hybrid", n_clients=len(profiles),
                velocity_decay=config.VELOCITY_DECAY,
                grid_size=config.GRID_SIZE,
            )

    def _plugin_configure_fit(
        self,
        server_round: int,
        parameters,
        all_clients: list,
        profiles: dict,
    ) -> list[tuple]:
        """
        Uses the estimator to select clients and allocate epochs.
        Returns a list of (client, cfg_dict) with the injected decisions.

        If use_estimator=False, returns the default selection with fixed epochs.
        """
        if not self._use_estimator or self._estimator is None:
            # Pure mode: every valid client participates, fixed epochs
            result = []
            for client in all_clients:
                cid = int(client.cid)
                if cid not in self.mobility_clients:
                    continue
                position = self.mobility_clients[cid].get_position(server_round)
                result.append((client, cid, position, {
                    "estimate": 0.5,
                    "alpha": 0.5,
                }, config.LOCAL_EPOCHS_BASE))
            return result

        # Plugin mode: query each candidate's MobilityClient
        client_estimates = []
        for client in all_clients:
            cid = int(client.cid)
            if cid not in self.mobility_clients:
                continue
            mob      = self.mobility_clients[cid].get_mobility(server_round)
            position = mob["position"]
            velocity = mob["velocity"]
            mode     = mob["mode"]
            cell_id  = (mob["cell_gx"], mob["cell_gy"])
            est = self._estimator.estimate(
                cid, position, velocity, mode, server_round,
                cell_id=cell_id,
            )
            client_estimates.append((client, cid, position, est))

        # Design B: explicit skip over pool + BUDGET_SAVING
        h_values_pool = [x[3]["estimate"] for x in client_estimates]
        mean_h   = float(np.mean(h_values_pool))   if h_values_pool else 1.0
        median_h = float(np.median(h_values_pool)) if h_values_pool else 1.0
        n_valid  = len(client_estimates)

        # Calibrate delta_m every round using the real client_log
        if (hasattr(self._estimator, "calibrate_mode_rates")
                and server_round > config.CALIBRATION_ROUNDS):
            self._estimator.client_log = self.client_log
            self._estimator.calibrate_mode_rates(self.client_log)
            self._calibrated = True

        # -- h_hat rank normalization ---------------------------------
        # h_norm = rank(h_hat_i) / (N-1)  ->  h_norm in [0.0, 1.0]
        # Makes the skip invariant to scale and to the distribution
        # shape (long-tailed GeoLife, more homogeneous datasets).
        # SKIP_FACTOR becomes directly the fraction of the pool skipped.
        # Applied before decide() in every method.
        # -- Diagnostic log -- captures h_raw BEFORE rank normalization --
        _h_raw_dict       = {x[1]: x[3]["estimate"] for x in client_estimates}
        _h_raw_arr        = np.array(list(_h_raw_dict.values()), dtype=float)
        _h_raw_pool_std   = float(np.std(_h_raw_arr))              if len(_h_raw_arr) > 1 else 0.0
        _h_raw_pool_range = float(_h_raw_arr.max()-_h_raw_arr.min()) if len(_h_raw_arr) > 1 else 0.0
        _h_raw_pool_mean  = float(np.mean(_h_raw_arr))             if len(_h_raw_arr) > 0 else 0.0

        if len(h_values_pool) > 1:
            _h_arr  = np.array(h_values_pool, dtype=float)
            _ranks  = np.argsort(np.argsort(_h_arr)).astype(float)
            _h_norm = _ranks / max(len(_h_arr) - 1, 1)
            _h_pos  = (len(_h_arr) - _ranks).astype(int)  # 1=most heterogeneous
            for _i, (_c, _cid, _pos, _est) in enumerate(client_estimates):
                client_estimates[_i] = (_c, _cid, _pos,
                                        {**_est,
                                         "estimate":   float(_h_norm[_i]),
                                         "h_rank_pos": int(_h_pos[_i]),
                                         "h_raw":      float(_h_raw_dict.get(_cid, 0.0)),
                                        })
            h_values_pool = [x[3]["estimate"] for x in client_estimates]

        # -- Explicit skip over the full pool ----------------------
        participating = []
        n_skipped_this_round = 0
        for client, cid, position, est in client_estimates:
            decision, _ = self._estimator.decide(
                cid, est["estimate"], mean_h=mean_h, median_h=median_h
            )
            if decision == "skip":
                n_skipped_this_round += 1
                # Streak -- how many consecutive rounds skipped
                # (mirrors FedAvgStrategy.configure_fit, for logging parity
                # between hybrid and fedprox/scaffold/fednova hybrid)
                _prev = self._client_consecutive.get(cid, {"skip": 0, "train": 0})
                _consec_skip  = _prev["skip"] + 1
                _consec_train = 0
                self._client_consecutive[cid] = {"skip": _consec_skip, "train": _consec_train}
                self.client_log.append({
                    "round":            server_round,
                    "client_id":        cid,
                    "decision":         "skip",
                    "h_rank_value":     float(est["estimate"]),
                    "h_rank_pos":       int(est.get("h_rank_pos", -1)),
                    "h_raw":            float(est.get("h_raw", -1.0)),
                    "h_raw_pool_std":   _h_raw_pool_std,
                    "h_raw_pool_range": _h_raw_pool_range,
                    "h_raw_pool_mean":  _h_raw_pool_mean,
                    "mode":             None,
                    "divergence":       None,
                    "velocity":         None,
                    "n_epochs":         0,
                    "loss_mean":        None,
                    "grad_norm_pre":    None,
                    "grad_norm_post":   None,
                    "grad_clip_rate":   None,
                    "consecutive_skip":  _consec_skip,
                    "consecutive_train": _consec_train,
                })
                continue
            participating.append((client, cid, position, est))

        # Estimator metrics for round_log (read in aggregate_fit)
        self._last_estimator_metrics = {
            "n_skipped":       n_skipped_this_round,
            "n_participating": len(participating),
            "skip_rate":       n_skipped_this_round / max(len(participating) + n_skipped_this_round, 1),
            "h_mean":          float(np.mean(h_values_pool)) if h_values_pool else 0.0,
            "h_std":           float(np.std(h_values_pool))  if h_values_pool else 0.0,
            "mean_h_pool":     mean_h,
            "median_h_pool":   median_h,
            "epochs_mean":     0.0,
        }

        # Enforce the minimum
        if len(participating) < config.MIN_SKIP_PARTICIPANTS:
            client_estimates_sorted = sorted(
                client_estimates,
                key=lambda x: x[3]["estimate"], reverse=True
            )
            for client, cid, position, est in client_estimates_sorted:
                if not any(c == cid for _, c, _, _ in participating):
                    participating.append((client, cid, position, est))
                if len(participating) >= config.MIN_SKIP_PARTICIPANTS:
                    break

        # Store configure_fit state for use in _plugin_post_aggregate
        # (h_rank_pos, h_raw, h_raw_pool_* per client) -- mirrors
        # FedAvgStrategy.configure_fit, needed so client_rank_log has the
        # same columns filled in for fedprox/scaffold/fednova hybrid.
        self._configure_fit_state = {
            cid: {
                "h_rank_value": float(est["estimate"]),
                "h_rank_pos":   int(est.get("h_rank_pos", -1)),
                "h_raw":        float(est.get("h_raw", -1.0)),
                "h_raw_pool_std":   _h_raw_pool_std,
                "h_raw_pool_range": _h_raw_pool_range,
                "h_raw_pool_mean":  _h_raw_pool_mean,
            }
            for _, cid, _, est in participating
        }

        # -- Proportional allocation with BUDGET_SAVING ----------------
        estimates = {cid: est["estimate"]
                     for _, cid, _, est in participating}
        epoch_alloc = self._estimator.allocate_epochs_proportional(
            estimates,
            base_epochs=config.LOCAL_EPOCHS_BASE,
            n_total_clients=n_valid,
            apply_budget=True,  # always hybrid in this context
        )

        # Update epochs_mean and h_hat components in the estimator metrics
        if epoch_alloc and hasattr(self, '_last_estimator_metrics'):
            self._last_estimator_metrics["epochs_mean"] = float(np.mean(list(epoch_alloc.values())))
        if hasattr(self, '_last_estimator_metrics') and participating:
            _ests = [e for _, _, _, e in participating]
            self._last_estimator_metrics.update({
                "temporal_mean":          float(np.mean([e.get("temporal", 0.0) for e in _ests])),
                "spatial_mean":           float(np.mean([e.get("spatial",  0.0) for e in _ests])),
                "alpha_mean":             float(np.mean([e.get("alpha",    0.5) for e in _ests])),
                "delta_m_mean":           float(np.mean([e.get("delta_m",  0.0) for e in _ests])),
                "h_temporal_contrib_mean":float(np.mean([e.get("h_temporal_contrib", 0.0) for e in _ests])),
                "h_spatial_contrib_mean": float(np.mean([e.get("h_spatial_contrib",  0.0) for e in _ests])),
                "geo_map_val_mean":       float(np.mean([e.get("geo_map_val",  0.0) for e in _ests])),
                "client_ema_val_mean":    float(np.mean([e.get("client_ema_val",0.0) for e in _ests])),
                "budget_saving":          config.BUDGET_SAVING,
                # map coverage per round
                "map_coverage_pct": float(
                    (self._estimator.geo_count > 0).sum()
                    / self._estimator.geo_count.size * 100
                ) if hasattr(self._estimator, "geo_count") else 0.0,
            })

        result = []
        for client, cid, position, est in participating:
            n_epochs = epoch_alloc.get(cid, config.LOCAL_EPOCHS_BASE)

            # detailed log + components
            self._estimator_log.append({
                "round":      server_round,
                "client_id":  cid,
                "h_estimate": float(est["estimate"]),
                "temporal":   float(est.get("temporal", 0.0)),
                "spatial":    float(est.get("spatial",  0.0)),
                "alpha":      float(est.get("alpha",    0.5)),
                "delta_m":    float(est.get("delta_m",  0.0)),
                "h_temporal_contrib": float(est.get("h_temporal_contrib", 0.0)),
                "h_spatial_contrib":  float(est.get("h_spatial_contrib",  0.0)),
                "geo_map_val":        float(est.get("geo_map_val",        0.0)),
                "client_ema_val":     float(est.get("client_ema_val",     0.0)),
                "decision":   "train",
                "epochs_allocated": n_epochs,
                "mode":       position.get("mode", "unknown"),
                "velocity":   float(
                    self.mobility_clients[cid].get_mobility(server_round)["velocity"]
                    if cid in self.mobility_clients else 0.0
                ),
                "mean_h":     mean_h,
                "median_h":   median_h,
                "skip_threshold": max(config.SKIP_FACTOR * mean_h,
                                      config.SKIP_FLOOR),
            })

            result.append((client, cid, position, est, n_epochs))

        return result

    def _plugin_post_aggregate(self, server_round, results, profiles):
        """
        Updates the estimator after aggregation with real client data.
        Collects client_log for continuous calibration.
        """
        if not self._use_estimator or self._estimator is None:
            return
        results = sorted(results, key=lambda x: int(x[0].cid))
        for _, fit_res in results:
            m = fit_res.metrics
            cid = m.get("client_id", -1)
            if cid < 0:
                continue
            div = m.get("divergence", 0.0)
            mode = m.get("mode", "unknown")
            velocity = m.get("velocity", 0.0)

            # Position via MobilityClient -- the server never accesses the profile directly
            position = self.mobility_clients[cid].get_position(server_round)                        if cid in self.mobility_clients else {}

            # cell_id reported by the client in fit()'s metrics
            cell_gx = m.get("cell_gx", None)
            cell_gy = m.get("cell_gy", None)
            cell_id = (int(cell_gx), int(cell_gy))                       if (cell_gx is not None and cell_gy is not None) else None

            # Update the estimator with the client's cell_id
            self._estimator.update(cid, div, position, cell_id=cell_id)

            # Streak counters -- same logic as FedAvgStrategy.aggregate_fit
            _prev_c = self._client_consecutive.get(cid, {"skip": 0, "train": 0})
            _consec_train = _prev_c["train"] + 1
            _consec_skip  = 0
            self._client_consecutive[cid] = {"skip": _consec_skip, "train": _consec_train}

            # State saved in _plugin_configure_fit for this client
            _cfs = self._configure_fit_state.get(cid, {})

            # Client log -- same schema as FedAvgStrategy.aggregate_fit,
            # for logging parity between hybrid and fedprox/scaffold/
            # fednova hybrid (client_rank_log.csv needs the same columns
            # filled in on both paths).
            self.client_log.append({
                "round":            server_round,
                "client_id":        cid,
                "decision":         "train",
                "h_rank_value":     _cfs.get("h_rank_value",     float(m.get("h_rank_norm", -1.0))),
                "h_rank_pos":       _cfs.get("h_rank_pos",       -1),
                "h_raw":            _cfs.get("h_raw",            -1.0),
                "h_raw_pool_std":   _cfs.get("h_raw_pool_std",   0.0),
                "h_raw_pool_range": _cfs.get("h_raw_pool_range", 0.0),
                "h_raw_pool_mean":  _cfs.get("h_raw_pool_mean",  0.0),
                "mode":             mode,
                "divergence":       div,
                "velocity":         float(velocity),
                "n_epochs":         int(m.get("local_epochs", 0)),
                "loss_mean":        float(m.get("loss", 0.0)),
                "grad_norm_pre":    float(m.get("grad_norm_pre_mean",  0.0)),
                "grad_norm_post":   float(m.get("grad_norm_post_mean", 0.0)),
                "grad_clip_rate":   float(m.get("grad_clip_rate",      0.0)),
                "consecutive_skip":  _consec_skip,
                "consecutive_train": _consec_train,
            })

    def _plugin_wandb_log(self, wandb_run, server_round, log_dict):
        """
        Adds full estimator metrics to the wandb log_dict.
        Called by the aggregate_fit of the algorithms that use this mixin
        (FedProx, SCAFFOLD, FedNova, FedDyn).
        Identical to the FedAvg hybrid's logging -- same metrics for a fair comparison.
        """
        if not self._use_estimator or not self._estimator_log:
            return log_dict

        # Filter this round's decisions
        round_decisions = [
            d for d in self._estimator_log
            if d["round"] == server_round
        ]
        if not round_decisions:
            return log_dict

        h_vals      = [d["h_estimate"] for d in round_decisions]
        epochs_vals = [d["epochs_allocated"] for d in round_decisions]

        # Skip: n_skipped stored in configure_fit via _n_skipped_last_round
        n_participating = len(round_decisions)
        n_skipped = getattr(self, '_n_skipped_last_round', 0)

        # h_hat components
        _mean_h   = round_decisions[0].get("mean_h", 0.0)
        _median_h = round_decisions[0].get("median_h", 0.0)
        _skip_thr = round_decisions[0].get("skip_threshold",
                        max(config.SKIP_FACTOR * _mean_h, config.SKIP_FLOOR))

        temporal_vals = [d.get("temporal", 0.0) for d in round_decisions]
        spatial_vals  = [d.get("spatial",  0.0) for d in round_decisions]
        alpha_vals    = [d.get("alpha",    0.5) for d in round_decisions]
        delta_m_vals  = [d.get("delta_m",  0.0) for d in round_decisions]
        h_t_contrib   = [d.get("h_temporal_contrib", 0.0) for d in round_decisions]
        h_s_contrib   = [d.get("h_spatial_contrib",  0.0) for d in round_decisions]

        log_dict.update({
            "estimator/h_mean":    float(np.mean(h_vals)),
            "estimator/h_min":     float(np.min(h_vals)),
            "estimator/h_max":     float(np.max(h_vals)),
            "estimator/h_std":     float(np.std(h_vals)),
            "estimator/epochs_mean": float(np.mean(epochs_vals)),
            "estimator/epochs_min":  float(np.min(epochs_vals)),
            "estimator/epochs_max":  float(np.max(epochs_vals)),
            "estimator/n_participating": n_participating,
            "estimator/n_skipped":   n_skipped,
            "estimator/mean_h_pool":    _mean_h,
            "estimator/median_h_pool":  _median_h,
            "estimator/skip_threshold": _skip_thr,
            "estimator/skip_rate": n_skipped / max(n_participating + n_skipped, 1),
            # h_hat components
            "estimator/temporal_mean": float(np.mean(temporal_vals)),
            "estimator/spatial_mean":  float(np.mean(spatial_vals)),
            "estimator/alpha_mean":    float(np.mean(alpha_vals)),
            "estimator/delta_m_mean":  float(np.mean(delta_m_vals)),
            "estimator/h_temporal_contrib_mean": float(np.mean(h_t_contrib)),
            "estimator/h_spatial_contrib_mean":  float(np.mean(h_s_contrib)),
            "estimator/geo_map_val_mean": float(np.mean(
                [d.get("geo_map_val", 0.0) for d in round_decisions])),
            "estimator/client_ema_val_mean": float(np.mean(
                [d.get("client_ema_val", 0.0) for d in round_decisions])),
            "estimator/budget_saving": config.BUDGET_SAVING,
        })
        return log_dict


# ================================================================
#  FEDPROX -- Client
# ================================================================

class FedProxClient(fl.client.NumPyClient):
    """
    FedProx client.

    Difference from FedAvg: the local loss includes the proximal term
    that penalizes drift relative to the global model:

    L_fedprox = L_task(w) + (mu/2) * ||w - w_global||^2

    Parameter mu: controls the trade-off
      mu = 0     -> FedAvg (no penalty)
      mu large   -> local model stays close to the global one
    """

    def __init__(
        self,
        client_id: int,
        dataset:   TransitionPredictionDataset,
        profile:   dict,
        mu:        float = config.FEDPROX_MU,
        device:    str   = "cpu",
        dataset_name: str = None,
    ):
        self.cid     = client_id
        self.dataset = dataset
        self.profile = profile
        self.mu      = mu
        self.device  = device
        self._dataset_name = dataset_name
        torch.manual_seed(config.ACTUAL_SEED)
        self.model   = create_model(self._dataset_name)

    def get_parameters(self, cfg):
        return get_parameters(self.model)

    def fit(self, parameters, cfg):
        round_num = cfg.get("round", 0)
        n_epochs  = cfg.get("local_epochs", config.LOCAL_EPOCHS_BASE)
        _seed_worker(config.ACTUAL_SEED, self.cid, round_num)
        # dynamic mu injected by the estimator plugin
        mu_override = cfg.get("mu_dynamic", None)
        if mu_override is not None:
            self.mu = float(mu_override)

        # Rebuild position from the scalar fields
        if "pos_lat" in cfg:
            position = _cfg_to_pos(cfg)
        else:
            position = self.profile["position_seq"][
                (round_num - 1) % config.N_ROUNDS
            ]

        # Keep the global model for the proximal term
        global_params = [p.copy() for p in parameters]
        set_parameters(self.model, parameters)

        # MOBIS: client_user_id in cfg -> the participant's own dataset
        if cfg.get("client_user_id"):
            local_data = get_client_dataset(
                self.dataset,
                user_id=cfg["client_user_id"],
            )
        else:
            local_data = get_local_dataset(
                self.dataset, position, radius_km=3.0
            )
        if len(local_data) == 0:
            return get_parameters(self.model), 0, {
                "skipped": True
            }

        # Deterministic generator tied to the global seed and the client
        g = torch.Generator()
        g.manual_seed(config.ACTUAL_SEED + int(self.cid))

        loader = DataLoader(
            local_data, batch_size=config.BATCH_SIZE,
            shuffle=True, num_workers=0,
            collate_fn=collate_with_mode,
            generator   = g  # <-- determinism fix
        )

        metrics = self._train_fedprox(
            loader, global_params, n_epochs
        )
        local_p    = get_parameters(self.model)
        divergence = compute_gradient_divergence(
            local_p, global_params
        )

        # Free memory
        del global_params

        return local_p, len(local_data), {
            "loss":           float(metrics["loss"]),
            "proximal_term":  float(metrics["proximal_term"]),
            "divergence":     float(divergence),
            "gradient_steps": int(metrics.get("gradient_steps", 0)),
            "local_epochs":   int(n_epochs),
            "n_samples":      len(local_data),
            "client_id":      self.cid,
            "round":          round_num,
            "mobility_group": self.profile["mobility_group"],
            "mode":           position.get("mode", "unknown"),
            "velocity":       float(self.profile["velocity_seq"][
                                  (round_num - 1) % config.N_ROUNDS
                              ]),
        }

    def evaluate(self, parameters, cfg):
        round_num = cfg.get("round", 0)

        if "pos_lat" in cfg:
            position = _cfg_to_pos(cfg)
        else:
            position = self.profile["position_seq"][
                (round_num - 1) % config.N_ROUNDS
            ]

        set_parameters(self.model, parameters)
        _uid  = self.profile.get("user_id")
        _type = self.profile.get("type", "")
        if _uid and _type == "real_mobis":
            eval_data = get_client_dataset(self.dataset, user_id=_uid)
        else:
            eval_data = get_local_dataset(
                self.dataset, position, radius_km=3.0
            )
        if len(eval_data) == 0:
            return 0.0, 0, {"mae_dist": 0.0, "mae_time": 0.0}

        loader  = DataLoader(eval_data, batch_size=32,
                              shuffle=False, num_workers=0,
                              collate_fn=collate_with_mode)
        metrics = evaluate_model(
            self.model, loader, self.dataset, self.device
        )
        extra = {}
        for mode, vals in metrics.get(
                "mae_by_mode", {}).items():
            extra[f"mode_dist_{mode}"] = float(vals["mae_dist"])
            extra[f"mode_time_{mode}"] = float(vals["mae_time"])
            extra[f"mode_n_{mode}"]    = float(vals["n"])
        return float(metrics["loss"]), len(eval_data), {
            "mae_dist": float(metrics["mae_dist"]),
            "mae_time": float(metrics["mae_time"]),
            **extra,
        }

    def _train_fedprox(
        self,
        loader:        DataLoader,
        global_params: list[np.ndarray],
        n_epochs:      int,
    ) -> dict:
        """
        Training with the proximal loss:
        L = MSE(pred, target) + (mu/2) * ||w - w_global||^2
        """
        self.model.to(self.device)
        self.model.train()

        # Frozen reference of the global model
        global_model = create_model(self._dataset_name).to(self.device)
        set_parameters(global_model, global_params)
        for p in global_model.parameters():
            p.requires_grad = False

        # Unified SGD + momentum -- aligns with the FedProx original paper (Li et al., MLSys 2020).
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr           = config.LEARNING_RATE,
            momentum     = config.SGD_MOMENTUM,
            weight_decay = config.SGD_WEIGHT_DECAY,
        )

        total_loss, total_prox, total_n, gradient_steps = 0.0, 0.0, 0, 0

        for _ in range(n_epochs):
            for batch in loader:
                features = batch[0].to(self.device)
                targets  = batch[1].to(self.device)

                optimizer.zero_grad()
                preds = self.model(features)

                # Task loss (Huber or MSE, dataset-dependent -- see model.py)
                task_loss = compute_task_loss(preds, targets, self._dataset_name)

                # Proximal term: penalizes drift from the global model
                prox_term = 0.0
                for w, w_g in zip(
                    self.model.parameters(),
                    global_model.parameters(),
                ):
                    prox_term += ((w - w_g) ** 2).sum()
                prox_term = (self.mu / 2) * prox_term

                loss = task_loss + prox_term
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )
                optimizer.step()

                total_loss += task_loss.item() * len(targets)
                total_prox += prox_term.item() * len(targets)
                total_n    += len(targets)
                gradient_steps += 1

        # Cleanup -- global_model is a temporary copy, free it
        del global_model, optimizer
        self.model.cpu()

        return {
            "loss":          total_loss / max(total_n, 1),
            "proximal_term": total_prox / max(total_n, 1),
            "gradient_steps":  gradient_steps,
        }


# ================================================================
#  SCAFFOLD -- Client
# ================================================================

class SCAFFOLDClient(fl.client.NumPyClient):
    """
    SCAFFOLD client.

    Keeps a local control variate c_k that estimates the expected
    local gradient. The corrected gradient is:

    corrected_grad = grad F_k(w) - c_k + c_global

    The control variate c_k is updated after every round:
    c_k_new = c_k - c_global + (w_global - w_local) / (K * lr)
    """

    def __init__(
        self,
        client_id: int,
        dataset:   TransitionPredictionDataset,
        profile:   dict,
        device:    str = "cpu",
        dataset_name: str = None,
    ):
        self.cid     = client_id
        self.dataset = dataset
        self.profile = profile
        self.device  = device
        self._dataset_name = dataset_name
        torch.manual_seed(config.ACTUAL_SEED)
        self.model   = create_model(self._dataset_name)

        # Local control variate c_k
        self.control_variate: Optional[list[np.ndarray]] = None

    def get_parameters(self, cfg):
        return get_parameters(self.model)

    def fit(self, parameters, cfg):
        round_num  = cfg.get("round", 0)
        n_epochs   = cfg.get("local_epochs", config.LOCAL_EPOCHS_BASE)
        _seed_worker(config.ACTUAL_SEED, self.cid, round_num)
        # bypass mode -- uses plain SGD when h_hat indicates a stable region
        use_bypass = cfg.get("scaffold_bypass", False)

        # Rebuild position from the scalar fields
        if "pos_lat" in cfg:
            position = _cfg_to_pos(cfg)
        else:
            position = self.profile["position_seq"][
                (round_num - 1) % config.N_ROUNDS
            ]

        # Receive the global control variate (base64 str)
        c_global_flat = cfg.get("c_global", None)

        global_params = [p.copy() for p in parameters]
        set_parameters(self.model, parameters)

        # Initialize c_k with zeros on the first round
        if self.control_variate is None:
            self.control_variate = [
                np.zeros_like(p) for p in parameters
            ]

        # Deserialize c_global
        if c_global_flat is not None:
            c_global = _deserialize_control(
                c_global_flat, parameters
            )
        else:
            c_global = [np.zeros_like(p) for p in parameters]

        local_data = get_local_dataset(
            self.dataset, position, radius_km=3.0
        ) if not cfg.get("client_user_id") else get_client_dataset(
            self.dataset, user_id=cfg["client_user_id"]
        )
        if len(local_data) == 0:
            return get_parameters(self.model), 0, {
                "skipped":   True,
                "c_k_delta": _serialize_control(
                    [np.zeros_like(p)
                     for p in self.control_variate]
                ),
            }

        # Deterministic generator tied to the global seed and the client
        g = torch.Generator()
        g.manual_seed(config.ACTUAL_SEED + int(self.cid))

        loader = DataLoader(
            local_data, batch_size=config.BATCH_SIZE,
            shuffle=True, num_workers=0,
            collate_fn=collate_with_mode,
            generator   = g  # <-- determinism fix
        )

        metrics, n_steps = self._train_scaffold(
            loader, c_global, n_epochs, bypass=use_bypass
        )

        local_p = get_parameters(self.model)

        # c_k update — skip if bypass mode (no control variates used)
        if use_bypass:
            c_k_delta = [np.zeros_like(p) for p in local_p]
        else:
            # c_k_new = c_k - c_global + (w_before - w_after) / (K*lr)
            c_k_delta = []
            for ck, cg, w_b, w_a in zip(
                self.control_variate,
                c_global,
                global_params,
                local_p,
            ):
                delta = (
                    -cg
                    + (w_b - w_a) / max(n_steps * config.LEARNING_RATE,
                                         1e-8)
                )
                c_k_delta.append(delta)
                ck += delta

        divergence = compute_gradient_divergence(
            local_p, global_params
        )

        # Free memory -- release large temporary arrays
        del global_params, c_global, c_global_flat
        self.model.cpu()

        return local_p, len(local_data), {
            "loss":           float(metrics["loss"]),
            "divergence":     float(divergence),
            "gradient_steps": int(metrics.get("gradient_steps", 0)),
            "scaffold_steps": int(metrics.get("scaffold_steps", 0)),
            "sgd_steps":      int(metrics.get("sgd_steps", 0)),
            "bypass":         use_bypass,
            "local_epochs":   int(n_epochs),
            "n_samples":      len(local_data),
            "client_id":      self.cid,
            "round":          round_num,
            "mobility_group": self.profile["mobility_group"],
            "mode":           position.get("mode", "unknown"),
            "velocity":       float(self.profile["velocity_seq"][
                                  (round_num - 1) % config.N_ROUNDS
                              ]),
            # Send the c_k delta to the server (base64 str)
            "c_k_delta":      _serialize_control(c_k_delta),
        }

    def evaluate(self, parameters, cfg):
        round_num = cfg.get("round", 0)

        if "pos_lat" in cfg:
            position = _cfg_to_pos(cfg)
        else:
            position = self.profile["position_seq"][
                (round_num - 1) % config.N_ROUNDS
            ]

        set_parameters(self.model, parameters)
        _uid  = self.profile.get("user_id")
        _type = self.profile.get("type", "")
        if _uid and _type == "real_mobis":
            eval_data = get_client_dataset(self.dataset, user_id=_uid)
        else:
            eval_data = get_local_dataset(
                self.dataset, position, radius_km=3.0
            )
        if len(eval_data) == 0:
            return 0.0, 0, {"mae_dist": 0.0, "mae_time": 0.0}

        loader  = DataLoader(eval_data, batch_size=32,
                              shuffle=False, num_workers=0,
                              collate_fn=collate_with_mode)
        metrics = evaluate_model(
            self.model, loader, self.dataset, self.device
        )
        extra = {}
        for mode, vals in metrics.get(
                "mae_by_mode", {}).items():
            extra[f"mode_dist_{mode}"] = float(vals["mae_dist"])
            extra[f"mode_time_{mode}"] = float(vals["mae_time"])
            extra[f"mode_n_{mode}"]    = float(vals["n"])
        return float(metrics["loss"]), len(eval_data), {
            "mae_dist": float(metrics["mae_dist"]),
            "mae_time": float(metrics["mae_time"]),
            **extra,
        }

    def _train_scaffold(
        self,
        loader:   DataLoader,
        c_global: list[np.ndarray],
        n_epochs: int,
        bypass:   bool = False,
    ) -> tuple[dict, int]:
        """
        SCAFFOLD training with the corrected gradient:
        corrected_grad = grad F_k(w) - c_k + c_global

        If bypass=True, uses plain SGD (no control-variate correction)
        to save compute.
        """
        self.model.to(self.device)
        self.model.train()

        # SCAFFOLD uses momentum=0.0 intentionally.
        # Momentum accumulates a velocity buffer across federated rounds,
        # contaminating the SCAFFOLD correction grad ← grad - c_k + c_global.
        # SCAFFOLD's own control variates provide the acceleration mechanism —
        # external momentum is redundant and causes instability.
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr           = config.LEARNING_RATE,
            momentum     = 0.0,
            weight_decay = config.SGD_WEIGHT_DECAY,
        )

        total_loss, total_n, n_steps = 0.0, 0, 0
        scaffold_steps, sgd_steps = 0, 0

        if not bypass:
            ck_t = [torch.tensor(c, dtype=torch.float32,
                                  device=self.device)
                    for c in self.control_variate]
            cg_t = [torch.tensor(c, dtype=torch.float32,
                                  device=self.device)
                    for c in c_global]

        for _ in range(n_epochs):
            for batch in loader:
                features = batch[0].to(self.device)
                targets  = batch[1].to(self.device)

                optimizer.zero_grad()
                preds = self.model(features)
                loss  = compute_task_loss(preds, targets, self._dataset_name)
                loss.backward()

                if not bypass:
                    # SCAFFOLD correction: grad <- grad - c_k + c_global
                    with torch.no_grad():
                        for param, ck, cg in zip(
                            self.model.parameters(), ck_t, cg_t
                        ):
                            if param.grad is not None:
                                param.grad.add_(-ck + cg)
                    scaffold_steps += 1
                else:
                    sgd_steps += 1

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )
                optimizer.step()

                total_loss += loss.item() * len(targets)
                total_n    += len(targets)
                n_steps    += 1

        # Cleanup -- temporary control-variate tensors and optimizer
        if not bypass:
            del ck_t, cg_t
        del optimizer

        return (
            {"loss": total_loss / max(total_n, 1),
             "gradient_steps": n_steps,
             "scaffold_steps": scaffold_steps,
             "sgd_steps": sgd_steps},
            max(n_steps, 1),
        )


# ================================================================
#  FLOWER STRATEGIES
# ================================================================

class FedProxStrategy(EstimatorMixin, fl.server.strategy.FedAvg):
    """
    FedProx strategy -- pure or with the estimator plugin.

    When use_estimator=True:
    - Client selection by h_hat
    - Dynamic mu: mu = FEDPROX_MU_MIN + h_hat * (FEDPROX_MU_MAX - FEDPROX_MU_MIN)
    - Epochs proportional to h_hat

    dedicated_channel=False isolates the common channel (selection+epochs
    via h_hat, controlled by use_estimator) from the dedicated channel
    (dynamic mu) -- used by the fedprox_hybrid_static_mu variant, which
    selects/allocates epochs via h_hat but keeps mu fixed (same value as
    fedprox_pure). Goal: isolate what the dedicated channel contributes
    beyond the common channel, which random_skip already tests on its own
    (see KB_DECISAO_ESCOPO_RANDOM_SKIP_v1.md -- random_skip cannot test
    the dedicated channel, only the common one).
    """

    def __init__(self, profiles: dict, wandb_run=None,
                 use_estimator=False, dedicated_channel=True, **kwargs):
        self._dedicated_channel = dedicated_channel
        def _agg_fit(metrics):
            losses = [m.get("loss",       0.0) for _, m in metrics]
            divs   = [m.get("divergence", 0.0) for _, m in metrics]
            return {"loss": float(np.mean(losses)),
                    "divergence": float(np.mean(divs))}
        # Extract dataset before passing **kwargs to Flower
        _dataset = kwargs.pop("dataset", None)
        super().__init__(fit_metrics_aggregation_fn=_agg_fit, **kwargs)
        self.round_log:  list[dict] = []
        self.total_grad_steps: int = 0
        self._val_log:   dict[int, float] = {}
        self._pbar       = None
        self._wandb_run  = wandb_run
        self._val_ds     = None
        self._device     = "cpu"
        self.global_eval_log: list[dict] = []
        self._final_params = None
        self._plugin_init(profiles, use_estimator, dataset=_dataset)
        # memory and time monitor
        from resource_monitor import ResourceMonitor
        self._monitor = ResourceMonitor()

    def configure_fit(self, server_round, parameters,
                       client_manager):
                       
        import random
        random.seed(config.ACTUAL_SEED + server_round)
               
        all_clients = client_manager.sample(
            num_clients     = min(config.N_CLIENTS,
                                   client_manager.num_available()),
            min_num_clients = config.MIN_FIT_CLIENTS,
        )

        all_clients = sorted(all_clients, key=lambda c: int(c.cid))

        selected = self._plugin_configure_fit(
            server_round, parameters, all_clients,
            {cid: mc.profile for cid, mc in self.mobility_clients.items()}
        )

        ins_list = []
        for item in selected:
            if len(item) == 5:
                client, cid, position, est, n_epochs = item
            else:
                client, cid, position, est, n_epochs = *item, config.LOCAL_EPOCHS_BASE

            cfg = {
                "round":        server_round,
                "local_epochs": n_epochs,
                **_pos_to_cfg(position),
            }

            # dynamic mu proportional to h_hat -- only if the dedicated
            # channel is active (dedicated_channel=False keeps mu fixed,
            # same as fedprox_pure, while selection/epochs stay via h_hat)
            if self._use_estimator and self._dedicated_channel:
                h = est.get("estimate", 0.5) if isinstance(est, dict) else 0.5
                mu_dyn = (config.FEDPROX_MU_MIN
                          + h * (config.FEDPROX_MU_MAX - config.FEDPROX_MU_MIN))
                cfg["mu_dynamic"] = float(mu_dyn)

            # MOBIS: the real participant's user_id -- same reason as
            # FedAvgStrategy. Empty for GeoLife (original behavior).
            cfg["client_user_id"] = str(
                self.mobility_clients[cid].profile.get("user_id", "")
            ) if (cid in self.mobility_clients and
                  self.mobility_clients[cid].profile.get("type", "")
                  == "real_mobis") else ""

            ins_list.append((client, flc.FitIns(parameters, cfg)))
        return ins_list

    def aggregate_fit(self, server_round, results, failures):
        self._monitor.start_round()  # round timing
        if self._pbar is None:
            try:
                from tqdm import tqdm as _tqdm
                self._pbar = _tqdm(
                    total=config.N_ROUNDS, desc=f"  FedProx       ",
                    unit="rd", dynamic_ncols=True, colour="green",
                    postfix={"loss":"—","val":"—","div":"—","MAE":"—"},
                )
            except ImportError: pass

        if not results:
            return None, {}
        # Clients dispatched with no local samples -- logged before being
        # discarded (same logic as FedAvgStrategy.aggregate_fit).
        _no_data_results = [(cp, r) for cp, r in results if r.num_examples == 0]
        results = [(cp, r) for cp, r in results if r.num_examples > 0]
        _log_train_no_data(
            self.client_log, self._client_consecutive,
            self._configure_fit_state, self.mobility_clients,
            server_round, _no_data_results,
        )
        if not results:
            print(f"[WARN] FedProx round {server_round}: 0 samples -- skipped.")
            return None, {}
        results = sorted(results, key=lambda x: int(x[0].cid))
        agg, _ = super().aggregate_fit(
            server_round, results, failures
        )
        losses = [r.metrics.get("loss", 0)
                  for _, r in results]
        divs   = [r.metrics.get("divergence", 0)
                  for _, r in results]
        round_grad_steps = sum(r.metrics.get("gradient_steps", config.LOCAL_EPOCHS_BASE) for _, r in results if r.num_examples > 0)
        self.total_grad_steps += round_grad_steps
        # Estimator metrics (from this round's configure_fit)
        _em = getattr(self, "_last_estimator_metrics", {})

        log = {
            "round":     server_round,
            "mean_loss": float(np.mean(losses)),
            "mean_div":  float(np.mean(divs)),
            "n_clients": len(results),
            "round_grad_steps": round_grad_steps,
            "total_grad_steps": self.total_grad_steps,
            # Estimator / cost
            "n_skipped":       _em.get("n_skipped",       0),
            "n_participating": _em.get("n_participating", len(results)),
            "skip_rate":       _em.get("skip_rate",       0.0),
            "h_mean":          _em.get("h_mean",          0.0),
            "h_std":           _em.get("h_std",           0.0),
            "epochs_mean":     _em.get("epochs_mean",     0.0),
            "temporal_mean":           _em.get("temporal_mean",           0.0),
            "spatial_mean":            _em.get("spatial_mean",            0.0),
            "alpha_mean":              _em.get("alpha_mean",              0.5),
            "delta_m_mean":            _em.get("delta_m_mean",            0.0),
            "h_temporal_contrib_mean": _em.get("h_temporal_contrib_mean", 0.0),
            "h_spatial_contrib_mean":  _em.get("h_spatial_contrib_mean",  0.0),
            "geo_map_val_mean":        _em.get("geo_map_val_mean",        0.0),
            "client_ema_val_mean":     _em.get("client_ema_val_mean",     0.0),
            "budget_saving":           _em.get("budget_saving", config.BUDGET_SAVING),
            "map_coverage_pct":        _em.get("map_coverage_pct",        0.0),
            # Global gradient stats aggregated per round
            # Extracted directly from results for compatibility with every algorithm
            "grad_norm_pre_mean":  float(np.mean([r.metrics.get("grad_norm_pre_mean",  0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
            "grad_norm_pre_std":   float(np.mean([r.metrics.get("grad_norm_pre_std",   0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
            "grad_norm_post_mean": float(np.mean([r.metrics.get("grad_norm_post_mean", 0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
            "grad_clip_rate":      float(np.mean([r.metrics.get("grad_clip_rate",      0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
        }
        self.round_log.append(log)
        # -- Unified validation every VAL_EVERY rounds ------------
        if (self._val_ds is not None
                and server_round % config.VAL_EVERY == 0):
            from run_experiment import _global_eval_callback
            ge = _global_eval_callback(
                server_round, agg,
                self._val_ds, self._device,
                self._wandb_run,
                method_name="fedprox",
            )
            if ge:
                self.global_eval_log.append(ge)
        if self._wandb_run is not None:
            try:
                n_active = sum(1 for _, fr in results if fr.num_examples > 0)
                log_dict = {
                    "train/loss":      log["mean_loss"],
                    "train/div":       log["mean_div"],
                    "train/n_clients": log["n_clients"],
                    "cost/round_grad_steps": log.get("round_grad_steps", 0),
                    "cost/total_grad_steps": log.get("total_grad_steps", 0),
                    "cost/n_active_clients": n_active,
                }
                log_dict = self._plugin_wandb_log(
                    self._wandb_run, server_round, log_dict)
                log_dict.update(self._monitor.end_round())
                self._wandb_run.log(log_dict, step=server_round)
            except Exception:
                pass
        if self._pbar is not None:
            val_str = (
                f"{self._val_log[server_round - 1]:.4f}"
                if (server_round - 1) in self._val_log else "—"
            )
            self._pbar.set_postfix(
                loss=f"{log['mean_loss']:.4f}",
                val =val_str,
                div =f"{log['mean_div']:.4f}",
                MAE ="—", refresh=False,
            )
            self._pbar.update(1)
        else:
            print(f"  [FedProx R{server_round:03d}] "
                  f"loss={log['mean_loss']:.4f} "
                  f"div={log['mean_div']:.4f}")
        self._final_params = agg
        self._plugin_post_aggregate(server_round, results,
            {cid: mc.profile for cid, mc in self.mobility_clients.items()})
        return agg, log

    def configure_evaluate(self, server_round, parameters,
                            client_manager):
        # gate by VAL_EVERY
        if server_round % config.VAL_EVERY != 0:
            return []
        clients = client_manager.sample(
            num_clients     = config.MIN_EVAL_CLIENTS,
            min_num_clients = config.MIN_EVAL_CLIENTS,
        )
        return [(c, flc.EvaluateIns(parameters, {
            "round": server_round,
            **_pos_to_cfg(
                self.mobility_clients[int(c.cid)].get_position(server_round)
            ),
        })) for c in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        results = sorted(results, key=lambda x: int(x[0].cid))
        mae_dists = [r.metrics.get("mae_dist", 0)
                     for _, r in results]
        mae_times = [r.metrics.get("mae_time", 0)
                     for _, r in results]
        val_loss = float(np.mean([r.loss for _, r in results]))
        self._val_log[server_round] = val_loss

        from collections import defaultdict as _dd
        mode_dist: dict = _dd(list)
        mode_time: dict = _dd(list)
        for _, r in results:
            for key, val in r.metrics.items():
                if key.startswith("mode_dist_"):
                    mode_dist[key[10:]].append(val)
                elif key.startswith("mode_time_"):
                    mode_time[key[10:]].append(val)
        mae_by_mode = {
            m: {"mae_dist": float(np.mean(mode_dist[m])),
                "mae_time": float(np.mean(mode_time[m]))}
            for m in mode_dist
        }

        if self._wandb_run is not None:
            try:
                log_dict = {
                    "eval/val_loss": val_loss,
                    "eval/mae_dist": float(np.mean(mae_dists)),
                    "eval/mae_time": float(np.mean(mae_times)),
                }
                for mode, vals in mae_by_mode.items():
                    log_dict[f"eval/mae_dist_{mode}"] = vals["mae_dist"]
                    log_dict[f"eval/mae_time_{mode}"] = vals["mae_time"]
                self._wandb_run.log(log_dict, step=server_round)
            except Exception:
                pass
        if self._pbar is not None:
            self._pbar.set_postfix(
                loss=self._pbar.postfix.get("loss", "—"),
                val =f"{val_loss:.4f}",
                div =self._pbar.postfix.get("div",  "—"),
                MAE =f"{np.mean(mae_dists):.0f}m",
                refresh=True,
            )
        else:
            print(f"  [FedProx Eval R{server_round:03d}] "
                  f"val_loss={val_loss:.4f}  "
                  f"MAE dist={np.mean(mae_dists):.1f}m "
                  f"MAE time={np.mean(mae_times):.1f}s")
        return (val_loss,
                {"mae_dist": float(np.mean(mae_dists)),
                 "mae_time": float(np.mean(mae_times)),
                 **{f"mae_dist_{m}": v["mae_dist"]
                    for m, v in mae_by_mode.items()},
                 **{f"mae_time_{m}": v["mae_time"]
                    for m, v in mae_by_mode.items()}})


class SCAFFOLDStrategy(EstimatorMixin, fl.server.strategy.FedAvg):
    """
    SCAFFOLD strategy -- pure or with the estimator plugin.

    When use_estimator=True:
    - Client selection by h_hat
    - Bypass: clients with h_hat < SCAFFOLD_BYPASS_THRESHOLD use plain SGD
    - Epochs proportional to h_hat
    - Tracks scaffold_steps vs sgd_steps separately

    dedicated_channel=False isolates the common channel (selection+epochs
    via h_hat, controlled by use_estimator) from the dedicated channel
    (adaptive bypass) -- used by the scaffold_hybrid_no_bypass variant,
    which selects/allocates epochs via h_hat but never triggers the
    bypass (SCAFFOLD correction always active, same as scaffold_pure).
    Same goal as FedProxStrategy's dedicated_channel -- isolate what the
    dedicated channel contributes.
    """

    def __init__(self, profiles: dict,
                 n_clients: int = config.N_CLIENTS,
                 wandb_run=None,
                 use_estimator=False,
                 dedicated_channel=True,
                 **kwargs):
        self._dedicated_channel = dedicated_channel
        def _agg_fit(metrics):
            losses = [m.get("loss",       0.0) for _, m in metrics]
            divs   = [m.get("divergence", 0.0) for _, m in metrics]
            return {"loss": float(np.mean(losses)),
                    "divergence": float(np.mean(divs))}
        _dataset = kwargs.pop("dataset", None)
        super().__init__(fit_metrics_aggregation_fn=_agg_fit, **kwargs)
        self.round_log: list[dict] = []
        self.total_grad_steps: int = 0
        self.total_scaffold_steps: int = 0
        self.total_sgd_steps: int = 0
        self._val_log: dict[int, float] = {}
        self._pbar      = None
        self._wandb_run = wandb_run
        self._val_ds    = None
        self._device    = "cpu"
        self.global_eval_log: list[dict] = []
        self._final_params = None
        self._c_global: Optional[list[np.ndarray]] = None
        self._n_clients = n_clients
        self._client_controls: dict[int, list[np.ndarray]] = {}
        self._plugin_init(profiles, use_estimator, dataset=_dataset)
        # memory and time monitor
        from resource_monitor import ResourceMonitor
        self._monitor = ResourceMonitor()

    def configure_fit(self, server_round, parameters,
                       client_manager):

        import random
        random.seed(config.ACTUAL_SEED + server_round)

        all_clients = client_manager.sample(
            num_clients     = min(config.N_CLIENTS,
                                   client_manager.num_available()),
            min_num_clients = config.MIN_FIT_CLIENTS,
        )
        
        all_clients = sorted(all_clients, key=lambda c: int(c.cid))

        selected = self._plugin_configure_fit(
            server_round, parameters, all_clients,
            {cid: mc.profile for cid, mc in self.mobility_clients.items()}
        )

        c_global_serialized = (
            _serialize_control(self._c_global)
            if self._c_global is not None
            else None
        )

        ins_list = []
        for item in selected:
            if len(item) == 5:
                client, cid, position, est, n_epochs = item
            else:
                client, cid, position, est, n_epochs = *item, config.LOCAL_EPOCHS_BASE

            cfg = {
                "round":        server_round,
                "local_epochs": n_epochs,
                **_pos_to_cfg(position),
            }
            if c_global_serialized is not None:
                cfg["c_global"] = c_global_serialized

            # bypass when h_hat indicates a stable region -- only if the
            # dedicated channel is active (dedicated_channel=False never
            # triggers the bypass, same as scaffold_pure, while
            # selection/epochs stay via h_hat)
            if self._use_estimator and self._dedicated_channel:
                h = est.get("estimate", 0.5) if isinstance(est, dict) else 0.5
                if h < config.SCAFFOLD_BYPASS_THRESHOLD:
                    cfg["scaffold_bypass"] = True
                else:
                    cfg["scaffold_bypass"] = False

            # MOBIS: the real participant's user_id -- same reason as the
            # other strategies. Empty for GeoLife (original behavior).
            cfg["client_user_id"] = str(
                self.mobility_clients[cid].profile.get("user_id", "")
            ) if (cid in self.mobility_clients and
                  self.mobility_clients[cid].profile.get("type", "")
                  == "real_mobis") else ""

            ins_list.append((client, flc.FitIns(parameters, cfg)))
        return ins_list

    def aggregate_fit(self, server_round, results, failures):
        self._monitor.start_round()  # round timing
        if self._pbar is None:
            try:
                from tqdm import tqdm as _tqdm
                self._pbar = _tqdm(
                    total=config.N_ROUNDS, desc=f"  SCAFFOLD      ",
                    unit="rd", dynamic_ncols=True, colour="green",
                    postfix={"loss":"—","val":"—","div":"—","MAE":"—"},
                )
            except ImportError: pass

        if not results:
            return None, {}
        # Clients dispatched with no local samples -- logged before being
        # discarded (same logic as FedAvgStrategy.aggregate_fit).
        _no_data_results = [(cp, r) for cp, r in results if r.num_examples == 0]
        results = [(cp, r) for cp, r in results if r.num_examples > 0]
        _log_train_no_data(
            self.client_log, self._client_consecutive,
            self._configure_fit_state, self.mobility_clients,
            server_round, _no_data_results,
        )
        if not results:
            print(f"[WARN] SCAFFOLD round {server_round}: 0 samples -- skipped.")
            return None, {}

        # Deterministic order -- see the equivalent note in FedAvgStrategy.aggregate_fit
        results = sorted(results, key=lambda x: int(x[0].cid))

        agg_params, _ = super().aggregate_fit(
            server_round, results, failures
        )
        if agg_params is None:
            return None, {}

        # B4: update per-client c_k and recompute c_global as mean
        agg_p = flc.parameters_to_ndarrays(agg_params)
        results = sorted(results, key=lambda x: int(x[0].cid))
        for _, fit_res in results:
            cid = fit_res.metrics.get("client_id", -1)
            delta_flat = fit_res.metrics.get("c_k_delta", None)
            if delta_flat is not None and cid >= 0:
                delta = _deserialize_control(delta_flat, agg_p)
                if cid not in self._client_controls:
                    self._client_controls[cid] = [
                        np.zeros_like(p) for p in agg_p
                    ]
                for i in range(len(self._client_controls[cid])):
                    self._client_controls[cid][i] += delta[i]

        # Recompute c_global as mean of ALL stored c_k
        if self._client_controls:
            all_ck = list(self._client_controls.values())
            n_stored = len(all_ck)
            self._c_global = [
                sum(ck[i] for ck in all_ck) / n_stored
                for i in range(len(all_ck[0]))
            ]
        results = sorted(results, key=lambda x: int(x[0].cid))
        losses = [r.metrics.get("loss", 0)
                  for _, r in results]
        divs   = [r.metrics.get("divergence", 0)
                  for _, r in results]
        round_grad_steps = sum(r.metrics.get("gradient_steps", config.LOCAL_EPOCHS_BASE) for _, r in results if r.num_examples > 0)
        # separate accounting: scaffold steps vs sgd steps
        round_scaffold_steps = sum(r.metrics.get("scaffold_steps", 0) for _, r in results)
        round_sgd_steps = sum(r.metrics.get("sgd_steps", 0) for _, r in results)
        self.total_grad_steps += round_grad_steps
        self.total_scaffold_steps += round_scaffold_steps
        self.total_sgd_steps += round_sgd_steps
        # Estimator metrics (from this round's configure_fit)
        _em = getattr(self, "_last_estimator_metrics", {})

        log = {
            "round":     server_round,
            "mean_loss": float(np.mean(losses)),
            "mean_div":  float(np.mean(divs)),
            "n_clients": len(results),
            "round_grad_steps": round_grad_steps,
            "total_grad_steps": self.total_grad_steps,
            "round_scaffold_steps": round_scaffold_steps,
            "round_sgd_steps": round_sgd_steps,
            "total_scaffold_steps": self.total_scaffold_steps,
            "total_sgd_steps": self.total_sgd_steps,
            # Estimator / cost
            "n_skipped":       _em.get("n_skipped",       0),
            "n_participating": _em.get("n_participating", len(results)),
            "skip_rate":       _em.get("skip_rate",       0.0),
            "h_mean":          _em.get("h_mean",          0.0),
            "h_std":           _em.get("h_std",           0.0),
            "epochs_mean":     _em.get("epochs_mean",     0.0),
            "temporal_mean":           _em.get("temporal_mean",           0.0),
            "spatial_mean":            _em.get("spatial_mean",            0.0),
            "alpha_mean":              _em.get("alpha_mean",              0.5),
            "delta_m_mean":            _em.get("delta_m_mean",            0.0),
            "h_temporal_contrib_mean": _em.get("h_temporal_contrib_mean", 0.0),
            "h_spatial_contrib_mean":  _em.get("h_spatial_contrib_mean",  0.0),
            "geo_map_val_mean":        _em.get("geo_map_val_mean",        0.0),
            "client_ema_val_mean":     _em.get("client_ema_val_mean",     0.0),
            "budget_saving":           _em.get("budget_saving", config.BUDGET_SAVING),
            "map_coverage_pct":        _em.get("map_coverage_pct",        0.0),
            # Global gradient stats aggregated per round
            # Extracted directly from results for compatibility with every algorithm
            "grad_norm_pre_mean":  float(np.mean([r.metrics.get("grad_norm_pre_mean",  0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
            "grad_norm_pre_std":   float(np.mean([r.metrics.get("grad_norm_pre_std",   0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
            "grad_norm_post_mean": float(np.mean([r.metrics.get("grad_norm_post_mean", 0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
            "grad_clip_rate":      float(np.mean([r.metrics.get("grad_clip_rate",      0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
        }
        self.round_log.append(log)
        # -- Unified validation every VAL_EVERY rounds ------------
        if (self._val_ds is not None
                and server_round % config.VAL_EVERY == 0):
            from run_experiment import _global_eval_callback
            ge = _global_eval_callback(
                server_round, agg_params,
                self._val_ds, self._device,
                self._wandb_run,
                method_name="scaffold",
            )
            if ge:
                self.global_eval_log.append(ge)
        if self._wandb_run is not None:
            try:
                log_dict = {
                    "train/loss":      log["mean_loss"],
                    "train/div":       log["mean_div"],
                    "train/n_clients": log["n_clients"],
                    "cost/round_grad_steps": log.get("round_grad_steps", 0),
                    "cost/total_grad_steps": log.get("total_grad_steps", 0),
                    "cost/round_scaffold_steps": log.get("round_scaffold_steps", 0),
                    "cost/round_sgd_steps": log.get("round_sgd_steps", 0),
                    "cost/total_scaffold_steps": self.total_scaffold_steps,
                    "cost/total_sgd_steps": self.total_sgd_steps,
                }
                log_dict = self._plugin_wandb_log(
                    self._wandb_run, server_round, log_dict)
                # memory and time metrics
                log_dict.update(self._monitor.end_round())
                self._wandb_run.log(log_dict, step=server_round)
            except Exception:
                pass
        if self._pbar is not None:
            val_str = (
                f"{self._val_log[server_round - 1]:.4f}"
                if (server_round - 1) in self._val_log else "—"
            )
            self._pbar.set_postfix(
                loss=f"{log['mean_loss']:.4f}",
                val =val_str,
                div =f"{log['mean_div']:.4f}",
                MAE ="—", refresh=False,
            )
            self._pbar.update(1)
        else:
            print(f"  [SCAFFOLD R{server_round:03d}] "
                  f"loss={log['mean_loss']:.4f} "
                  f"div={log['mean_div']:.4f}")
        self._final_params = agg_params
        self._plugin_post_aggregate(server_round, results,
            {cid: mc.profile for cid, mc in self.mobility_clients.items()})
        return agg_params, log

    def configure_evaluate(self, server_round, parameters,
                            client_manager):
        # gate by VAL_EVERY
        if server_round % config.VAL_EVERY != 0:
            return []
        clients = client_manager.sample(
            num_clients     = config.MIN_EVAL_CLIENTS,
            min_num_clients = config.MIN_EVAL_CLIENTS,
        )
        return [(c, flc.EvaluateIns(parameters, {
            "round": server_round,
            **_pos_to_cfg(
                self.mobility_clients[int(c.cid)].get_position(server_round)
            ),
        })) for c in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        results = sorted(results, key=lambda x: int(x[0].cid))
        mae_dists = [r.metrics.get("mae_dist", 0)
                     for _, r in results]
        mae_times = [r.metrics.get("mae_time", 0)
                     for _, r in results]
        val_loss  = float(np.mean([r.loss for _, r in results]))
        self._val_log[server_round] = val_loss

        # Agrega MAE real do modelo por modal
        from collections import defaultdict as _dd
        mode_dist: dict = _dd(list)
        mode_time: dict = _dd(list)
        for _, r in results:
            for key, val in r.metrics.items():
                if key.startswith("mode_dist_"):
                    mode_dist[key[10:]].append(val)
                elif key.startswith("mode_time_"):
                    mode_time[key[10:]].append(val)
        mae_by_mode = {
            m: {"mae_dist": float(np.mean(mode_dist[m])),
                "mae_time": float(np.mean(mode_time[m]))}
            for m in mode_dist
        }

        if self._wandb_run is not None:
            try:
                log_dict = {
                    "eval/val_loss": val_loss,
                    "eval/mae_dist": float(np.mean(mae_dists)),
                    "eval/mae_time": float(np.mean(mae_times)),
                }
                for mode, vals in mae_by_mode.items():
                    log_dict[f"eval/mae_dist_{mode}"] = vals["mae_dist"]
                    log_dict[f"eval/mae_time_{mode}"] = vals["mae_time"]
                self._wandb_run.log(log_dict, step=server_round)
            except Exception:
                pass
        if self._pbar is not None:
            self._pbar.set_postfix(
                loss=self._pbar.postfix.get("loss", "—"),
                val =f"{val_loss:.4f}",
                div =self._pbar.postfix.get("div",  "—"),
                MAE =f"{np.mean(mae_dists):.0f}m",
                refresh=True,
            )
        else:
            print(f"  [SCAFFOLD Eval R{server_round:03d}] "
                  f"val_loss={val_loss:.4f}  "
                  f"MAE dist={np.mean(mae_dists):.1f}m "
                  f"MAE time={np.mean(mae_times):.1f}s")
        return (val_loss,
                {"mae_dist": float(np.mean(mae_dists)),
                 "mae_time": float(np.mean(mae_times)),
                 **{f"mae_dist_{m}": v["mae_dist"]
                    for m, v in mae_by_mode.items()},
                 **{f"mae_time_{m}": v["mae_time"]
                    for m, v in mae_by_mode.items()}})



# ================================================================
#  FEDDYN (Acar et al., ICLR 2021, arXiv:2111.04263)
# ================================================================

class FedDynClient(fl.client.NumPyClient):
    """
    FedDyn client (Algorithm 1, Acar et al. 2021).

    Minimizes locally:
      L_k(theta) - <prev_grad_k, theta> + (alpha/2)*||theta - theta_global||^2

    `prev_grad_k` is a per-client persistent state (starts at zero),
    updated every round the client participates in:
      prev_grad_k <- prev_grad_k - alpha*(theta_k^new - theta_global)

    alpha can be fixed (config.FEDDYN_ALPHA, when the strategy's
    `dedicated_channel` is off) or dynamic by h_hat (cfg["alpha_dynamic"],
    injected by FedDynStrategy when it's on) -- the same pattern as
    FedProx's dynamic mu.

    **Design history:** an early version had `alpha_delta` use each
    round's dynamic alpha_k, summed into a server-side accumulator (`g`)
    that never resets and was rescaled by a fixed alpha_ref -- this
    compounded an imbalance over 80 rounds and made `feddyn_hybrid`
    diverge. A first fix used a fixed alpha_ref in `alpha_delta` too,
    which resolved the divergence but made the accumulator deaf to
    h_hat (alpha_ref cancels out algebraically in the server's division,
    so it stopped representing anything real).

    **Final design (this one):** `alpha_delta` goes back to using the
    REAL alpha_k for the round (`alpha`, below) -- faithful to what
    actually produced the delta. What changes is the SERVER side
    (`FedDynStrategy.aggregate_fit`): instead of dividing `g` by an
    external constant, it divides by the real running average of the
    alpha_k values that actually went into `g` -- this keeps h_hat
    responsiveness in both channels (the proximal term's strength here,
    and its weight in the persistent accumulator) without
    reintroducing the imbalance. In `pure`/`hybrid_static_alpha`,
    alpha_k is always `config.FEDDYN_ALPHA` -- behavior is unchanged on
    both sides. This isn't an exact match to the paper's derivation
    under a time-varying alpha (the paper assumes a single global
    alpha; there's no documented "correct" generalization -- the same
    caveat already noted for FedProx's dynamic mu) -- it's the most
    defensible engineering choice found.
    """

    def __init__(
        self,
        client_id: int,
        dataset:   TransitionPredictionDataset,
        profile:   dict,
        device:    str = "cpu",
        dataset_name: str = None,
    ):
        self.cid     = client_id
        self.dataset = dataset
        self.profile = profile
        self.device  = device
        self._dataset_name = dataset_name
        torch.manual_seed(config.ACTUAL_SEED)
        self.model   = create_model(self._dataset_name)

        # Persistent state: tracked gradient (prev_grad_k)
        self.prev_grad: Optional[list[np.ndarray]] = None

    def get_parameters(self, cfg):
        return get_parameters(self.model)

    def fit(self, parameters, cfg):
        round_num = cfg.get("round", 0)
        n_epochs  = cfg.get("local_epochs", config.LOCAL_EPOCHS_BASE)
        _seed_worker(config.ACTUAL_SEED, self.cid, round_num)
        # dynamic alpha injected by the estimator plugin (dedicated channel)
        alpha = float(cfg.get("alpha_dynamic", config.FEDDYN_ALPHA))

        # Rebuild position from the scalar fields
        if "pos_lat" in cfg:
            position = _cfg_to_pos(cfg)
        else:
            position = self.profile["position_seq"][
                (round_num - 1) % config.N_ROUNDS
            ]

        # Keep the global model for the quadratic term and the prev_grad update
        global_params = [p.copy() for p in parameters]
        set_parameters(self.model, parameters)

        # Initialize prev_grad_k with zeros on the first participation
        if self.prev_grad is None:
            self.prev_grad = [np.zeros_like(p) for p in parameters]

        local_data = get_local_dataset(
            self.dataset, position, radius_km=3.0
        ) if not cfg.get("client_user_id") else get_client_dataset(
            self.dataset, user_id=cfg["client_user_id"]
        )
        if len(local_data) == 0:
            return get_parameters(self.model), 0, {
                "skipped":     True,
                "alpha_used":  alpha,
                "alpha_delta": _serialize_control(
                    [np.zeros_like(p) for p in self.prev_grad]
                ),
            }

        # Deterministic generator tied to the global seed and the client
        g = torch.Generator()
        g.manual_seed(config.ACTUAL_SEED + int(self.cid))

        loader = DataLoader(
            local_data, batch_size=config.BATCH_SIZE,
            shuffle=True, num_workers=0,
            collate_fn=collate_with_mode,
            generator   = g  # <-- determinism fix
        )

        metrics = self._train_feddyn(
            loader, global_params, n_epochs, alpha
        )
        local_p = get_parameters(self.model)

        # alpha_delta = alpha_k*(theta_new - theta_global) -- uses this
        # round's REAL `alpha` (dynamic, not a fixed reference); see the
        # class docstring for why. Sent to the server to update g
        # (FedDynStrategy.aggregate_fit) and also updates prev_grad_k
        # locally (subtracts the same delta).
        alpha_delta = [
            alpha * (local_p[i] - global_params[i])
            for i in range(len(local_p))
        ]
        for i in range(len(self.prev_grad)):
            self.prev_grad[i] = self.prev_grad[i] - alpha_delta[i]

        divergence = compute_gradient_divergence(
            local_p, global_params
        )

        del global_params

        return local_p, len(local_data), {
            "loss":           float(metrics["loss"]),
            "divergence":     float(divergence),
            "gradient_steps": int(metrics.get("gradient_steps", 0)),
            "local_epochs":   int(n_epochs),
            "n_samples":      len(local_data),
            "client_id":      self.cid,
            "round":          round_num,
            "mobility_group": self.profile["mobility_group"],
            "mode":           position.get("mode", "unknown"),
            "velocity":       float(self.profile["velocity_seq"][
                                  (round_num - 1) % config.N_ROUNDS
                              ]),
            "alpha_used":     alpha,
            # Sends alpha_k*(theta_new-theta_global) to the server (base64
            # str) -- this round's REAL alpha, also used by the server to
            # track the running average of alpha (see aggregate_fit)
            "alpha_delta":    _serialize_control(alpha_delta),
        }

    def evaluate(self, parameters, cfg):
        round_num = cfg.get("round", 0)

        if "pos_lat" in cfg:
            position = _cfg_to_pos(cfg)
        else:
            position = self.profile["position_seq"][
                (round_num - 1) % config.N_ROUNDS
            ]

        set_parameters(self.model, parameters)
        _uid  = self.profile.get("user_id")
        _type = self.profile.get("type", "")
        if _uid and _type == "real_mobis":
            eval_data = get_client_dataset(self.dataset, user_id=_uid)
        else:
            eval_data = get_local_dataset(
                self.dataset, position, radius_km=3.0
            )
        if len(eval_data) == 0:
            return 0.0, 0, {"mae_dist": 0.0, "mae_time": 0.0}

        loader  = DataLoader(eval_data, batch_size=32,
                              shuffle=False, num_workers=0,
                              collate_fn=collate_with_mode)
        metrics = evaluate_model(
            self.model, loader, self.dataset, self.device
        )
        extra = {}
        for mode, vals in metrics.get(
                "mae_by_mode", {}).items():
            extra[f"mode_dist_{mode}"] = float(vals["mae_dist"])
            extra[f"mode_time_{mode}"] = float(vals["mae_time"])
            extra[f"mode_n_{mode}"]    = float(vals["n"])
        return float(metrics["loss"]), len(eval_data), {
            "mae_dist": float(metrics["mae_dist"]),
            "mae_time": float(metrics["mae_time"]),
            **extra,
        }

    def _train_feddyn(
        self,
        loader:        DataLoader,
        global_params: list[np.ndarray],
        n_epochs:      int,
        alpha:         float,
    ) -> dict:
        """
        Training with FedDyn's dynamic regularization:
        L = L_task(theta) - <prev_grad, theta> + (alpha/2)*||theta - theta_global||^2
        """
        self.model.to(self.device)
        self.model.train()

        # Frozen reference of the global model (quadratic term)
        global_model = create_model(self._dataset_name).to(self.device)
        set_parameters(global_model, global_params)
        for p in global_model.parameters():
            p.requires_grad = False

        # Client's tracked gradient, frozen (linear term)
        prev_grad_t = [
            torch.tensor(g, dtype=torch.float32, device=self.device)
            for g in self.prev_grad
        ]

        # Same optimizer as the other algorithms (SGD+momentum) -- the
        # linear/quadratic term is already part of the loss, no special
        # optimizer is needed.
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr           = config.LEARNING_RATE,
            momentum     = config.SGD_MOMENTUM,
            weight_decay = config.SGD_WEIGHT_DECAY,
        )

        total_loss, total_lin, total_quad, total_n, gradient_steps = \
            0.0, 0.0, 0.0, 0, 0

        for _ in range(n_epochs):
            for batch in loader:
                features = batch[0].to(self.device)
                targets  = batch[1].to(self.device)

                optimizer.zero_grad()
                preds = self.model(features)

                task_loss = compute_task_loss(preds, targets, self._dataset_name)  # same loss as FedProx/SCAFFOLD/FedNova

                lin_term  = 0.0
                quad_term = 0.0
                for w, w_g, pg in zip(
                    self.model.parameters(),
                    global_model.parameters(),
                    prev_grad_t,
                ):
                    lin_term  += (pg * w).sum()
                    quad_term += ((w - w_g) ** 2).sum()
                quad_term = (alpha / 2) * quad_term

                loss = task_loss - lin_term + quad_term
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )
                optimizer.step()

                total_loss += task_loss.item() * len(targets)
                total_lin  += float(lin_term) * len(targets) if isinstance(lin_term, torch.Tensor) else 0.0
                total_quad += quad_term.item() * len(targets)
                total_n    += len(targets)
                gradient_steps += 1

        del global_model, optimizer, prev_grad_t
        self.model.cpu()

        return {
            "loss":          total_loss / max(total_n, 1),
            "linear_term":   total_lin  / max(total_n, 1),
            "quadratic_term": total_quad / max(total_n, 1),
            "gradient_steps": gradient_steps,
        }


class FedDynStrategy(EstimatorMixin, fl.server.strategy.FedAvg):
    """
    FedDyn strategy (Acar et al., ICLR 2021) -- pure or with the
    estimator plugin.

    When use_estimator=True:
    - Client selection and epochs via h_hat (common channel, same as the others)
    When use_estimator=True AND dedicated_channel=True (default):
    - Dynamic alpha: alpha = FEDDYN_ALPHA_MIN + h_hat * (FEDDYN_ALPHA_MAX - FEDDYN_ALPHA_MIN)
      (the same pattern as FedProx's dynamic mu)
    When dedicated_channel=False: alpha stays fixed at FEDDYN_ALPHA even
    with use_estimator=True -- isolates the common channel's contribution
    on its own ("FedNova-style": h_hat only decides selection/epochs,
    nothing else).

    **The server aggregation is NOT a weighted FedAvg** -- it follows
    Algorithm 1 from the paper (Acar et al. 2021), generalized to a
    per-client/per-round alpha:
      theta^t = UNweighted_average(theta_k^t, k in P^t) - g^t/alpha_bar
      g^t = g^{t-1} - (1/N_total)*sum_{k in P^t} alpha_k*(theta_k^t - theta^{t-1})
      alpha_bar = running average of every alpha_k that has gone into g
    Each client sends `alpha_k*(theta_k^t-theta^{t-1})` pre-computed
    (`alpha_delta` in the fit metrics, using that round's REAL alpha_k --
    see `FedDynClient.fit`) along with `alpha_used` (alpha_k itself); the
    server sums the deltas into `g` AND accumulates `alpha_used` into a
    running average (`self._alpha_sum`/`self._alpha_n`), used as the
    divisor instead of a fixed constant. With alpha_k always equal to
    `config.FEDDYN_ALPHA` (`pure`/`hybrid_static_alpha`), alpha_bar
    converges exactly to `FEDDYN_ALPHA` -- reducing to the paper's
    original formula, with unchanged behavior.

    **Design history:** an early version's rescale divided by a FIXED
    `alpha_ref` (`config.FEDDYN_ALPHA`) while `g` accumulated a
    per-term DYNAMIC alpha_k -- this breaks the algebraic identity that
    cancels alpha out in the rescale (it only cancels when EVERY term
    uses the SAME alpha). With alpha_k varying per client/round, each
    term carried an implicit `alpha_k/alpha_ref` weight, and since `g`
    never resets (it sums from round 1), that imbalance compounded over
    80 rounds until training diverged. Fixing it by also using a fixed
    `alpha_ref` in the client's `alpha_delta` resolved the divergence
    but made the accumulator deaf to h_hat (alpha_ref cancels out
    algebraically in the division, so it no longer represents anything
    real).
    **Final design (this one):** `alpha_delta` goes back to using the
    REAL alpha_k (faithful to what produced that delta -- both of
    h_hat's channels stay active: the local proximal term's strength,
    and its weight in the accumulator), and it's the RESCALE that
    self-normalizes by the real running average of the alpha_k values
    that actually went into `g`, instead of a misaligned external
    constant. This preserves h_hat responsiveness without
    reintroducing the imbalance. The minimum alpha_k is
    `FEDDYN_ALPHA_MIN=0.001` (never zero) -- no risk of division by
    zero; with dozens of clients per round the average converges
    quickly to the real mean of h_hat. This isn't an exact match to the
    paper under a time-varying alpha (there's no documented "correct"
    generalization -- the same caveat already noted for FedProx's
    dynamic mu) -- it's the most defensible engineering choice found:
    stable, and faithful to the alpha actually used rather than a
    fictitious reference.
    """

    def __init__(self, profiles: dict,
                 n_clients: int = config.N_CLIENTS,
                 wandb_run=None,
                 use_estimator=False,
                 dedicated_channel=True,
                 **kwargs):
        def _agg_fit(metrics):
            losses = [m.get("loss",       0.0) for _, m in metrics]
            divs   = [m.get("divergence", 0.0) for _, m in metrics]
            return {"loss": float(np.mean(losses)),
                    "divergence": float(np.mean(divs))}
        _dataset = kwargs.pop("dataset", None)
        super().__init__(fit_metrics_aggregation_fn=_agg_fit, **kwargs)
        self.round_log: list[dict] = []
        self.total_grad_steps: int = 0
        self._val_log: dict[int, float] = {}
        self._pbar      = None
        self._wandb_run = wandb_run
        self._val_ds    = None
        self._device    = "cpu"
        self.global_eval_log: list[dict] = []
        self._final_params = None
        self._g: Optional[list[np.ndarray]] = None   # accumulated state g=h/alpha
        self._alpha_sum: float = 0.0    # running sum of the alpha_k values actually used in g
        self._alpha_n:   int   = 0      # number of client-round contributions summed into g
        self._n_clients = n_clients
        self._dedicated_channel = dedicated_channel
        self._plugin_init(profiles, use_estimator, dataset=_dataset)
        from resource_monitor import ResourceMonitor
        self._monitor = ResourceMonitor()

    def configure_fit(self, server_round, parameters,
                       client_manager):

        import random
        random.seed(config.ACTUAL_SEED + server_round)

        all_clients = client_manager.sample(
            num_clients     = min(config.N_CLIENTS,
                                   client_manager.num_available()),
            min_num_clients = config.MIN_FIT_CLIENTS,
        )
        all_clients = sorted(all_clients, key=lambda c: int(c.cid))

        selected = self._plugin_configure_fit(
            server_round, parameters, all_clients,
            {cid: mc.profile for cid, mc in self.mobility_clients.items()}
        )

        ins_list = []
        for item in selected:
            if len(item) == 5:
                client, cid, position, est, n_epochs = item
            else:
                client, cid, position, est, n_epochs = *item, config.LOCAL_EPOCHS_BASE

            cfg = {
                "round":        server_round,
                "local_epochs": n_epochs,
                **_pos_to_cfg(position),
            }

            # dynamic alpha proportional to h_hat -- only if the dedicated
            # channel is active (dedicated_channel=False -> alpha stays
            # fixed, same value as feddyn_pure, but selection/epochs still go via h_hat)
            if self._use_estimator and self._dedicated_channel:
                h = est.get("estimate", 0.5) if isinstance(est, dict) else 0.5
                alpha_dyn = (config.FEDDYN_ALPHA_MIN
                             + h * (config.FEDDYN_ALPHA_MAX - config.FEDDYN_ALPHA_MIN))
                cfg["alpha_dynamic"] = float(alpha_dyn)

            cfg["client_user_id"] = str(
                self.mobility_clients[cid].profile.get("user_id", "")
            ) if (cid in self.mobility_clients and
                  self.mobility_clients[cid].profile.get("type", "")
                  == "real_mobis") else ""

            ins_list.append((client, flc.FitIns(parameters, cfg)))
        return ins_list

    def aggregate_fit(self, server_round, results, failures):
        self._monitor.start_round()
        if self._pbar is None:
            try:
                from tqdm import tqdm as _tqdm
                self._pbar = _tqdm(
                    total=config.N_ROUNDS, desc=f"  FedDyn        ",
                    unit="rd", dynamic_ncols=True, colour="green",
                    postfix={"loss":"—","val":"—","div":"—","MAE":"—"},
                )
            except ImportError: pass

        if not results:
            return None, {}
        _no_data_results = [(cp, r) for cp, r in results if r.num_examples == 0]
        results = [(cp, r) for cp, r in results if r.num_examples > 0]
        _log_train_no_data(
            self.client_log, self._client_consecutive,
            self._configure_fit_state, self.mobility_clients,
            server_round, _no_data_results,
        )
        if not results:
            print(f"[WARN] FedDyn round {server_round}: 0 samples -- skipped.")
            return None, {}
        results = sorted(results, key=lambda x: int(x[0].cid))

        # -- FedDyn aggregation (Algorithm 1) -- NOT a weighted FedAvg --
        client_params = [
            flc.parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]
        n_shapes = len(client_params[0])

        # theta^t = UNweighted average of theta_k^t (every client weighs the same, not by n_k)
        theta_mean = [
            np.mean([cp[i] for cp in client_params], axis=0)
            for i in range(n_shapes)
        ]

        # g^t = g^{t-1} - (1/N_total)*sum(alpha_delta_k) (already computed by the client)
        if self._g is None:
            self._g = [np.zeros_like(p) for p in theta_mean]

        sum_alpha_delta = [np.zeros_like(p) for p in theta_mean]
        for _, fit_res in results:
            delta_flat = fit_res.metrics.get("alpha_delta", None)
            if delta_flat is not None:
                delta = _deserialize_control(delta_flat, theta_mean)
                for i in range(n_shapes):
                    sum_alpha_delta[i] += delta[i]

        for i in range(n_shapes):
            self._g[i] = (
                self._g[i] - sum_alpha_delta[i] / max(self._n_clients, 1)
            )

        # Tracks the running average of the alpha_k values REALLY used in
        # `g` (only the clients that contributed, i.e. `results` already
        # filtered by num_examples>0) -- this is what the rescale below
        # uses, not an external constant. See the class docstring for the reasoning.
        for _, fit_res in results:
            a_used = fit_res.metrics.get("alpha_used", None)
            if a_used is not None:
                self._alpha_sum += float(a_used)
                self._alpha_n   += 1
        alpha_ref_dynamic = self._alpha_sum / max(self._alpha_n, 1)

        # theta^t = mean(theta_k^t) - g^t/alpha_bar. `g` accumulates
        # alpha_k*delta_k using each contribution's REAL alpha; dividing
        # by the real running average (alpha_bar, not an external
        # constant) keeps the division dimensionally consistent with what
        # actually went into `g`, avoiding the imbalance that caused
        # divergence without deafening the accumulator to h_hat. In
        # `pure`/`hybrid_static_alpha`, alpha_k is always FEDDYN_ALPHA ->
        # alpha_bar converges exactly to FEDDYN_ALPHA, unchanged behavior.
        theta_new = [
            theta_mean[i] - self._g[i] / alpha_ref_dynamic
            for i in range(n_shapes)
        ]
        agg_params = flc.ndarrays_to_parameters(theta_new)

        losses = [r.metrics.get("loss", 0)
                  for _, r in results]
        divs   = [r.metrics.get("divergence", 0)
                  for _, r in results]
        round_grad_steps = sum(r.metrics.get("gradient_steps", config.LOCAL_EPOCHS_BASE) for _, r in results if r.num_examples > 0)
        self.total_grad_steps += round_grad_steps
        alpha_used_vals = [r.metrics.get("alpha_used", config.FEDDYN_ALPHA)
                            for _, r in results]
        # Estimator metrics (from this round's configure_fit)
        _em = getattr(self, "_last_estimator_metrics", {})

        log = {
            "round":     server_round,
            "mean_loss": float(np.mean(losses)),
            "mean_div":  float(np.mean(divs)),
            "n_clients": len(results),
            "round_grad_steps": round_grad_steps,
            "total_grad_steps": self.total_grad_steps,
            "alpha_used_mean": float(np.mean(alpha_used_vals)),
            "alpha_used_std":  float(np.std(alpha_used_vals)),
            # Estimator / cost
            "n_skipped":       _em.get("n_skipped",       0),
            "n_participating": _em.get("n_participating", len(results)),
            "skip_rate":       _em.get("skip_rate",       0.0),
            "h_mean":          _em.get("h_mean",          0.0),
            "h_std":           _em.get("h_std",           0.0),
            "epochs_mean":     _em.get("epochs_mean",     0.0),
            "temporal_mean":           _em.get("temporal_mean",           0.0),
            "spatial_mean":            _em.get("spatial_mean",            0.0),
            "alpha_mean":              _em.get("alpha_mean",              0.5),
            "delta_m_mean":            _em.get("delta_m_mean",            0.0),
            "h_temporal_contrib_mean": _em.get("h_temporal_contrib_mean", 0.0),
            "h_spatial_contrib_mean":  _em.get("h_spatial_contrib_mean",  0.0),
            "geo_map_val_mean":        _em.get("geo_map_val_mean",        0.0),
            "client_ema_val_mean":     _em.get("client_ema_val_mean",     0.0),
            "budget_saving":           _em.get("budget_saving", config.BUDGET_SAVING),
            "map_coverage_pct":        _em.get("map_coverage_pct",        0.0),
            "grad_norm_pre_mean":  float(np.mean([r.metrics.get("grad_norm_pre_mean",  0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
            "grad_norm_pre_std":   float(np.mean([r.metrics.get("grad_norm_pre_std",   0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
            "grad_norm_post_mean": float(np.mean([r.metrics.get("grad_norm_post_mean", 0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
            "grad_clip_rate":      float(np.mean([r.metrics.get("grad_clip_rate",      0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
        }
        self.round_log.append(log)
        if (self._val_ds is not None
                and server_round % config.VAL_EVERY == 0):
            from run_experiment import _global_eval_callback
            ge = _global_eval_callback(
                server_round, agg_params,
                self._val_ds, self._device,
                self._wandb_run,
                method_name="feddyn",
            )
            if ge:
                self.global_eval_log.append(ge)
        if self._wandb_run is not None:
            try:
                n_active = sum(1 for _, fr in results if fr.num_examples > 0)
                log_dict = {
                    "train/loss":      log["mean_loss"],
                    "train/div":       log["mean_div"],
                    "train/n_clients": log["n_clients"],
                    "cost/round_grad_steps": log.get("round_grad_steps", 0),
                    "cost/total_grad_steps": log.get("total_grad_steps", 0),
                    "cost/n_active_clients": n_active,
                    "feddyn/alpha_used_mean": log["alpha_used_mean"],
                }
                log_dict = self._plugin_wandb_log(
                    self._wandb_run, server_round, log_dict)
                log_dict.update(self._monitor.end_round())
                self._wandb_run.log(log_dict, step=server_round)
            except Exception:
                pass
        if self._pbar is not None:
            val_str = (
                f"{self._val_log[server_round - 1]:.4f}"
                if (server_round - 1) in self._val_log else "—"
            )
            self._pbar.set_postfix(
                loss=f"{log['mean_loss']:.4f}",
                val =val_str,
                div =f"{log['mean_div']:.4f}",
                MAE ="—", refresh=False,
            )
            self._pbar.update(1)
        else:
            print(f"  [FedDyn R{server_round:03d}] "
                  f"loss={log['mean_loss']:.4f} "
                  f"div={log['mean_div']:.4f}")
        self._final_params = agg_params
        self._plugin_post_aggregate(server_round, results,
            {cid: mc.profile for cid, mc in self.mobility_clients.items()})
        return agg_params, log

    def configure_evaluate(self, server_round, parameters,
                            client_manager):
        if server_round % config.VAL_EVERY != 0:
            return []
        clients = client_manager.sample(
            num_clients     = config.MIN_EVAL_CLIENTS,
            min_num_clients = config.MIN_EVAL_CLIENTS,
        )
        return [(c, flc.EvaluateIns(parameters, {
            "round": server_round,
            **_pos_to_cfg(
                self.mobility_clients[int(c.cid)].get_position(server_round)
            ),
        })) for c in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        results = sorted(results, key=lambda x: int(x[0].cid))
        mae_dists = [r.metrics.get("mae_dist", 0)
                     for _, r in results]
        mae_times = [r.metrics.get("mae_time", 0)
                     for _, r in results]
        val_loss = float(np.mean([r.loss for _, r in results]))
        self._val_log[server_round] = val_loss

        from collections import defaultdict as _dd
        mode_dist: dict = _dd(list)
        mode_time: dict = _dd(list)
        for _, r in results:
            for key, val in r.metrics.items():
                if key.startswith("mode_dist_"):
                    mode_dist[key[10:]].append(val)
                elif key.startswith("mode_time_"):
                    mode_time[key[10:]].append(val)
        mae_by_mode = {
            m: {"mae_dist": float(np.mean(mode_dist[m])),
                "mae_time": float(np.mean(mode_time[m]))}
            for m in mode_dist
        }

        if self._wandb_run is not None:
            try:
                log_dict = {
                    "eval/val_loss": val_loss,
                    "eval/mae_dist": float(np.mean(mae_dists)),
                    "eval/mae_time": float(np.mean(mae_times)),
                }
                for mode, vals in mae_by_mode.items():
                    log_dict[f"eval/mae_dist_{mode}"] = vals["mae_dist"]
                    log_dict[f"eval/mae_time_{mode}"] = vals["mae_time"]
                self._wandb_run.log(log_dict, step=server_round)
            except Exception:
                pass
        if self._pbar is not None:
            self._pbar.set_postfix(
                loss=self._pbar.postfix.get("loss", "—"),
                val =f"{val_loss:.4f}",
                div =self._pbar.postfix.get("div",  "—"),
                MAE =f"{np.mean(mae_dists):.0f}m",
                refresh=True,
            )
        else:
            print(f"  [FedDyn Eval R{server_round:03d}] "
                  f"val_loss={val_loss:.4f}  "
                  f"MAE dist={np.mean(mae_dists):.1f}m "
                  f"MAE time={np.mean(mae_times):.1f}s")
        return (val_loss,
                {"mae_dist": float(np.mean(mae_dists)),
                 "mae_time": float(np.mean(mae_times)),
                 **{f"mae_dist_{m}": v["mae_dist"]
                    for m, v in mae_by_mode.items()},
                 **{f"mae_time_{m}": v["mae_time"]
                    for m, v in mae_by_mode.items()}})


# ================================================================
#  FEDNOVA -- Client
# ================================================================

class FedNovaClient(fl.client.NumPyClient):
    """
    FedNova client (Wang et al., NeurIPS 2020).

    Solves the objective-inconsistency problem that arises when
    clients run heterogeneous numbers of local steps -- exactly the
    hybrid estimator's case, which varies epochs per client based on h_hat.

    Mechanism:
      Instead of sending delta_w = w_local - w_global directly,
      FedNova normalizes the update by the number of steps:

        d_k = (1 / tau_k) * sum_{t=0}^{tau_k-1} grad F_k(w_t)

      where tau_k is client k's total number of local steps.
      The server aggregates the normalized updates:

        w_global <- w_global - eta * (sum_k p_k * d_k)

      This guarantees convergence even with heterogeneous steps,
      without needing control variates (SCAFFOLD) or a proximal
      penalty (FedProx).

    Why this is the correct baseline:
      The hybrid estimator uses variable epochs (n_epochs proportional
      to h_hat). Without FedNova, a reviewer might question whether the
      gain comes from the estimator or simply from adaptive epochs.
      FedNova isolates that confound: it also uses heterogeneous steps,
      but without mobility awareness.

    Reference:
      Wang, J. et al. "Tackling the objective inconsistency
      problem in heterogeneous federated optimization."
      NeurIPS 2020.
    """

    def __init__(
        self,
        client_id: int,
        dataset:   TransitionPredictionDataset,
        profile:   dict,
        device:    str = "cpu",
        dataset_name: str = None,
    ):
        self.cid     = client_id
        self.dataset = dataset
        self.profile = profile
        self.device  = device
        self._dataset_name = dataset_name
        torch.manual_seed(config.ACTUAL_SEED)
        self.model   = create_model(self._dataset_name)

    def get_parameters(self, cfg):
        return get_parameters(self.model)

    def fit(self, parameters, cfg):
        round_num = cfg.get("round", 0)
        n_epochs  = cfg.get("local_epochs", config.LOCAL_EPOCHS_BASE)
        _seed_worker(config.ACTUAL_SEED, self.cid, round_num)

        if "pos_lat" in cfg:
            position = _cfg_to_pos(cfg)
        else:
            position = self.profile["position_seq"][
                (round_num - 1) % config.N_ROUNDS
            ]

        global_params = [p.copy() for p in parameters]
        set_parameters(self.model, parameters)

        # MOBIS: client_user_id in cfg -> the participant's own dataset
        if cfg.get("client_user_id"):
            local_data = get_client_dataset(
                self.dataset,
                user_id=cfg["client_user_id"],
            )
        else:
            local_data = get_local_dataset(
                self.dataset, position, radius_km=3.0
            )
        if len(local_data) == 0:
            return get_parameters(self.model), 0, {
                "skipped": True
            }

        # Deterministic generator tied to the global seed and the client
        g = torch.Generator()
        g.manual_seed(config.ACTUAL_SEED + int(self.cid))

        loader = DataLoader(
            local_data, batch_size=config.BATCH_SIZE,
            shuffle=True, num_workers=0,
            collate_fn=collate_with_mode,
            generator   = g  # <-- determinism fix
        )

        # Local training and tau_k step count
        local_p, tau_k, train_loss = self._train_fednova(
            loader, n_epochs
        )

        # Normalized update: d_k = (w_global - w_local) / tau_k
        # Note: we send the raw update + tau_k to the server, which
        # performs the normalization at aggregation time.
        divergence = compute_gradient_divergence(
            local_p, global_params
        )

        # Free memory
        del global_params

        return local_p, len(local_data), {
            "loss":           float(train_loss),
            "divergence":     float(divergence),
            "tau":            tau_k,          # local steps
            "gradient_steps": tau_k,          # alias for the cost metric
            "local_epochs":   int(n_epochs),
            "n_samples":      len(local_data),
            "client_id":      self.cid,
            "round":          round_num,
            "mobility_group": self.profile["mobility_group"],
            "mode":           position.get("mode", "unknown"),
            "velocity":       float(self.profile["velocity_seq"][
                                  (round_num - 1) % config.N_ROUNDS
                              ]),
        }

    def evaluate(self, parameters, cfg):
        round_num = cfg.get("round", 0)

        if "pos_lat" in cfg:
            position = _cfg_to_pos(cfg)
        else:
            position = self.profile["position_seq"][
                (round_num - 1) % config.N_ROUNDS
            ]

        set_parameters(self.model, parameters)
        _uid  = self.profile.get("user_id")
        _type = self.profile.get("type", "")
        if _uid and _type == "real_mobis":
            eval_data = get_client_dataset(self.dataset, user_id=_uid)
        else:
            eval_data = get_local_dataset(
                self.dataset, position, radius_km=3.0
            )
        if len(eval_data) == 0:
            return 0.0, 0, {"mae_dist": 0.0, "mae_time": 0.0}

        loader  = DataLoader(eval_data, batch_size=32,
                              shuffle=False, num_workers=0,
                              collate_fn=collate_with_mode)
        metrics = evaluate_model(
            self.model, loader, self.dataset, self.device
        )
        extra = {}
        for mode, vals in metrics.get(
                "mae_by_mode", {}).items():
            extra[f"mode_dist_{mode}"] = float(vals["mae_dist"])
            extra[f"mode_time_{mode}"] = float(vals["mae_time"])
            extra[f"mode_n_{mode}"]    = float(vals["n"])
        return float(metrics["loss"]), len(eval_data), {
            "mae_dist": float(metrics["mae_dist"]),
            "mae_time": float(metrics["mae_time"]),
            **extra,
        }

    def _train_fednova(
        self,
        loader:   DataLoader,
        n_epochs: int,
    ) -> tuple[list[np.ndarray], int, float]:
        """
        Standard local SGD training.
        Returns: (local_parameters, n_steps_tau, mean_loss)
        The server uses tau to normalize the update.
        """
        self.model.to(self.device)
        self.model.train()

        # momentum=0.0 intentionally, like SCAFFOLD. FedNova was proposed
        # for vanilla SGD (Wang et al., NeurIPS 2020). With momentum > 0,
        # local steps stop being independent -- the momentum buffer
        # persists across batches/epochs, contaminating the tau_k
        # normalization. FedNova already has its own correction mechanism
        # via tau_k -- external momentum is redundant and causes
        # instability at aggregation, especially with variable pools
        # (the estimator's skip).
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr           = config.LEARNING_RATE,
            momentum     = 0.0,           # vanilla SGD, as in the original paper
            weight_decay = config.SGD_WEIGHT_DECAY,
        )

        total_loss, total_n, tau = 0.0, 0, 0

        for _ in range(n_epochs):
            for batch in loader:
                features = batch[0].to(self.device)
                targets  = batch[1].to(self.device)

                optimizer.zero_grad()
                preds = self.model(features)
                loss  = compute_task_loss(preds, targets, self._dataset_name)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )
                optimizer.step()

                total_loss += loss.item() * len(targets)
                total_n    += len(targets)
                tau        += 1

        local_p = get_parameters(self.model)

        # Cleanup -- optimizer and move the model back to CPU
        del optimizer
        self.model.cpu()

        return local_p, max(tau, 1), total_loss / max(total_n, 1)


# ================================================================
#  FEDNOVA -- Strategy
# ================================================================

class FedNovaStrategy(EstimatorMixin, fl.server.strategy.FedAvg):
    """
    FedNova strategy -- pure or with the estimator plugin.

    When use_estimator=True:
    - Client selection by h_hat
    - Dynamic epochs proportional to h_hat (FedNova already normalizes by tau_k)
    """

    def __init__(self, profiles: dict, wandb_run=None,
                 use_estimator=False, **kwargs):
        def _agg_fit(metrics):
            losses = [m.get("loss",       0.0) for _, m in metrics]
            divs   = [m.get("divergence", 0.0) for _, m in metrics]
            return {"loss": float(np.mean(losses)),
                    "divergence": float(np.mean(divs))}
        _dataset = kwargs.pop("dataset", None)
        super().__init__(fit_metrics_aggregation_fn=_agg_fit, **kwargs)
        self.round_log: list[dict] = []
        self.total_grad_steps: int = 0
        self._val_log:  dict[int, float] = {}
        self._pbar      = None
        self._wandb_run = wandb_run
        self._val_ds    = None
        self._device    = "cpu"
        self.global_eval_log: list[dict] = []
        self._final_params = None
        self._prev_global_weights = None
        self._plugin_init(profiles, use_estimator, dataset=_dataset)
        # memory and time monitor
        from resource_monitor import ResourceMonitor
        self._monitor = ResourceMonitor()

    def configure_fit(self, server_round, parameters,
                       client_manager):
        # Capture the global weights before sending them to clients.
        # Flower consumes initial_parameters after the 1st round, which
        # would otherwise cause a NoneType error in FedNova's aggregate_fit.
        if self._prev_global_weights is None and parameters is not None:
            self._prev_global_weights = flc.parameters_to_ndarrays(
                parameters
            )
            
        import random
        random.seed(config.ACTUAL_SEED + server_round)

        all_clients = client_manager.sample(
            num_clients     = min(config.N_CLIENTS,
                                   client_manager.num_available()),
            min_num_clients = config.MIN_FIT_CLIENTS,
        )
        
        all_clients = sorted(all_clients, key=lambda c: int(c.cid))

        selected = self._plugin_configure_fit(
            server_round, parameters, all_clients,
            {cid: mc.profile for cid, mc in self.mobility_clients.items()}
        )

        ins_list = []
        for item in selected:
            if len(item) == 5:
                client, cid, position, est, n_epochs = item
            else:
                client, cid, position, est, n_epochs = *item, config.LOCAL_EPOCHS_BASE

            cfg = {
                "round":        server_round,
                "local_epochs": n_epochs,
                **_pos_to_cfg(position),
            }
            # MOBIS: the real participant's user_id -- same reason as the
            # other strategies. Empty for GeoLife (original behavior).
            cfg["client_user_id"] = str(
                self.mobility_clients[cid].profile.get("user_id", "")
            ) if (cid in self.mobility_clients and
                  self.mobility_clients[cid].profile.get("type", "")
                  == "real_mobis") else ""
            ins_list.append((client, flc.FitIns(parameters, cfg)))
        return ins_list

    def aggregate_fit(self, server_round, results, failures):
        self._monitor.start_round()  # round timing
        # Lazy init of the tqdm progress bar
        if self._pbar is None:
            try:
                from tqdm import tqdm as _tqdm
                self._pbar = _tqdm(
                    total=config.N_ROUNDS, desc=f"  FedNova       ",
                    unit="rd", dynamic_ncols=True, colour="blue",
                    postfix={"loss":"—","val":"—","div":"—","MAE":"—"},
                )
            except ImportError: pass

        if not results:
            return None, {}

        # -- FedNova aggregation: normalizes by tau_k -----------
        # _prev_global_weights already captured in configure_fit
        w_global = self._prev_global_weights
        results = sorted(results, key=lambda x: int(x[0].cid))

        # Clients dispatched with no local samples -- FedNova doesn't
        # filter `results` (a tau_k=0 weight already neutralizes them in
        # the weighted sum), but _plugin_post_aggregate discards these
        # entries for lacking client_id in fit_res.metrics; logged here
        # before that happens (same logic as the other algorithms).
        _no_data_results = [(cp, fit_res) for cp, fit_res in results
                             if fit_res.num_examples == 0]
        _log_train_no_data(
            self.client_log, self._client_consecutive,
            self._configure_fit_state, self.mobility_clients,
            server_round, _no_data_results,
        )

        total_samples = sum(fit_res.num_examples
                            for _, fit_res in results)
        if total_samples == 0:
            agg_standard, _ = super().aggregate_fit(
                server_round, results, failures
            )
            return agg_standard, {}

        # delta_w_k = w_local_k - w_global
        # SGD with momentum produces: w_local = w_global - lr * sum_t(m_t)
        # where m_t is the momentum buffer. So:
        #   delta_w_k = w_local - w_global = -lr * sum_t(m_t)
        # The delta is already in weight scale (lr is implicitly included).
        #
        # FedNova normalizes by effective steps a_k to make clients with
        # different tau_k comparable:
        #   d_k = delta_w_k / a_k
        #
        # Aggregated: delta = sum_k (n_k/N) * d_k
        # Applied as: w_new = w_global + tau_eff * delta
        #   where tau_eff = sum_k (n_k/N) * a_k (weighted average effective steps)
        #
        # This guarantees the update's scale is equivalent to an SGD run
        # with tau_eff steps on the full dataset.
        #
        # Ref: Wang et al., NeurIPS 2020, Eq. (4)-(5)
        # momentum=0.0 on the client -> a_k = tau_k (exact vanilla SGD).
        # With momentum=0, (1 - 0^tau) / (1 - 0) = 1/1 = 1 for every
        # tau != 0, which would be incorrect -- the correct formula for
        # momentum=0 is a_k = tau_k.
        weighted_deltas = None
        tau_eff = 0.0  # weighted average effective steps

        for _, fit_res in results:
            tau_k    = max(fit_res.metrics.get("tau", 1), 1)
            n_k      = fit_res.num_examples
            w_local  = flc.parameters_to_ndarrays(fit_res.parameters)
            weight   = n_k / total_samples

            # Vanilla SGD -> a_k = tau_k (independent steps)
            a_k = float(tau_k)

            tau_eff += weight * a_k

            # d_k = delta_w_k / a_k (normalized by effective steps)
            if weighted_deltas is None:
                weighted_deltas = [
                    weight * (wl - wg) / a_k
                    for wl, wg in zip(w_local, w_global)
                ]
            else:
                weighted_deltas = [
                    wd + weight * (wl - wg) / a_k
                    for wd, wl, wg in zip(
                        weighted_deltas, w_local, w_global
                    )
                ]

        if weighted_deltas is not None:
            # w_new = w_global + tau_eff * delta
            nova_params = [
                wg + tau_eff * delta
                for wg, delta in zip(w_global, weighted_deltas)
            ]
            agg_params = flc.ndarrays_to_parameters(nova_params)
            # Update the global reference for the next round
            self._prev_global_weights = [p.copy() for p in nova_params]
        else:
            agg_standard, _ = super().aggregate_fit(
                server_round, results, failures
            )
            if agg_standard is not None:
                agg_params = agg_standard
                self._prev_global_weights = flc.parameters_to_ndarrays(
                    agg_params
                )
            else:
                # No valid client -- keep the previous weights
                agg_params = flc.ndarrays_to_parameters(
                    self._prev_global_weights
                )
        
        results = sorted(results, key=lambda x: int(x[0].cid))
        
        losses = [r.metrics.get("loss", 0) for _, r in results]
        divs   = [r.metrics.get("divergence", 0) for _, r in results]
        taus   = [r.metrics.get("tau", 1) for _, r in results]

        round_grad_steps = sum(r.metrics.get("gradient_steps", config.LOCAL_EPOCHS_BASE) for _, r in results if r.num_examples > 0)
        self.total_grad_steps += round_grad_steps
        # Estimator metrics (from this round's configure_fit)
        _em = getattr(self, "_last_estimator_metrics", {})

        log = {
            "round":      server_round,
            "mean_loss":  float(np.mean(losses)),
            "mean_div":   float(np.mean(divs)),
            "mean_tau":   float(np.mean(taus)),
            "n_clients":  len(results),
            "round_grad_steps": round_grad_steps,
            "total_grad_steps": self.total_grad_steps,
            # Estimator / cost
            "n_skipped":       _em.get("n_skipped",       0),
            "n_participating": _em.get("n_participating", len(results)),
            "skip_rate":       _em.get("skip_rate",       0.0),
            "h_mean":          _em.get("h_mean",          0.0),
            "h_std":           _em.get("h_std",           0.0),
            "epochs_mean":     _em.get("epochs_mean",     0.0),
            "temporal_mean":           _em.get("temporal_mean",           0.0),
            "spatial_mean":            _em.get("spatial_mean",            0.0),
            "alpha_mean":              _em.get("alpha_mean",              0.5),
            "delta_m_mean":            _em.get("delta_m_mean",            0.0),
            "h_temporal_contrib_mean": _em.get("h_temporal_contrib_mean", 0.0),
            "h_spatial_contrib_mean":  _em.get("h_spatial_contrib_mean",  0.0),
            "geo_map_val_mean":        _em.get("geo_map_val_mean",        0.0),
            "client_ema_val_mean":     _em.get("client_ema_val_mean",     0.0),
            "budget_saving":           _em.get("budget_saving", config.BUDGET_SAVING),
            "map_coverage_pct":        _em.get("map_coverage_pct",        0.0),
            # Global gradient stats aggregated per round
            # Extracted directly from results for compatibility with every algorithm
            "grad_norm_pre_mean":  float(np.mean([r.metrics.get("grad_norm_pre_mean",  0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
            "grad_norm_pre_std":   float(np.mean([r.metrics.get("grad_norm_pre_std",   0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
            "grad_norm_post_mean": float(np.mean([r.metrics.get("grad_norm_post_mean", 0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
            "grad_clip_rate":      float(np.mean([r.metrics.get("grad_clip_rate",      0.0) for _, r in results if r.num_examples > 0])) if results else 0.0,
        }
        self.round_log.append(log)
        # -- Unified validation every VAL_EVERY rounds ------------
        if (self._val_ds is not None
                and server_round % config.VAL_EVERY == 0):
            from run_experiment import _global_eval_callback
            ge = _global_eval_callback(
                server_round, agg_params,
                self._val_ds, self._device,
                self._wandb_run,
                method_name="fednova",
            )
            if ge:
                self.global_eval_log.append(ge)
        if self._wandb_run is not None:
            try:
                n_active = sum(1 for _, fr in results if fr.num_examples > 0)
                log_dict = {
                    "train/loss":      log["mean_loss"],
                    "train/div":       log["mean_div"],
                    "train/n_clients": log["n_clients"],
                    "cost/round_grad_steps": log.get("round_grad_steps", 0),
                    "cost/total_grad_steps": log.get("total_grad_steps", 0),
                    "cost/n_active_clients": n_active,
                }
                log_dict = self._plugin_wandb_log(
                    self._wandb_run, server_round, log_dict)
                log_dict.update(self._monitor.end_round())
                self._wandb_run.log(log_dict, step=server_round)
            except Exception:
                pass

        if self._pbar is not None:
            val_str = (
                f"{self._val_log[server_round - 1]:.4f}"
                if (server_round - 1) in self._val_log else "—"
            )
            self._pbar.set_postfix(
                loss=f"{log['mean_loss']:.4f}",
                val =val_str,
                div =f"{log['mean_div']:.4f}",
                MAE ="—", refresh=False,
            )
            self._pbar.update(1)
        else:
            print(f"  [FedNova R{server_round:03d}] "
                  f"loss={log['mean_loss']:.4f} "
                  f"div={log['mean_div']:.4f} "
                  f"τ_mean={log['mean_tau']:.1f}")

        self._final_params = agg_params
        self._plugin_post_aggregate(server_round, results,
            {cid: mc.profile for cid, mc in self.mobility_clients.items()})
        return agg_params, log

    def configure_evaluate(self, server_round, parameters,
                            client_manager):
        # gate by VAL_EVERY
        if server_round % config.VAL_EVERY != 0:
            return []
        clients = client_manager.sample(
            num_clients     = config.MIN_EVAL_CLIENTS,
            min_num_clients = config.MIN_EVAL_CLIENTS,
        )
        return [(c, flc.EvaluateIns(parameters, {
            "round": server_round,
            **_pos_to_cfg(
                self.mobility_clients[int(c.cid)].get_position(server_round)
            ),
        })) for c in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        results = sorted(results, key=lambda x: int(x[0].cid))
        mae_dists = [r.metrics.get("mae_dist", 0)
                     for _, r in results]
        mae_times = [r.metrics.get("mae_time", 0)
                     for _, r in results]
        val_loss  = float(np.mean([r.loss for _, r in results]))
        self._val_log[server_round] = val_loss

        from collections import defaultdict as _dd
        mode_dist: dict = _dd(list)
        mode_time: dict = _dd(list)
        for _, r in results:
            for key, val in r.metrics.items():
                if key.startswith("mode_dist_"):
                    mode_dist[key[10:]].append(val)
                elif key.startswith("mode_time_"):
                    mode_time[key[10:]].append(val)
        mae_by_mode = {
            m: {"mae_dist": float(np.mean(mode_dist[m])),
                "mae_time": float(np.mean(mode_time[m]))}
            for m in mode_dist
        }

        if self._wandb_run is not None:
            try:
                log_dict = {
                    "eval/val_loss": val_loss,
                    "eval/mae_dist": float(np.mean(mae_dists)),
                    "eval/mae_time": float(np.mean(mae_times)),
                }
                for mode, vals in mae_by_mode.items():
                    log_dict[f"eval/mae_dist_{mode}"] = vals["mae_dist"]
                    log_dict[f"eval/mae_time_{mode}"] = vals["mae_time"]
                self._wandb_run.log(log_dict, step=server_round)
            except Exception:
                pass
        if self._pbar is not None:
            self._pbar.set_postfix(
                loss=self._pbar.postfix.get("loss", "—"),
                val =f"{val_loss:.4f}",
                div =self._pbar.postfix.get("div",  "—"),
                MAE =f"{np.mean(mae_dists):.0f}m",
                refresh=True,
            )
        else:
            print(f"  [FedNova Eval R{server_round:03d}] "
                  f"val_loss={val_loss:.4f}  "
                  f"MAE dist={np.mean(mae_dists):.1f}m "
                  f"MAE time={np.mean(mae_times):.1f}s")
        return (val_loss,
                {"mae_dist": float(np.mean(mae_dists)),
                 "mae_time": float(np.mean(mae_times)),
                 **{f"mae_dist_{m}": v["mae_dist"]
                    for m, v in mae_by_mode.items()},
                 **{f"mae_time_{m}": v["mae_time"]
                    for m, v in mae_by_mode.items()}})


# -- Factory functions -----------------------------------------------------

# -- Per-worker cache: avoids reloading from disk on every client_fn() --
# Flower reuses Ray actors -- one actor can serve multiple CIDs. The
# cache guarantees the pickle is read once per worker process.


def make_fedprox_client_fn(
    dataset:  TransitionPredictionDataset,
    profiles: dict,
    device:   str = "cpu",
    dataset_name: str = None,
):
    # -- Anti-OOM: shared file cache (see make_client_fn) --
    import pickle, tempfile, os
    _cache_path = os.path.join(
        tempfile.gettempdir(),
        f"fl_shared_data_seed{config.ACTUAL_SEED}.pkl"
    )
    if not os.path.exists(_cache_path):
        with open(_cache_path, "wb") as f:
            pickle.dump({"dataset": dataset, "profiles": profiles}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    cache_path = _cache_path

    def client_fn(cid: str) -> fl.client.Client:
        # --- Seed the Ray worker ---
        # cudnn.deterministic and benchmark are set by _seed_worker()
        # inside client.fit() -- NOT set here in the closure, since
        # torch.backends.cudnn is a CudnnModule that Ray's cloudpickle
        # cannot serialize.
        worker_seed = config.ACTUAL_SEED + int(cid)
        import random
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)
        # -----------------------------------
        ds, prof = _load_shared_data(cache_path)
        return FedProxClient(
            client_id = int(cid),
            dataset   = ds,
            profile   = prof[int(cid)],
            device    = device,
            dataset_name = dataset_name,
        ).to_client()
    return client_fn


def make_scaffold_client_fn(
    dataset:  TransitionPredictionDataset,
    profiles: dict,
    device:   str = "cpu",
    dataset_name: str = None,
):
    # -- Anti-OOM: shared file cache (see make_client_fn) --
    import pickle, tempfile, os
    _cache_path = os.path.join(
        tempfile.gettempdir(),
        f"fl_shared_data_seed{config.ACTUAL_SEED}.pkl"
    )
    if not os.path.exists(_cache_path):
        with open(_cache_path, "wb") as f:
            pickle.dump({"dataset": dataset, "profiles": profiles}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    cache_path = _cache_path

    def client_fn(cid: str) -> fl.client.Client:
        # --- Seed the Ray worker ---
        worker_seed = config.ACTUAL_SEED + int(cid)
        import random
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)
        # -----------------------------------
        ds, prof = _load_shared_data(cache_path)
        return SCAFFOLDClient(
            client_id = int(cid),
            dataset   = ds,
            profile   = prof[int(cid)],
            device    = device,
            dataset_name = dataset_name,
        ).to_client()
    return client_fn



def make_fednova_client_fn(
    dataset:  TransitionPredictionDataset,
    profiles: dict,
    device:   str = "cpu",
    dataset_name: str = None,
):
    # -- Anti-OOM: shared file cache (see make_client_fn) --
    import pickle, tempfile, os
    _cache_path = os.path.join(
        tempfile.gettempdir(),
        f"fl_shared_data_seed{config.ACTUAL_SEED}.pkl"
    )
    if not os.path.exists(_cache_path):
        with open(_cache_path, "wb") as f:
            pickle.dump({"dataset": dataset, "profiles": profiles}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    cache_path = _cache_path

    def client_fn(cid: str) -> fl.client.Client:
        # --- Seed the Ray worker ---
        worker_seed = config.ACTUAL_SEED + int(cid)
        import random
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)
        # -----------------------------------
        ds, prof = _load_shared_data(cache_path)
        return FedNovaClient(
            client_id = int(cid),
            dataset   = ds,
            profile   = prof[int(cid)],
            device    = device,
            dataset_name = dataset_name,
        ).to_client()
    return client_fn


def make_feddyn_client_fn(
    dataset:  TransitionPredictionDataset,
    profiles: dict,
    device:   str = "cpu",
    dataset_name: str = None,
):
    # -- Anti-OOM: shared file cache (see make_client_fn) --
    import pickle, tempfile, os
    _cache_path = os.path.join(
        tempfile.gettempdir(),
        f"fl_shared_data_seed{config.ACTUAL_SEED}.pkl"
    )
    if not os.path.exists(_cache_path):
        with open(_cache_path, "wb") as f:
            pickle.dump({"dataset": dataset, "profiles": profiles}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    cache_path = _cache_path

    def client_fn(cid: str) -> fl.client.Client:
        # --- Seed the Ray worker ---
        worker_seed = config.ACTUAL_SEED + int(cid)
        import random
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)
        # -----------------------------------
        ds, prof = _load_shared_data(cache_path)
        return FedDynClient(
            client_id = int(cid),
            dataset   = ds,
            profile   = prof[int(cid)],
            device    = device,
            dataset_name = dataset_name,
        ).to_client()
    return client_fn


# -- Control variate serialization (SCAFFOLD) -----------------------------
# Flower's metrics only accept primitive types. To transmit numpy
# control-variate arrays, we serialize them as a base64 str.

import base64
import io

def _serialize_control(
    arrays: list[np.ndarray],
) -> str:
    """Serializes a list of numpy arrays to a base64 string."""
    buf = io.BytesIO()
    np.save(buf, np.array(arrays, dtype=object),
            allow_pickle=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _deserialize_control(
    s:        str,
    template: list[np.ndarray],
) -> list[np.ndarray]:
    """Deserializes a base64 string to a list of numpy arrays."""
    try:
        buf = io.BytesIO(base64.b64decode(s.encode("utf-8")))
        arr = np.load(buf, allow_pickle=True)
        return [np.array(a, dtype=np.float32) for a in arr]
    except Exception:
        return [np.zeros_like(p) for p in template]
