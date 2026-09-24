"""
geolife_loader.py
-----------------
Loads GeoLife and produces:
  1. Training samples: (GPS window -> [dist, time] until transition)
  2. Mobility profiles: position + speed + mode per FL round

GeoLife directory structure on disk after download:
  GEOLIFE_ROOT/
    Data/
      000/
        Trajectory/
          20081023055305.plt   <- GPS trajectory
          ...
        labels.txt             <- transport-mode annotations (only ~70 users have them)
      001/
        ...

.plt file format:
  Line 1-6: header (ignore)
  Line 7+:  lat, lon, 0, altitude, days_since_1899, date, time

labels.txt format:
  Start Time,End Time,Transportation Mode
  2008/10/23 05:53:05,2008/10/23 06:15:30,walk
  2008/10/23 06:15:30,2008/10/23 07:02:00,subway
"""

import os
import pickle
import numpy as np
from datetime import datetime
from typing import Optional

from geo_utils import (haversine, bearing, velocity_kmh,
                        normalize_position)
import config


# -- Main loading -----------------------------------------------

def load_geolife(
    n_clients:   int = config.N_CLIENTS,
    seed:        int = config.EXPERIMENT_SEEDS[0],
    samples_cache: str = config.SAMPLES_CACHE,
    profiles_cache: str = config.PROFILES_CACHE,
) -> tuple[list[dict], dict]:
    """
    Loads GeoLife and returns:
      samples  : list of samples {features, targets, meta}
      profiles : mobility profiles per FL client
    """
    if (os.path.exists(samples_cache)):
        print("[GeoLife] Loading cache...")
        with open(samples_cache, "rb") as f:
            cache_data = pickle.load(f)
        # Handle both old format (list) and new format (dict)
        if isinstance(cache_data, dict):
            samples = cache_data["samples"]
            profiles = cache_data["profiles"]
            val_samples = cache_data.get("val_samples", [])
            val_profiles = cache_data.get("val_profiles", {})
        else:
            # Old format compatibility
            samples = cache_data
            profiles = {}
            val_samples = []
            val_profiles = {}
            if os.path.exists(profiles_cache):
                with open(profiles_cache, "rb") as f:
                    profiles = pickle.load(f)
        print(f"  {len(samples)} training samples, "
              f"{len(profiles)} training profiles, "
              f"{len(val_samples)} val samples, "
              f"{len(val_profiles)} val profiles")
        return samples, profiles, val_samples, val_profiles

    _check_geolife_files()

    data_dir = _resolved_data_dir  # resolved by _check_geolife_files
    user_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    # Keep only users that have labels.txt
    labeled_users = [
        u for u in user_dirs
        if os.path.exists(
            os.path.join(data_dir, u, "labels.txt")
        )
    ]

    np.random.seed(seed)
    np.random.shuffle(labeled_users)
    # select extra clients for validation-only (never train)
    n_val = getattr(config, "N_VAL_CLIENTS", 0)
    n_total_needed = n_clients + n_val
    selected = labeled_users[:n_total_needed]

    print(f"[GeoLife] {len(labeled_users)} users with labels, "
          f"using {len(selected)} ({n_clients} train + {n_val} val)")

    all_samples = []
    val_samples = []
    profiles    = {}
    val_profiles = {}
    cid_seq = 0
    val_cid_seq = 0

    for i, user_id in enumerate(selected):
        user_path = os.path.join(data_dir, user_id)
        print(f"  User {i:02d} ({user_id})...", end=" ")

        annotated = _load_annotated_trajectories(user_path)
        if not annotated:
            print("no valid data, skipping")
            continue

        user_samples = _extract_samples(annotated,
                                         user_id=user_id)
        if len(user_samples) == 0:
            print("0 samples (no valid transitions), skipping")
            continue

        # validation clients are filled first — guarantees val_samples is never empty
        if val_cid_seq < n_val:
            val_samples.extend(user_samples)
            val_profiles[val_cid_seq] = _build_profile(
                annotated, config.N_ROUNDS, user_id
            )
            print(f"{len(user_samples)} samples [validation]")
            val_cid_seq += 1
        else:
            all_samples.extend(user_samples)
            profiles[cid_seq] = _build_profile(
                annotated, config.N_ROUNDS, user_id
            )
            print(f"{len(user_samples)} samples [train], "
                  f"group={profiles[cid_seq]['mobility_group']}")
            cid_seq += 1

    n_valid = len(profiles)
    n_val_valid = len(val_profiles)
    print(f"\n[GeoLife] Train: {len(all_samples)} samples "
          f"-- {n_valid} clients")
    print(f"[GeoLife] Validation: {len(val_samples)} samples "
          f"-- {n_val_valid} dedicated clients")
    _print_sample_stats(all_samples)

    # Save cache
    os.makedirs(os.path.dirname(samples_cache), exist_ok=True)
    cache_data = {
        "samples": all_samples,
        "profiles": profiles,
        "val_samples": val_samples,
        "val_profiles": val_profiles,
    }
    with open(samples_cache, "wb") as f:
        pickle.dump(cache_data, f)

    return all_samples, profiles, val_samples, val_profiles


# -- Loading annotated trajectories -------------------------------

def _load_annotated_trajectories(
    user_path: str,
) -> list[dict]:
    """
    Combines .plt files with labels.txt to obtain trajectories
    with a transport mode annotated per GPS point.
    """
    labels_path = os.path.join(user_path, "labels.txt")
    traj_dir    = os.path.join(user_path, "Trajectory")

    if not os.path.exists(labels_path):
        return []

    # Load mode annotations
    labels = _load_labels(labels_path)
    if not labels:
        return []

    # Load all GPS points
    all_points = []
    for plt_file in sorted(os.listdir(traj_dir)):
        if not plt_file.endswith(".plt"):
            continue
        pts = _load_plt(os.path.join(traj_dir, plt_file))
        all_points.extend(pts)

    if not all_points:
        return []

    # Sort by timestamp
    all_points.sort(key=lambda p: p["ts"])

    # Annotate each point with its corresponding mode
    annotated = _annotate_points(all_points, labels)

    # Keep only points within Beijing
    b = config.BEIJING_BBOX
    annotated = [
        p for p in annotated
        if (b["lat_min"] <= p["lat"] <= b["lat_max"] and
            b["lon_min"] <= p["lon"] <= b["lon_max"])
    ]

    return annotated


def _load_labels(path: str) -> list[dict]:
    """Loads GeoLife's labels.txt."""
    labels = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                mode = parts[2].strip().lower()
                # Normalize mode-name variants
                mode = _normalize_mode(mode)
                if mode not in config.TRANSPORT_MODES:
                    continue
                try:
                    t_start = _parse_geolife_time(parts[0])
                    t_end   = _parse_geolife_time(parts[1])
                    labels.append({
                        "t_start": t_start,
                        "t_end":   t_end,
                        "mode":    mode,
                    })
                except Exception:
                    continue
    except Exception:
        return []
    return labels


def _load_plt(path: str) -> list[dict]:
    """Loads a GeoLife .plt file."""
    points = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for _ in range(6):
                next(f)  # skip 6 header lines
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    ts  = _parse_geolife_time(
                        f"{parts[5]} {parts[6]}"
                    )
                    points.append({
                        "lat": lat, "lon": lon, "ts": ts
                    })
                except Exception:
                    continue
    except Exception:
        return []
    return points


def _annotate_points(
    points: list[dict],
    labels: list[dict],
) -> list[dict]:
    """Assigns a transport mode to each GPS point."""
    annotated = []
    for p in points:
        mode = None
        for label in labels:
            if label["t_start"] <= p["ts"] <= label["t_end"]:
                mode = label["mode"]
                break
        if mode is not None:
            annotated.append({**p, "mode": mode})
    return annotated


# -- Training sample extraction ---------------------------------

def _extract_samples(annotated: list[dict],
                      user_id: str = "unknown") -> list[dict]:
    """
    Extracts samples (GPS window -> [dist, time] until transition).

    For each point p inside a stable-mode segment, computes the
    distance and time until the next mode change. That is the
    model's target variable.

    The sliding window of the previous WINDOW_SIZE points forms
    the input.
    """
    if len(annotated) < config.WINDOW_SIZE + 2:
        return []

    # Identify transition positions
    transitions = []
    for i in range(1, len(annotated)):
        if annotated[i]["mode"] != annotated[i-1]["mode"]:
            transitions.append({
                "idx":       i,
                "ts":        annotated[i]["ts"],
                "lat":       annotated[i]["lat"],
                "lon":       annotated[i]["lon"],
                "mode_from": annotated[i-1]["mode"],
                "mode_to":   annotated[i]["mode"],
            })

    if not transitions:
        return []

    samples = []
    k = config.WINDOW_SIZE

    for t_idx, transition in enumerate(transitions):
        trans_point_idx = transition["idx"]

        # Segment points before this transition
        start_idx = (
            transitions[t_idx - 1]["idx"]
            if t_idx > 0 else 0
        )
        segment_indices = range(
            max(start_idx, k - 1),
            trans_point_idx,
        )

        for p_idx in segment_indices:
            # Window of k points ending at p_idx
            window = annotated[p_idx - k + 1: p_idx + 1]
            if len(window) < k:
                continue

            # Compute window features
            features = _compute_window_features(window)
            if features is None:
                continue

            # Compute target: distance and time until the transition
            p_cur = annotated[p_idx]
            dist_m = haversine(
                p_cur["lat"], p_cur["lon"],
                transition["lat"], transition["lon"],
            )
            time_s = max(0.0,
                (transition["ts"] - p_cur["ts"]).total_seconds()
            )

            # Filter outliers
            if (dist_m < config.MIN_DIST_THRESHOLD or
                    dist_m > config.MAX_DIST_THRESHOLD or
                    time_s > config.MAX_TIME_THRESHOLD):
                continue

            norm_lat, norm_lon = normalize_position(
                p_cur["lat"], p_cur["lon"],
                **config.BEIJING_BBOX,
            )

            samples.append({
                "features":  features,
                "targets":   np.array(
                    [dist_m, time_s],
                    dtype=np.float32,
                ),
                "mode":      p_cur["mode"],
                "mode_to":   transition["mode_to"],
                "lat":       p_cur["lat"],
                "lon":       p_cur["lon"],
                "norm_lat":  norm_lat,
                "norm_lon":  norm_lon,
                "user_id":   user_id,
                "timestamp": p_cur["ts"],
            })

    return samples


def _compute_window_features(
    window: list[dict],
) -> Optional[np.ndarray]:
    """
    Extracts features from a window of k GPS points.

    Features per point (N_FEATURES = 9):
      0  instantaneous speed (km/h)
      1  acceleration (delta speed / delta time)
      2  bearing rate (deg/s)
      3  normalized latitude [0, 1]
      4  normalized longitude [0, 1]
      5  encoded current mode (0-5)
      6  normalized hour of day (0-1)
      7  day of week (0-6)
      8  cumulative distance in the segment (m)
    """
    k = len(window)
    features = np.zeros((k, config.N_FEATURES), dtype=np.float32)

    b = config.BEIJING_BBOX
    dist_accum = 0.0
    prev_speed = 0.0
    prev_bearing = None

    for i, p in enumerate(window):
        norm_lat, norm_lon = normalize_position(
            p["lat"], p["lon"], **b
        )
        hour_norm = p["ts"].hour / 24.0
        dow       = p["ts"].weekday()
        mode_idx  = config.MODE2IDX.get(p["mode"], 0)

        if i == 0:
            speed    = 0.0
            accel    = 0.0
            bear_rate = 0.0
        else:
            prev = window[i - 1]
            dt   = max(
                (p["ts"] - prev["ts"]).total_seconds(), 1.0
            )
            speed = velocity_kmh(
                prev["lat"], prev["lon"],
                p["lat"],    p["lon"], dt
            )
            accel = (speed - prev_speed) / dt
            dist_accum += haversine(
                prev["lat"], prev["lon"],
                p["lat"],    p["lon"],
            )
            b_now = bearing(
                prev["lat"], prev["lon"],
                p["lat"],    p["lon"],
            )
            if prev_bearing is not None:
                diff = abs(b_now - prev_bearing)
                diff = min(diff, 360 - diff)
                bear_rate = diff / dt
            else:
                bear_rate = 0.0
            prev_bearing = b_now

        prev_speed = speed

        features[i] = [
            min(speed, 200.0),
            np.clip(accel, -10, 10),
            min(bear_rate, 30.0),
            norm_lat,
            norm_lon,
            float(mode_idx),
            hour_norm,
            float(dow),
            dist_accum / 1000.0,  # in km
        ]

    if not np.all(np.isfinite(features)):
        return None

    return features


# -- Mobility profiles for the estimator -------------------------

def _build_profile(
    annotated: list[dict],
    n_rounds:  int,
    user_id:   str,
) -> dict:
    """
    User mobility profile for the FL server.
    Each FL round = one real position and speed from the trajectory.
    """
    step = max(len(annotated) // n_rounds, 1)
    position_seq = []
    velocity_seq = []

    for r in range(n_rounds):
        idx  = min(r * step, len(annotated) - 1)
        p    = annotated[idx]
        norm_lat, norm_lon = normalize_position(
            p["lat"], p["lon"], **config.BEIJING_BBOX
        )

        if idx == 0:
            v = 0.0
        else:
            prev = annotated[idx - 1]
            dt   = max(
                (p["ts"] - prev["ts"]).total_seconds(), 1.0
            )
            v = velocity_kmh(
                prev["lat"], prev["lon"],
                p["lat"],    p["lon"], dt,
            )

        position_seq.append({
            "lat":      p["lat"],
            "lon":      p["lon"],
            "norm_lat": norm_lat,
            "norm_lon": norm_lon,
            "mode":     p["mode"],
            "user_id":  user_id,   # FL privacy: client trains only on own data
        })
        velocity_seq.append(float(min(v, 200.0)))

    mean_v = float(np.mean(velocity_seq))
    return {
        "position_seq":  position_seq,
        "velocity_seq":  velocity_seq,
        "mean_velocity": mean_v,
        "mobility_group": _classify_mobility(mean_v),
        "user_id":       user_id,
        "type":          "real_geolife",
    }


# -- Utilities -----------------------------------------------------

def _parse_geolife_time(s: str) -> datetime:
    s = s.strip().replace("/", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized time format: {s}")


def _normalize_mode(mode: str) -> str:
    mapping = {
        "walking": "walk", "on foot": "walk",
        "bicycle": "bike", "cycling": "bike",
        "driving": "car",  "taxi": "car",
        "automobile": "car",
        "bus": "bus",
        "subway": "subway", "metro": "subway",
        "train": "train",  "rail": "train",
    }
    return mapping.get(mode, mode)


def _classify_mobility(mean_v: float) -> str:
    for group, params in config.MOBILITY_GROUPS.items():
        if mean_v <= params["velocity_max"]:
            return group
    return "alta"  # must match the last key of config.MOBILITY_GROUPS, not a label to translate


def _check_geolife_files() -> None:
    """
    Checks whether GeoLife's Data/ directory exists.
    Tries multiple paths to handle different extraction layouts
    of the original zip:

      Official zip structure:
        Geolife Trajectories 1.3/
          Data/
            000/ 001/ ... 181/

      Paths tried (in order):
        1. GEOLIFE_ROOT/Data/                 <- configured path
        2. GEOLIFE_ROOT/Geolife Trajectories 1.3/Data/
        3. GEOLIFE_ROOT/../Geolife Trajectories 1.3/Data/
        4. Recursive search for Data/000/ under GEOLIFE_ROOT
    """
    candidates = [
        os.path.join(config.GEOLIFE_ROOT, "Data"),
        os.path.join(config.GEOLIFE_ROOT,
                     "Geolife Trajectories 1.3", "Data"),
        os.path.join(os.path.dirname(config.GEOLIFE_ROOT),
                     "Geolife Trajectories 1.3", "Data"),
        os.path.join(os.path.dirname(config.GEOLIFE_ROOT),
                     "data", "Geolife Trajectories 1.3", "Data"),
    ]

    # Recursive search up to 3 levels from the parent directory
    for search_root in [config.GEOLIFE_ROOT,
                         os.path.dirname(config.GEOLIFE_ROOT)]:
        if not os.path.exists(search_root):
            continue
        for dirpath, dirnames, _ in os.walk(search_root):
            if "Data" in dirnames:
                candidate = os.path.join(dirpath, "Data")
                try:
                    subdirs = os.listdir(candidate)
                    if any(d.isdigit() for d in subdirs):
                        candidates.insert(0, candidate)
                        break
                except PermissionError:
                    pass
            depth = dirpath[len(search_root):].count(os.sep)
            if depth >= 3:
                dirnames.clear()

    for candidate in candidates:
        if os.path.exists(candidate):
            # Found it -- update the resolved root via the module
            # variable for later use
            _set_resolved_root(candidate)
            return

    raise FileNotFoundError(
        f"GeoLife's Data/ directory was not found.\n"
        f"  Configured GEOLIFE_ROOT: {config.GEOLIFE_ROOT}\n"
        f"  Paths tried:\n"
        + "\n".join(f"    {c}" for c in candidates)
        + "\n\n"
        "Expected structure after extracting the zip:\n"
        "  <GEOLIFE_ROOT>/Data/000/Trajectory/*.plt\n"
        "  <GEOLIFE_ROOT>/Data/000/labels.txt\n\n"
        "Option 1 -- move the Data/ folder to the configured path:\n"
        f"  mv 'Geolife Trajectories 1.3/Data' {config.GEOLIFE_ROOT}/\n\n"
        "Option 2 -- adjust GEOLIFE_ROOT in config.py to point to the\n"
        "  folder that contains 'Data/', e.g.:\n"
        "  GEOLIFE_ROOT = './data/Geolife Trajectories 1.3'\n\n"
        "Download: https://www.microsoft.com/en-us/research/publication/"
        "geolife-gps-trajectory-dataset-user-guide/"
    )


# Holds the resolved Data/ path for use in load_geolife
_resolved_data_dir: str = ""


def _set_resolved_root(data_dir: str) -> None:
    global _resolved_data_dir
    _resolved_data_dir = data_dir
    print(f"[GeoLife] Data/ found at: {data_dir}")


def _print_sample_stats(samples: list[dict]) -> None:
    from collections import Counter
    modes  = Counter(s["mode"] for s in samples)
    dists  = [s["targets"][0] for s in samples]
    times  = [s["targets"][1] for s in samples]
    print("\n  Distribution by mode:")
    for m in config.TRANSPORT_MODES:
        n = modes.get(m, 0)
        print(f"    {m:8s}: {n:5d} samples")
    print(f"  Distance until transition: "
          f"mean={np.mean(dists):.0f}m, "
          f"std={np.std(dists):.0f}m")
    print(f"  Time until transition:     "
          f"mean={np.mean(times):.0f}s, "
          f"std={np.std(times):.0f}s")
