"""
fl_utils.py
-----------
Utilities shared between fl_algorithms.py and the FL clients.

Centralizes functions that used to be duplicated in
fl_client_strategy.py and fl_algorithms.py:
  - _seed_worker
  - _pos_to_cfg / _cfg_to_pos
  - _load_shared_data / _worker_data_cache
  - make_client_fn (generic factory)
  - cleanup_pkl_cache

Consolidated here because the duplicated implementation was the root
cause of several bugs — any change had to be replicated in 5+ separate
places.
"""

import glob
import os
import pickle
import random
import tempfile

import numpy as np
import torch
import flwr as fl

import config


# -- Per-worker cache -----------------------------------------------------
# Flower reuses Ray actors — one actor can serve multiple CIDs.
# The cache guarantees the pickle is read once per worker.
_worker_data_cache: dict = {}


def cleanup_pkl_cache() -> None:
    """
    Remove cached .pkl files from previous runs in /tmp.

    Called at the start of run_experiment.py to avoid accumulating
    stale files across runs. Each run creates new caches keyed by the
    current seed — previous ones are garbage.
    """
    stale = glob.glob("/tmp/fl_shared_data_seed*.pkl")
    if stale:
        for f in stale:
            try:
                os.remove(f)
            except OSError:
                pass  # already removed by another process
        print(f"  [Cache] {len(stale)} .pkl file(s) removed from /tmp")
    else:
        print("  [Cache] No stale .pkl found in /tmp")


def _seed_worker(seed: int, client_id: int, round_num: int) -> None:
    """
    Determinism — sets a unique seed per (seed, client, round).
    Guarantees reproducibility regardless of Ray actor scheduling order.
    """
    worker_seed = seed + client_id * 10000 + round_num
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**31))
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _pos_to_cfg(pos: dict, prefix: str = "pos_") -> dict:
    """
    Serialize a position dict into primitive scalar fields.

    Flower only accepts primitive types in config dicts
    (int, float, str, bool, bytes). The "position" dict is flattened
    into scalar fields by the server before entering FitIns, and
    reassembled on the client after receiving the config.
    """
    return {
        prefix + "lat":      float(pos.get("lat",      0.0)),
        prefix + "lon":      float(pos.get("lon",      0.0)),
        prefix + "norm_lat": float(pos.get("norm_lat", 0.0)),
        prefix + "norm_lon": float(pos.get("norm_lon", 0.0)),
        prefix + "mode":     str(pos.get("mode",      "unknown")),
        prefix + "user_id":  str(pos.get("user_id",   "")),
    }


def _cfg_to_pos(cfg: dict, prefix: str = "pos_") -> dict:
    """Rebuild a position dict from its scalar fields."""
    return {
        "lat":      cfg.get(prefix + "lat",      0.0),
        "lon":      cfg.get(prefix + "lon",      0.0),
        "norm_lat": cfg.get(prefix + "norm_lat", 0.0),
        "norm_lon": cfg.get(prefix + "norm_lon", 0.0),
        "mode":     cfg.get(prefix + "mode",     "unknown"),
        "user_id":  cfg.get(prefix + "user_id",  None),
    }


def _load_shared_data(cache_path: str):
    """Load dataset and profiles from disk, cached per worker."""
    global _worker_data_cache
    if cache_path not in _worker_data_cache:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        _worker_data_cache[cache_path] = (
            data["dataset"], data["profiles"]
        )
    return _worker_data_cache[cache_path]


def make_client_fn(
    dataset,
    profiles: dict,
    client_class,
    device: str = "cpu",
    **client_kwargs,
):
    """
    Generic client_fn factory for any type of FL client.

    Parameters
    ----------
    dataset         : TransitionPredictionDataset
    profiles        : dict of profiles keyed by client_id
    client_class    : client class (FedAvgClient, FedProxClient, etc.)
    device          : training device
    **client_kwargs : extra arguments passed to the client constructor
                      (e.g. mu=0.01 for FedProxClient)

    Anti-OOM: serializes the dataset via pickle to a temp file.
    Each Ray actor loads it from disk on first use.
    The OS caches it in page cache — the second read is ~instant.
    """
    _cache_path = os.path.join(
        tempfile.gettempdir(),
        f"fl_shared_data_seed{config.ACTUAL_SEED}.pkl"
    )
    with open(_cache_path, "wb") as f:
        pickle.dump(
            {"dataset": dataset, "profiles": profiles},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    cache_path = _cache_path  # only the lightweight str is captured in the closure

    def client_fn(cid: str) -> fl.client.Client:
        worker_seed = config.ACTUAL_SEED + int(cid)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

        ds, prof = _load_shared_data(cache_path)
        return client_class(
            client_id=int(cid),
            dataset=ds,
            profile=prof[int(cid)],
            device=device,
            **client_kwargs,
        ).to_client()

    return client_fn
