"""
resource_monitor.py — v10
-------------------------
Lightweight memory/time instrumentation for the FL experiment.

Reports to wandb each round:
  - memory/rss_gb          : main process RSS (GB)
  - memory/rss_children_gb : RSS of Ray worker processes (GB)
  - memory/gpu_allocated_mb: allocated VRAM (MB, if CUDA is available)
  - memory/gpu_max_allocated_mb: peak VRAM (MB)
  - time/round_seconds     : wall-clock time of the round
  - time/total_seconds     : cumulative time since the method started

Usage in strategies:
  from resource_monitor import ResourceMonitor
  self._monitor = ResourceMonitor()       # in __init__
  self._monitor.start_round()             # at the start of aggregate_fit
  metrics = self._monitor.end_round()     # at the end of aggregate_fit
  log_dict.update(metrics)                # before wandb.log
"""

import time
import os


class ResourceMonitor:
    """Resource monitor for one FL method run."""

    def __init__(self):
        self._method_start = time.perf_counter()
        self._round_start = None

    def start_round(self):
        """Mark the start of a round (call at the start of aggregate_fit)."""
        self._round_start = time.perf_counter()

    def end_round(self) -> dict:
        """
        Collect resource metrics and return a dict for wandb.
        Call at the end of aggregate_fit, before wandb.log.
        """
        now = time.perf_counter()
        metrics = {}

        # -- Time -------------------------------------------------
        if self._round_start is not None:
            metrics["time/round_seconds"] = now - self._round_start
        metrics["time/total_seconds"] = now - self._method_start

        # -- RAM memory (main process) -----------------------------
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            metrics["memory/rss_gb"] = proc.memory_info().rss / (1024**3)

            # RSS of child processes (Ray workers)
            children_rss = sum(
                c.memory_info().rss for c in proc.children(recursive=True)
            )
            metrics["memory/rss_children_gb"] = children_rss / (1024**3)
        except (ImportError, Exception):
            # psutil not installed or process not accessible
            pass

        # -- GPU memory (CUDA) --------------------------------------
        try:
            import torch
            if torch.cuda.is_available():
                metrics["memory/gpu_allocated_mb"] = (
                    torch.cuda.memory_allocated() / (1024**2)
                )
                metrics["memory/gpu_max_allocated_mb"] = (
                    torch.cuda.max_memory_allocated() / (1024**2)
                )
        except Exception:
            pass

        return metrics
