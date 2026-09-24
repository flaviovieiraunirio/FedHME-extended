"""
run_experiment.py
-----------------
Runs the complete FedHME experiment with ALL baselines --
internal estimators and external FL algorithms.

Comparison structure:

  GROUP 1 -- Estimator baselines (same FL algorithm,
             different estimation strategies):
    B0  FedAvg + no_estimate    : no estimation
    B1  FedAvg + temporal_only  : history only
    B2  FedAvg + spatial_only   : geographic map only
    B3  FedAvg + velocity_only  : alpha by physical velocity
    P   FedAvg + hybrid         : PROPOSAL

  GROUP 2 -- FL algorithm baselines (state of the art,
             no heterogeneity estimator):
    McMahan et al. 2017     : plain FedAvg
    Li et al. 2020          : FedProx (mu=0.01)
    Karimireddy et al. 2020 : SCAFFOLD
    Wang et al. 2020        : FedNova

  The comparison between groups is the central argument:
  - Group 1 shows the gain from the heterogeneity estimate
  - Group 2 shows the gain versus algorithmic state of the art
  - The proposal should beat both groups in model MAE with
    lower energy cost than B0 and the Group 2 algorithms

Usage:
  python run_experiment.py
"""

import os
import json
import warnings
import numpy as np
import torch
import flwr as fl

warnings.filterwarnings("ignore")

# Silence Flower/Ray INFO logs during the simulation so the tqdm
# progress bar shows up without interference. Errors (ERROR/WARNING)
# remain visible.
import logging
logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("ray").setLevel(logging.ERROR)

import config
from geolife_loader      import load_geolife
from mobis_loader         import load_mobis
from dataset             import TransitionPredictionDataset
from fl_utils      import (make_client_fn, cleanup_pkl_cache)
from fl_algorithms import (FedAvgClient, FedAvgStrategy,
                            make_fedavg_client_fn,
                            make_fedprox_client_fn,
                            make_scaffold_client_fn,
                            make_fednova_client_fn,
                            make_feddyn_client_fn,
                            FedProxStrategy,
                            SCAFFOLDStrategy,
                            FedNovaStrategy,
                            FedDynStrategy)
from model               import create_model, get_parameters



# -- wandb integration ------------------------------------------------------

import os
# Required by PyTorch to force deterministic operations in LSTMs/RNNs
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.use_deterministic_algorithms(True)


# -- Dataset characterization plots -----------------------------------------

def plot_dataset_characterization(
    samples: list[dict],
    profiles: dict,
    out_dir: str = config.RESULTS_DIR,
    wandb_run=None,
) -> None:
    """
    Generates 6 dataset characterization plots BEFORE training.
    Saved to out_dir/plots/ and logged to wandb as images.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter, defaultdict

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    wandb_images = {}

    # -- 1. GPS scatter plot + transitions ----------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    lats = [s["lat"] for s in samples]
    lons = [s["lon"] for s in samples]
    modes = [s.get("mode", "unknown") for s in samples]
    mode_colors = {
        "walk": "#2ca02c", "bike": "#ff7f0e", "bus": "#1f77b4",
        "car": "#d62728", "subway": "#9467bd", "train": "#8c564b",
        "unknown": "#7f7f7f",
    }
    for mode in config.TRANSPORT_MODES:
        m_lats = [la for la, m in zip(lats, modes) if m == mode]
        m_lons = [lo for lo, m in zip(lons, modes) if m == mode]
        if m_lats:
            ax.scatter(m_lons, m_lats, s=1, alpha=0.3,
                       c=mode_colors.get(mode, "#7f7f7f"), label=mode)

    # Correct bounding box per dataset -- not hardcoded to Beijing
    _current_dataset = getattr(config, "CURRENT_DATASET", "geolife")
    if "mobis" in _current_dataset:
        bbox      = config.MOBIS_BBOX
        ds_label  = "MOBIS-eta"
    else:
        bbox      = config.BEIJING_BBOX
        ds_label  = "GeoLife"
    ax.axhline(bbox["lat_min"], color="red", ls="--", alpha=0.5)
    ax.axhline(bbox["lat_max"], color="red", ls="--", alpha=0.5)
    ax.axvline(bbox["lon_min"], color="red", ls="--", alpha=0.5)
    ax.axvline(bbox["lon_max"], color="red", ls="--", alpha=0.5)
    # Fit axis limits to the dataset's bbox (not to the raw data)
    ax.set_xlim(bbox["lon_min"] - 0.1, bbox["lon_max"] + 0.1)
    ax.set_ylim(bbox["lat_min"] - 0.05, bbox["lat_max"] + 0.05)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{ds_label} -- {len(samples)} samples, "
                 f"{len(profiles)} clients")
    ax.legend(markerscale=5, fontsize=8)
    fig.tight_layout()
    path1 = os.path.join(plots_dir, "dataset_scatter.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    wandb_images["dataset/scatter_gps"] = path1

    # -- 2. Sample distribution per client ------------------------
    samples_by_client = defaultdict(int)
    for s in samples:
        samples_by_client[s.get("user_id", "?")] += 1
    counts = sorted(samples_by_client.values(), reverse=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(counts)), counts, color="#1f77b4")
    ax.set_xlabel("Client (sorted by count)")
    ax.set_ylabel("Samples")
    ax.set_title(f"Sample distribution per client "
                 f"(total={len(samples)}, min={min(counts)}, "
                 f"max={max(counts)})")
    fig.tight_layout()
    path2 = os.path.join(plots_dir, "dataset_samples_per_client.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    wandb_images["dataset/samples_per_client"] = path2

    # -- 3. Mode distribution per client (heatmap) -----------------
    user_ids = sorted(samples_by_client.keys())
    mode_matrix = defaultdict(lambda: defaultdict(int))
    for s in samples:
        uid = s.get("user_id", "?")
        mode_matrix[uid][s.get("mode", "unknown")] += 1

    modes_list = config.TRANSPORT_MODES
    matrix = np.zeros((len(user_ids), len(modes_list)))
    for i, uid in enumerate(user_ids):
        for j, m in enumerate(modes_list):
            matrix[i, j] = mode_matrix[uid].get(m, 0)

    fig, ax = plt.subplots(figsize=(8, max(4, len(user_ids) * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(modes_list)))
    ax.set_xticklabels(modes_list, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Client")
    ax.set_title("Samples by mode x client")
    fig.colorbar(im, ax=ax, label="Samples")
    fig.tight_layout()
    path3 = os.path.join(plots_dir, "dataset_mode_per_client.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    wandb_images["dataset/mode_per_client"] = path3

    # -- 4. Target distribution -------------------------------------
    dists = [s["targets"][0] for s in samples]
    times = [s["targets"][1] for s in samples]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(dists, bins=50, color="#2ca02c", alpha=0.7, edgecolor="white")
    ax1.set_xlabel("Distance to transition (m)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"dist_to_transition (median={np.median(dists):.0f}m)")
    ax2.hist(times, bins=50, color="#ff7f0e", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Time to transition (s)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"time_to_transition (median={np.median(times):.0f}s)")
    fig.tight_layout()
    path4 = os.path.join(plots_dir, "dataset_target_distribution.png")
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    wandb_images["dataset/target_distribution"] = path4

    # -- 5. Velocity profile per mode (boxplot) ----------------------
    vel_by_mode = defaultdict(list)
    for s in samples:
        # Mean velocity over the whole window (feature 0). An earlier
        # version used s["features"][0][0] -- the first point's
        # velocity, which is always 0.
        vel_by_mode[s.get("mode", "unknown")].append(
            float(np.mean(s["features"][:, 0]))
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    modes_with_data = [m for m in modes_list if vel_by_mode.get(m)]
    data = [vel_by_mode[m] for m in modes_with_data]
    bp = ax.boxplot(data, labels=modes_with_data, patch_artist=True)
    colors = [mode_colors.get(m, "#7f7f7f") for m in modes_with_data]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.axhspan(15, 25, alpha=0.15, color="yellow",
               label="Controlled range 15-25 km/h")
    ax.set_ylabel("Velocity (km/h)")
    ax.set_title("Velocity profile per transport mode")
    ax.legend()
    fig.tight_layout()
    path5 = os.path.join(plots_dir, "dataset_velocity_by_mode.png")
    fig.savefig(path5, dpi=150)
    plt.close(fig)
    wandb_images["dataset/velocity_by_mode"] = path5

    # -- 6. Transition map (mode_from -> mode_to pairs) --------------
    transitions = Counter()
    for s in samples:
        mf = s.get("mode", "unknown")
        mt = s.get("mode_to", "unknown")
        if mf != mt:
            transitions[(mf, mt)] += 1

    if transitions:
        trans_modes = sorted(set(
            m for pair in transitions for m in pair
        ))
        trans_matrix = np.zeros((len(trans_modes), len(trans_modes)))
        for (mf, mt), cnt in transitions.items():
            i = trans_modes.index(mf)
            j = trans_modes.index(mt)
            trans_matrix[i, j] = cnt

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(trans_matrix, cmap="Blues")
        ax.set_xticks(range(len(trans_modes)))
        ax.set_xticklabels(trans_modes, rotation=45, ha="right")
        ax.set_yticks(range(len(trans_modes)))
        ax.set_yticklabels(trans_modes)
        ax.set_xlabel("Destination mode")
        ax.set_ylabel("Origin mode")
        ax.set_title("Transition frequency between modes")
        # Annotate cells
        for i in range(len(trans_modes)):
            for j in range(len(trans_modes)):
                v = int(trans_matrix[i, j])
                if v > 0:
                    ax.text(j, i, str(v), ha="center", va="center",
                            fontsize=7,
                            color="white" if v > trans_matrix.max()*0.6
                            else "black")
        fig.colorbar(im, ax=ax, label="Samples")
        fig.tight_layout()
        path6 = os.path.join(plots_dir, "dataset_transitions.png")
        fig.savefig(path6, dpi=150)
        plt.close(fig)
        wandb_images["dataset/transitions"] = path6

    # -- Log to wandb -------------------------------------------
    if wandb_run is not None:
        try:
            import wandb
            for key, path in wandb_images.items():
                wandb_run.log({key: wandb.Image(path)}, step=0)
        except Exception:
            pass

    print(f"[Dataset] {len(wandb_images)} plots saved to {plots_dir}/")


# -- Geographic divergence heatmap -------------------------------------------

def _plot_geo_heatmap(
    estimator,
    method_name: str,
    seed:        int,
    out_dir:     str,
    wandb_run=None,
) -> None:
    """
    Generates a heatmap of the gradient divergence accumulated on the
    estimator's geographic map over the course of the experiment.

    The map covers the Beijing bounding box defined in config.py.
    Each cell represents the mean divergence observed in that
    geographic region across every round of the experiment.

    Saved to:
      out_dir/plots/heatmap_{method_name}_seed{seed}.png
    And sent to wandb if wandb_run is provided.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[WARNING] matplotlib not installed -- heatmap not generated.")
        return

    if not hasattr(estimator, "get_geo_heatmap"):
        return

    hm = estimator.get_geo_heatmap()
    geo_map   = hm["geo_map"]
    counts    = hm["counts"]
    lat_min   = hm["lat_min"]
    lat_max   = hm["lat_max"]
    lon_min   = hm["lon_min"]
    lon_max   = hm["lon_max"]
    grid_size = hm["grid_size"]

    # Mask cells with no data (count == 1 = initial value)
    mask = counts <= 1

    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    # -- Main figure: divergence heatmap ---------------------------
    n_maps = 2 if (hm.get("loss_map") is not None) else 1
    fig, axes = plt.subplots(1, n_maps,
                              figsize=(6 * n_maps, 5))
    if n_maps == 1:
        axes = [axes]

    def _draw_map(ax, data, title, cmap):
        import numpy as np
        masked = np.ma.masked_where(mask, data)
        im = ax.imshow(
            masked,
            origin    = "upper",
            cmap      = cmap,
            aspect    = "auto",
            extent    = [lon_min, lon_max, lat_min, lat_max],
            interpolation = "bilinear",
        )
        plt.colorbar(im, ax=ax, shrink=0.8, label="mean divergence")
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude",  fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(True, alpha=0.2, linewidth=0.5)

    _draw_map(axes[0], geo_map,
              f"Gradient divergence\n{method_name} -- seed {seed}",
              "hot_r")

    if n_maps == 2 and hm["loss_map"] is not None:
        _draw_map(axes[1], hm["loss_map"],
                  f"Global model loss\n{method_name} -- seed {seed}",
                  "Blues")

    fig.suptitle(
        f"Geographic map -- {method_name} (seed={seed})\n"
        f"Beijing {lat_min:.2f}-{lat_max:.2f} deg N  "
        f"{lon_min:.2f}-{lon_max:.2f} deg E  "
        f"grid {grid_size}x{grid_size}",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()

    fname = f"heatmap_{method_name}_seed{seed}.png"
    fpath = os.path.join(out_dir, "plots", fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [heatmap] {fpath}")

    # -- Figure 2: geo_count (observation density) -----------------
    import numpy as np
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    counts_masked = np.ma.masked_where(mask, counts)
    im2 = ax2.imshow(
        counts_masked,
        origin="upper", cmap="Greens", aspect="auto",
        extent=[lon_min, lon_max, lat_min, lat_max],
        interpolation="bilinear",
    )
    plt.colorbar(im2, ax=ax2, shrink=0.8, label="observations")
    ax2.set_xlabel("Longitude", fontsize=10)
    ax2.set_ylabel("Latitude",  fontsize=10)
    ax2.set_title(
        f"Observation density (geo_count)\n"
        f"{method_name} -- seed {seed}",
        fontsize=11,
    )
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax2.grid(True, alpha=0.2, linewidth=0.5)
    fig2.tight_layout()

    fname2 = f"heatmap_counts_{method_name}_seed{seed}.png"
    fpath2 = os.path.join(out_dir, "plots", fname2)
    fig2.savefig(fpath2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [heatmap counts] {fpath2}")

    # -- Send to wandb --------------------------------------------
    if wandb_run is not None:
        try:
            import wandb
            wandb_run.log({
                f"heatmap/grad_{method_name}": wandb.Image(
                    fpath,
                    caption=(f"Geographic divergence -- "
                             f"{method_name} seed={seed}")
                ),
                f"heatmap/counts_{method_name}": wandb.Image(
                    fpath2,
                    caption=(f"Observation density -- "
                             f"{method_name} seed={seed}")
                ),
            })
        except Exception as e:
            print(f"  [WARNING] wandb heatmap failed: {e}")



# -- Periodic global evaluation -----------------------------------------------

def _global_eval_callback(
    server_round: int,
    parameters,
    val_ds,
    device:     str,
    wandb_run=None,
    method_name: str = "",
) -> dict:
    """
    Evaluates the global model on the FULL val_ds.
    Called by aggregate_fit every VAL_EVERY rounds.
    Runs in the main process (not the Ray workers) -- no FL overhead.

    Returns metrics, or {} on error.
    This is the source of the paper's table values:
      - Convergence curve: eval_global/mae_dist per round
      - Final MAE: mean of the last N evaluations
      - MAE by mode: stratified over the full dataset
    """
    try:
        from model import create_model, set_parameters, evaluate_model
        from torch.utils.data import DataLoader
        from dataset import collate_with_mode
        import flwr.common as flc

        model = create_model()
        set_parameters(model, flc.parameters_to_ndarrays(parameters))

        loader = DataLoader(
            val_ds, batch_size=64,
            shuffle=False, num_workers=0,
            collate_fn=collate_with_mode,
        )
        metrics = evaluate_model(model, loader, val_ds, device)

        result = {
            "round":    server_round,
            "mae_dist": metrics["mae_dist"],
            "mae_time": metrics["mae_time"],
            "val_loss": metrics["loss"],
            "mae_by_mode": metrics.get("mae_by_mode", {}),
        }

        if wandb_run is not None:
            try:
                log_dict = {
                    "eval_global/mae_dist": metrics["mae_dist"],
                    "eval_global/mae_time": metrics["mae_time"],
                    "eval_global/val_loss": metrics["loss"],
                }
                for mode, vals in metrics.get("mae_by_mode", {}).items():
                    log_dict[f"eval_global/mae_dist_{mode}"] = vals["mae_dist"]
                    log_dict[f"eval_global/mae_time_{mode}"] = vals["mae_time"]
                wandb_run.log(log_dict, step=server_round)
            except Exception:
                pass

        print(f"  [Global Eval R{server_round:03d}] "
              f"MAE dist={metrics['mae_dist']:.1f}m  "
              f"MAE time={metrics['mae_time']:.1f}s  "
              f"val_loss={metrics['loss']:.4f}")
        return result

    except Exception as e:
        print(f"  [Global Eval] Error in round {server_round}: {e}")
        return {}


def _compute_final_global_metrics(
    global_eval_log: list[dict],
    last_n: int = 10,
) -> dict:
    """
    Computes the mean of the last last_n global evaluations.
    This is the value that goes into the paper's tables --
    more robust than the point value of the last round.
    """
    if not global_eval_log:
        return {}
    recent = global_eval_log[-last_n:]
    mae_dists = [e["mae_dist"] for e in recent]
    mae_times = [e["mae_time"] for e in recent]
    val_losses = [e["val_loss"] for e in recent]

    # Aggregate MAE by mode
    from collections import defaultdict
    mode_dists = defaultdict(list)
    mode_times = defaultdict(list)
    for e in recent:
        for mode, vals in e.get("mae_by_mode", {}).items():
            mode_dists[mode].append(vals["mae_dist"])
            mode_times[mode].append(vals["mae_time"])

    result = {
        "mae_dist_global": float(np.mean(mae_dists)),
        "mae_time_global": float(np.mean(mae_times)),
        "val_loss_global": float(np.mean(val_losses)),
        "n_evals":         len(recent),
        "mae_by_mode_global": {
            m: {
                "mae_dist": float(np.mean(mode_dists[m])),
                "mae_time": float(np.mean(mode_times[m])),
            }
            for m in mode_dists
        },
    }
    return result


def _wandb_init(method_name: str, seed: int, extra_config: dict = None):
    """
    Starts a wandb run for a specific method/seed combination.
    Each method+seed combination becomes a separate run in the
    dashboard, making comparison and filtering easier.
    Returns the run (or None if wandb is disabled/not installed).
    """
    if not config.WANDB_ENABLED:
        return None
    try:
        import wandb
        run_config = {
            "method":          method_name,
            "method_display":  config.METHOD_DISPLAY_NAMES.get(method_name, method_name),
            "seed":            seed,
            "n_clients":       config.N_CLIENTS,
            "n_rounds":        config.N_ROUNDS,
            "min_fit_clients": config.MIN_FIT_CLIENTS,
            "local_epochs":    config.LOCAL_EPOCHS_BASE,
            "batch_size":      config.BATCH_SIZE,
            "learning_rate":   config.LEARNING_RATE,
            "window_size":     config.WINDOW_SIZE,
            "fedprox_mu":      config.FEDPROX_MU,
            "use_gpu":         config.USE_GPU,
        }
        if extra_config:
            run_config.update(extra_config)
        display_name = config.METHOD_DISPLAY_NAMES.get(method_name, method_name)
        # Run name: display name + seed + code version
        run = wandb.init(
            project   = config.WANDB_PROJECT,
            entity    = config.WANDB_ENTITY,
            name      = f"{display_name}_seed{seed}_{config.CODE_VERSION}",
            group     = f"seed{seed}",
            tags      = config.WANDB_TAGS + [display_name, f"seed{seed}"],
            config    = run_config,
            reinit    = True,
        )
        return run
    except ImportError:
        print("[WARNING] wandb not installed. Run: pip install wandb")
        return None
    except Exception as e:
        print(f"[WARNING] wandb failed to start: {e}")
        return None



def _wandb_log_final(run, mae_dist: float, mae_time: float,
                      final_loss: float, analysis: dict = None):
    """
    Sends final metrics to wandb at the end of the method.

    Uses run.log() (not just summary) so per-mode metrics show up in
    wandb's Charts, not just the summary table. step=0 ensures they
    appear as a single point on the final chart.
    """
    if run is None:
        return
    try:
        import wandb
        # Aggregate metrics -- summary for the comparison table
        run.summary.update({
            "final/mae_dist": mae_dist,
            "final/mae_time": mae_time,
            "final/loss":     final_loss,
        })
        # Per-mode metrics -- log() so they show up in Charts
        if analysis:
            modal_metrics = {}
            for mode, val in analysis.get("by_mode", {}).items():
                # by_mode from the estimator = estimation error (h_hat
                # vs. real divergence), logged separately so it isn't
                # confused with the model's MAE
                modal_metrics[f"final/estimator_error_{mode}"] = val
                run.summary[f"final/estimator_error_{mode}"] = val
            if modal_metrics:
                run.log(modal_metrics)
        run.finish()
    except Exception:
        pass

# -- Helper functions -------------------------------------------

def _initial_params():
    return fl.common.ndarrays_to_parameters(
        get_parameters(create_model())
    )


def _base_strategy_kwargs():
    return dict(
        fraction_fit          = config.FRACTION_FIT,
        min_fit_clients       = config.MIN_FIT_CLIENTS,
        min_evaluate_clients  = config.MIN_EVAL_CLIENTS,
        min_available_clients = config.N_CLIENTS,
        initial_parameters    = _initial_params(),
    )


def _run_simulation(client_fn, strategy):
    # Concurrent workers = min(
    #   floor(RAY_NUM_CPUS / RAY_CPUS_PER_CLIENT),
    #   floor(RAY_NUM_GPUS / RAY_GPUS_PER_CLIENT)  <- if USE_GPU=True
    # )
    # Tune the RAY_* parameters in config.py to calibrate.

    # Re-seed Flower's server-side sampling -- this must run on EVERY
    # invocation, not just once at the top of the file.
    import random
    random.seed(config.ACTUAL_SEED)
    np.random.seed(config.ACTUAL_SEED)
    torch.manual_seed(config.ACTUAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.ACTUAL_SEED)
        
    client_resources = {
        "num_cpus": config.RAY_CPUS_PER_CLIENT,
        "num_gpus": config.RAY_GPUS_PER_CLIENT if config.USE_GPU else 0.0,
    }
    ray_init_args = {
        "num_cpus":          config.RAY_NUM_CPUS,
        "num_gpus":          config.RAY_NUM_GPUS if config.USE_GPU else 0,
        "include_dashboard": False,
    }
    
    return fl.simulation.start_simulation(
        client_fn        = client_fn,
        num_clients      = config.N_CLIENTS,
        config           = fl.server.ServerConfig(
                               num_rounds=config.N_ROUNDS),
        strategy         = strategy,
        client_resources = client_resources,
        ray_init_args    = ray_init_args,
    )


def _extract_final_metrics(history, round_log):
    """Extracts the final MAE and final loss from the history."""
    mae_dist, mae_time = 0.0, 0.0
    dist_hist = history.metrics_distributed.get("mae_dist", [])
    time_hist = history.metrics_distributed.get("mae_time", [])
    if dist_hist:
        mae_dist = dist_hist[-1][1]
    if time_hist:
        mae_time = time_hist[-1][1]
    final_loss = (round_log[-1]["mean_loss"]
                  if round_log else 0.0)
    return mae_dist, mae_time, final_loss


# -- Group 1 run: estimator baselines --------------------------

def run_estimator_group(
    dataset:  object,
    profiles: dict,
    device:   str,
    val_ds:   object = None,
    seed:     int = config.EXPERIMENT_SEEDS[0],
    selected_methods: set = None,
) -> dict:
    """Runs the estimators with FedAvg as the base algorithm."""
    results = {}
    _pending_csv = {}  # accumulates data for the CSV -- held_out filled in later

    for est_name in config.ESTIMATOR_NAMES:
        if selected_methods is not None and est_name not in selected_methods:
            continue
        display_est = config.METHOD_DISPLAY_NAMES.get(est_name, est_name)
        print(f"\n{'-'*50}")
        print(f" Estimator: {display_est}  (seed={seed})")
        print(f"{'-'*50}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        import random; random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # -- wandb: start a run for this method+seed ----------
        wandb_run = _wandb_init(est_name, seed)
        strategy = FedAvgStrategy(
            profiles       = profiles,
            dataset        = dataset,  # to create MobilityClients
            estimator_name = est_name,
            n_clients      = config.N_CLIENTS,
            wandb_run      = wandb_run,
            **_base_strategy_kwargs(),
        )

        # Pass val_ds and device for periodic global evaluation
        strategy._val_ds = val_ds
        strategy._device = device

        # compute per-mode weights (hybrid only)
        strategy._mode_weights = strategy.compute_mode_weights(dataset)

        history = _run_simulation(
            make_fedavg_client_fn(
                dataset, profiles, device,
                dataset_name=config.CURRENT_DATASET,
            ),
            strategy,
        )
        mae_d, mae_t, loss = _extract_final_metrics(
            history, strategy.round_log
        )

        # Global metrics -- mean of the last 10 rounds -- paper table
        global_metrics = _compute_final_global_metrics(
            strategy.global_eval_log
        )
        if global_metrics:
            mae_d = global_metrics["mae_dist_global"]
            mae_t = global_metrics["mae_time_global"]
            if wandb_run is not None:
                try:
                    wandb_run.summary.update({
                        "article/mae_dist": mae_d,
                        "article/mae_time": mae_t,
                        **{f"article/mae_dist_{m}": v["mae_dist"]
                           for m, v in global_metrics
                           .get("mae_by_mode_global", {}).items()},
                    })
                except Exception:
                    pass

        # -- wandb: send final metrics (per-round log already real-time) --
        _wandb_log_final(
            wandb_run, mae_d, mae_t, loss,
            analysis=strategy.get_analysis(),
        )

        # -- Local CSV (held_out filled in in _create_seed_eval_run) --
        _summary = {
            "method":              est_name,
            "seed":                seed,
            "mae_dist_final":      mae_d,
            "mae_time_final":      mae_t,
            "held_out_mae_dist":   "",
            "held_out_mae_time":   "",
            "held_out_mse_dist":   "",
            "held_out_mse_time":   "",
            "held_out_rmse_dist":  "",
            "held_out_rmse_time":  "",
            "held_out_mape_dist":  "",
            "held_out_mape_time":  "",
            "held_out_mape_n_excluded_dist": "",
            "held_out_mape_n_excluded_time": "",
            "held_out_smape_dist": "",
            "held_out_smape_time": "",
            "held_out_r2_dist":    "",
            "held_out_r2_time":    "",
            "budget_saving":       config.BUDGET_SAVING,
            "skip_factor":         config.SKIP_FACTOR,
            "total_grad_steps":    strategy.total_grad_steps,
        }
        # Save the CSV immediately (without held-out for now)
        _save_run_csv(
            est_name, seed,
            strategy.global_eval_log,
            strategy.round_log,
            _summary,
        )
        # Detailed per-estimate log
        _save_estimator_detail_log(
            est_name, seed,
            getattr(strategy.estimator, "log", []),
            _summary,
        )
        # Per-client rank and decision history
        _save_client_rank_log(
            est_name, seed,
            getattr(strategy, "client_log", []),
            _summary,
        )
        _pending_csv[est_name] = {
            "global_eval_log": strategy.global_eval_log,
            "round_log":       strategy.round_log,
            "summary":         _summary,
        }
        # Store in results so _create_seed_eval_run can access it
        results[f"__pending_csv_{est_name}"] = _pending_csv[est_name]

        # -- Geographic divergence heatmap --------------------
        _plot_geo_heatmap(
            strategy.estimator, est_name, seed,
            config.RESULTS_DIR, wandb_run,
        )

        results[est_name] = {
            "group":          "estimator",
            "round_log":      strategy.round_log,
            "analysis":       strategy.get_analysis(),
            "final_mae_dist": mae_d,
            "final_mae_time": mae_t,
            "final_loss":     loss,
            "final_params":   strategy._final_params,
            "total_grad_steps": strategy.total_grad_steps,
            "h4_data":        strategy.h4_data,
            "velocity_controlled": (
                strategy.estimator.get_velocity_controlled_errors()
            ),
            # client_log for _save_client_rank_log
            "client_log":           getattr(strategy, "client_log", []),
            "estimator_detail_log": getattr(strategy.estimator, "log", []),
        }

        # Free memory between estimators
        import gc
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
                print(f"  [cleanup] Ray shutdown after {display_est}")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # aggregated = results in this context (1 seed, no multi-seed aggregation)
    return results, _pending_csv, results


# -- Group 2 run: FL algorithm baselines ------------------------



def _print_h4_analysis(h4_data: list[dict]) -> None:
    """
    Checks whether clients whose data is closer to a mode transition
    have higher gradient divergence.

    Splits the data into tertiles of dist_to_transition and compares
    the mean divergence of each tertile. If the tertile closest to
    the transition has significantly higher divergence, the
    hypothesis is confirmed.
    """
    if len(h4_data) < 9:
        print(f"   (only {len(h4_data)} points -- insufficient)")
        return

    dists = np.array([d["dist_to_transition"] for d in h4_data])
    divs  = np.array([d["divergence"]         for d in h4_data])

    # Spearman correlation (does not assume linearity)
    from scipy.stats import spearmanr
    corr, p_val = spearmanr(dists, divs)
    print(f"   Spearman rho = {corr:.3f}  (p = {p_val:.4f})")
    # Negative correlation expected: smaller distance -> higher divergence
    if corr < 0 and p_val < 0.05:
        print(f"   -> CONFIRMED: significant negative correlation")
    elif corr < 0:
        print(f"   -> TREND: negative correlation but p > 0.05")
    else:
        print(f"   -> NOT CONFIRMED: correlation >= 0")

    # Tertile analysis
    t1 = np.percentile(dists, 33.3)
    t2 = np.percentile(dists, 66.6)
    close  = divs[dists <= t1]
    mid    = divs[(dists > t1) & (dists <= t2)]
    far    = divs[dists > t2]
    print(f"   Tertiles of distance to transition:")
    print(f"     Close    (<={t1:.0f}m) : mean div = "
          f"{np.mean(close):.2f} +/- {np.std(close):.2f}  (n={len(close)})")
    print(f"     Mid      ({t1:.0f}-{t2:.0f}m): mean div = "
          f"{np.mean(mid):.2f} +/- {np.std(mid):.2f}  (n={len(mid)})")
    print(f"     Far      (>{t2:.0f}m) : mean div = "
          f"{np.mean(far):.2f} +/- {np.std(far):.2f}  (n={len(far)})")

    # Per-mode analysis
    from collections import defaultdict
    by_mode: dict[str, list] = defaultdict(list)
    for d in h4_data:
        by_mode[d["mode"]].append(
            (d["dist_to_transition"], d["divergence"])
        )
    print(f"   By transport mode:")
    for mode in sorted(by_mode):
        pairs = by_mode[mode]
        if len(pairs) < 3:
            continue
        m_dists = [p[0] for p in pairs]
        m_divs  = [p[1] for p in pairs]
        r, p = spearmanr(m_dists, m_divs)
        print(f"     {mode:8s}: rho={r:+.3f} (p={p:.3f}, n={len(pairs)})")


# -- Controlled-velocity experiment (15-25 km/h) -----------------

def _print_velocity_controlled(all_results: dict) -> None:
    """
    Compares the estimator's error for velocity_only vs. hybrid,
    filtered to the 15-25 km/h range where bike, bus and car coexist.

    If velocity_only has a similar error across all modes but hybrid
    differentiates them (lower error for modes with a specific
    delta_m), it demonstrates that mode adds information that
    physical velocity alone does not capture.
    """
    vc_hybrid = all_results.get("hybrid", {}).get(
        "velocity_controlled", {})
    vc_velocity = all_results.get("velocity_only", {}).get(
        "velocity_controlled", {})

    if not vc_hybrid and not vc_velocity:
        return

    print(f"\n{'='*65}")
    print(" Controlled-velocity experiment (15-25 km/h)")
    print(f" Isolates the contribution of mode: within the same velocity")
    print(f" range, does transport mode better predict heterogeneity?")
    print(f"{'='*65}")

    all_modes = sorted(
        set(list(vc_hybrid.keys()) + list(vc_velocity.keys()))
    )

    if not all_modes:
        print("  Not enough data in the 15-25 km/h range.")
        return

    print(f"\n{'Mode':10s} {'velocity_only':20s} {'hybrid':20s} {'Delta':8s}")
    print("-" * 62)

    for mode in all_modes:
        v_data = vc_velocity.get(mode)
        h_data = vc_hybrid.get(mode)
        v_str = (f"{v_data['mae']:.2f}+/-{v_data['std']:.2f} "
                 f"(n={v_data['n']})" if v_data else "-")
        h_str = (f"{h_data['mae']:.2f}+/-{h_data['std']:.2f} "
                 f"(n={h_data['n']})" if h_data else "-")
        if v_data and h_data:
            delta = h_data["mae"] - v_data["mae"]
            d_str = f"{delta:+.2f}"
        else:
            d_str = "-"
        print(f"{mode:10s} {v_str:20s} {h_str:20s} {d_str:8s}")

    # Interpretation
    if vc_velocity and vc_hybrid:
        v_maes = [v["mae"] for v in vc_velocity.values()]
        h_maes = [v["mae"] for v in vc_hybrid.values()]
        v_range = max(v_maes) - min(v_maes) if len(v_maes) > 1 else 0
        h_range = max(h_maes) - min(h_maes) if len(h_maes) > 1 else 0
        print(f"\n  Spread across modes (MAE range):")
        print(f"    velocity_only: {v_range:.2f}  "
              f"(expected low -- same velocity for all)")
        print(f"    hybrid:        {h_range:.2f}  "
              f"(expected high -- delta_m differentiates modes)")
        if h_range > v_range * 1.2:
            print(f"  -> hybrid differentiates modes within the same "
                  f"velocity range -- mode adds information")
        else:
            print(f"  -> the difference between spreads is not conclusive")


# -- Cost-efficiency analysis ----------------------------------

def print_cost_efficiency(all_results: dict) -> None:
    """
    Cost-efficiency table: MAE x total cost per method.

    Metrics reported:
      - Total cost (client-epochs): sum of all epochs across all
        clients over all rounds
      - Final MAE dist (meters)
      - Efficiency ratio: (MAE_base / MAE_method) / (cost_method / cost_base)
        where base = no_estimate. Efficiency > 1.0 = better cost-benefit
        than the baseline.
    """
    print(f"\n{'='*75}")
    print(" Cost-efficiency analysis (Green FL)")
    print(f" Fixed budget: N_clients x E_base epochs per round")
    print(f"{'='*75}")

    # Correct baseline per method: each hybrid vs. its corresponding pure
    BASELINE_MAP = {
        "hybrid":          "no_estimate",
        "fedprox_hybrid":  "fedprox_pure",
        "scaffold_hybrid": "scaffold_pure",
        "fednova_hybrid":  "fednova_pure",
    }

    no_est = all_results.get("no_estimate", {})
    if no_est.get("total_grad_steps", 0) == 0:
        print("  (no cost data for no_estimate)")
        return

    print(f"\n{'Method':20s} {'MAE dist':>10s} {'Grad Steps':>12s} "
          f"{'Efficiency':>12s} {'MAE/Cost':>12s} {'Baseline':>15s}")
    print("-" * 85)

    order = (
        config.ESTIMATOR_NAMES
        + ["fedprox_pure", "fedprox_hybrid_static_mu", "fedprox_hybrid", "scaffold_pure", "scaffold_hybrid_no_bypass", "scaffold_hybrid", "fednova_pure", "fednova_hybrid", "feddyn_pure", "feddyn_hybrid_static_alpha", "feddyn_hybrid"]
    )

    for name in order:
        res = all_results.get(name)
        if res is None:
            continue
        mae  = res.get("final_mae_dist", float("nan"))
        cost = res.get("total_grad_steps", 0)
        if cost == 0 or mae != mae:  # nan check
            continue

        baseline_name = BASELINE_MAP.get(name, None)
        if baseline_name is not None:
            base = all_results.get(baseline_name, {})
            base_mae  = base.get("final_mae_dist", float("nan"))
            base_cost = base.get("total_grad_steps", 0)
        else:
            base_mae  = float("nan")
            base_cost = 0

        # Efficiency ratio: > 1.0 = better than the baseline
        if baseline_name and base_cost > 0 and base_mae == base_mae and mae > 0:
            quality_gain = base_mae / mae
            cost_ratio   = cost / base_cost
            efficiency   = quality_gain / cost_ratio if cost_ratio > 0 else 0
        else:
            efficiency = float("nan")

        # MAE per unit of cost (lower = better)
        mae_per_cost = mae * cost / 1000  # scaled for readability

        marker = " <-" if name == "hybrid" else ""
        ref_label = baseline_name if baseline_name else "(ref)"
        display = config.METHOD_DISPLAY_NAMES.get(name, name)
        print(f"{display:25s} {mae:10.1f} {cost:10d} "
              f"{efficiency:12.3f} {mae_per_cost:12.1f} {ref_label:>15s}"
              + marker)

    print(f"\n  Efficiency > 1.0 = better cost-benefit than FedAvg")
    print(f"  Efficiency = (MAE_base/MAE) / (cost/cost_base)")
    print(f"  Cost = total gradient steps (clients x epochs x batches)")


def print_cost_efficiency_aggregated(
    results_by_seed: dict,
) -> None:
    """
    Aggregated version (mean +/- std over seeds) of the efficiency table.
    """
    from collections import defaultdict

    # Correct baseline per method: each hybrid vs. its corresponding pure
    BASELINE_MAP = {
        "hybrid":          "no_estimate",
        "fedprox_hybrid":  "fedprox_pure",
        "scaffold_hybrid": "scaffold_pure",
        "fednova_hybrid":  "fednova_pure",
    }

    # Collect per method
    data: dict[str, dict[str, list]] = defaultdict(
        lambda: {"mae": [], "cost": [], "eff": []}
    )

    for seed, seed_results in results_by_seed.items():
        for name, res in seed_results.items():
            mae  = res.get("final_mae_dist", float("nan"))
            cost = res.get("total_grad_steps", 0)
            if cost == 0 or mae != mae:
                continue

            baseline_name = BASELINE_MAP.get(name, None)
            if baseline_name is not None:
                base     = seed_results.get(baseline_name, {})
                base_mae  = base.get("final_mae_dist", float("nan"))
                base_cost = base.get("total_grad_steps", 0)
                if base_cost > 0 and base_mae == base_mae and mae > 0:
                    eff = (base_mae / mae) / (cost / base_cost)
                else:
                    eff = float("nan")
            else:
                eff = float("nan")

            data[name]["mae"].append(mae)
            data[name]["cost"].append(cost)
            if eff == eff:  # not nan
                data[name]["eff"].append(eff)

    print(f"\n{'='*80}")
    print(" Aggregated cost-efficiency (mean +/- std over seeds)")
    print(f"{'='*80}")
    print(f"\n{'Method':20s} {'MAE dist':>14s} {'Cost':>14s} "
          f"{'Efficiency':>14s}")
    print("-" * 66)

    order = (
        config.ESTIMATOR_NAMES
        + ["fedprox_pure", "fedprox_hybrid_static_mu", "fedprox_hybrid", "scaffold_pure", "scaffold_hybrid_no_bypass", "scaffold_hybrid", "fednova_pure", "fednova_hybrid", "feddyn_pure", "feddyn_hybrid_static_alpha", "feddyn_hybrid"]
    )

    for name in order:
        d = data.get(name)
        if not d or not d["mae"]:
            continue
        marker = " <-" if name == "hybrid" else ""
        display = config.METHOD_DISPLAY_NAMES.get(name, name)
        print(
            f"{display:25s} "
            f"{np.mean(d['mae']):7.1f}+/-{np.std(d['mae']):5.1f} "
            f"{np.mean(d['cost']):7.0f}+/-{np.std(d['cost']):5.0f} "
            f"{np.mean(d['eff']):7.3f}+/-{np.std(d['eff']):5.3f}"
            + marker
        )

    # -- Log as wandb.Table -------------------------------------
    try:
        import wandb
        if wandb.run is not None:
            table = wandb.Table(
                columns=["Method", "MAE dist (mean)", "MAE dist (std)",
                         "Grad Steps (mean)", "Grad Steps (std)",
                         "Efficiency (mean)", "Efficiency (std)"]
            )
            for name in order:
                d = data.get(name)
                if not d or not d["mae"]:
                    continue
                display = config.METHOD_DISPLAY_NAMES.get(name, name)
                table.add_data(
                    display,
                    round(float(np.mean(d["mae"])), 1),
                    round(float(np.std(d["mae"])), 1),
                    round(float(np.mean(d["cost"])), 0),
                    round(float(np.std(d["cost"])), 0),
                    round(float(np.mean(d["eff"])), 3),
                    round(float(np.std(d["eff"])), 3),
                )
            wandb.log({"cost_efficiency_table": table})
    except Exception:
        pass


# -- Multi-seed aggregation ------------------------------------

def _aggregate_results(
    results_by_seed: dict[int, dict],
) -> dict:
    """
    Computes the mean and standard deviation of every metric
    across all seeds.

    Returns a dict with structure:
      aggregated[method] = {
          "mae_dist_mean": float,
          "mae_dist_std":  float,
          "mae_time_mean": float,
          "mae_time_std":  float,
          "loss_mean":     float,
          "loss_std":      float,
      }
    """
    # Collect values per method
    values: dict[str, dict[str, list]] = {}

    for seed, seed_results in results_by_seed.items():
        for method, res in seed_results.items():
            if method not in values:
                values[method] = {
                    "mae_dist": [], "mae_time": [], "loss": []
                }
            values[method]["mae_dist"].append(
                res.get("final_mae_dist", float("nan"))
            )
            values[method]["mae_time"].append(
                res.get("final_mae_time", float("nan"))
            )
            values[method]["loss"].append(
                res.get("final_loss", float("nan"))
            )

    aggregated = {}
    for method, v in values.items():
        aggregated[method] = {
            "mae_dist_mean": float(np.nanmean(v["mae_dist"])),
            "mae_dist_std":  float(np.nanstd (v["mae_dist"])),
            "mae_time_mean": float(np.nanmean(v["mae_time"])),
            "mae_time_std":  float(np.nanstd (v["mae_time"])),
            "loss_mean":     float(np.nanmean(v["loss"])),
            "loss_std":      float(np.nanstd (v["loss"])),
            "n_runs":        len(v["mae_dist"]),
        }
    return aggregated


def print_aggregated_comparison(aggregated: dict) -> None:
    """
    Prints the final table with mean +/- standard deviation.
    Standard scientific-paper format.
    """
    print(f"\n{'='*75}")
    print(f" Final results -- mean +/- std over "
          f"{list(aggregated.values())[0]['n_runs']} seeds")
    print(f"{'='*75}")
    print(f"{'Method':25s} {'MAE dist (m)':22s} "
          f"{'MAE time (s)':22s} {'Loss':15s}")
    print("-" * 75)

    all_methods = (
        list(config.ESTIMATOR_NAMES)
        + ["fedprox_pure", "fedprox_hybrid_static_mu", "fedprox_hybrid", "scaffold_pure", "scaffold_hybrid_no_bypass", "scaffold_hybrid", "fednova_pure", "fednova_hybrid", "feddyn_pure", "feddyn_hybrid_static_alpha", "feddyn_hybrid"]
    )

    for name in all_methods:
        if name not in aggregated:
            continue
        a = aggregated[name]
        marker = " <-" if name == "hybrid" else ""
        mae_d = f"{a['mae_dist_mean']:.1f} +/- {a['mae_dist_std']:.1f}"
        mae_t = f"{a['mae_time_mean']:.1f} +/- {a['mae_time_std']:.1f}"
        loss  = f"{a['loss_mean']:.4f} +/- {a['loss_std']:.4f}"
        print(f"{name:25s} {mae_d:22s} {mae_t:22s} "
              f"{loss:15s}" + marker)

    print(f"{'='*75}")
    print(f" Format: mean +/- standard deviation (n={list(aggregated.values())[0]['n_runs']} runs)")


def print_statistical_tests(
    results_by_seed: dict[int, dict],
) -> None:
    """
    Wilcoxon test between the proposal (hybrid) and each baseline.
    Wilcoxon is preferable to a t-test for small samples (3-5 seeds)
    since it does not assume normality.

    Reports the p-value and whether the difference is significant
    (p < 0.05).
    """
    from scipy.stats import wilcoxon

    print(f"\n{'='*65}")
    print(f" Statistical tests -- Wilcoxon signed-rank")
    print(f" Proposal (hybrid) vs. each baseline")
    print(f" H0: there is no difference between the methods")
    print(f"{'='*65}")

    # Collect the hybrid's MAE per seed
    hybrid_scores = [
        results_by_seed[s].get("hybrid", {}).get(
            "final_mae_dist", float("nan")
        )
        for s in sorted(results_by_seed.keys())
    ]

    all_methods = (
        [m for m in config.ESTIMATOR_NAMES if m != "hybrid"]
        + ["fedprox_pure", "fedprox_hybrid_static_mu", "fedprox_hybrid", "scaffold_pure", "scaffold_hybrid_no_bypass", "scaffold_hybrid", "fednova_pure", "fednova_hybrid", "feddyn_pure", "feddyn_hybrid_static_alpha", "feddyn_hybrid"]
    )

    print(f"\n{'Baseline':25s} {'p-value':12s} {'Result':20s}")
    print("-" * 57)

    for name in all_methods:
        scores = [
            results_by_seed[s].get(name, {}).get(
                "final_mae_dist", float("nan")
            )
            for s in sorted(results_by_seed.keys())
        ]

        # Filter NaN
        valid = [
            (h, b) for h, b in zip(hybrid_scores, scores)
            if not (np.isnan(h) or np.isnan(b))
        ]

        if len(valid) < 2:
            print(f"{name:25s} {'N/A':12s} insufficient samples")
            continue

        h_vals = [v[0] for v in valid]
        b_vals = [v[1] for v in valid]

        if h_vals == b_vals:
            print(f"{name:25s} {'N/A':12s} identical values")
            continue

        try:
            stat, p = wilcoxon(h_vals, b_vals)
            sig = "significant" if p < 0.05 else "not significant"
            better = ("hybrid better"
                      if np.mean(h_vals) < np.mean(b_vals)
                      else "baseline better")
            print(f"{name:25s} {p:12.4f} {sig} ({better})")
        except Exception as e:
            print(f"{name:25s} error: {e}")

    print(f"\n Note: p < 0.05 rejects H0 with 95% confidence")
    print(f" With only {len(results_by_seed)} seeds, interpret")
    print(f" with caution -- use 5 seeds for the final version")



# -- Plot generation ---------------------------------------------

def _plot_results(
    results_by_seed: dict,
    aggregated:      dict,
    out_dir:         str,
) -> None:
    """
    Generates and saves 4 plots to out_dir/plots/:
      1. convergencia_loss.png  -- mean loss per round (all seeds)
      2. mae_dist_final.png     -- final MAE dist (mean +/- std, bars)
      3. divergencia.png        -- gradient divergence per round
      4. mae_por_modo.png       -- estimator MAE per transport mode
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed -- plots not generated.")
        print("          Run: pip install matplotlib")
        return

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Color palette per method
    COLORS = {
        "hybrid":        "#378ADD",
        "no_estimate":   "#1D9E75",
        "temporal_only": "#D85A30",
        "spatial_only":  "#7F77DD",
        "velocity_only": "#BA7517",
        "random_skip":   "#999999",
        "fedprox":       "#888780",
        "scaffold":      "#D4537E",
    }
    DASHES = {
        "fedprox":  (4, 3),
        "scaffold": (4, 3),
    }
    all_methods = list(config.ESTIMATOR_NAMES) + ["fedprox", "scaffold"]

    # -- Collect loss per round (mean over seeds) --------------
    loss_by_method: dict[str, list[float]] = {}
    div_by_method:  dict[str, list[float]] = {}

    for method in all_methods:
        round_losses = {}
        round_divs   = {}
        for seed_res in results_by_seed.values():
            if method not in seed_res:
                continue
            for entry in seed_res[method].get("round_log", []):
                r = entry["round"]
                round_losses.setdefault(r, []).append(
                    entry.get("mean_loss", float("nan"))
                )
                round_divs.setdefault(r, []).append(
                    entry.get("mean_div", float("nan"))
                )
        if round_losses:
            rounds = sorted(round_losses)
            loss_by_method[method] = [
                float(np.nanmean(round_losses[r])) for r in rounds
            ]
            div_by_method[method] = [
                float(np.nanmean(round_divs.get(r, [float("nan")])))
                for r in rounds
            ]

    rounds_x = list(range(1, config.N_ROUNDS + 1))

    # -- Plot 1: Loss convergence -------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in all_methods:
        if method not in loss_by_method:
            continue
        vals = loss_by_method[method]
        x    = rounds_x[:len(vals)]
        ax.plot(
            x, vals,
            label     = method,
            color     = COLORS.get(method, "#333"),
            dashes    = DASHES.get(method, (None, None)),
            linewidth = 2.2 if method == "hybrid" else 1.5,
        )
    ax.set_xlabel("FL round", fontsize=12)
    ax.set_ylabel("Mean loss", fontsize=12)
    ax.set_title("Convergence -- loss per round", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    bs_sf_prefix = f"bs{config.BUDGET_SAVING:.2f}_sf{config.SKIP_FACTOR:.2f}_"
    path1 = os.path.join(plots_dir, f"{bs_sf_prefix}convergencia_loss.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path1}")

    # -- Plot 2: Final MAE dist (bars with error) ---------------
    methods_with_data = [m for m in all_methods if m in aggregated]
    means = [aggregated[m]["mae_dist_mean"] for m in methods_with_data]
    stds  = [aggregated[m]["mae_dist_std"]  for m in methods_with_data]
    colors_bar = [COLORS.get(m, "#888") for m in methods_with_data]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        range(len(methods_with_data)), means,
        yerr       = stds,
        color      = colors_bar,
        capsize    = 5,
        edgecolor  = "white",
        linewidth  = 0.5,
        error_kw   = dict(elinewidth=1.2, ecolor="#444"),
    )
    # Highlight the proposal's bar
    if "hybrid" in methods_with_data:
        idx = methods_with_data.index("hybrid")
        bars[idx].set_edgecolor("#1a1a1a")
        bars[idx].set_linewidth(2)

    ax.set_xticks(range(len(methods_with_data)))
    ax.set_xticklabels(methods_with_data, rotation=25,
                        ha="right", fontsize=10)
    ax.set_ylabel("MAE dist (m)", fontsize=12)
    ax.set_title(
        f"Final distance MAE -- mean +/- std "
        f"({list(aggregated.values())[0]['n_runs']} seeds)",
        fontsize=13,
    )
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path2 = os.path.join(plots_dir, f"{bs_sf_prefix}mae_dist_final.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path2}")

    # -- Plot 3: Gradient divergence -----------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in all_methods:
        if method not in div_by_method:
            continue
        vals = div_by_method[method]
        if all(np.isnan(v) for v in vals):
            continue
        x = rounds_x[:len(vals)]
        ax.plot(
            x, vals,
            label     = method,
            color     = COLORS.get(method, "#333"),
            dashes    = DASHES.get(method, (None, None)),
            linewidth = 2.2 if method == "hybrid" else 1.5,
        )
    ax.set_xlabel("FL round", fontsize=12)
    ax.set_ylabel("Mean divergence", fontsize=12)
    ax.set_title("Gradient divergence per round", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path3 = os.path.join(plots_dir, f"{bs_sf_prefix}divergencia.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path3}")

    # -- Plot 4: MAE per transport mode ---------------------------
    modes = config.TRANSPORT_MODES
    estimator_methods = [
        m for m in config.ESTIMATOR_NAMES
        if m in results_by_seed[
            list(results_by_seed.keys())[0]
        ]
    ]

    mode_data: dict[str, dict[str, float]] = {}
    for method in estimator_methods:
        bm = {}
        for seed_res in results_by_seed.values():
            if method not in seed_res:
                continue
            by_mode = (seed_res[method]
                       .get("analysis", {})
                       .get("by_mode", {}))
            for mode, val in by_mode.items():
                bm.setdefault(mode, []).append(val)
        mode_data[method] = {
            m: float(np.nanmean(bm[m])) if m in bm else float("nan")
            for m in modes
        }

    x     = np.arange(len(modes))
    width = 0.8 / max(len(estimator_methods), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(estimator_methods):
        vals = [mode_data[method].get(m, float("nan")) for m in modes]
        offset = (i - len(estimator_methods) / 2 + 0.5) * width
        ax.bar(
            x + offset, vals,
            width      = width * 0.9,
            label      = method,
            color      = COLORS.get(method, "#888"),
            edgecolor  = "white",
            linewidth  = 0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=11)
    ax.set_ylabel("Estimator MAE", fontsize=12)
    ax.set_title("Estimator MAE per transport mode", fontsize=13)
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path4 = os.path.join(plots_dir, f"{bs_sf_prefix}mae_por_modo.png")
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path4}")

    print(f"\n  Plots saved to: {plots_dir}/")

    # Copy the plots to Google Drive
    _copy_to_gdrive([path1, path2, path3, path4], label="plots")




# -- Export comparison CSVs and final analysis --------------------------------

def _save_comparison_csv(
    seed: int,
    seed_results: dict,
) -> None:
    """
    Saves the pure vs. hybrid comparison CSV per algorithm.
    Each hybrid's baseline is its corresponding pure variant.
    FedAvg: hybrid vs no_estimate.
    Generated at results/csv/comparison_seed{N}_bs{X}_sf{Y}.csv
    """
    import csv as _csv, os as _os

    try:
        csv_dir = _os.path.join(config.RESULTS_DIR, "csv")
        _os.makedirs(csv_dir, exist_ok=True)
        bs = config.BUDGET_SAVING
        sf = config.SKIP_FACTOR
        path = _os.path.join(csv_dir,
            f"comparison_seed{seed}_bs{float(bs):.2f}_sf{float(sf):.2f}.csv")

        # pure -> hybrid pairs. Each hybrid is compared against its pure.
        # FedAvg has no "pure" -- uses no_estimate as the baseline.
        pairs = [
            ("no_estimate",    "hybrid"),
            ("fedprox_pure",   "fedprox_hybrid"),
            ("scaffold_pure",  "scaffold_hybrid"),
            ("fednova_pure",   "fednova_hybrid"),
        ]

        rows = []
        for base_name, hybrid_name in pairs:
            base_res   = seed_results.get(base_name)
            hybrid_res = seed_results.get(hybrid_name)
            if base_res is None and hybrid_res is None:
                continue

            def _get(res, key):
                return res.get(key, float("nan")) if res else float("nan")

            base_mae   = _get(base_res,   "final_mae_dist")
            base_cost  = _get(base_res,   "total_grad_steps")
            hybrid_mae = _get(hybrid_res, "final_mae_dist")
            hybrid_cost= _get(hybrid_res, "total_grad_steps")

            # Efficiency: quality gain per cost relative to the pure variant
            if (base_cost > 0 and hybrid_cost > 0
                    and base_mae == base_mae and hybrid_mae == hybrid_mae):
                quality_gain = base_mae / hybrid_mae if hybrid_mae > 0 else float("nan")
                cost_ratio   = hybrid_cost / base_cost
                efficiency   = quality_gain / cost_ratio if cost_ratio > 0 else float("nan")
            else:
                efficiency = float("nan")

            # Add the pure row
            if base_res:
                rows.append({
                    "algorithm":         base_name.replace("_pure","").replace("no_estimate","fedavg"),
                    "variant":           "pure" if "_pure" in base_name else "no_estimate",
                    "method":            base_name,
                    "seed":              seed,
                    "budget_saving":     bs,
                    "skip_factor":       sf,
                    "mae_dist_final":    _get(base_res, "final_mae_dist"),
                    "mae_time_final":    _get(base_res, "final_mae_time"),
                    "held_out_mae_dist": _get(base_res, "held_out_mae_dist"),
                    "total_grad_steps":  _get(base_res, "total_grad_steps"),
                    "baseline_method":   "",
                    "efficiency_vs_pure": float("nan"),
                })
            # Add the hybrid row
            if hybrid_res:
                rows.append({
                    "algorithm":         hybrid_name.replace("_hybrid",""),
                    "variant":           "hybrid",
                    "method":            hybrid_name,
                    "seed":              seed,
                    "budget_saving":     bs,
                    "skip_factor":       sf,
                    "mae_dist_final":    _get(hybrid_res, "final_mae_dist"),
                    "mae_time_final":    _get(hybrid_res, "final_mae_time"),
                    "held_out_mae_dist": _get(hybrid_res, "held_out_mae_dist"),
                    "total_grad_steps":  _get(hybrid_res, "total_grad_steps"),
                    "baseline_method":   base_name,
                    "efficiency_vs_pure": round(efficiency, 4) if efficiency == efficiency else float("nan"),
                })

        if not rows:
            return

        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        print(f"  [CSV] comparison_seed{seed}_bs{float(bs):.2f}_sf{float(sf):.2f}.csv -> {csv_dir}/")
        _copy_to_gdrive([path], label=f"comparison_seed{seed}")

    except Exception as _e:
        print(f"  [CSV] ERROR saving comparison: {_e}")
        import traceback; traceback.print_exc()


def _save_wilcoxon_csv(
    results_by_seed: dict,
) -> None:
    """
    Saves a CSV with the Wilcoxon test results.
    Comparison: each hybrid vs. its corresponding pure variant.
    Requires n_seeds >= 2 for the statistical test.
    With 1 seed: saves a descriptive comparison without a p-value.
    Generated at results/csv/wilcoxon_bs{X}_sf{Y}.csv
    """
    import csv as _csv, os as _os

    try:
        csv_dir = _os.path.join(config.RESULTS_DIR, "csv")
        _os.makedirs(csv_dir, exist_ok=True)
        bs = config.BUDGET_SAVING
        sf = config.SKIP_FACTOR
        path = _os.path.join(csv_dir,
            f"wilcoxon_bs{float(bs):.2f}_sf{float(sf):.2f}.csv")

        n_seeds = len(results_by_seed)
        seeds   = sorted(results_by_seed.keys())

        # pure -> hybrid pairs for comparison
        pairs = [
            ("no_estimate",   "hybrid",          "fedavg"),
            ("fedprox_pure",  "fedprox_hybrid",  "fedprox"),
            ("scaffold_pure", "scaffold_hybrid", "scaffold"),
            ("fednova_pure",  "fednova_hybrid",  "fednova"),
            ("feddyn_pure",   "feddyn_hybrid",   "feddyn"),
        ]

        rows = []
        for pure_name, hybrid_name, algo in pairs:
            pure_scores   = [results_by_seed[s].get(pure_name,   {}).get("final_mae_dist", float("nan")) for s in seeds]
            hybrid_scores = [results_by_seed[s].get(hybrid_name, {}).get("final_mae_dist", float("nan")) for s in seeds]

            valid = [(h, p) for h, p in zip(hybrid_scores, pure_scores)
                     if not (np.isnan(h) or np.isnan(p))]

            row = {
                "algorithm":          algo,
                "pure_method":        pure_name,
                "hybrid_method":      hybrid_name,
                "n_seeds":            len(valid),
                "hybrid_mae_mean":    round(float(np.mean([v[0] for v in valid])), 2) if valid else float("nan"),
                "pure_mae_mean":      round(float(np.mean([v[1] for v in valid])), 2) if valid else float("nan"),
                "hybrid_better":      (np.mean([v[0] for v in valid]) < np.mean([v[1] for v in valid])) if valid else float("nan"),
                "p_value":            float("nan"),
                "significant":        False,
                "note":               "",
            }

            if len(valid) < 2:
                row["note"] = f"insufficient for Wilcoxon (n={len(valid)}) -- use >= 2 seeds"
            elif [v[0] for v in valid] == [v[1] for v in valid]:
                row["note"] = "identical scores -- no statistical difference"
            else:
                try:
                    from scipy.stats import wilcoxon as _wilcoxon
                    stat, p = _wilcoxon([v[0] for v in valid], [v[1] for v in valid])
                    row["p_value"]   = round(float(p), 6)
                    row["significant"] = p < 0.05
                    row["note"]      = "Wilcoxon signed-rank"
                except Exception as _we:
                    row["note"] = f"Wilcoxon error: {_we}"

            rows.append(row)

        if not rows:
            return

        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        print(f"  [CSV] wilcoxon_bs{float(bs):.2f}_sf{float(sf):.2f}.csv -> {csv_dir}/ (n_seeds={n_seeds})")
        _copy_to_gdrive([path], label="wilcoxon")

    except Exception as _e:
        print(f"  [CSV] ERROR saving wilcoxon: {_e}")
        import traceback; traceback.print_exc()


def _save_aggregated_csv(
    results_by_seed: dict,
    aggregated: dict,
) -> None:
    """
    Saves a CSV with aggregated results (mean +/- std) over all seeds.
    Generated at results/csv/aggregated_bs{X}_sf{Y}.csv
    """
    import csv as _csv, os as _os
    from collections import defaultdict

    try:
        csv_dir = _os.path.join(config.RESULTS_DIR, "csv")
        _os.makedirs(csv_dir, exist_ok=True)
        bs = config.BUDGET_SAVING
        sf = config.SKIP_FACTOR
        path = _os.path.join(csv_dir,
            f"aggregated_bs{float(bs):.2f}_sf{float(sf):.2f}.csv")

        baseline_name = "no_estimate"
        eff_data = defaultdict(lambda: {"mae": [], "cost": [], "eff": [],
                                         "held_out": []})

        for seed, seed_results in results_by_seed.items():
            base = seed_results.get(baseline_name, {})
            base_mae  = base.get("final_mae_dist", float("nan"))
            base_cost = base.get("total_grad_steps", 0)
            for name, res in seed_results.items():
                if name.startswith("__"):
                    continue
                mae  = res.get("final_mae_dist", float("nan"))
                cost = res.get("total_grad_steps", 0)
                held = res.get("held_out_mae_dist", float("nan"))
                eff_data[name]["mae"].append(mae)
                eff_data[name]["cost"].append(cost)
                eff_data[name]["held_out"].append(held)
                if base_cost > 0 and mae == mae and base_mae == base_mae:
                    eff = (base_mae / mae) / (cost / base_cost) if cost > 0 else float("nan")
                    eff_data[name]["eff"].append(eff)

        rows = []
        order = (
            list(config.ESTIMATOR_NAMES)
            + ["fedprox_pure", "fedprox_hybrid",
               "scaffold_pure", "scaffold_hybrid",
               "fednova_pure", "fednova_hybrid"]
        )
        for name in order:
            d = eff_data.get(name)
            if not d or not d["mae"]:
                continue
            # aggregated from _aggregate_results
            agg = aggregated.get(name, {})
            rows.append({
                "method":               name,
                "budget_saving":        bs,
                "skip_factor":          sf,
                "n_seeds":              len(d["mae"]),
                "mae_dist_mean":        round(float(np.nanmean(d["mae"])), 2),
                "mae_dist_std":         round(float(np.nanstd(d["mae"])), 2),
                "mae_time_mean":        round(float(agg.get("mae_time_mean", float("nan"))), 2),
                "mae_time_std":         round(float(agg.get("mae_time_std",  float("nan"))), 2),
                "held_out_mae_mean":    round(float(np.nanmean(d["held_out"])), 2),
                "held_out_mae_std":     round(float(np.nanstd(d["held_out"])), 2),
                "grad_steps_mean":      round(float(np.nanmean(d["cost"])), 0),
                "grad_steps_std":       round(float(np.nanstd(d["cost"])), 0),
                "efficiency_mean":      round(float(np.nanmean(d["eff"])), 4) if d["eff"] else float("nan"),
                "efficiency_std":       round(float(np.nanstd(d["eff"])), 4)  if d["eff"] else float("nan"),
            })

        if not rows:
            return

        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        print(f"  [CSV] aggregated_bs{float(bs):.2f}_sf{float(sf):.2f}.csv -> {csv_dir}/")
        _copy_to_gdrive([path], label="aggregated")

    except Exception as _e:
        print(f"  [CSV] ERROR saving aggregated: {_e}")
        import traceback; traceback.print_exc()


# -- Local export of per-run metrics -------------------------------------------


# -- Copy to Google Drive (mounted folder) --------------------------------

def _copy_to_gdrive(src_paths: list, label: str = "") -> None:
    """
    Copies files to GDRIVE_RESULTS_DIR if configured.
    Uses shutil -- no external dependencies.
    Drive for Desktop syncs automatically when there is a connection.
    """
    import shutil as _shutil

    gdrive = getattr(config, "GDRIVE_RESULTS_DIR", "").strip()
    if not gdrive:
        return

    import os as _os
    try:
        _os.makedirs(gdrive, exist_ok=True)
        copied = 0
        for src_path in src_paths:
            if src_path and _os.path.isfile(src_path):
                dst = _os.path.join(gdrive, _os.path.basename(src_path))
                _shutil.copy2(src_path, dst)
                copied += 1
        if copied:
            tag = f" [{label}]" if label else ""
            print(f"  [GDrive]{tag} {copied} file(s) -> {gdrive}/")
    except Exception as e:
        print(f"  [GDrive] WARNING: failed to copy{' '+label if label else ''}: {e}")

def _save_run_csv(
    method_name: str,
    seed: int,
    global_eval_log: list,
    round_log: list,
    summary: dict,
):
    """
    Saves metrics to a local CSV at the end of each run.
    Independent of wandb -- works offline.

    Files generated at results/csv/:
      {method}_seed{N}_bs{X}_sf{Y}_rounds.csv   -- per-round metrics
      {method}_seed{N}_bs{X}_sf{Y}_summary.csv  -- final result
    """
    import csv as _csv, os as _os

    try:
        csv_dir = _os.path.join(config.RESULTS_DIR, "csv")
        _os.makedirs(csv_dir, exist_ok=True)
        bs  = summary.get("budget_saving", config.BUDGET_SAVING)
        sf  = summary.get("skip_factor",   config.SKIP_FACTOR)
        prefix = f"{method_name}_seed{seed}_bs{float(bs):.2f}_sf{float(sf):.2f}"

        # -- Rounds CSV -----------------------------------------------
        eval_by_round = {e["round"]: e for e in (global_eval_log or [])}
        rounds_path = _os.path.join(csv_dir, f"{prefix}_rounds.csv")
        fieldnames = [
            "round", "train_loss", "divergence", "n_clients",
            "round_grad_steps", "total_grad_steps",
            "n_skipped", "n_participating", "skip_rate",
            "h_mean", "h_std", "epochs_mean",
            "temporal_mean", "spatial_mean", "alpha_mean", "delta_m_mean",
            "h_temporal_contrib_mean", "h_spatial_contrib_mean",
            "geo_map_val_mean", "client_ema_val_mean", "budget_saving",
            "map_coverage_pct",
            # Global gradient per round
            "grad_norm_pre_mean", "grad_norm_pre_std",
            "grad_norm_post_mean", "grad_clip_rate",
            "mae_dist_global", "mae_time_global", "val_loss_global",
        ]
        with open(rounds_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in (round_log or []):
                rnd = r.get("round", "")
                ev  = eval_by_round.get(rnd, {})
                w.writerow({
                    "round":            rnd,
                    "train_loss":       r.get("mean_loss", ""),
                    "divergence":       r.get("mean_div", ""),
                    "n_clients":        r.get("n_clients", ""),
                    "round_grad_steps": r.get("round_grad_steps", ""),
                    "total_grad_steps": r.get("total_grad_steps", ""),
                    "n_skipped":        r.get("n_skipped", ""),
                    "n_participating":  r.get("n_participating", ""),
                    "skip_rate":        r.get("skip_rate", ""),
                    "h_mean":           r.get("h_mean", ""),
                    "h_std":            r.get("h_std", ""),
                    "epochs_mean":      r.get("epochs_mean", ""),
                    "temporal_mean":    r.get("temporal_mean", ""),
                    "spatial_mean":     r.get("spatial_mean", ""),
                    "alpha_mean":       r.get("alpha_mean", ""),
                    "delta_m_mean":     r.get("delta_m_mean", ""),
                    "h_temporal_contrib_mean": r.get("h_temporal_contrib_mean", ""),
                    "h_spatial_contrib_mean":  r.get("h_spatial_contrib_mean", ""),
                    "geo_map_val_mean": r.get("geo_map_val_mean", ""),
                    "client_ema_val_mean": r.get("client_ema_val_mean", ""),
                    "budget_saving":    r.get("budget_saving", ""),
                    "map_coverage_pct": r.get("map_coverage_pct", ""),
                    # Global gradient
                    "grad_norm_pre_mean":  r.get("grad_norm_pre_mean", ""),
                    "grad_norm_pre_std":   r.get("grad_norm_pre_std", ""),
                    "grad_norm_post_mean": r.get("grad_norm_post_mean", ""),
                    "grad_clip_rate":      r.get("grad_clip_rate", ""),
                    "mae_dist_global":  ev.get("mae_dist", ""),
                    "mae_time_global":  ev.get("mae_time", ""),
                    "val_loss_global":  ev.get("val_loss", ""),
                })

        # -- Summary CSV ------------------------------------------------
        summary_path = _os.path.join(csv_dir, f"{prefix}_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(summary.keys()))
            w.writeheader()
            w.writerow(summary)

        print(f"  [CSV] {prefix}_rounds.csv + _summary.csv -> {csv_dir}/")

        # Copy to Google Drive
        _copy_to_gdrive([rounds_path, summary_path], label=prefix)

    except Exception as _e:
        print(f"  [CSV] ERROR saving CSV for {method_name}: {_e}")
        import traceback; traceback.print_exc()


def _save_estimator_detail_log(
    method_name: str,
    seed: int,
    estimator_detail_log: list,
    summary: dict,
) -> None:
    """
    Saves the detailed per-estimate log to a separate CSV.
    Fields: is_first_round, cell_obs_count, cell_id_gx/gy.
    File: results/csv/{method}_seed{N}_bs{X}_sf{Y}_estimator_log.csv
    """
    import csv as _csv, os as _os
    if not estimator_detail_log:
        return
    try:
        csv_dir = _os.path.join(config.RESULTS_DIR, "csv")
        _os.makedirs(csv_dir, exist_ok=True)
        bs = summary.get("budget_saving", config.BUDGET_SAVING)
        sf = summary.get("skip_factor",   config.SKIP_FACTOR)
        prefix   = f"{method_name}_seed{seed}_bs{float(bs):.2f}_sf{float(sf):.2f}"
        log_path = _os.path.join(csv_dir, f"{prefix}_estimator_log.csv")

        fieldnames = [
            "round", "client_id", "mode", "velocity", "alpha", "delta_m",
            "estimate", "temporal", "spatial",
            "h_temporal_contrib", "h_spatial_contrib",
            "geo_map_val", "client_ema_val",
            "is_first_round", "cell_obs_count",
            "cell_id_gx", "cell_id_gy",
        ]
        with open(log_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for e in estimator_detail_log:
                w.writerow({k: e.get(k, "") for k in fieldnames})

        print(f"  [CSV] {prefix}_estimator_log.csv "
              f"({len(estimator_detail_log):,} entries) -> {csv_dir}/")
        _copy_to_gdrive([log_path], label=f"{prefix}_estimator_log")

    except Exception as _e:
        print(f"  [CSV] ERROR saving estimator_log for {method_name}: {_e}")
        import traceback; traceback.print_exc()


def _save_client_rank_log(
    method_name: str,
    seed: int,
    client_log: list,
    summary: dict,
) -> None:
    """
    Saves the per-client rank and decision history to a separate CSV.
    Used to detect concentration of participation.
    File: results/csv/{method}_seed{N}_bs{X}_sf{Y}_client_rank_log.csv
    """
    import csv as _csv, os as _os
    if not client_log:
        return
    try:
        csv_dir = _os.path.join(config.RESULTS_DIR, "csv")
        _os.makedirs(csv_dir, exist_ok=True)
        bs = summary.get("budget_saving", config.BUDGET_SAVING)
        sf = summary.get("skip_factor",   config.SKIP_FACTOR)
        prefix   = f"{method_name}_seed{seed}_bs{float(bs):.2f}_sf{float(sf):.2f}"
        log_path = _os.path.join(csv_dir, f"{prefix}_client_rank_log.csv")

        fieldnames = [
            "round", "client_id", "decision",
            "h_rank_value", "h_rank_pos",
            "h_raw", "h_raw_pool_std", "h_raw_pool_range", "h_raw_pool_mean",
            "mode", "divergence", "velocity", "n_epochs",
            "loss_mean", "grad_norm_pre", "grad_norm_post", "grad_clip_rate",
            "consecutive_skip", "consecutive_train",
        ]
        with open(log_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for e in client_log:
                w.writerow({k: e.get(k, "") for k in fieldnames})

        print(f"  [CSV] {prefix}_client_rank_log.csv "
              f"({len(client_log):,} entries) -> {csv_dir}/")

    except Exception as _e:
        print(f"  [CSV] ERROR saving client_rank_log for {method_name}: {_e}")
        import traceback; traceback.print_exc()


def _save_clients_meta(
    seed: int,
    profiles: dict,
    samples: list,
    train_ds: "TransitionPredictionDataset",
    val_ds_internal,
) -> None:
    """
    Exports n_k (training samples per client in the FL pool) -- used for
    the nominal weights p_k = n_k/n in the theoretical chi-squared
    formalization. Without this, the calculation assumes a uniform p,
    a problematic approximation on GeoLife (client sample counts vary
    by a ratio of roughly 5000:1) -- kept identical on MOBIS for
    consistency between datasets.

    Includes ineligible clients (n_train_samples=0 -- these fall into
    create_datasets()'s internal validation/test split for this seed
    and therefore never train), to distinguish "insufficient data"
    from "discarded by the estimator" -- both produce decision="skip"/
    absence from client_rank_log, but for different reasons.

    Pure instrumentation: reads counts already materialized in
    train_ds/val_ds_internal/samples, does not influence client
    selection, epoch allocation, or training.

    File: results/csv/clients_meta_seed{N}.csv
    """
    import csv as _csv, os as _os
    from collections import Counter

    train_counts  = Counter(s.get("user_id") for s in train_ds.samples)
    val_counts    = (Counter(s.get("user_id") for s in val_ds_internal.samples)
                      if val_ds_internal is not None else Counter())
    window_counts = Counter(s.get("user_id") for s in samples)

    try:
        csv_dir = _os.path.join(config.RESULTS_DIR, "csv")
        _os.makedirs(csv_dir, exist_ok=True)
        log_path = _os.path.join(csv_dir, f"clients_meta_seed{seed}.csv")

        fieldnames = [
            "client_id", "user_id",
            "n_train_samples", "n_val_samples", "n_windows",
            "eligible",
        ]
        with open(log_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for cid, profile in sorted(profiles.items()):
                uid  = profile.get("user_id")
                n_tr = train_counts.get(uid, 0)
                w.writerow({
                    "client_id":       cid,
                    "user_id":         uid,
                    "n_train_samples": n_tr,
                    "n_val_samples":   val_counts.get(uid, 0),
                    "n_windows":       window_counts.get(uid, 0),
                    "eligible":        n_tr > 0,
                })

        print(f"  [CSV] clients_meta_seed{seed}.csv "
              f"({len(profiles)} clients) -> {csv_dir}/")

    except Exception as _e:
        print(f"  [CSV] ERROR saving clients_meta_seed{seed}: {_e}")
        import traceback; traceback.print_exc()


def _reorder_for_spatial_diversity(
    samples: list,
    seed:    int,
) -> list:
    """
    Reorders the samples so that the user_ids selected for held-out
    (the first 15% after split_samples' shuffle) cover diverse
    geographic regions instead of being concentrated.

    Strategy: groups user_ids by geographic quadrant and interleaves
    one from each quadrant -- so split_samples' shuffle ends up
    distributing the held-out set geographically.

    Used only for MOBIS (GeoLife never calls this function).
    """
    from collections import defaultdict
    import numpy as np

    # Group samples by user_id
    by_user = defaultdict(list)
    for s in samples:
        by_user[s.get("user_id", "unknown")].append(s)

    # Compute mean position per user
    user_pos = {}
    for uid, samps in by_user.items():
        lats = [s["lat"] for s in samps if s.get("lat")]
        lons = [s["lon"] for s in samps if s.get("lon")]
        if lats and lons:
            user_pos[uid] = (float(np.mean(lats)), float(np.mean(lons)))

    if len(user_pos) < 4:
        return samples  # too few users -- not worth reordering

    # Split into quadrants by the median point
    lats_all = [p[0] for p in user_pos.values()]
    lons_all = [p[1] for p in user_pos.values()]
    lat_mid  = float(np.median(lats_all))
    lon_mid  = float(np.median(lons_all))

    quadrants = defaultdict(list)
    for uid, (lat, lon) in user_pos.items():
        q = (int(lat >= lat_mid), int(lon >= lon_mid))
        quadrants[q].append(uid)

    # Shuffle within each quadrant (deterministic per seed)
    rng = np.random.default_rng(seed + 9999)
    for q in quadrants:
        rng.shuffle(quadrants[q])

    # Interleave user_ids from the 4 quadrants
    # Result: the first 15% cover every quadrant
    reordered_ids = []
    max_len = max(len(v) for v in quadrants.values())
    for i in range(max_len):
        for q_uids in quadrants.values():
            if i < len(q_uids):
                reordered_ids.append(q_uids[i])

    # Rebuild samples in the new user_id order
    result = []
    for uid in reordered_ids:
        result.extend(by_user[uid])
    # Add users with no position at the end
    covered = set(reordered_ids)
    for uid, samps in by_user.items():
        if uid not in covered:
            result.extend(samps)

    return result


def main():
    import argparse
    _all_methods = [
        "no_estimate", "hybrid",
        "temporal_only", "spatial_only", "velocity_only", "mode_only",
        "random_skip",  # exploratory -- see config.ESTIMATOR_NAMES
        "fedprox_pure",  "fedprox_hybrid_static_mu",  "fedprox_hybrid",
        "scaffold_pure", "scaffold_hybrid_no_bypass", "scaffold_hybrid",
        "fednova_pure",  "fednova_hybrid",
        "feddyn_pure",   "feddyn_hybrid_static_alpha",  "feddyn_hybrid",
    ]
    parser = argparse.ArgumentParser(
        description="FedHME -- FL experiment (GeoLife + MOBIS)"
    )
    parser.add_argument(
        "--methods", nargs="+", choices=_all_methods, default=None,
        metavar="METHOD",
        help=("Methods to run. Options: " + ", ".join(_all_methods) +
              ". Default: all."),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        metavar="S",
        help="Seeds to run (e.g.: --seeds 0 1 2). Default: config.EXPERIMENT_SEEDS.",
    )
    parser.add_argument(
        "--dataset", choices=["geolife", "mobis_eta", "mobis_eta_purpose"],
        default="geolife",
        help="Dataset to run. Default: geolife.",
    )
    parser.add_argument(
        "--budget-saving", type=float, default=None,
        metavar="B",
        help="Overrides BUDGET_SAVING (e.g.: 0.8). Default: value in config.py.",
    )
    parser.add_argument(
        "--skip-factor", type=float, default=None,
        metavar="F",
        help="Overrides SKIP_FACTOR (e.g.: 0.5). Default: value in config.py.",
    )
    parser.add_argument(
        "--random-skip-mode", choices=["paired", "unpaired"], default=None,
        help=("Per-round participant mode for the exploratory random_skip "
              "baseline. 'unpaired': fixed m(r) (--random-skip-target-n). "
              "'paired': m(r) = the hybrid's real n_participating in the "
              "same round/seed (requires --random-skip-paired-dir). "
              "Default: value in config.py (unpaired)."),
    )
    parser.add_argument(
        "--random-skip-target-n", type=int, default=None,
        metavar="N",
        help=("Fixed m(r) for random_skip in unpaired mode. Default: "
              "value in config.py (None -- recomputed each round as "
              "round(n_valid(r) * (1 - skip_factor)), recommended; a "
              "fixed constant here leaks hybrid-derived information "
              "into random_skip's epoch allocation)."),
    )
    parser.add_argument(
        "--random-skip-paired-dir", type=str, default=None,
        metavar="DIR",
        help=("csv/ folder of an official hybrid run (same dataset/"
              "seeds), used to pair m(r) in random_skip's paired mode. "
              "Required if --random-skip-mode paired."),
    )
    parser.add_argument(
        "--velocity-decay", type=float, default=None,
        metavar="D",
        help=("Overrides VELOCITY_DECAY (e.g.: 0.001). Default: value in "
              "config.py (0.05). velocity_decay=config.VELOCITY_DECAY must "
              "be passed explicitly to create_estimator() for this "
              "override to take effect -- __init__'s default is fixed at "
              "import time, not recomputed per call."),
    )
    parser.add_argument(
        "--mobis-window-size", type=int, default=None,
        metavar="W",
        help=("Overrides MOBIS_WINDOW_SIZE (e.g.: 5). Default: value in "
              "config.py (10, inherited from GeoLife's WINDOW_SIZE -- "
              "never calibrated for MOBIS). W=10 legs on MOBIS corresponds "
              "to roughly 1.7-2.5 days of mobility (legs are not "
              "continuous in time, mean duration 16.8min), very different "
              "from W=10 GPS points on GeoLife (~20-50s) -- a candidate "
              "explanation for the training stagnation observed on MOBIS. "
              "Only takes effect with --dataset mobis_eta or "
              "mobis_eta_purpose."),
    )
    parser.add_argument(
        "--mobis-period", choices=["pre_covid", "covid", "post_covid"],
        default=None, metavar="PERIOD",
        help=("Filters MOBIS by collection period (pre_covid: before "
              "2020-03-02; covid: 2020-03-02 to 2021-12-31; post_covid: "
              "from 2022-01-01 onward). Only takes effect with "
              "--dataset mobis_eta or mobis_eta_purpose. Default: no "
              "filter (the whole dataset, original behavior)."),
    )
    parser.add_argument(
        "--mobis-stratify-mode", action="store_true",
        help=("Reserves a minimum number of participants per predominant "
              "mode (walk/bike/bus/car/subway/train) when selecting the "
              "N_CLIENTS sample, before filling the rest by random "
              "shuffle -- see --mobis-min-per-mode. Default: off "
              "(plain shuffle, original behavior). Only takes effect "
              "with --dataset mobis_eta or mobis_eta_purpose."),
    )
    parser.add_argument(
        "--mobis-min-per-mode", type=int, default=3, metavar="N",
        help=("Minimum number of participants per predominant mode to "
              "reserve when --mobis-stratify-mode is active. Ignored "
              "if --mobis-stratify-mode is not passed. Default: 3."),
    )
    parser.add_argument(
        "--mobis-purpose-filter", nargs="+", metavar="PURPOSE",
        default=None,
        help=("Filters MOBIS participants by the predominant purpose of "
              "their activities (whole clients only). Accepts one or "
              "more purposes: Home Work Leisure Shopping "
              "Other Errand Assistance Education. "
              "Examples: "
              "--mobis-purpose-filter Work Errand Education "
              "(local mobility); "
              "--mobis-purpose-filter Leisure Shopping "
              "(mixed/long-distance mobility). "
              "Default: no filter (all eligible participants). "
              "Only takes effect with --dataset mobis_eta or "
              "mobis_eta_purpose."),
    )
    parser.add_argument(
        "--mobis-region", choices=["zurich"], default=None,
        metavar="REGION",
        help=("Filters MOBIS participants by a named geographic region "
              "(currently only 'zurich'). Keeps participants whose "
              "fraction of legs with destination inside the region's "
              "bbox is >= --mobis-region-frac (default 0.50). "
              "Implemented in mobis_loader.py (region_filter) -- did not "
              "exist in any earlier version of the code. Default: no "
              "filter. Only takes effect with --dataset mobis_eta or "
              "mobis_eta_purpose."),
    )
    parser.add_argument(
        "--mobis-region-frac", type=float, default=0.50,
        metavar="FRAC",
        help=("Minimum fraction of legs with destination in the region "
              "required by --mobis-region. Ignored if --mobis-region is "
              "not passed. Default: 0.50 (same threshold as "
              "check_zurich_viability.py)."),
    )
    parser.add_argument(
        "--grid-size", type=int, default=None,
        metavar="G",
        help=("Overrides GRID_SIZE (e.g.: 10). Default: value in "
              "config.py (40). Needed for grid-resolution comparisons "
              "(e.g. Zurich G40 vs G10)."),
    )
    args = parser.parse_args()

    # Apply hyperparameter overrides from the CLI
    if args.budget_saving is not None:
        config.BUDGET_SAVING = args.budget_saving
        print(f"[CLI] BUDGET_SAVING={config.BUDGET_SAVING}")
    if args.skip_factor is not None:
        config.SKIP_FACTOR = args.skip_factor
        print(f"[CLI] SKIP_FACTOR={config.SKIP_FACTOR}")
    if args.random_skip_mode is not None:
        config.RANDOM_SKIP_MODE = args.random_skip_mode
        print(f"[CLI] RANDOM_SKIP_MODE={config.RANDOM_SKIP_MODE}")
    if args.random_skip_target_n is not None:
        config.RANDOM_SKIP_TARGET_N = args.random_skip_target_n
        print(f"[CLI] RANDOM_SKIP_TARGET_N={config.RANDOM_SKIP_TARGET_N}")
    if args.random_skip_paired_dir is not None:
        config.RANDOM_SKIP_PAIRED_HYBRID_DIR = args.random_skip_paired_dir
        print(f"[CLI] RANDOM_SKIP_PAIRED_HYBRID_DIR={config.RANDOM_SKIP_PAIRED_HYBRID_DIR}")
    if config.RANDOM_SKIP_MODE == "paired" and not config.RANDOM_SKIP_PAIRED_HYBRID_DIR:
        parser.error("--random-skip-mode paired requires --random-skip-paired-dir "
                      "(or RANDOM_SKIP_PAIRED_HYBRID_DIR in config.py)")
    if args.velocity_decay is not None:
        config.VELOCITY_DECAY = args.velocity_decay
        print(f"[CLI] VELOCITY_DECAY={config.VELOCITY_DECAY}")
    if args.mobis_window_size is not None:
        config.MOBIS_WINDOW_SIZE = args.mobis_window_size
        print(f"[CLI] MOBIS_WINDOW_SIZE={config.MOBIS_WINDOW_SIZE}")
    if args.mobis_period is not None:
        print(f"[CLI] MOBIS period={args.mobis_period}")
    if args.mobis_stratify_mode:
        print(f"[CLI] MOBIS stratify_by_mode=True "
              f"(min_per_mode={args.mobis_min_per_mode})")
    if args.mobis_purpose_filter is not None:
        print(f"[CLI] MOBIS purpose_filter={args.mobis_purpose_filter}")
    if args.mobis_region is not None:
        print(f"[CLI] MOBIS region_filter={args.mobis_region} "
              f"(frac>={args.mobis_region_frac:.0%})")
    if args.grid_size is not None:
        config.GRID_SIZE = args.grid_size
        print(f"[CLI] GRID_SIZE={config.GRID_SIZE}")

    selected_methods = set(args.methods) if args.methods else None
    dataset_name     = args.dataset
    if dataset_name not in config.SUPPORTED_DATASETS:
        parser.error(
            f"Unsupported --dataset {dataset_name!r}. "
            f"Supported: {config.SUPPORTED_DATASETS}."
        )
    config.CURRENT_DATASET = dataset_name  # used by model.py (mobis_eta_purpose)
    # Per-dataset RESULTS_DIR -- a single shared "./results" would risk
    # overlap between GeoLife and MOBIS runs done in sequence; without
    # this, running the grid on both datasets back to back would
    # overwrite the first one's results.
    config.RESULTS_DIR = f"./results_{dataset_name}"

    # Clean up .pkl caches from previous runs in /tmp
    cleanup_pkl_cache()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    if config.USE_GPU and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    seeds = args.seeds if args.seeds is not None else config.EXPERIMENT_SEEDS
    


    print(f"\n{'='*65}")
    print(f" FedHME -- Mode Transition Prediction ({dataset_name})")
    _n_methods = len(selected_methods) if selected_methods else (len(config.ESTIMATOR_NAMES) + 6)
    print(f" {len(seeds)} runs x {_n_methods} methods")
    print(f" Version: {config.CODE_VERSION}")
    print(f"{'='*65}")
    print(f" Device      : {device}")
    if selected_methods:
        print(f" Methods     : {', '.join(sorted(selected_methods))}")
    print(f" Seeds       : {seeds}")
    print(f" Clients     : {config.N_CLIENTS}")
    print(f" Rounds      : {config.N_ROUNDS}")
    print(f" Grid        : {config.GRID_SIZE}")
    print(f"{'='*65}\n")

    results_by_seed: dict[int, dict] = {}
    test_ds_by_seed: dict[int, object] = {}

    for seed in seeds:
    
        
        config.ACTUAL_SEED = seed

        print(f"\n{'='*65}")
        print(f" SEED = {seed}")
        print(f"{'='*65}")

        # -- Different clients per seed ------------------------
        # Each seed selects a different subset of GeoLife's 69
        # labeled clients. Cached per seed.
        print(f"\n[1/4] Loading {dataset_name.upper()} (seed={seed})...")
        try:
            if dataset_name == "geolife":
                samples_cache  = config.SAMPLES_CACHE.format(seed=seed)
                profiles_cache = config.PROFILES_CACHE.format(seed=seed)
                samples, profiles, val_samples, val_profiles = load_geolife(
                    n_clients      = config.N_CLIENTS,
                    seed           = seed,
                    samples_cache  = samples_cache,
                    profiles_cache = profiles_cache,
                )
            elif dataset_name in ("mobis_eta", "mobis_eta_purpose"):
                task = "eta" if dataset_name == "mobis_eta" else "eta_purpose"
                # Set N_FEATURES before creating the model (only used as a
                # shape fallback in dataset.py when len(samples)==0 -- the
                # actual model creation goes through create_model(), which
                # already reads MOBIS_N_FEATURES/MOBIS_N_FEATURES_PURPOSE
                # directly)
                if dataset_name == "mobis_eta":
                    config.N_FEATURES = config.MOBIS_N_FEATURES
                else:
                    config.N_FEATURES = config.MOBIS_N_FEATURES_PURPOSE
                samples, profiles, val_samples, val_profiles = load_mobis(
                    n_clients = config.N_CLIENTS,
                    seed      = seed,
                    task      = task,
                    period    = args.mobis_period,
                    stratify_by_mode = args.mobis_stratify_mode,
                    min_per_mode     = args.mobis_min_per_mode,
                    purpose_filter   = args.mobis_purpose_filter,
                    region_filter    = args.mobis_region,
                    region_frac_min  = args.mobis_region_frac,
                )
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        except (FileNotFoundError, ValueError) as e:
            print(f"\n[ERROR] {e}")
            continue

        if not profiles:
            print("[ERROR] No profile loaded.")
            continue

        # MOBIS: reorder for spatial diversity in the held-out set
        # (the first 15% after split_samples' shuffle cover diverse
        # geographic regions instead of being concentrated)
        if dataset_name in ("mobis_eta", "mobis_eta_purpose"):
            samples = _reorder_for_spatial_diversity(samples, seed)

        n_valid = len(profiles)
        _orig_n = config.N_CLIENTS
        config.N_CLIENTS       = n_valid
        config.MIN_FIT_CLIENTS = min(config.MIN_FIT_CLIENTS, n_valid)
        config.MIN_EVAL_CLIENTS = min(config.MIN_EVAL_CLIENTS, n_valid)

        # -- Per-seed dataset characterization -----------------
        print(f"\n[1.5/4] Dataset plots (seed={seed})...")
        _ds_wandb = _wandb_init(
            f"dataset_seed{seed}", seed)
        plot_dataset_characterization(
            samples, profiles,
            out_dir=config.RESULTS_DIR,
            wandb_run=_ds_wandb,
        )
        if _ds_wandb is not None:
            try: _ds_wandb.finish()
            except Exception: pass

        # -- Create datasets for this seed ----------------------
        from dataset import create_datasets, TransitionPredictionDataset
        _use_log1p_target = dataset_name in getattr(config, "LOG1P_TARGET_DATASETS", ())
        train_ds, val_ds_internal, test_ds, test_uids = create_datasets(
            samples,
            test_user_ratio = 0.15,
            val_ratio       = 0.15,
            seed            = seed,
            use_log1p_target = _use_log1p_target,
        )
        test_ds_by_seed[seed] = test_ds

        # Per-client n_k (weights p_k=n_k/n for the theoretical
        # chi-squared formalization) -- pure instrumentation, see
        # _save_clients_meta's docstring.
        _save_clients_meta(seed, profiles, samples, train_ds, val_ds_internal)

        # val_ds from dedicated clients (never participate in training)
        # Uses the full trajectory -- no temporal split
        if val_samples:
            val_ds = TransitionPredictionDataset(
                val_samples,
                fit_scaler  = False,
                scaler      = train_ds.scaler,
                target_mean = train_ds.target_mean,
                target_std  = train_ds.target_std,
                split_name  = "val_dedicated",
                use_log1p_target = _use_log1p_target,
            )
            print(f"[Val] {len(val_ds)} samples from "
                  f"{len(val_profiles)} dedicated clients")
        else:
            # Fallback: use the temporal split's val_ds
            _, val_ds, _, _ = create_datasets(
                samples, test_user_ratio=0.15,
                val_ratio=0.15, seed=seed,
                use_log1p_target=_use_log1p_target,
            )
            print("[Val] No dedicated clients -- using the temporal split")

        # -- Deterministic seed ---------------------------------
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # -- Run estimators -------------------------------------
        print(f"\n[2/4] Estimators (seed={seed})...")
        r_g1, _pending_csv_g1, _results_g1 = run_estimator_group(
                                    train_ds, profiles,
                                    device, val_ds=val_ds,
                                    seed=seed,
                                    selected_methods=selected_methods)

        # -- Run algorithms individually -------------------------
        print(f"\n[3/4] FL algorithms (seed={seed})...")
        r_g2, _pending_csv_g2 = _run_algorithms_individually(
            train_ds, profiles, device,
            val_ds=val_ds, seed=seed,
            selected_methods=selected_methods,
        )

        # seed_results uses _results_g1 (has final_params) instead of r_g1 (aggregated)
        results_by_seed[seed] = {**_results_g1, **r_g2}
        _pending_csv_all = {**_pending_csv_g1, **_pending_csv_g2}

        # -- Per-seed evaluation run --------------------------
        _create_seed_eval_run(
            seed, results_by_seed[seed],
            test_ds, val_ds, device,
            pending_csv=_pending_csv_all,
        )

        # Restore N_CLIENTS for the next seed
        config.N_CLIENTS = _orig_n

    # -- Final evaluation run ----------------------------------
    _create_final_eval_run(
        results_by_seed, test_ds_by_seed, device,
    )


def _run_algorithms_individually(
    dataset, profiles, device,
    val_ds=None, seed=0,
    selected_methods: set = None,
) -> dict:
    """
    Runs each FL algorithm in two variants:
    - Pure: standard behavior from the literature
    - +Hybrid: with the estimator plugin for selection, epochs, mu/bypass

    FedProx and SCAFFOLD get a 3rd variant that isolates the dedicated
    channel (dynamic mu / adaptive bypass) from the common channel
    (selection+epochs via h_hat, already tested by random_skip -- see
    KB_DECISAO_ESCOPO_RANDOM_SKIP_v1.md): fedprox_hybrid_static_mu and
    scaffold_hybrid_no_bypass select/allocate epochs via h_hat but keep
    the dedicated channel off (mu fixed / bypass never triggers, same as
    pure). FedNova doesn't get a 3rd variant -- no dedicated channel, it
    only normalizes by tau_k, a mechanism of the algorithm itself, not of
    FedHME.

    FedDyn (Acar et al. 2021) also gets 3 variants -- "pure" (fixed
    alpha, no estimator), "hybrid_static_alpha" (dedicated channel via
    h_hat, fixed alpha -- "FedNova style": selection/epochs only),
    "hybrid" (dedicated channel + dynamic alpha via h_hat -- "FedProx
    style"). Same dedicated_channel mechanism as FedProx/SCAFFOLD, kept
    as a separate name (hybrid_static_alpha vs. hybrid_static_mu) since
    FedDyn's dedicated channel modulates alpha, not mu.
    """
    results = {}
    _pending_csv = {}  # accumulates data for the CSV -- held_out filled in later

    _default_variants = [("pure", False, None), ("hybrid", True, None)]
    algo_configs = [
        ("fedprox",  FedProxStrategy,  make_fedprox_client_fn,
         [("pure", False, None),
          ("hybrid_static_mu", True, False),
          ("hybrid", True, True)]),
        ("scaffold", SCAFFOLDStrategy, make_scaffold_client_fn,
         [("pure", False, None),
          ("hybrid_no_bypass", True, False),
          ("hybrid", True, True)]),
        ("fednova",  FedNovaStrategy,  make_fednova_client_fn,  _default_variants),
        ("feddyn",   FedDynStrategy,   make_feddyn_client_fn,
         [("pure", False, None),
          ("hybrid_static_alpha", True, False),
          ("hybrid", True, True)]),
    ]

    for name, StrategyClass, make_fn, variant_specs in algo_configs:
        variants = [
            (v, use_e, ded_ch) for v, use_e, ded_ch in variant_specs
            if selected_methods is None or f"{name}_{v}" in selected_methods
        ]
        if not variants:
            continue
        for variant, use_est, ded_ch in variants:
            run_name = f"{name}_{variant}"
            display_run = config.METHOD_DISPLAY_NAMES.get(run_name, run_name)
            print(f"\n  [{display_run}] seed={seed}")
            import random
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            try:
                wandb_run = _wandb_init(run_name, seed)
                result = _run_single_algorithm(
                    run_name, StrategyClass, make_fn,
                    dataset, profiles, device,
                    val_ds=val_ds, seed=seed,
                    wandb_run=wandb_run,
                    use_estimator=use_est,
                    dedicated_channel=ded_ch,
                )
                results[run_name] = result
            except Exception as e:
                print(f"  [ERROR] {display_run} failed: {e}")
                import traceback; traceback.print_exc()
                print(f"  Continuing with the rest...")
            finally:
                # Pending CSV -- held_out filled in later
                if run_name in results:
                    _res = results[run_name]
                    _summary = {
                        "method":            run_name,
                        "seed":              seed,
                        "mae_dist_final":    _res.get("final_mae_dist", ""),
                        "mae_time_final":    _res.get("final_mae_time", ""),
                        "held_out_mae_dist": "",
                        "held_out_mae_time": "",
                        "held_out_mse_dist":   "",
                        "held_out_mse_time":   "",
                        "held_out_rmse_dist":  "",
                        "held_out_rmse_time":  "",
                        "held_out_mape_dist":  "",
                        "held_out_mape_time":  "",
                        "held_out_mape_n_excluded_dist": "",
                        "held_out_mape_n_excluded_time": "",
                        "held_out_smape_dist": "",
                        "held_out_smape_time": "",
                        "held_out_r2_dist":    "",
                        "held_out_r2_time":    "",
                        "budget_saving":     config.BUDGET_SAVING,
                        "skip_factor":       config.SKIP_FACTOR,
                        "total_grad_steps":  _res.get("total_grad_steps", ""),
                    }
                    # Save the CSV immediately (without held-out for now)
                    # _create_seed_eval_run will fill in the held-out later
                    _save_run_csv(
                        run_name, seed,
                        _res.get("global_eval_log", []),
                        _res.get("round_log", []),
                        _summary,
                    )
                    # Detailed per-estimate log
                    _save_estimator_detail_log(
                        run_name, seed,
                        _res.get("estimator_detail_log", []),
                        _summary,
                    )
                    # Per-client rank and decision history
                    _save_client_rank_log(
                        run_name, seed,
                        _res.get("client_log", []),
                        _summary,
                    )
                    _pending_csv[run_name] = {
                        "global_eval_log": _res.get("global_eval_log", []),
                        "round_log":       _res.get("round_log", []),
                        "summary":         _summary,
                    }
                # Free memory between runs
                import gc
                try:
                    import ray
                    if ray.is_initialized():
                        ray.shutdown()
                        print(f"  [cleanup] Ray shutdown after {display_run}")
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    return results, _pending_csv


def _run_single_algorithm(
    name, StrategyClass, make_fn,
    dataset, profiles, device,
    val_ds=None, seed=0, wandb_run=None,
    use_estimator=False, dedicated_channel=None,
) -> dict:
    """Runs a single FL algorithm and returns the results."""

    # Ensure the initial weights are born identical
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    from model import create_model, get_parameters
    import flwr.common as flc

    initial_model = create_model()
    initial_params = flc.ndarrays_to_parameters(
        get_parameters(initial_model)
    )

    kwargs = {
        "initial_parameters":    initial_params,
        "min_fit_clients":       config.MIN_FIT_CLIENTS,
        "min_evaluate_clients":  config.MIN_EVAL_CLIENTS,
        "min_available_clients": config.MIN_FIT_CLIENTS,
        "profiles":              profiles,
        "dataset":               dataset,  # to create MobilityClients in the strategies
        "wandb_run":             wandb_run,
        "use_estimator":         use_estimator,
    }
    if "scaffold" in name.lower():
        kwargs["n_clients"] = config.N_CLIENTS
    if "feddyn" in name.lower():
        kwargs["n_clients"] = config.N_CLIENTS
    # dedicated_channel exists on FedProx/SCAFFOLD/FedDyn's constructors
    # (not FedNova -- no dedicated channel, see docstring above) --
    # thread it through for all three, not just feddyn.
    if ("fedprox" in name.lower() or "scaffold" in name.lower()
            or "feddyn" in name.lower()):
        if dedicated_channel is not None:
            kwargs["dedicated_channel"] = dedicated_channel

    strategy = StrategyClass(**kwargs)
    strategy._val_ds = val_ds
    strategy._device = device

    client_fn = make_fn(dataset, profiles, device,
                         dataset_name=config.CURRENT_DATASET)

    history = _run_simulation(
        client_fn   = client_fn,
        strategy    = strategy,
    )

    # Final metrics
    mae_d, mae_t, loss = float("nan"), float("nan"), float("nan")
    if strategy.global_eval_log:
        last_evals = strategy.global_eval_log[-3:]
        mae_d = float(np.mean([e["mae_dist"]
                               for e in last_evals]))
        mae_t = float(np.mean([e["mae_time"]
                               for e in last_evals]))
        loss = float(np.mean([e["val_loss"]
                              for e in last_evals]))

    _wandb_log_final(wandb_run, mae_d, mae_t, loss)
    strategy._final_params = getattr(strategy, '_final_params', None)

    return {
        "group":          "algorithm",
        "round_log":      strategy.round_log,
        "global_eval_log": getattr(strategy, 'global_eval_log', []),
        "final_mae_dist": mae_d,
        "final_mae_time": mae_t,
        "final_loss":     loss,
        "final_params":   strategy._final_params,
        "total_grad_steps": strategy.total_grad_steps,
        # estimator_detail_log is read from strategy._estimator.log
        # (not the Mixin's internal strategy._estimator_log, which only
        # accumulates participants) -- mirrors what FedAvgStrategy uses
        # (strategy.estimator.log), populated for the WHOLE pool on
        # every estimate() call in _plugin_configure_fit, regardless of
        # skip. This is what makes estimator_log.csv complete for
        # fedprox_hybrid/scaffold_hybrid/fednova_hybrid.
        "estimator_detail_log": getattr(
            getattr(strategy, '_estimator', None), 'log', []),
        "client_log":     getattr(strategy, 'client_log', []),
        "total_scaffold_steps": getattr(strategy, 'total_scaffold_steps', 0),
        "total_sgd_steps": getattr(strategy, 'total_sgd_steps', 0),
    }


def _create_seed_eval_run(
    seed: int,
    seed_results: dict,
    test_ds,
    val_ds,
    device: str,
    pending_csv: dict = None,
) -> None:
    """
    Creates the wandb evaluation run for one seed.
    Contains: held-out, cost-efficiency, heatmaps, tables.
    """
    # Extract pending_csv embedded in seed_results (FedAvg/hybrid via run_estimator_group)
    if pending_csv is None:
        pending_csv = {}
    for k, v in seed_results.items():
        if k.startswith("__pending_csv_"):
            method_key = k[len("__pending_csv_"):]
            pending_csv[method_key] = v

    eval_run = _wandb_init(f"eval_seed{seed}", seed)

    # -- Held-out evaluation -----------------------------------
    from torch.utils.data import DataLoader
    from model import create_model, set_parameters, evaluate_model
    from dataset import collate_with_mode
    import flwr.common as flc

    holdout_rows = []

    if len(test_ds) > 0:
        loader = DataLoader(
            test_ds, batch_size=64,
            shuffle=False, num_workers=0,
            collate_fn=collate_with_mode,
        )

        print(f"\n  Held-out evaluation (seed={seed}):")
        print(f"{'Method':25s} {'MAE dist':>10s} {'MAE time':>10s} "
              f"{'RMSE dist':>10s} {'R2 dist':>8s}")
        print("-" * 60)

        for name, res in seed_results.items():
            final_params = res.get("final_params")
            if final_params is None:
                continue
            try:
                # Ensure the initial weights are born identical
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                model = create_model()
                weights = flc.parameters_to_ndarrays(final_params)
                set_parameters(model, weights)
                metrics = evaluate_model(
                    model, loader, test_ds, device
                )
                # Real RMSE (computed per sample in evaluate_model)
                rmse_d = metrics.get("rmse_dist", 0.0)
                r2_d   = metrics.get("r2_dist", float("nan"))

                marker = " <-" if name == "hybrid" else ""
                display = config.METHOD_DISPLAY_NAMES.get(name, name)
                print(f"{display:25s} {metrics['mae_dist']:10.1f} "
                      f"{metrics['mae_time']:10.1f} "
                      f"{rmse_d:10.1f} {r2_d:8.3f}" + marker)

                holdout_rows.append({
                    "method":   name,
                    "mae_dist": metrics["mae_dist"],
                    "mae_time": metrics["mae_time"],
                    "seed":     seed,
                })

                # Store held-out metrics in results
                res["held_out_mae_dist"]   = metrics["mae_dist"]
                res["held_out_mae_time"]   = metrics["mae_time"]
                res["held_out_mse_dist"]   = metrics["mse_dist"]
                res["held_out_mse_time"]   = metrics["mse_time"]
                res["held_out_rmse_dist"]  = metrics["rmse_dist"]
                res["held_out_rmse_time"]  = metrics["rmse_time"]
                res["held_out_mape_dist"]  = metrics["mape_dist"]
                res["held_out_mape_time"]  = metrics["mape_time"]
                res["held_out_mape_n_excluded_dist"] = metrics["mape_n_excluded_dist"]
                res["held_out_mape_n_excluded_time"] = metrics["mape_n_excluded_time"]
                res["held_out_smape_dist"] = metrics["smape_dist"]
                res["held_out_smape_time"] = metrics["smape_time"]
                res["held_out_r2_dist"]    = metrics["r2_dist"]
                res["held_out_r2_time"]    = metrics["r2_time"]

                # Update the CSV with the held-out values
                if pending_csv and name in pending_csv:
                    _s = pending_csv[name]["summary"]
                    _s["held_out_mae_dist"]   = metrics["mae_dist"]
                    _s["held_out_mae_time"]   = metrics["mae_time"]
                    _s["held_out_mse_dist"]   = metrics["mse_dist"]
                    _s["held_out_mse_time"]   = metrics["mse_time"]
                    _s["held_out_rmse_dist"]  = metrics["rmse_dist"]
                    _s["held_out_rmse_time"]  = metrics["rmse_time"]
                    _s["held_out_mape_dist"]  = metrics["mape_dist"]
                    _s["held_out_mape_time"]  = metrics["mape_time"]
                    _s["held_out_mape_n_excluded_dist"] = metrics["mape_n_excluded_dist"]
                    _s["held_out_mape_n_excluded_time"] = metrics["mape_n_excluded_time"]
                    _s["held_out_smape_dist"] = metrics["smape_dist"]
                    _s["held_out_smape_time"] = metrics["smape_time"]
                    _s["held_out_r2_dist"]    = metrics["r2_dist"]
                    _s["held_out_r2_time"]    = metrics["r2_time"]


            except Exception as e:
                print(f"{name:25s} error: {e}")

    # -- Save local CSVs (with held_out filled in) -------------
    if pending_csv:
        for name, data in pending_csv.items():
            _save_run_csv(
                name, seed,
                data.get("global_eval_log", []),
                data.get("round_log", []),
                data["summary"],
            )

    # -- Cost-efficiency -----------------------------------------
    print_cost_efficiency(seed_results)
    _save_comparison_csv(seed, seed_results)

    # -- Log to wandb ----------------------------------------------
    if eval_run is not None:
        try:
            import wandb

            # Held-out table
            if holdout_rows:
                ht = wandb.Table(
                    columns=["Method", "MAE dist", "MAE time", "Seed"]
                )
                for r in holdout_rows:
                    ht.add_data(r["method"], r["mae_dist"],
                                r["mae_time"], r["seed"])
                eval_run.log({"held_out_results": ht})

            # Cost-efficiency table
            _log_cost_table_to_wandb(seed_results, eval_run)

            # Heatmaps
            plots_dir = os.path.join(config.RESULTS_DIR, "plots")
            for name in seed_results:
                hm_path = os.path.join(
                    plots_dir,
                    f"heatmap_{name}_seed{seed}.png"
                )
                if os.path.exists(hm_path):
                    eval_run.log({
                        f"heatmap/{name}": wandb.Image(hm_path)
                    })

            eval_run.finish()
        except Exception as e:
            print(f"  [WARNING] wandb eval_seed failed: {e}")


def _create_final_eval_run(
    results_by_seed: dict,
    test_ds_by_seed: dict,
    device: str,
) -> None:
    """
    Creates the final (cross-seed) wandb evaluation run.
    Contains: aggregated tables, statistical tests, plots.
    """
    seeds = list(results_by_seed.keys())
    if not seeds:
        return

    final_run = _wandb_init("eval_final", seeds[0])

    print(f"\n{'='*65}")
    print(f" FINAL EVALUATION -- {len(seeds)} seeds")
    print(f"{'='*65}")

    # -- Aggregate results ---------------------------------------
    aggregated = _aggregate_results(results_by_seed)

    # -- Separate validation + held-out tables -------------------
    print(f"\n Results -- Validation (global evaluation during training):")
    print_aggregated_comparison(aggregated)

    print(f"\n Results -- Held-out (generalization to new clients):")
    _print_held_out_aggregated(results_by_seed)

    # -- Statistical tests ----------------------------------------
    print_statistical_tests(results_by_seed)
    _save_wilcoxon_csv(results_by_seed)

    # -- Aggregated cost-efficiency ---------------------------------
    print_cost_efficiency_aggregated(results_by_seed)
    aggregated2 = _aggregate_results(results_by_seed)
    _save_aggregated_csv(results_by_seed, aggregated2)

    # -- Save JSON --------------------------------------------------
    out = os.path.join(config.RESULTS_DIR,
                       "fedhme_full_results.json")
    results_for_json = {}
    for s, methods in results_by_seed.items():
        results_for_json[s] = {
            m: {k: v for k, v in d.items() if k != "final_params"}
            for m, d in methods.items()
        }
    with open(out, "w") as f:
        json.dump({
            "aggregated":      aggregated,
            "results_by_seed": results_for_json,
            "config": {
                "version":     config.CODE_VERSION,
                "n_clients":   config.N_CLIENTS,
                "n_rounds":    config.N_ROUNDS,
                "grid_size":   config.GRID_SIZE,
                "seeds":       seeds,
            }
        }, f, indent=2, default=str)
    print(f"\n Results in: {out}")

    # -- Final plots -------------------------------------------
    print("\n Generating final plots...")
    _plot_results(results_by_seed, aggregated, config.RESULTS_DIR)

    # -- Log to wandb ----------------------------------------------
    if final_run is not None:
        try:
            import wandb

            # Aggregated held-out table
            _log_held_out_table(results_by_seed, final_run)

            # Final plots
            plots_dir = os.path.join(config.RESULTS_DIR, "plots")
            for fname in ["convergencia_loss.png",
                          "mae_dist_final.png",
                          "divergencia.png",
                          "mae_por_modo.png"]:
                path = os.path.join(plots_dir, fname)
                if os.path.exists(path):
                    final_run.log({
                        f"final/{fname.replace('.png', '')}":
                            wandb.Image(path)
                    })

            final_run.finish()
        except Exception as e:
            print(f"  [WARNING] wandb eval_final failed: {e}")


def _log_cost_table_to_wandb(seed_results: dict, run) -> None:
    """Logs the cost-efficiency table on a wandb run."""
    import wandb
    baseline = seed_results.get("no_estimate", {})
    base_mae  = baseline.get("final_mae_dist", float("nan"))
    base_cost = baseline.get("total_grad_steps", 0)
    if base_cost == 0:
        return

    table = wandb.Table(
        columns=["Method", "MAE dist", "Grad Steps", "Efficiency"]
    )
    for name, res in seed_results.items():
        mae  = res.get("final_mae_dist", float("nan"))
        cost = res.get("total_grad_steps", 0)
        if cost == 0 or mae != mae:
            continue
        eff = (base_mae / mae) / (cost / base_cost)
        display = config.METHOD_DISPLAY_NAMES.get(name, name)
        table.add_data(display, round(mae, 1), cost, round(eff, 3))
    run.log({"cost_efficiency": table})


def _print_held_out_aggregated(results_by_seed: dict) -> None:
    """Prints the aggregated held-out table (mean +/- std)."""
    from collections import defaultdict
    data = defaultdict(lambda: {"dist": [], "time": []})

    for seed, methods in results_by_seed.items():
        for name, res in methods.items():
            d = res.get("held_out_mae_dist")
            t = res.get("held_out_mae_time")
            if d is not None:
                data[name]["dist"].append(d)
            if t is not None:
                data[name]["time"].append(t)

    order = config.ESTIMATOR_NAMES + ["fedprox_pure", "fedprox_hybrid_static_mu", "fedprox_hybrid", "scaffold_pure", "scaffold_hybrid_no_bypass", "scaffold_hybrid", "fednova_pure", "fednova_hybrid", "feddyn_pure", "feddyn_hybrid_static_alpha", "feddyn_hybrid"]
    print(f"{'Method':25s} {'MAE dist (m)':22s} {'MAE time (s)':22s}")
    print("-" * 75)

    for name in order:
        d = data.get(name)
        if not d or not d["dist"]:
            continue
        marker = " <-" if name == "hybrid" else ""
        display = config.METHOD_DISPLAY_NAMES.get(name, name)
        print(f"{display:25s} "
              f"{np.mean(d['dist']):8.1f} +/- {np.std(d['dist']):6.1f}   "
              f"{np.mean(d['time']):8.1f} +/- {np.std(d['time']):6.1f}"
              + marker)


def _log_held_out_table(results_by_seed: dict, run) -> None:
    """Logs the aggregated held-out table to wandb."""
    import wandb
    from collections import defaultdict
    data = defaultdict(lambda: {"dist": [], "time": []})

    for seed, methods in results_by_seed.items():
        for name, res in methods.items():
            d = res.get("held_out_mae_dist")
            t = res.get("held_out_mae_time")
            if d is not None:
                data[name]["dist"].append(d)
                data[name]["time"].append(t or 0)

    table = wandb.Table(
        columns=["Method", "MAE dist (mean)", "MAE dist (std)",
                 "MAE time (mean)", "MAE time (std)", "N seeds"]
    )
    order = config.ESTIMATOR_NAMES + ["fedprox_pure", "fedprox_hybrid_static_mu", "fedprox_hybrid", "scaffold_pure", "scaffold_hybrid_no_bypass", "scaffold_hybrid", "fednova_pure", "fednova_hybrid", "feddyn_pure", "feddyn_hybrid_static_alpha", "feddyn_hybrid"]
    for name in order:
        d = data.get(name)
        if not d or not d["dist"]:
            continue
        table.add_data(
            name,
            round(float(np.mean(d["dist"])), 1),
            round(float(np.std(d["dist"])), 1),
            round(float(np.mean(d["time"])), 1),
            round(float(np.std(d["time"])), 1),
            len(d["dist"]),
        )
    run.log({"held_out_aggregated": table})

    print(f"\n Note: MAE on fully held-out users")
    print(f"       demonstrates generalization to new clients")


if __name__ == "__main__":
    main()
