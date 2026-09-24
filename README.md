# FedHME — Federated Heterogeneity-aware Mobility Estimator

**FedHME** is a scheduling plugin for Federated Learning that estimates
client heterogeneity using spatio-temporal mobility context. It operates
as an orthogonal layer on top of any FL aggregation algorithm — selecting
clients and allocating local epochs without modifying the aggregation rule
or client update.

Implemented as part of a doctoral research on Green AI and Sustainable
Federated Learning for Intelligent Transportation Systems (ITS).

This single codebase runs both supported datasets (GeoLife and MOBIS) —
dataset-specific behavior (loss function, target transform, feature count)
is a runtime branch on `--dataset`, controlled by `config.py`:

| Behavior | GeoLife | MOBIS | Driven by |
|---|---|---|---|
| Target transform | raw (no transform) | `log1p` before Z-score | `config.LOG1P_TARGET_DATASETS` |
| Training loss | MSE | Huber (`config.HUBER_DELTA`) | `config.HUBER_LOSS_DATASETS` |
| Feature count | 9 (`config.N_FEATURES`) | 26 / 27 (`config.MOBIS_N_FEATURES[_PURPOSE]`) | `create_model(dataset_name)` |
| Window source | `geolife_loader.py` | `mobis_loader.py` (region filter, feature engineering) | `--dataset` |

---

## How It Works

At each round, the server estimates a heterogeneity score **ĥ** for each
client candidate using two components:

```
ĥ = α(v, m) · temporal + (1 − α(v, m)) · spatial

α(v, m) = exp(−(v · δ_v + δ_m))
```

- **temporal** — EMA of the client's historical gradient divergence
- **spatial** — smoothed value of the geographic divergence map at the
  client's grid cell
- **α(v, m)** — confidence weight modulated by client velocity `v` (km/h)
  and transport mode `m`; faster clients have sparser geographic coverage
  per cell, reducing spatial reliability

Clients are ranked by **ĥ** within each round (rank normalization). The
bottom 50% are skipped. Participating clients receive local epochs
proportional to their rank-normalized ĥ, within a fixed computational
budget.

The plugin integrates with five aggregation algorithms:

| Algorithm | FedHME intervention |
|---|---|
| FedAvg | Client selection + epochs ∝ ĥ |
| FedProx | Client selection + epochs ∝ ĥ + dynamic µ ∝ ĥ (dedicated channel) |
| SCAFFOLD | Client selection + epochs ∝ ĥ + SGD bypass when ĥ < 0.3 (dedicated channel) |
| FedNova | Client selection + τ_k ∝ ĥ (via epochs); normalisation by a_k preserved — no dedicated channel |
| FedDyn | Client selection + epochs ∝ ĥ + dynamic α ∝ ĥ (dedicated channel) |

For FedProx, SCAFFOLD and FedDyn, the **common channel** (client
selection + epoch allocation via ĥ) and the **dedicated channel**
(dynamic µ / adaptive bypass / dynamic α — the algorithm's own
hyperparameter modulated by ĥ) can be isolated independently — see the
`*_hybrid_static_mu`, `*_hybrid_no_bypass` and `*_hybrid_static_alpha`
variants in [Available Methods](#available-methods).

---

## Requirements

```bash
pip install torch flwr ray[default] scikit-learn numpy matplotlib psutil
# Optional — real-time monitoring
pip install wandb
```

Python 3.10+ recommended. Tested on Ubuntu 24 (WSL2) with CUDA 12.

**GeoLife:** [GPS Trajectories 1.3](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)
— download and extract to `./data/Geolife Trajectories 1.3/`.

**MOBIS:** ETH Zurich / IVT — ~4,600 participants, 6.27M legs, Switzerland,
Sep/2019-Jan/2023. Access requires a data-sharing agreement with
[IVT ETH Zurich](https://ivtmobis.ethz.ch). Expected structure:

```
./data/mobis/tracking/legs.csv
./data/mobis/tracking/activities.csv
./data/mobis/enrichments/covid_externalities.csv
```

Both datasets' cache/raw data live under the same `./data/` folder
(different filenames per dataset, see [Project Structure](#project-structure)).

---

## Quick Start

```bash
# GeoLife — all methods, default seeds [0, 1, 2]
python run_experiment.py --dataset geolife

# MOBIS (ETA task) — all methods, default seeds
python run_experiment.py --dataset mobis_eta

# MOBIS purpose-aware ETA variant
python run_experiment.py --dataset mobis_eta_purpose

# Run only FedAvg baselines with a single seed
python run_experiment.py --dataset geolife --methods no_estimate hybrid --seeds 0

# Run all hybrid methods across all seeds
python run_experiment.py --dataset mobis_eta \
  --methods hybrid fedprox_hybrid scaffold_hybrid fednova_hybrid feddyn_hybrid \
  --seeds 0 1 2
```

`--dataset` defaults to `geolife` if omitted, but always pass it
explicitly to avoid ambiguity when running MOBIS.

---

## Command-Line Arguments

All arguments are optional. Defaults come from `config.py`.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | choice | `geolife` | `geolife`, `mobis_eta` or `mobis_eta_purpose`. |
| `--methods` | list | all | Methods to run. See available methods below. |
| `--seeds` | list of int | `[0, 1, 2]` | Random seeds for reproducibility. |
| `--budget-saving` | float | `0.8` | Fraction of epoch budget saved by the hybrid. `0.0` = full budget (same as baseline); `0.8` = uses 20% of total budget. |
| `--skip-factor` | float | `0.5` | Fraction of client pool skipped per round. `0.5` = bottom 50% by rank-normalised ĥ are skipped. |
| `--velocity-decay` | float | `config.py` (`0.05`) | Overrides `VELOCITY_DECAY` (δ_v) in α(v, m). Applies to either dataset. |
| `--grid-size` | int | `config.py` (`40`) | Overrides the geographic grid resolution. Applies to either dataset (e.g. Zurich G40 vs G10 on MOBIS). |
| `--mobis-window-size` | int | `config.py` (`10`) | Overrides the number of legs per window. MOBIS only. |
| `--mobis-period` | choice | none | Filters MOBIS by collection period (`pre_covid`, `covid`, `post_covid`). MOBIS only. |
| `--mobis-stratify-mode` | flag | off | Reserves a minimum number of participants per predominant transport mode. MOBIS only. |
| `--mobis-min-per-mode` | int | `3` | Minimum participants per mode when `--mobis-stratify-mode` is set. MOBIS only. |
| `--mobis-purpose-filter` | list | none | Filters MOBIS participants by predominant activity purpose. MOBIS only. |
| `--mobis-region` | choice | none | Restricts participants to a named geographic region (currently only `zurich`). MOBIS only. |
| `--mobis-region-frac` | float | `0.50` | Minimum fraction of a participant's legs whose destination falls inside `--mobis-region`'s bounding box. Ignored unless `--mobis-region` is set. MOBIS only. |

### Available Methods

| Method name | Description |
|---|---|
| `no_estimate` | FedAvg — no estimator (baseline) |
| `hybrid` | FedAvg + FedHME plugin |
| `temporal_only` | Ablation — ĥ uses only the temporal component |
| `spatial_only` | Ablation — ĥ uses only the spatial component |
| `velocity_only` | Ablation — α set purely by client velocity |
| `mode_only` | Ablation — α set purely by transport mode |
| `random_skip` | Baseline — random client skip (Power-of-Choice/Oort-style), no ĥ |
| `fedprox_pure` | FedProx — no estimator |
| `fedprox_hybrid_static_mu` | FedProx + common channel (selection/epochs via ĥ), dedicated channel off — µ fixed, same value as `fedprox_pure` |
| `fedprox_hybrid` | FedProx + FedHME plugin (common + dedicated channel — dynamic µ) |
| `scaffold_pure` | SCAFFOLD — no estimator |
| `scaffold_hybrid_no_bypass` | SCAFFOLD + common channel, dedicated channel off — bypass never triggers, same as `scaffold_pure` |
| `scaffold_hybrid` | SCAFFOLD + FedHME plugin (common + dedicated channel — adaptive bypass) |
| `fednova_pure` | FedNova — no estimator |
| `fednova_hybrid` | FedNova + FedHME plugin |
| `feddyn_pure` | FedDyn — no estimator, fixed α |
| `feddyn_hybrid_static_alpha` | FedDyn + common channel, dedicated channel off — α fixed |
| `feddyn_hybrid` | FedDyn + FedHME plugin (common + dedicated channel — dynamic α) |

The `*_hybrid_static_mu` / `*_hybrid_no_bypass` / `*_hybrid_static_alpha`
variants exist to isolate what the dedicated channel contributes beyond
the common channel — the common channel alone is already tested by
`random_skip`, which cannot reach the dedicated channel by design.

### Examples

```bash
# Grid search over budget-saving and skip-factor (single seed for speed)
python run_experiment.py --dataset geolife --methods hybrid --seeds 0 --budget-saving 0.3
python run_experiment.py --dataset geolife --methods hybrid --seeds 0 --budget-saving 0.5 --skip-factor 0.3
python run_experiment.py --dataset geolife --methods hybrid --seeds 0 --budget-saving 0.8 --skip-factor 0.5

# Isolate the dedicated channel's contribution (FedProx, GeoLife)
python run_experiment.py --dataset geolife \
  --methods fedprox_pure fedprox_hybrid_static_mu fedprox_hybrid --seeds 0 1 2 3 4

# Restrict MOBIS to the Zurich sub-region, coarser grid (G=10)
python run_experiment.py --dataset mobis_eta --mobis-region zurich --grid-size 10 --seeds 0

# Final run with optimal configuration (all seeds, all methods)
python run_experiment.py --dataset geolife --budget-saving 0.8 --skip-factor 0.5 --seeds 0 1 2
```

---

## Configuration — `config.py`

Parameters that cannot be set via CLI must be edited in `config.py`.

### Hardware

| Parameter | Default | Description |
|---|---|---|
| `USE_GPU` | `True` | Use CUDA if available. Set `False` to force CPU. |
| `RAY_NUM_CPUS` | `8` | Total CPUs allocated to Ray. |
| `RAY_NUM_GPUS` | `1` | Total GPUs allocated to Ray. |
| `RAY_CPUS_PER_CLIENT` | `4` | CPUs per worker → `floor(RAY_NUM_CPUS / RAY_CPUS_PER_CLIENT)` simultaneous workers. |
| `RAY_GPUS_PER_CLIENT` | `0.5` | GPU fraction per worker. Match to CPUs to avoid idle workers. |

### FL Experiment

| Parameter | Default | Description |
|---|---|---|
| `N_CLIENTS` | `40` | Total client pool size. |
| `N_ROUNDS` | `80` | Number of FL rounds. |
| `MIN_FIT_CLIENTS` | `10` | Minimum clients required to start a round (Flower). |
| `MIN_SKIP_PARTICIPANTS` | `5` | Minimum participants after skip (safety guarantee). |
| `CALIBRATION_ROUNDS` | `5` | Warm-up rounds before δ_m recalibration begins. |
| `LOCAL_EPOCHS_BASE` | `3` | Base local epochs per client. |
| `EXPERIMENT_SEEDS` | `[0, 1, 2]` | Default seeds (overridable via `--seeds`). |

### Model

| Parameter | Default | Description |
|---|---|---|
| `HIDDEN_DIM` | `64` | LSTM hidden dimension. |
| `N_LSTM_LAYERS` | `2` | Number of LSTM layers. |
| `DROPOUT` | `0.4` | Dropout rate. |
| `BATCH_SIZE` | `64` | Training batch size. |
| `LEARNING_RATE` | `0.01` | SGD learning rate (all methods). |
| `SGD_MOMENTUM` | `0.9` | SGD momentum (FedAvg, FedProx, SCAFFOLD). FedNova uses `0.0` by design. |
| `SGD_WEIGHT_DECAY` | `1e-4` | Weight decay (all methods). |
| `HUBER_DELTA` | `1.0` | δ for the Huber training loss — used only for datasets in `HUBER_LOSS_DATASETS`. |
| `LOG1P_TARGET_DATASETS` | `("mobis_eta", "mobis_eta_purpose")` | Datasets whose target gets `log1p`-transformed before Z-score (`dataset.py`). GeoLife: no transform. |
| `HUBER_LOSS_DATASETS` | `("mobis_eta", "mobis_eta_purpose")` | Datasets trained with Huber loss instead of MSE (`model.py`, `fl_algorithms.py`). |

### FedHME Estimator

| Parameter | Default | Description |
|---|---|---|
| `GRID_SIZE` (G) | `40` | Geographic grid resolution (G×G cells). Overridable via `--grid-size`. |
| `GEO_MAP_DECAY` (β) | `0.3` | EMA weight for the spatial divergence map. Lower = longer memory. |
| `CLIENT_EMA_ALPHA` | `0.3` | EMA weight for the client temporal history. |
| `VELOCITY_DECAY` (δ_v) | `0.05` | Velocity coefficient in α(v, m). Faster clients → lower α → more spatial weight. Overridable via `--velocity-decay`. |
| `MODE_DECAY` | `1.0` | Scale factor for δ_m (neutral at 1.0 — reserved for future calibration). |
| `SKIP_FACTOR` | `0.5` | Skip bottom fraction of pool by rank-normalised ĥ (overridable via CLI). |
| `MAX_CONSECUTIVE_SKIPS` | `3` | Maximum consecutive skips per client before forced participation. |
| `BUDGET_SAVING` | `0.8` | Epoch budget saving fraction (overridable via CLI). |
| `MOBIS_WINDOW_SIZE` | `10` | Number of legs per window. MOBIS only. Overridable via `--mobis-window-size`. |

### Algorithm-specific

| Parameter | Default | Description |
|---|---|---|
| `FEDPROX_MU` | `0.01` | Fixed µ for FedProx pure. |
| `FEDPROX_MU_MIN` (µ_min) | `0.001` | Minimum µ for FedProx hybrid (low heterogeneity). |
| `FEDPROX_MU_MAX` (µ_max) | `0.1` | Maximum µ for FedProx hybrid (high heterogeneity). |
| `SCAFFOLD_BYPASS_THRESHOLD` | `0.3` | ĥ_norm below which SCAFFOLD bypasses control variates and uses plain SGD. |
| `FEDDYN_ALPHA` | `0.01` | Fixed α for FedDyn pure / static-alpha hybrid. |
| `FEDDYN_ALPHA_MIN` / `FEDDYN_ALPHA_MAX` | `0.001` / `0.1` | Dynamic α range for FedDyn hybrid (dedicated channel). |

### Data

| Parameter | Default | Description |
|---|---|---|
| `GEOLIFE_ROOT` | `./data/Geolife Trajectories 1.3` | Path to the GeoLife dataset root. |
| `MOBIS_ROOT` | `./data/mobis` | Path to the MOBIS dataset root. |
| `MOBIS_MAX_PARTICIPANTS` | `4500` | Cap on participants loaded as FL clients. |
| `SUPPORTED_DATASETS` | `["geolife", "mobis_eta", "mobis_eta_purpose"]` | Datasets `run_experiment.py` accepts for `--dataset`. |
| `RESULTS_DIR` | `./results_{dataset}` | Output directory for CSVs and plots — set per-dataset at runtime, to avoid overlap when running both back to back. |
| `GDRIVE_RESULTS_DIR` | `""` | Optional Google Drive path for automatic backup (WSL2 Drive for Desktop). Leave empty to disable. |

### Monitoring

| Parameter | Default | Description |
|---|---|---|
| `WANDB_ENABLED` | `False` | Enable Weights & Biases logging. Run `wandb login` first. |
| `WANDB_PROJECT` | `fedhme-experiment` | W&B project name. |

---

## Output Files

Results are saved to `results_{dataset}/` after each run, independent of W&B.

```
results_{dataset}/
  csv/
    {method}_seed{N}_bs{B}_sf{F}_rounds.csv    ← per-round metrics
    {method}_seed{N}_bs{B}_sf{F}_summary.csv   ← final MAE + held-out
  plots/
    dataset/   ← GPS scatter, samples per client, mode distribution
    {method}/  ← MAE curves, skip rate, epoch allocation
```

### `rounds.csv` columns

```
round, train_loss, divergence, n_clients,
round_grad_steps, total_grad_steps,
n_skipped, n_participating, skip_rate,
h_mean, h_std, epochs_mean,
temporal_mean, spatial_mean, alpha_mean, delta_m_mean,
h_temporal_contrib_mean, h_spatial_contrib_mean,
geo_map_val_mean, client_ema_val_mean, budget_saving,
mae_dist_global, mae_time_global, val_loss_global
```

### `summary.csv` columns

```
method, seed, mae_dist_final, mae_time_final,
held_out_mae_dist, held_out_mae_time,
held_out_mse_dist, held_out_mse_time,
held_out_rmse_dist, held_out_rmse_time,
held_out_mape_dist, held_out_mape_time,
held_out_mape_n_excluded_dist, held_out_mape_n_excluded_time,
held_out_smape_dist, held_out_smape_time,
held_out_r2_dist, held_out_r2_time,
budget_saving, skip_factor, total_grad_steps
```

---

## Project Structure

```
.
├── run_experiment.py      # Entry point — CLI, orchestration, CSV output
├── fl_algorithms.py       # FedAvg, FedProx, SCAFFOLD, FedNova, FedDyn (pure + hybrid variants)
├── estimator.py           # HybridEstimator — ĥ calculation, skip, epoch allocation
├── dataset.py             # TransitionPredictionDataset — log1p target transform (MOBIS only) + inverse
├── model.py                # TransitionLSTM + Huber/MSE loss (compute_task_loss) + training/evaluation
├── geolife_loader.py      # GeoLife parsing, sample extraction, profile building
├── mobis_loader.py        # MOBIS parsing, region filter, feature engineering
├── geo_utils.py           # Haversine, bearing, normalise_position
├── fl_utils.py            # Shared utilities (seed, serialisation, Ray cache)
├── fl_client_strategy.py  # Compatibility re-exports (legacy)
├── config.py              # All hyperparameters and paths, both datasets
├── resource_monitor.py    # RAM/GPU/time instrumentation
└── data/
    ├── Geolife Trajectories 1.3/   ← GeoLife dataset (not included, see Requirements)
    └── mobis/                       ← MOBIS dataset (not included, see Requirements)
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{vieira2026fedhme,
  title   = {FedHME: A Heterogeneity-aware Mobility Estimator Plugin
             for Sustainable Federated Learning in ITS},
  author  = {Vieira, Flávio and ...},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2026},
  note    = {Full citation details to be added upon publication}
}
```

A prior conference version was presented at IEEE VTC Fall 2026.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
