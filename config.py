"""
config.py — FedHME experiment configuration
-------------------------------------------
Hyperparameters, paths, and flags for the FedHME conference experiment.
All parameters that cannot be set via CLI must be edited here.
"""

# -- Code version (shown in the wandb run name) ----------------
CODE_VERSION = "v31_unified_geolife_mobis"

# -- Datasets supported by this (unified) copy of the code ----------------
# Unlike the split geolife/ and mobis/ folders this was merged from, this
# copy runs either dataset from the same code: model.py/dataset.py branch
# on dataset_name (see LOG1P_TARGET_DATASETS/HUBER_LOSS_DATASETS below and
# create_model()'s dataset_name argument) instead of hardcoding one
# dataset's settings. run_experiment.py still refuses to start with a
# --dataset not listed here, as a plain input-validation check now, not a
# folder-identity gate.
SUPPORTED_DATASETS = ["geolife", "mobis_eta", "mobis_eta_purpose"]

# -- Google Drive (folder mounted via Drive for Desktop) --------------------
# WSL2: the Windows path G:\My Drive\... shows up as /mnt/g/My Drive/...
# Leave "" to disable copying to Drive.
# NOTE: fill in the correct path to your Google Drive folder!
#GDRIVE_RESULTS_DIR = ""  # e.g. "/mnt/g/My Drive/FL_Experiments"
GDRIVE_RESULTS_DIR = ""  # set to your Google Drive path to enable auto-backup



# -- FL experiment ----------------------------------------------
# -- Ray resources per worker (anti-OOM and GPU control) ----------------------
#
# Simultaneous-worker formula:
#   workers = min(
#       floor(RAY_NUM_CPUS  / RAY_CPUS_PER_CLIENT),
#       floor(RAY_NUM_GPUS  / RAY_GPUS_PER_CLIENT),  <- only if USE_GPU=True
#   )
#
# Reference for a machine with 16 CPUs / 15.5 GB RAM / 1 GPU with 4 GB VRAM:
#
#   CPU (RAM ~0.8 GB/worker + 3.7 GB driver process):
#     RAY_CPUS_PER_CLIENT = 2  -> 4 workers  (~7 GB RAM)   <- with GPU
#     RAY_CPUS_PER_CLIENT = 4  -> 2 workers  (~5 GB RAM)   <- conservative
#
#   GPU (VRAM ~205 MB/worker):
#     RAY_GPUS_PER_CLIENT = 0.25 -> 4 workers  (~820 MB VRAM) <- recommended
#     RAY_GPUS_PER_CLIENT = 0.50 -> 2 workers  (~410 MB VRAM) <- conservative
#     RAY_GPUS_PER_CLIENT = 1.0  -> 1 worker   (~205 MB VRAM) <- safe
#
# Tip: keep workers_cpu == workers_gpu to avoid idle workers.
#   E.g.: RAY_CPUS_PER_CLIENT=2 + RAY_GPUS_PER_CLIENT=0.25 -> 4 workers each.

USE_GPU             = True   # False -> forces CPU even if a GPU is available

RAY_NUM_CPUS        = 8      # total CPUs given to Ray
RAY_NUM_GPUS        = 1      # total GPUs given to Ray (this machine has 1)

RAY_CPUS_PER_CLIENT = 4      # CPUs per worker  -> 2 simultaneous workers
RAY_GPUS_PER_CLIENT = 0.5    # GPU fraction per worker -> 2 simultaneous workers

N_CLIENTS          = 40      # uses 40 of the 69 labelled GeoLife users
N_ROUNDS           = 80
CALIBRATION_ROUNDS = 5
MIN_FIT_CLIENTS    = 10
MIN_EVAL_CLIENTS   = 5
FRACTION_FIT       = 0.25    # adjusted: 10/40 = 0.25

# -- Seeds for multiple runs (mean +/- std) ---------------------
EXPERIMENT_SEEDS   = [0, 1, 2]   # minimum for the conference paper
ACTUAL_SEED = 0


# dedicated validation clients (never participate in training)
# These clients are evaluated every round but never participate in FL.
# Their val_loss is stable because it uses the full trajectory, not a temporal split.
N_VAL_CLIENTS      = 5           # kept separate from the training pool

# -- Prediction task ----------------------------------------------
WINDOW_SIZE        = 10
N_FEATURES         = 9
N_OUTPUTS          = 2

# Quality filters
MIN_DIST_THRESHOLD = 10
MAX_DIST_THRESHOLD = 5000
MAX_TIME_THRESHOLD = 1800

# -- LSTM model ---------------------------------------------------
HIDDEN_DIM         = 64
N_LSTM_LAYERS      = 2
DROPOUT            = 0.4      # raised from 0.2 (overfitting observed)
BATCH_SIZE         = 64       # raised from 32
LOCAL_EPOCHS_BASE  = 3
VAL_EVERY          = 5   # unified validation every N rounds

# ── Unified optimizer: SGD + Momentum for all 8 methods ──────────────────────
# SGD aligns with original FL papers (FedProx, SCAFFOLD, FedNova).
# FedNova uses momentum=0.0 by design (see FedNovaStrategy).
# Reference: Li et al. MLSys 2020; Karimireddy et al. ICML 2020; Wang et al. NeurIPS 2020.
LEARNING_RATE      = 0.01    # SGD requires higher LR than Adam
SGD_MOMENTUM       = 0.9     # momentum for all methods except FedNova
SGD_WEIGHT_DECAY   = 1e-4

# Huber loss instead of MSE. delta=1.0 is the PyTorch/literature default —
# pred/target are already in log1p+Z-score space (see dataset.py), so
# delta=1 corresponds to "1 standard deviation of the transformed target".
# Motivation: the target's heavy-tailed distribution (MOBIS: median 2.6km,
# p95 12.5km, cap 200km) made MSE (squared error) concentrate the gradient
# on a minority of tail outliers — consistent with a low average grad_norm
# but a grad_clip_rate of 2.1% (a signature of an asymmetric gradient, not
# a uniformly small one).
HUBER_DELTA = 1.0

# Datasets that use log1p on the target (dataset.py) / Huber loss instead
# of MSE (model.py, fl_algorithms.py). Currently the same set for both --
# the two are orthogonal switches (target transform vs. loss function)
# that happen to have been tuned together as one MOBIS-specific package
# ("v12") -- kept as two separate constants rather than one flag so a
# future change to only one of them doesn't need a new constant name.
LOG1P_TARGET_DATASETS = ("mobis_eta", "mobis_eta_purpose")
HUBER_LOSS_DATASETS   = ("mobis_eta", "mobis_eta_purpose")

# -- Weighted loss per transport mode ----------------------------
# Only the hybrid uses weighted loss -- baselines use each dataset's
# standard loss (MSE for GeoLife, Huber for MOBIS -- see
# HUBER_LOSS_DATASETS above).
# weight(mode) = 1 / sqrt(freq(mode)), normalized to sum to N_modes.
# Balances learning between dominant and rare transitions.
WEIGHTED_LOSS_ENABLED = False  # False = standard loss for everyone

# -- FedHME hybrid estimator ------------------------------------
GRID_SIZE             = 40       # 40x40 grid ~= 1km^2/cell over Beijing
GRID_SIZES_COMPARE    = [40, 20, 10]  # map-resolution comparison
HISTORY_WINDOW        = 5
VELOCITY_DECAY        = 0.05
MODE_DECAY            = 1.0
# Periodic delta_m recalibration: every N rounds after the initial
# calibration, checks whether new modes appeared and recalculates delta_m.
# -- Skip factor: client skipped if rank-normalised h_hat < SKIP_FACTOR -----
# Relative threshold — scale-invariant via rank normalisation.
SKIP_FACTOR           = 0.5   # skip if h_hat < 50% of the round average
SKIP_FLOOR            = 0.0   # obsolete — kept for compatibility
MAX_CONSECUTIVE_SKIPS = 3     # safeguard: avoids indefinitely skipping a client
MIN_SKIP_PARTICIPANTS = 5     # Design B: minimum participants per round

# -- random_skip: participant target per round (m(r)) ------------------------
# Exploratory baseline — see BaselineRandomSkip.get_target_n_participating in
# estimator.py. m(r) is a FIXED/external target (following the
# Power-of-Choice/Oort pattern), it does not emerge from independent
# per-client decisions.
#
# m(r) must NEVER be derived from h_hat or from the hybrid — dividing
# E_total by an m(r) copied from the hybrid ("paired") would indirectly
# leak an h_hat-based measurement into random_skip's epoch accounting.
#   Default (RANDOM_SKIP_TARGET_N=None): m(r) = round(n_valid(r) * (1 -
#     SKIP_FACTOR)), recomputed each round purely from the global cutoff
#     (SKIP_FACTOR) and the REAL pool size sampled that round
#     (n_valid — a fact of the simulation/dataset, not an h_hat
#     measurement; hybrid/no_estimate/random_skip always see the same
#     n_valid per round). Adapts on its own to any dataset (GeoLife has a
#     variable pool, ~11-18; MOBIS has a fixed pool of ~40) without any
#     calibration needed.
#   RANDOM_SKIP_TARGET_N = <int>: manual override, a fixed constant —
#     exploratory/debugging use, ignores the round's real pool size.
#   "paired": kept only for completeness/historical comparison (see
#     get_target_n_participating) — uses the hybrid's real n_participating.
#     NOT a fair comparison for the reason above; do not use it to draw
#     conclusions.
RANDOM_SKIP_MODE = "unpaired"          # "unpaired" | "paired"
RANDOM_SKIP_TARGET_N = None            # None = recomputed per round (recommended); int = fixed manual constant
RANDOM_SKIP_PAIRED_HYBRID_DIR = None   # required in "paired" mode, e.g. ".../mobis_oficial_v2/results_20260822/csv"

# -- EMA parameters -----------------------------------------------------------
GEO_MAP_DECAY      = 0.3   # weight of the new observation in the map (EMA)
CLIENT_EMA_ALPHA   = 0.3   # weight of the new divergence in the client's EMA

# -- Budget saving ---------------------------------------------------------------
# 0.0 = full budget (same as no_estimate); 0.8 = uses 20% of total budget.
# Confirmed optimal by grid search (BS=0.80, SF=0.50).
BUDGET_SAVING      = 0.8

# -- GeoLife data --------------------------------------------------
# Download: https://www.microsoft.com/en-us/research/publication/
#           geolife-gps-trajectory-dataset-user-guide/
#
# The zip extracts to: "Geolife Trajectories 1.3/Data/000..181/"
#
# Option A (recommended) — point GEOLIFE_ROOT at the folder
#   that contains "Data/":
#   GEOLIFE_ROOT = "./data/Geolife Trajectories 1.3"
#
# Option B — move Data/ into ./data/geolife/:
#   mv "Geolife Trajectories 1.3/Data" ./data/geolife/
#   GEOLIFE_ROOT = "./data/geolife"
#
# The code detects both layouts automatically.
GEOLIFE_ROOT   = "./data/Geolife Trajectories 1.3"
SAMPLES_CACHE  = "./data/geolife_samples_v10_seed{seed}.pkl"
PROFILES_CACHE = "./data/geolife_profiles_v10_seed{seed}.pkl"

BEIJING_BBOX = {
    "lat_min": 39.75, "lat_max": 40.10,
    "lon_min": 116.10, "lon_max": 116.65,
}

TRANSPORT_MODES = ["walk", "bike", "bus", "car", "subway", "train",
                   "moto", "moto_scooter", "scooter"]
MODE2IDX        = {m: i for i, m in enumerate(TRANSPORT_MODES)}


# -- Weights & Biases (wandb) --------------------------------------------------
# Real-time online monitoring of the experiment.
# View results at: https://wandb.ai
#
# To use:
#   1. pip install wandb
#   2. wandb login   (only the first time — opens a browser to authenticate)
#   3. Run normally: python run_experiment.py
#
# To disable without removing the code:
#   WANDB_ENABLED = False
#   or set the environment variable: WANDB_MODE=disabled
#
# WANDB_PROJECT : project name on the dashboard (groups all runs)
# WANDB_ENTITY  : your wandb user or organization (None = use the account default)
# WANDB_TAGS    : list of tags to filter runs on the dashboard
WANDB_ENABLED = False
WANDB_PROJECT = "fedhme-experiment"
WANDB_ENTITY  = None          # e.g. "your-team" — None uses the account default
WANDB_TAGS    = ["geolife", "federated-learning", "heterogeneity"]
# -- Results ------------------------------------------------------
RESULTS_DIR = "./results"

MOBILITY_GROUPS = {
    "estatico": {"velocity_max":  5,  "fraction": 0.20},
    "baixa":    {"velocity_max": 20,  "fraction": 0.35},
    "media":    {"velocity_max": 60,  "fraction": 0.30},
    "alta":     {"velocity_max": 200, "fraction": 0.15},
}

# -- Execution control ---------------------------------------------------------

# -- Estimator as an orthogonal plugin --------------------------
# The estimator stops competing with aggregation algorithms and instead
# complements them. Each algorithm is run TWICE: pure and with the
# estimator plugin.
#
# Group 1: FedAvg + hybrid estimator (the proposal's main reference)
# Group 2: Pure algorithms vs. Algorithms + hybrid

# -- Per-component ablation set -----------------------------------
# temporal_only/spatial_only/velocity_only isolate one component of the
# estimator's alpha blend each; mode_only does the same for the per-mode
# delta_m term (see BaselineModal in estimator.py for the exact formulas).
ESTIMATOR_NAMES = [
    "hybrid",         # FedAvg + FedHME
    "no_estimate",    # plain FedAvg (baseline)
    "temporal_only",  # B1 — temporal component only (EMA of divergence)
    "spatial_only",   # B2 — spatial component only (geographic map / cell)
    "velocity_only",  # B3 — temporal/spatial blend weighted by velocity only
    "mode_only",      # B4 — per-mode calibrated delta_m only
    # Exploratory baseline, outside the estimator's official ablation set:
    # uniform random skip + uniform epoch allocation.
    "random_skip",
]

# -- Naming for the paper / reports / plots --------------------------------------
# Internal keys kept for code and wandb compatibility.
# Display names used in tables, plots, and result logs.
METHOD_DISPLAY_NAMES = {
    "no_estimate":      "FedAvg",
    "hybrid":           "FedAvg + FedHME",
    "temporal_only":    "FedAvg + FedHME (temporal-only)",
    "spatial_only":     "FedAvg + FedHME (spatial-only)",
    "velocity_only":    "FedAvg + FedHME (velocity-only)",
    "mode_only":        "FedAvg + FedHME (mode-only)",
    "random_skip":      "FedAvg + random skip (exploratory)",
    "fedprox_pure":               "FedProx",
    "fedprox_hybrid_static_mu":   "FedProx + FedHME (fixed mu)",
    "fedprox_hybrid":             "FedProx + FedHME",
    "scaffold_pure":              "SCAFFOLD",
    "scaffold_hybrid_no_bypass":  "SCAFFOLD + FedHME (no bypass)",
    "scaffold_hybrid":            "SCAFFOLD + FedHME",
    "fednova_pure":     "FedNova",
    "fednova_hybrid":   "FedNova + FedHME",
    "feddyn_pure":               "FedDyn",
    "feddyn_hybrid_static_alpha": "FedDyn + FedHME (fixed alpha)",
    "feddyn_hybrid":              "FedDyn + FedHME",
    # Adam vs. SGD comparison variants (optional, if there's room in the paper)
    "hybrid_adam":      "FedAvg + FedHME (Adam)",
    "hybrid_sgd":       "FedAvg + FedHME (SGD)",
}

# Group 2: pure algorithms + algorithms with the plugin
# Each is run as "{algo}_pure" and "{algo}_hybrid"

# -- FedProx ------------------------------------------------------
FEDPROX_MU             = 0.01    # fixed mu (pure version)
FEDPROX_MU_MIN         = 0.001   # minimum mu (plugin — stable region)
FEDPROX_MU_MAX         = 0.1     # maximum mu (plugin — high heterogeneity)

# -- SCAFFOLD -------------------------------------------------
# h_hat threshold below which SCAFFOLD bypasses
# (uses plain SGD instead of the control-variate algebra)
SCAFFOLD_BYPASS_THRESHOLD = 0.3  # h_hat < 0.3 -> stable region -> bypass

# -- FedDyn (Acar et al., ICLR 2021, arXiv:2111.04263) ----------
# Fixed alpha (pure/common-channel-only version) and a dynamic range
# (dedicated channel, alpha = FEDDYN_ALPHA_MIN +
# h_hat*(FEDDYN_ALPHA_MAX-FEDDYN_ALPHA_MIN), the same pattern as FedProx's
# dynamic mu) — the range is copied from FedProx since it plays the same
# role (a proximal-term-like coefficient); NOT calibrated, same caveat
# already noted for the original FedProx range.
FEDDYN_ALPHA           = 0.01    # fixed alpha (pure/common-channel-only version)
FEDDYN_ALPHA_MIN        = 0.001   # minimum alpha (dedicated channel — stable region)
FEDDYN_ALPHA_MAX        = 0.1     # maximum alpha (dedicated channel — high heterogeneity)

# -- FedNova (Wang et al., NeurIPS 2020) ---------------------------------------
# FedNova uses momentum=0.0 (vanilla SGD) — momentum breaks tau_k normalisation.
# Aliases kept for code compatibility.
FEDNOVA_MOMENTUM     = SGD_MOMENTUM      # alias -> SGD_MOMENTUM
FEDNOVA_WEIGHT_DECAY = SGD_WEIGHT_DECAY  # alias -> SGD_WEIGHT_DECAY

# -- MOBIS / MOBIS-Covid ------------------------------------------
# Dataset: ~4,598 participants, 6.27M legs, Switzerland, Sep/2019-Jan/2023
# Access: requires an agreement with IVT ETH Zurich (https://ivtmobis.ethz.ch)
# Expected structure:
#   MOBIS_ROOT/tracking/legs.csv
#   MOBIS_ROOT/tracking/activities.csv
#   MOBIS_ROOT/enrichments/covid_externalities.csv
MOBIS_ROOT = "./data/mobis"
MOBIS_BBOX = {
    # Swiss territory + margin — covers all participants in the dataset
    "lat_min": 45.80, "lat_max": 47.80,
    "lon_min":  5.90, "lon_max": 10.65,
}
MOBIS_MAX_PARTICIPANTS = 4500
MOBIS_WINDOW_SIZE      = WINDOW_SIZE   # window over legs (= 10 by default)
MOBIS_SAMPLES_CACHE    = "./data/mobis_samples_v1_{task}_seed{seed}.pkl"
MOBIS_N_FEATURES       = 26  # +weekday, +gap_norm, +same_trip_prev, +trip_pos_norm,
                              # +mode_run_len_norm, +mode_dist_accum_norm, +mode_dur_accum_norm,
                              # mode 1-hot (+N_MODES-1), hour/weekday sin+cos (+2 each),
                              # gap measured from the end of the previous leg (not its start) + log1p
MOBIS_N_FEATURES_PURPOSE = 27  # MOBIS_N_FEATURES + purpose
