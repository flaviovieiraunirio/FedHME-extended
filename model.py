"""
model.py
--------
LSTM for transport-mode transition point prediction.

Input : sequence of k GPS feature vectors
           shape (batch, WINDOW_SIZE, N_FEATURES)
Output   : [normalized_distance, normalized_time]
           shape (batch, 2)

LSTM is used because:
  1. Temporal dependency is central — deceleration patterns predict mode transitions
  2. Lightweight enough for IoT devices (hidden_dim=64)
  3. Well-documented structured sparsity in LSTMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

import config


class TransitionLSTM(nn.Module):
    """
    LSTM with a simple temporal attention layer.
    Attention weights recent points in the window more heavily --
    which makes sense for predicting an imminent transition (behavior
    in the last few seconds is more informative than the start of the
    window).
    """

    def __init__(
        self,
        n_features:  int   = config.N_FEATURES,
        hidden_dim:  int   = config.HIDDEN_DIM,
        n_layers:    int   = config.N_LSTM_LAYERS,
        n_outputs:   int   = config.N_OUTPUTS,
        dropout:     float = config.DROPOUT,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size   = n_features,
            hidden_size  = hidden_dim,
            num_layers   = n_layers,
            batch_first  = True,
            dropout      = dropout if n_layers > 1 else 0.0,
        )

        # Temporal attention -- plain or with context (config.ATTENTION_CONTEXT)
        # Plain:     Linear(H->1) -- ~64 parameters
        # Context:   Linear(H->H//2) -> Tanh -> Linear(H//2->1) -- ~2K parameters (+6%)
        if getattr(config, 'ATTENTION_CONTEXT', False):
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.attention = nn.Linear(hidden_dim, 1)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features)
        returns: (batch, n_outputs)
        """
        # lstm_out: (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)

        # Attention: weight per time step
        attn_scores = self.attention(lstm_out)   # (batch, seq, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)  # (batch, H)

        return self.head(context)


# -- Flower interface -------------------------------------------

def create_model(dataset_name: str = None) -> nn.Module:
    # An explicit dataset_name takes priority over config.CURRENT_DATASET.
    # config.CURRENT_DATASET is a dynamic attribute (never declared in
    # config.py, only set at runtime by main()) -- it does not survive
    # across the Ray worker process boundary (each worker does a fresh
    # `import config`, without ever running main()). When create_model()
    # is called inside an FL client (which runs in a worker),
    # getattr(config, "CURRENT_DATASET", "") always comes back "" there,
    # even with --dataset mobis_eta, and the model would be built with
    # GeoLife's N_FEATURES (9) instead of MOBIS_N_FEATURES (13) -- a size
    # mismatch when loading the global model's parameters. Calls made in
    # the driver process (e.g. _initial_params) keep working via
    # config.CURRENT_DATASET, without needing to pass the argument.
    _dataset = dataset_name if dataset_name is not None else getattr(config, "CURRENT_DATASET", "")
    if _dataset == "mobis_eta_purpose":
        # ETA regression with purpose as an extra feature
        n_feat = getattr(config, "MOBIS_N_FEATURES_PURPOSE", 14)
        return TransitionLSTM(n_outputs=config.N_OUTPUTS, n_features=n_feat)
    if _dataset == "mobis_eta":
        # Explicit branch -- this used to silently fall through to the
        # default (config.N_FEATURES, GeoLife's value) because both
        # happened to be 9. That stopped being a coincidence once
        # MOBIS_N_FEATURES became 13: without this branch, the model
        # would be instantiated with the wrong input_size (9) while the
        # DataLoader delivered 13-column tensors, breaking in forward().
        n_feat = getattr(config, "MOBIS_N_FEATURES", 13)
        return TransitionLSTM(n_outputs=config.N_OUTPUTS, n_features=n_feat)
    return TransitionLSTM()


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [v.cpu().numpy()
            for v in model.state_dict().values()]


def set_parameters(
    model:      nn.Module,
    parameters: list[np.ndarray],
) -> None:
    state_dict = dict(zip(
        model.state_dict().keys(),
        [torch.tensor(p) for p in parameters],
    ))
    model.load_state_dict(state_dict, strict=True)


def _huber_loss(
    pred:      torch.Tensor,
    target:    torch.Tensor,
    delta:     float = 1.0,
    reduction: str   = "mean",
) -> torch.Tensor:
    """
    Manual Huber loss implementation (does not depend on F.huber_loss
    being available in the installed torch version).

    |error| <= delta: quadratic (0.5*error^2), same as MSE -- good for
                       fine adjustment near the target.
    |error| >  delta: linear (delta*(|error|-0.5*delta)) -- grows only
                       linearly, so it doesn't let tail outliers
                       dominate the gradient the way squared error would.

    delta=1.0 (PyTorch/literature default) because pred/target here are
    already in log1p + Z-score space -- delta=1 means "one standard
    deviation of the transformed target", a reasonable scale without
    extra calibration.
    """
    diff      = pred - target
    abs_diff  = diff.abs()
    quadratic = torch.clamp(abs_diff, max=delta)
    linear    = abs_diff - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    raise ValueError(f"unknown reduction: {reduction}")


def compute_task_loss(
    preds:        torch.Tensor,
    targets:      torch.Tensor,
    dataset_name: str = None,
) -> torch.Tensor:
    """
    Per-dataset task loss: Huber for datasets in
    config.HUBER_LOSS_DATASETS (MOBIS), plain MSE otherwise (GeoLife).

    Single place for this choice so fl_algorithms.py's five client
    classes (FedAvg via train_local() below, FedProx/SCAFFOLD/FedDyn/
    FedNova via their own inline training loops) don't each repeat the
    same dataset_name check.
    """
    if dataset_name in getattr(config, "HUBER_LOSS_DATASETS", ()):
        return _huber_loss(preds, targets, delta=config.HUBER_DELTA)
    return F.mse_loss(preds, targets)


# -- Training and evaluation --------------------------------------

def train_local(
    model:    nn.Module,
    loader:   DataLoader,
    n_epochs: int,
    lr:       float = config.LEARNING_RATE,
    device:   str   = "cpu",
    mode_weights: dict = None,
    dataset_name: str  = None,
) -> dict:
    """
    Local training.

    If mode_weights is provided (hybrid only), applies weighted loss per
    mode: weight(mode) = 1/sqrt(freq(mode)). Baselines receive
    mode_weights=None -> standard loss. The loss itself (Huber vs. MSE)
    depends on dataset_name -- see compute_task_loss().
    """
    model.to(device)
    model.train()
    use_huber = dataset_name in getattr(config, "HUBER_LOSS_DATASETS", ())

    # Unified SGD + momentum for all methods.
    # SGD aligns with FL literature (FedProx, SCAFFOLD, FedNova)
    # and avoids client drift amplified by Adam's adaptive momentum.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr           = lr,
        momentum     = config.SGD_MOMENTUM,
        weight_decay = config.SGD_WEIGHT_DECAY,
    )

    total_loss, total_samples, gradient_steps = 0.0, 0, 0

    # Per-batch gradient-norm tracking (diagnostic)
    grad_norms_pre_clip  = []   # L2 norm before clip_grad_norm_
    grad_norms_post_clip = []   # L2 norm after clip_grad_norm_
    n_clipped_batches    = 0    # batches where the clip was active

    for _ in range(n_epochs):
        for batch in loader:
            features = batch[0].to(device)  # (B, W, F)
            targets  = batch[1].to(device)  # (B, 2)

            optimizer.zero_grad()
            preds = model(features)

            # weighted loss per transport mode (Huber or MSE, see use_huber)
            if mode_weights is not None and len(batch) > 4:
                modes = batch[4]  # list of strings
                weights = torch.tensor(
                    [mode_weights.get(m, 1.0) for m in modes],
                    dtype=torch.float32, device=device,
                )  # (B,)
                # per-sample loss, then weighted
                if use_huber:
                    per_sample_loss = _huber_loss(
                        preds, targets, delta=config.HUBER_DELTA, reduction="none"
                    ).mean(dim=1)  # (B,)
                else:
                    per_sample_loss = ((preds - targets) ** 2).mean(dim=1)  # (B,)
                loss = (per_sample_loss * weights).mean()
            else:
                loss = compute_task_loss(preds, targets, dataset_name)

            loss.backward()

            # -- Collect PRE-clip gradient norm (diagnostic) --
            # Global L2 norm across all parameters with a gradient.
            # Reflects the real gradient magnitude before any clipping.
            pre_clip_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float('inf')  # no clip -- measurement only
                ).item()
            )
            grad_norms_pre_clip.append(pre_clip_norm)

            # -- Effective clip (original behavior preserved) --
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # -- Collect POST-clip gradient norm --
            post_clip_norm = float(np.sqrt(sum(
                p.grad.detach().norm().item() ** 2
                for p in model.parameters()
                if p.grad is not None
            )))
            grad_norms_post_clip.append(post_clip_norm)

            # Count batches where clipping was active
            if pre_clip_norm > 1.0:
                n_clipped_batches += 1

            optimizer.step()

            total_loss    += loss.item() * len(targets)
            total_samples += len(targets)
            gradient_steps += 1

    # Clean up temporary optimizer and move the model back to CPU
    del optimizer
    model.cpu()

    return {
        "loss":                total_loss / max(total_samples, 1),
        "samples":             total_samples,
        "gradient_steps":      gradient_steps,
        # Global gradient stats per client
        # Mean and std over all batches of local training
        "grad_norm_pre_mean":  float(np.mean(grad_norms_pre_clip))  if grad_norms_pre_clip  else 0.0,
        "grad_norm_pre_std":   float(np.std(grad_norms_pre_clip))   if grad_norms_pre_clip  else 0.0,
        "grad_norm_post_mean": float(np.mean(grad_norms_post_clip)) if grad_norms_post_clip else 0.0,
        "grad_norm_post_std":  float(np.std(grad_norms_post_clip))  if grad_norms_post_clip else 0.0,
        "grad_clip_rate":      n_clipped_batches / max(gradient_steps, 1),  # fraction of clipped batches
    }


def evaluate_model(
    model:   nn.Module,
    loader:  DataLoader,
    dataset: object,        # TransitionPredictionDataset
    device:  str = "cpu",
) -> dict:
    """
    Evaluates the model, reporting MAE in real units (meters and
    seconds) after denormalization.

    Also returns MAE stratified by transport mode (mae_by_mode) for
    diagnostics and wandb logging. The batch includes the transport
    mode as the 5th element returned by the dataset's __getitem__.
    """
    model.to(device)
    model.eval()

    all_preds   = []
    all_targets = []
    all_modes   = []

    with torch.no_grad():
        for batch in loader:
            features = batch[0].to(device)
            targets  = batch[1]
            preds    = model(features).cpu()
            all_preds.append(preds.numpy())
            all_targets.append(targets.numpy())
            # batch[4] = mode (str) -- added by __getitem__
            if len(batch) > 4:
                all_modes.extend(list(batch[4]))

    if not all_preds:
        return {"mae_dist": 0.0, "mae_time": 0.0,
                "mse_dist": 0.0, "mse_time": 0.0,
                "rmse_dist": 0.0, "rmse_time": 0.0,
                "mape_dist": float("nan"), "mape_time": float("nan"),
                "mape_n_excluded_dist": 0, "mape_n_excluded_time": 0,
                "smape_dist": 0.0, "smape_time": 0.0,
                "r2_dist": float("nan"), "r2_time": float("nan"),
                "loss": 0.0, "samples": 0, "mae_by_mode": {}}

    preds_norm   = np.concatenate(all_preds)
    targets_norm = np.concatenate(all_targets)

    # MSE in normalized space
    loss = float(np.mean((preds_norm - targets_norm) ** 2))

    # MAE in real units
    preds_real   = dataset.denormalize_targets(preds_norm)
    targets_real = dataset.denormalize_targets(targets_norm)
    abs_errors   = np.abs(preds_real - targets_real)   # (N, 2)
    mae          = np.mean(abs_errors, axis=0)

    # MAE stratified by transport mode
    mae_by_mode: dict[str, dict] = {}
    if all_modes and len(all_modes) == len(abs_errors):
        from collections import defaultdict
        errs_dist: dict[str, list] = defaultdict(list)
        errs_time: dict[str, list] = defaultdict(list)
        for i, mode in enumerate(all_modes):
            errs_dist[mode].append(float(abs_errors[i, 0]))
            errs_time[mode].append(float(abs_errors[i, 1]))
        for mode in errs_dist:
            mae_by_mode[mode] = {
                "mae_dist": float(np.mean(errs_dist[mode])),
                "mae_time": float(np.mean(errs_time[mode])),
                "n":        len(errs_dist[mode]),
            }

    # MSE/RMSE in real units
    mse  = np.mean(abs_errors ** 2, axis=0)   # (dist, time)
    rmse = np.sqrt(mse)

    # MAPE -- excludes samples whose real target is below the threshold
    # (avoids a percentage blow-up from dividing by near-zero; common
    # for very short legs).
    mape_min_target = 1.0  # meters (dist) / seconds (time)
    mask_dist = np.abs(targets_real[:, 0]) > mape_min_target
    mask_time = np.abs(targets_real[:, 1]) > mape_min_target
    mape_dist = (float(np.mean(abs_errors[mask_dist, 0]
                 / np.abs(targets_real[mask_dist, 0])) * 100)
                 if mask_dist.any() else float("nan"))
    mape_time = (float(np.mean(abs_errors[mask_time, 1]
                 / np.abs(targets_real[mask_time, 1])) * 100)
                 if mask_time.any() else float("nan"))

    # sMAPE -- robust to near-zero targets, no sample exclusion
    smape_denom = (np.abs(preds_real) + np.abs(targets_real)) / 2.0
    smape_denom = np.where(smape_denom < 1e-8, 1e-8, smape_denom)
    smape = np.mean(abs_errors / smape_denom, axis=0) * 100

    # R^2 (coefficient of determination)
    ss_res = np.sum((targets_real - preds_real) ** 2, axis=0)
    ss_tot = np.sum((targets_real - np.mean(targets_real, axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)

    return {
        "loss":        loss,
        "mae_dist":    float(mae[0]),   # meters -- aggregate
        "mae_time":    float(mae[1]),   # seconds -- aggregate
        "mse_dist":    float(mse[0]),
        "mse_time":    float(mse[1]),
        "rmse_dist":   float(rmse[0]),
        "rmse_time":   float(rmse[1]),
        "mape_dist":   mape_dist,       # %, excludes real target <= 1 unit
        "mape_time":   mape_time,
        "mape_n_excluded_dist": int((~mask_dist).sum()),
        "mape_n_excluded_time": int((~mask_time).sum()),
        "smape_dist":  float(smape[0]), # %, no exclusion
        "smape_time":  float(smape[1]),
        "r2_dist":     float(r2[0]),
        "r2_time":     float(r2[1]),
        "samples":     len(preds_norm),
        "mae_by_mode": mae_by_mode,     # per transport mode
    }
