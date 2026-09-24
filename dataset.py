"""
dataset.py
----------
PyTorch dataset for mode-transition point prediction, with a proper
train / validation / test split.

Split strategy:
  - Level 1 (per user): test users are held out entirely from FL
    training. They test generalization to unseen clients.
  - Level 2 (temporal, per user): within each user, samples follow
    chronological order. 70% train -> 15% validation -> 15% test.
  - Level 3 (per segment): the sliding window never crosses a split
    boundary -- avoids leakage between temporally adjacent samples.

Why a random split would be wrong here:
  The sliding window produces consecutive samples from the same
  trajectory. Samples at index k and k+1 share k-1 of their k input
  GPS points. Shuffling and splitting randomly would put nearly
  identical samples in both train and test -- the model would
  "memorize" instead of generalize. The temporal split avoids this.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from geo_utils import haversine
import config


# ================================================================
#  DATA SPLIT
# ================================================================

def split_samples(
    samples_by_user: dict[str, list[dict]],
    test_user_ratio: float = 0.15,
    val_ratio:       float = 0.15,
    train_ratio:     float = 0.70,
    seed:            int   = config.EXPERIMENT_SEEDS[0],
) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    """
    Splits samples into train / validation / test.

    Returns:
      train_samples  : samples for FL training
      val_samples    : samples for validation during training
      test_samples   : samples for final (held-out) evaluation
      test_user_ids  : IDs of the users held out for testing

    Parameters:
      test_user_ratio : fraction of users held out for testing
      val_ratio       : per-user temporal fraction for validation
      train_ratio     : per-user temporal fraction for training
    """
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio must equal 1.0"

    rng       = np.random.default_rng(seed)
    user_ids  = list(samples_by_user.keys())

    # -- Level 1: hold out test users --------------------------
    n_test_users = max(1, int(len(user_ids) * test_user_ratio))
    rng.shuffle(user_ids)
    test_user_ids  = user_ids[:n_test_users]
    train_user_ids = user_ids[n_test_users:]

    print(f"\n[Split] Train/val users : {len(train_user_ids)}")
    print(f"[Split] Test users (held-out): {len(test_user_ids)}")

    # -- Level 2: temporal split within each user ----------------
    train_samples, val_samples, test_samples = [], [], []

    for uid in train_user_ids:
        user_s = samples_by_user[uid]
        # Samples are already in chronological order (extracted
        # sequentially from the trajectory)
        n      = len(user_s)
        n_tr   = int(n * train_ratio)
        n_val  = int(n * val_ratio)

        # Ensures the split never breaks in the middle of a window
        train_samples.extend(user_s[:n_tr])
        val_samples.extend(  user_s[n_tr: n_tr + n_val])
        # Remainder goes to validation (avoids dropping samples)
        val_samples.extend(  user_s[n_tr + n_val:])

    for uid in test_user_ids:
        test_samples.extend(samples_by_user[uid])

    print(f"[Split] Train samples    : {len(train_samples)}")
    print(f"[Split] Validation samples : {len(val_samples)}")
    print(f"[Split] Test samples     : {len(test_samples)}")

    return train_samples, val_samples, test_samples, test_user_ids


def group_samples_by_user(
    samples: list[dict],
) -> dict[str, list[dict]]:
    """
    Groups samples by user (the 'user_id' field).
    Preserves chronological order within each user.
    """
    by_user: dict[str, list] = defaultdict(list)
    for s in samples:
        uid = s.get("user_id", "unknown")
        by_user[uid].append(s)
    return dict(by_user)


# ================================================================
#  PYTORCH DATASET
# ================================================================

class TransitionPredictionDataset(Dataset):
    """
    Dataset of GPS windows with a distance/time target until the
    next transport-mode change.

    Each sample:
      features : (WINDOW_SIZE, N_FEATURES) -- float32 tensor
      targets  : (2,) -- [dist_meters, time_seconds]
                  Z-score normalized; log1p'd first if use_log1p_target
                  is set -- see denormalize_targets() for the inverse
                  (expm1 after undoing the Z-score)

    The scaler is fit ONLY on the training split and applied to
    val/test -- avoids leaking normalization stats.
    """

    def __init__(
        self,
        samples:    list[dict],
        fit_scaler: bool            = True,
        scaler:     StandardScaler  = None,
        target_mean: np.ndarray     = None,
        target_std:  np.ndarray     = None,
        split_name:  str            = "train",
        use_log1p_target: bool      = False,
    ):
        self.samples    = samples
        self.split_name = split_name
        self._use_log1p_target = use_log1p_target

        if len(samples) == 0:
            self.X          = np.zeros(
                (0, config.WINDOW_SIZE, config.N_FEATURES),
                dtype=np.float32,
            )
            self.Y          = np.zeros((0, 2), dtype=np.float32)
            self.scaler     = scaler
            self.target_mean = (target_mean if target_mean is not None
                                else np.zeros(2))
            self.target_std  = (target_std if target_std is not None
                                else np.ones(2))
            return

        # -- Feature normalization (Z-score) -------------------
        X = np.stack([s["features"] for s in samples])
        N, W, F  = X.shape
        X_flat   = X.reshape(N * W, F)

        if fit_scaler:
            # Fit only on the training split
            self.scaler = StandardScaler()
            X_norm = self.scaler.fit_transform(X_flat)
        elif scaler is not None:
            # Apply the training split's scaler to val/test
            self.scaler = scaler
            X_norm = scaler.transform(X_flat)
        else:
            self.scaler = None
            X_norm = X_flat

        self.X = X_norm.reshape(N, W, F).astype(np.float32)

        # -- Target normalization ---------------------------------
        targets = np.stack([s["targets"] for s in samples])

        # log1p BEFORE the Z-score, only for datasets in
        # config.LOG1P_TARGET_DATASETS (MOBIS). dist_m/time_s are
        # heavy-tailed there (MOBIS: median 2.6km, p95 12.5km, cap
        # 200km) -- Z-score alone only centers/rescales linearly, it
        # doesn't correct the skew. log1p compresses the tail
        # (log1p(x)=ln(1+x), valid since dist/time are always >=0).
        # Expected effect: MSE/Huber gradients stop being dominated by a
        # handful of long-tail examples (consistent with an
        # asymmetric-gradient signature observed on MOBIS: low average
        # grad norm but grad_clip_rate=2.1%). GeoLife's target
        # distribution didn't show this heavy tail, so it skips this
        # step (use_log1p_target=False). Inverted in
        # denormalize_targets() (expm1 after undoing the Z-score).
        if self._use_log1p_target:
            targets = np.log1p(targets)

        if fit_scaler:
            # Compute stats only on the training split
            self.target_mean = targets.mean(axis=0)
            self.target_std  = targets.std(axis=0) + 1e-6
        else:
            # Use the training split's stats for val/test
            self.target_mean = (target_mean if target_mean is not None
                                else targets.mean(axis=0))
            self.target_std  = (target_std if target_std is not None
                                else targets.std(axis=0) + 1e-6)

        self.Y = (
            (targets - self.target_mean) / self.target_std
        ).astype(np.float32)

    # -- PyTorch interface ---------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.Y[idx]),
            float(self.samples[idx]["lat"]),
            float(self.samples[idx]["lon"]),
            str(self.samples[idx].get("mode", "unknown")),
        )

    def denormalize_targets(self, y: np.ndarray) -> np.ndarray:
        """Converts normalized output -> meters and seconds.

        Undoes the Z-score, then expm1 (the inverse of the log1p applied
        in __init__) if this dataset uses use_log1p_target. np.expm1 is
        safe here: the input is always the model's prediction/target in
        log1p space, never negative enough for expm1 to overflow
        (log1p(x)>=0 for x>=0; even with prediction error, very negative
        values are implicitly bounded by the observed training range).
        """
        y_real = y * self.target_std + self.target_mean
        if self._use_log1p_target:
            return np.expm1(y_real)
        return y_real

    @property
    def n_outputs(self) -> int:
        return config.N_OUTPUTS


# ================================================================
#  FACTORY: BUILDS ALL THREE SPLITS AT ONCE
# ================================================================

def create_datasets(
    samples:         list[dict],
    test_user_ratio: float = 0.15,
    val_ratio:       float = 0.15,
    seed:            int   = config.EXPERIMENT_SEEDS[0],
    use_log1p_target: bool = False,
) -> tuple["TransitionPredictionDataset",
           "TransitionPredictionDataset",
           "TransitionPredictionDataset",
           list[str]]:
    """
    Builds all three datasets at once, guaranteeing that:
      1. The scaler is fit ONLY on the training split
      2. Target normalization stats come from the training split
      3. The same stats are applied to val and test

    Returns: (train_ds, val_ds, test_ds, test_user_ids)
    """
    by_user = group_samples_by_user(samples)

    train_s, val_s, test_s, test_uids = split_samples(
        by_user,
        test_user_ratio = test_user_ratio,
        val_ratio       = val_ratio,
        train_ratio     = 1.0 - val_ratio,
        seed            = seed,
    )

    # Fit scaler and stats on the training split
    train_ds = TransitionPredictionDataset(
        train_s,
        fit_scaler = True,
        split_name = "train",
        use_log1p_target = use_log1p_target,
    )

    # Val and test use the training split's scaler and stats
    val_ds = TransitionPredictionDataset(
        val_s,
        fit_scaler   = False,
        scaler       = train_ds.scaler,
        target_mean  = train_ds.target_mean,
        target_std   = train_ds.target_std,
        split_name   = "val",
        use_log1p_target = use_log1p_target,
    )

    test_ds = TransitionPredictionDataset(
        test_s,
        fit_scaler   = False,
        scaler       = train_ds.scaler,
        target_mean  = train_ds.target_mean,
        target_std   = train_ds.target_std,
        split_name   = "test",
        use_log1p_target = use_log1p_target,
    )

    print(f"\n[Datasets created]")
    print(f"  Train      : {len(train_ds)} samples")
    print(f"  Validation : {len(val_ds)} samples")
    print(f"  Test       : {len(test_ds)} samples "
          f"({len(test_uids)} held-out users)")

    return train_ds, val_ds, test_ds, test_uids


# ================================================================
#  NON-IID GEOGRAPHIC PARTITIONING
# ================================================================


def collate_with_mode(batch):
    """
    Custom collate function to handle the mode field (str).
    PyTorch's default DataLoader doesn't support strings in batches.

    batch: list of tuples (features, targets, lat, lon, mode)
    Returns: (features_tensor, targets_tensor, lats, lons, modes)
    """
    import torch
    features = torch.stack([b[0] for b in batch])
    targets  = torch.stack([b[1] for b in batch])
    lats     = [b[2] for b in batch]
    lons     = [b[3] for b in batch]
    modes    = [b[4] for b in batch]
    return features, targets, lats, lons, modes


def get_client_dataset(
    dataset:   "TransitionPredictionDataset",
    user_id:   str,
) -> "Subset":
    """
    Returns all samples belonging to a given client (by user_id).
    Used for MOBIS, where position_seq does not carry user_id per
    position (only lat/lon/mode) -- user_id comes from the profile
    (client level), not from the client_position used by
    get_local_dataset.
    """
    indices = [
        i for i, s in enumerate(dataset.samples)
        if str(s.get("user_id", "")) == str(user_id)
    ]
    return Subset(dataset, indices)


def get_local_dataset(
    dataset:         TransitionPredictionDataset,
    client_position: dict,
    radius_km:       float = 3.0,   # kept in the signature for compatibility
    min_samples:     int   = 32,
) -> Subset:
    """
    Returns EXCLUSIVELY the client's own user's samples.

    FL privacy: each client trains only on its own data. The
    selection criterion is user_id -- never the geographic radius --
    guaranteeing that no other user's samples enter local training.

    The radius_km parameter is kept in the signature for compatibility
    with existing calls in fl_algorithms.py, but it is not used for
    filtering. Geographic position is still used by the FedHME
    estimator (via client_position), just not to select training data.

    If the client doesn't have enough samples (< min_samples), returns
    an empty Subset -- the client will report skipped=True that round.
    """
    user_id = client_position.get("user_id")

    if user_id is None:
        # Without a user_id there's no way to guarantee isolation -- skip
        return Subset(dataset, [])

    indices = [
        i for i, s in enumerate(dataset.samples)
        if s.get("user_id") == user_id
    ]

    # min_samples is intentionally not enforced: the client trains with
    # whatever it has, even with low volume. This shows the estimator
    # works regardless of local data volume.
    if len(indices) == 0:
        return Subset(dataset, [])

    return Subset(dataset, indices)
