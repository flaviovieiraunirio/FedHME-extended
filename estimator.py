"""
estimator.py — FedHME heterogeneity estimator
----------------------------------------
Hybrid heterogeneity estimator with a composite alpha.

Original contribution of this work:
  1. Composite alpha: physical velocity + per-mode delta_m
  2. delta_m calibrated empirically on GeoLife
  3. Geographic map of GRADIENT DIVERGENCE
  4. Decision made before training — novel temporal anticipation

Known limitation (future work):
  The estimator only uses gradient divergence as a heterogeneity
  signal. Clients with low gradient but high loss on the global
  model — indicating regions where the model still performs poorly —
  are misclassified as uninformative. A loss map would address this
  pathological case (left as future work).
"""

import numpy as np
import config

DEFAULT_MODE_RATES: dict[str, float] = {
    "walk":    0.02, "bike":    0.04,
    "bus":     0.06, "car":     0.05,
    "subway":  0.12, "train":   0.08,
    "unknown": 0.05,
}


# ================================================================
#  FedHME HYBRID ESTIMATOR
#  Geographic map: gradient divergence only
# ================================================================

class HybridEstimator:
    """
    h_hat = alpha(v,m) * temporal + (1-alpha) * spatial_grad

    where spatial_grad is the geographic map of gradient divergence
    -- the only spatial source in this version.
    """
    name = "hybrid"

    def __init__(
        self,
        n_clients:      int   = config.N_CLIENTS,
        grid_size:      int   = config.GRID_SIZE,
        history_window: int   = config.HISTORY_WINDOW,
        velocity_decay: float = config.VELOCITY_DECAY,
        mode_decay:     float = config.MODE_DECAY,
    ):
        self.grid_size      = grid_size
        self.history_window = history_window
        self.velocity_decay = velocity_decay
        self.mode_decay     = mode_decay

        # EMA scalar per client
        self.client_ema: dict[int, float] = {
            i: 0.0 for i in range(n_clients)
        }
        self.client_ema_init: dict[int, bool] = {
            i: False for i in range(n_clients)
        }
        # Kept for compatibility with calibrate_mode_rates
        self.client_history: dict[int, list[float]] = {
            i: [] for i in range(n_clients)
        }

        # Geographic map of GRADIENT DIVERGENCE
        # EMA with decay instead of cumulative average
        rng = np.random.default_rng(seed=config.ACTUAL_SEED)
        self.geo_map_grad = rng.uniform(0.001, 0.01, size=(grid_size, grid_size))
        self.geo_map_decay    = config.GEO_MAP_DECAY
        self.client_ema_alpha = config.CLIENT_EMA_ALPHA

        self.geo_count     = np.zeros((grid_size, grid_size))  # kept for compatibility

        self.mode_rates = DEFAULT_MODE_RATES.copy()

        self.consecutive_skips: dict[int, int] = {
            i: 0 for i in range(n_clients)
        }
        self.log: list[dict] = []

    # -- delta_m calibration -----------------------------------

    def calibrate_mode_rates(
        self,
        round_log: list[dict],
    ) -> dict[str, float]:
        """
        Calibrates delta_m from client_log (one entry per client),
        not round_log (entries aggregated per round).

        Reason: round_log stores per-round averages and has no
        individual client_id/mode/divergence/velocity fields, so
        entry.get("mode") would always return "unknown" there.

        The strategy keeps self.client_log with one entry per
        client per round during calibration; calibrate_mode_rates
        reads that log instead of round_log.
        """
        from collections import defaultdict
        delta_divs: dict[str, list] = defaultdict(list)
        velocities: dict[str, list] = defaultdict(list)

        source = getattr(self, "client_log", [])

        # track (divergence, round, mode) per client
        prev_info: dict[int, tuple] = {}

        for entry in source:
            cid   = entry.get("client_id")
            mode  = entry.get("mode", "unknown")
            div   = entry.get("divergence", 0.0)
            v     = entry.get("velocity",   0.0)
            rnd   = entry.get("round", 0)
            if cid is None:
                continue
            # Entries for skipped clients have divergence=None -- ignore
            # them for delta_m calibration
            if div is None or mode is None:
                continue
            if cid in prev_info:
                prev_div, prev_rnd, prev_mode = prev_info[cid]
                # normalize by the round gap
                gap = max(rnd - prev_rnd, 1)
                delta_per_round = abs(div - prev_div) / gap
                # attribute to the previous mode (origin of the change)
                delta_divs[prev_mode].append(delta_per_round)
                velocities[prev_mode].append(max(v, 1e-6))
            prev_info[cid] = (div, rnd, mode)

        calibrated = {}
        for mode in delta_divs:
            if len(delta_divs[mode]) < 3:
                calibrated[mode] = DEFAULT_MODE_RATES.get(mode, 0.05)
            else:
                calibrated[mode] = float(
                    np.mean(delta_divs[mode])
                    / np.mean(velocities[mode])
                )
        self.mode_rates.update(calibrated)
        print("\n[Estimator] delta_m calibrated:")
        for m, v in sorted(calibrated.items()):
            print(f"  {m:8s}: {v:.4f}")
        return calibrated

    def get_geo_heatmap(self) -> dict:
        """
        Returns the geographic divergence map with metadata for
        heatmap generation using real Beijing coordinates.

        The map covers the bounding box defined in config.BEIJING_BBOX.
        Each cell (i,j) represents the average gradient divergence
        observed in that region during the experiment.
        """
        bbox = config.BEIJING_BBOX
        return {
            "geo_map":  self.geo_map_grad.copy(),
            "counts":   self.geo_count.copy(),
            "lat_min":  bbox["lat_min"],
            "lat_max":  bbox["lat_max"],
            "lon_min":  bbox["lon_min"],
            "lon_max":  bbox["lon_max"],
            "grid_size": self.grid_size,
        }

    # -- Update after a round -----------------------------------

    def update(
        self,
        client_id:  int,
        divergence: float,
        position:   dict,
        local_loss: float = None,  # received but not used by this estimator
        cell_id:    tuple = None,  # (gx, gy) reported by MobilityClient
    ) -> None:
        # EMA scalar update per client
        if not self.client_ema_init[client_id]:
            self.client_ema[client_id] = divergence
            self.client_ema_init[client_id] = True
        else:
            alpha = self.client_ema_alpha
            self.client_ema[client_id] = (
                (1 - alpha) * self.client_ema[client_id]
                + alpha * divergence
            )

        # Kept for calibrate_mode_rates
        hist = self.client_history[client_id]
        hist.append(divergence)
        if len(hist) > self.history_window:
            hist.pop(0)

        # EMA update on the geographic divergence map
        # cell_id reported by MobilityClient -- avoids _to_grid on the server
        if cell_id is not None:
            gx, gy = cell_id
        else:
            gx, gy = self._to_grid(position)
        if self.geo_count[gx, gy] == 0:
            self.geo_map_grad[gx, gy] = divergence
        else:
            decay = self.geo_map_decay
            self.geo_map_grad[gx, gy] = (
                (1 - decay) * self.geo_map_grad[gx, gy]
                + decay * divergence
            )
        self.geo_count[gx, gy] += 1

        # local_loss is received but only recorded as data --
        # not used in the current estimator; reserved for future work

    # -- Estimate before a round ----------------------------------

    def estimate(
        self,
        client_id: int,
        position:  dict,
        velocity:  float,
        mode:      str   = "unknown",
        round_num: int   = 0,
        cell_id:   tuple = None,  # (gx, gy) from MobilityClient
    ) -> dict:
        # cell_id from MobilityClient is equivalent to _to_grid(position)
        if cell_id is not None:
            gx, gy = cell_id
        else:
            gx, gy = self._to_grid(position)

        # temporal component via EMA scalar
        if self.client_ema_init.get(client_id, False):
            temporal = float(self.client_ema[client_id])
        else:
            temporal = float(self.geo_map_grad.mean())

        # Spatial component -- gradient map only
        spatial = self._smoothed_map(self.geo_map_grad, gx, gy)

        # composite alpha: velocity + mode
        delta_m = self.mode_rates.get(mode, 0.05)
        alpha   = float(np.clip(
            np.exp(-(velocity * self.velocity_decay
                     + delta_m * self.mode_decay)),
            0.0, 1.0
        ))
        estimate = alpha * temporal + (1 - alpha) * spatial

        # If the estimate is ~0 due to missing spatial data (sparse
        # regions), fall back to the temporal component as a floor.
        # Avoids improperly skipping high-velocity clients whose
        # alpha~0 makes the estimate depend on an empty map.
        if estimate < 1e-6 and temporal > 0:
            estimate = temporal * 0.5  # conservative floor

        entry = {
            "estimate":  estimate,
            "temporal":  temporal,
            "spatial":   spatial,
            "alpha":     alpha,
            "velocity":  velocity,
            "mode":      mode,
            "delta_m":   delta_m,
            "round":     round_num,
            "client_id": client_id,
            # components for logging
            "h_temporal_contrib": float(alpha * temporal),
            "h_spatial_contrib":  float((1 - alpha) * spatial),
            "geo_map_val":        float(self.geo_map_grad[gx, gy]),
            "client_ema_val":     float(self.client_ema.get(client_id, 0.0)),
            "is_first_round": not self.client_ema_init.get(client_id, False),
            "cell_obs_count": int(self.geo_count[gx, gy]),
            "cell_id_gx":     int(gx),
            "cell_id_gy":     int(gy),
        }
        self.log.append(entry)
        return entry

    # -- Decision -------------------------------------------------

    def decide(
        self,
        client_id:   int,
        h:           float,
        mean_h:      float = None,
        median_h:    float = None,
        base_epochs: int   = config.LOCAL_EPOCHS_BASE,
    ) -> tuple[str, int]:
        """
        Decides whether the client participates in the round.

        Rank normalization (applied in fl_algorithms before this call):
          h = rank(h_hat_i) / (N-1)  ->  h in [0.0, 1.0]
          skip_threshold = SKIP_FACTOR  (e.g. 0.5 -> skips the bottom 50%)

        Advantage over SKIP_FACTOR x mean_h:
          - Invariant to the absolute scale of h_hat
          - Invariant to the distribution shape (long-tailed GeoLife,
            more homogeneous datasets)
          - SKIP_FACTOR has direct semantics: fraction of the pool to skip
          - No SKIP_FLOOR needed (rank is always well defined)

        Epoch allocation is NOT done here -- it is done collectively
        in the strategy's configure_fit(). This method only returns
        "skip"/0 or "train"/0.
        """
        skips = self.consecutive_skips[client_id]

        # Rank normalization is applied in fl_algorithms before decide().
        # h arrives as rank/N in [0.0, 1.0] -- SKIP_FACTOR is directly the
        # fraction to skip (0.5 -> bottom 50%, scale invariant).
        # mean_h and median_h are kept in the signature for compatibility.
        skip_threshold = config.SKIP_FACTOR

        if (h < skip_threshold and
                skips < config.MAX_CONSECUTIVE_SKIPS):
            self.consecutive_skips[client_id] += 1
            return "skip", 0
        self.consecutive_skips[client_id] = 0
        return "train", 0  # epochs are set by the strategy

    @staticmethod
    def allocate_epochs_proportional(
        estimates:    dict[int, float],
        base_epochs:  int = config.LOCAL_EPOCHS_BASE,
        n_total_clients: int = config.MIN_FIT_CLIENTS,
        e_min:        int = 1,
        e_max:        int = None,
        apply_budget: bool = False,  # True only for the hybrid estimator
    ) -> dict[int, int]:
        """
        Proportional epoch allocation under a fixed budget.

        E_total = n_total_clients * base_epochs
        E_k = round(E_total * h_hat_k / sum(h_hat_j))
        E_k = clamp(E_k, e_min, e_max)

        The total budget equals what no_estimate would spend with
        N clients x E_base fixed epochs -- a fair comparison.

        Args:
            estimates: {client_id: h_hat} of the clients that will train
            base_epochs: epochs the baseline uses per client
            n_total_clients: N used to compute E_total
            e_min: minimum epochs per client
            e_max: maximum epochs per client (None = 3x base)
        Returns:
            {client_id: n_epochs}
        """
        if e_max is None:
            e_max = base_epochs * 3

        if not estimates:
            return {}

        # BUDGET_SAVING limits the epoch budget -- applied only to the hybrid
        # apply_budget=True only on the hybrid's configure_fit calls
        if apply_budget:
            budget_factor = 1.0 - getattr(config, 'BUDGET_SAVING', 0.0)
            e_total = int(round(n_total_clients * base_epochs * budget_factor))
            e_total = max(e_total, len(estimates))  # guarantees >= 1 epoch per client
        else:
            e_total = n_total_clients * base_epochs
        h_sum = sum(estimates.values())

        if h_sum < 1e-10:
            # All h_hat are ~0 -- distribute uniformly
            return {cid: base_epochs for cid in estimates}

        # Proportional allocation
        raw = {
            cid: e_total * h / h_sum
            for cid, h in estimates.items()
        }

        # Clamp and round
        alloc = {
            cid: max(e_min, min(e_max, round(e)))
            for cid, e in raw.items()
        }

        return alloc

    # -- Analysis ---------------------------------------------------

    def record_true_divergence(
        self, round_num: int, client_id: int, true_div: float
    ) -> float:
        for e in reversed(self.log):
            if e["round"] == round_num and \
               e["client_id"] == client_id:
                e["true_divergence"] = true_div
                e["error"] = abs(e["estimate"] - true_div)
                return e["error"]
        return 0.0

    def get_errors_by_group(self, profiles: dict) -> dict:
        errs: dict[str, list] = {}
        for e in self.log:
            if "error" not in e:
                continue
            g = profiles[e["client_id"]].get("mobility_group", "?")
            errs.setdefault(g, []).append(e["error"])
        return {g: float(np.mean(v)) for g, v in errs.items()}

    def get_errors_by_mode(self) -> dict:
        errs: dict[str, list] = {}
        for e in self.log:
            if "error" not in e:
                continue
            errs.setdefault(e.get("mode", "?"), []).append(e["error"])
        return {m: float(np.mean(v)) for m, v in errs.items()}

    def get_convergence_curve(self) -> dict:
        by_round: dict[int, list] = {}
        for e in self.log:
            if "error" not in e:
                continue
            by_round.setdefault(e["round"], []).append(e["error"])
        return {r: float(np.mean(v))
                for r, v in sorted(by_round.items())}

    def get_velocity_controlled_errors(
        self, v_min: float = 15.0, v_max: float = 25.0,
    ) -> dict[str, dict]:
        """
        Estimation error filtered by a velocity band.

        Isolates the contribution of transport mode: in the 15-25 km/h
        band, bike, bus and car all coexist. If velocity_only has a
        similar error across all modes but hybrid differentiates them,
        mode is adding information that velocity alone doesn't capture.

        Returns: {mode: {"mae": float, "n": int, "errors": [...]}}
        """
        by_mode: dict[str, list] = {}
        for e in self.log:
            if "error" not in e:
                continue
            v = e.get("velocity", 0.0)
            if v_min <= v <= v_max:
                mode = e.get("mode", "unknown")
                by_mode.setdefault(mode, []).append(e["error"])
        return {
            m: {
                "mae": float(np.mean(errs)),
                "std": float(np.std(errs)),
                "n":   len(errs),
            }
            for m, errs in by_mode.items()
            if len(errs) >= 3  # minimum to be informative
        }

    # -- Internal helpers ------------------------------------------

    def _to_grid(self, position: dict) -> tuple[int, int]:
        gx = min(int(position["norm_lat"] * self.grid_size),
                 self.grid_size - 1)
        gy = min(int(position["norm_lon"] * self.grid_size),
                 self.grid_size - 1)
        return gx, gy

    def _smoothed_map(
        self, geo_map: np.ndarray, gx: int, gy: int
    ) -> float:
        """
        Weighted average of the 3x3 neighborhood.
        Only includes visited cells (geo_count > 0) -- a never-visited
        cell has divergence 0.0, which does not represent an absence
        of heterogeneity, but an absence of data. Including it would
        dilute the estimate.
        """
        g = self.grid_size
        total, weight = 0.0, 0.0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < g and 0 <= ny < g:
                    if self.geo_count[nx, ny] > 0:  # only visited cells
                        w = 1.0 if dx == 0 and dy == 0 else 0.5
                        total  += geo_map[nx, ny] * w
                        weight += w
        return float(total / weight) if weight > 0 else 0.0


# ================================================================
#  BASELINE ESTIMATORS
# ================================================================

class BaselineNoEstimate(HybridEstimator):
    name = "no_estimate"
    def estimate(self, client_id, position, velocity,
                 mode="unknown", round_num=0, cell_id=None) -> dict:
        # Uses the global mean observed divergence instead of a fixed 0.5,
        # so the estimator's error measures real deviation, not a scale artifact
        visited = self.geo_count > 0
        if visited.any():
            global_div_mean = float(
                self.geo_map_grad[visited].mean()
            )
        else:
            global_div_mean = 0.0
        e = {"estimate": global_div_mean, "alpha": 0.5,
             "client_id": client_id, "round": round_num,
             "mode": mode, "velocity": velocity,
             "temporal": global_div_mean, "spatial": global_div_mean,
             "delta_m": 0.0,
             "h_temporal_contrib": 0.5 * global_div_mean,
             "h_spatial_contrib":  0.5 * global_div_mean,
             "geo_map_val": global_div_mean,
             "client_ema_val": global_div_mean}
        self.log.append(e)
        return e
    def decide(self, client_id, h, mean_h=None, median_h=None,
               base_epochs=config.LOCAL_EPOCHS_BASE):
        return "train", base_epochs


class BaselineTemporal(HybridEstimator):
    name = "temporal_only"
    def estimate(self, client_id, position, velocity,
                 mode="unknown", round_num=0, cell_id=None) -> dict:
        if self.client_ema_init.get(client_id, False):
            temporal = float(self.client_ema[client_id])
        else:
            temporal = float(self.geo_map_grad.mean())
        e = {"estimate": temporal, "temporal": temporal,
             "spatial": 0.0, "alpha": 1.0,
             "client_id": client_id, "round": round_num,
             "mode": mode, "velocity": velocity}
        self.log.append(e)
        return e


class BaselineSpatial(HybridEstimator):
    name = "spatial_only"
    def estimate(self, client_id, position, velocity,
                 mode="unknown", round_num=0, cell_id=None) -> dict:
        gx, gy  = self._to_grid(position)
        spatial = self._smoothed_map(self.geo_map_grad, gx, gy)
        e = {"estimate": spatial, "temporal": 0.0,
             "spatial": spatial, "alpha": 0.0,
             "client_id": client_id, "round": round_num,
             "mode": mode, "velocity": velocity}
        self.log.append(e)
        return e


class BaselineVelocityOnly(HybridEstimator):
    name = "velocity_only"
    def estimate(self, client_id, position, velocity,
                 mode="unknown", round_num=0, cell_id=None) -> dict:
        gx, gy = self._to_grid(position)
        if self.client_ema_init.get(client_id, False):
            temporal = float(self.client_ema[client_id])
        else:
            temporal = float(self.geo_map_grad.mean())
        spatial = self._smoothed_map(self.geo_map_grad, gx, gy)
        alpha   = float(np.exp(-velocity * self.velocity_decay))
        e = {"estimate": alpha * temporal + (1 - alpha) * spatial,
             "temporal": temporal, "spatial": spatial,
             "alpha": alpha, "delta_m": 0.0,
             "client_id": client_id, "round": round_num,
             "mode": mode, "velocity": velocity}
        self.log.append(e)
        return e


class BaselineModal(HybridEstimator):
    """
    Formal ablation: isolates the contribution of transport mode
    within the same alpha formula as the hybrid, zeroing out the
    velocity term (symmetric to what BaselineVelocityOnly does by
    zeroing out the delta_m term):
        alpha = exp(-(velocity * velocity_decay + delta_m * mode_decay))
        mode_only:     alpha = exp(-delta_m * mode_decay)
        velocity_only: alpha = exp(-velocity * velocity_decay)
    temporal/spatial still enter the blend normally -- only the
    driver of the weight (alpha) changes.
    """
    name = "mode_only"
    def estimate(self, client_id, position, velocity,
                 mode="unknown", round_num=0, cell_id=None) -> dict:
        gx, gy = self._to_grid(position)
        if self.client_ema_init.get(client_id, False):
            temporal = float(self.client_ema[client_id])
        else:
            temporal = float(self.geo_map_grad.mean())
        spatial = self._smoothed_map(self.geo_map_grad, gx, gy)
        delta_m = self.mode_rates.get(mode, DEFAULT_MODE_RATES.get(mode, 0.05))
        alpha   = float(np.clip(np.exp(-delta_m * self.mode_decay), 0.0, 1.0))
        e = {"estimate": alpha * temporal + (1 - alpha) * spatial,
             "temporal": temporal, "spatial": spatial,
             "alpha": alpha, "delta_m": delta_m,
             "client_id": client_id, "round": round_num,
             "mode": mode, "velocity": velocity}
        self.log.append(e)
        return e


class BaselineRandomSkip(BaselineNoEstimate):
    """
    Exploratory "random_skip" baseline (uniform random skip + uniform
    epoch allocation) -- an ablation kept for exploratory comparison,
    outside the estimator's official ablation set (see BaselineModal
    and its siblings above for the formal per-component ablation).

    Mechanism -- m(r) is a FIXED, external target, not an emergent one:
      - RANDOM_SKIP_MODE="unpaired": m(r) = RANDOM_SKIP_TARGET_N,
        constant across all rounds.
      - RANDOM_SKIP_MODE="paired": m(r) = the hybrid's real
        n_participating IN THE SAME round/seed (read from
        RANDOM_SKIP_PAIRED_HYBRID_DIR) -- eliminates the gap by
        construction, a round-by-round matched comparison.
      - The consecutive-skip cap (MAX_CONSECUTIVE_SKIPS) stays ACTIVE
        and identical to the hybrid: clients at the cap are force-included,
        consuming part of the m(r) slots; a uniform draw (self._rng, a
        dedicated RNG never shared with SGD/shuffling/fraction_fit) fills
        the rest -- implemented in FedAvgStrategy.configure_fit (this
        cannot be coordinated to hit "exactly m(r) total" from inside
        decide(), which only sees one client at a time and doesn't know
        how many others were already chosen this round).
      - Epoch allocation: deterministic and uniform across participants
        (E_k(r) = E_total(r) / n_valid(r)), done in
        FedAvgStrategy.configure_fit -- not randomized, to avoid mixing
        selection and allocation as two separate axes of randomness.

    Design note: m(r) (the number of participants per round) is treated
    as a FIXED PARAMETER of the method rather than something that emerges
    from independent per-client draws, following the pattern used by
    Power-of-Choice (Cho, Wang & Joshi) and Oort (Lai et al., OSDI 2021).
    An earlier version let m(r) emerge from independent per-client
    Bernoulli draws (reusing HybridEstimator.decide()'s structure), but
    the resulting participant count was not comparable to the hybrid's,
    because the hybrid's h_hat has temporal persistence (a low-mobility
    client keeps a low rank for several consecutive rounds), so the
    consecutive-skip cap fires more often under the hybrid than under a
    memoryless random draw.

    Importantly, m(r) must never be derived from h_hat or from the
    hybrid's real participation count: dividing the epoch budget by an
    m(r) copied from the hybrid ("paired" mode) would indirectly leak an
    h_hat-based measurement into random_skip's epoch accounting. The
    default (RANDOM_SKIP_TARGET_N=None) instead recomputes m(r) each
    round purely from the global skip cutoff (SKIP_FACTOR) and the real
    sampled pool size that round (n_valid -- a fact of the
    simulation/dataset, not an h_hat measurement; hybrid/no_estimate/
    random_skip always see the same n_valid per round).

    estimate() (inherited from BaselineNoEstimate) only fills the log
    fields (h_estimate, alpha, etc.) with the same placeholder as
    no_estimate -- it is never used for selection.
    """
    name = "random_skip"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dedicated RNG, never shared with the SGD/shuffling/fraction_fit
        # RNG -- deterministic reproducibility per seed. Same seeding
        # pattern already used by HybridEstimator.__init__ (geo_map_grad).
        # Always called in the same order (client_estimates sorted by cid
        # in FedAvgStrategy.configure_fit).
        self._rng = np.random.default_rng(seed=config.ACTUAL_SEED)
        self._paired_n_participating: dict[int, int] = {}
        if config.RANDOM_SKIP_MODE == "paired":
            self._load_paired_n_participating()

    def _load_paired_n_participating(self) -> None:
        """
        Loads the real per-round n_participating from the rounds.csv
        of an official hybrid run (same dataset, same seed via
        config.ACTUAL_SEED) -- used as m(r) in "paired" mode.
        """
        import os
        import csv as _csv

        base_dir = config.RANDOM_SKIP_PAIRED_HYBRID_DIR
        if not base_dir:
            raise ValueError(
                "RANDOM_SKIP_MODE='paired' requires "
                "RANDOM_SKIP_PAIRED_HYBRID_DIR pointing to the csv/ "
                "folder of an official hybrid run (same dataset, same "
                "seeds) -- see config.py."
            )
        path = os.path.join(
            base_dir,
            f"hybrid_seed{config.ACTUAL_SEED}_bs{config.BUDGET_SAVING:.2f}"
            f"_sf{config.SKIP_FACTOR:.2f}_rounds.csv",
        )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[paired random_skip] hybrid rounds.csv not found: "
                f"{path} -- run the hybrid for this dataset/seed first, "
                f"or adjust RANDOM_SKIP_PAIRED_HYBRID_DIR."
            )
        with open(path) as f:
            for row in _csv.DictReader(f):
                self._paired_n_participating[int(row["round"])] = int(
                    float(row["n_participating"])
                )

    def get_target_n_participating(self, server_round: int, n_valid: int) -> int:
        """
        m(r) -- the participant target for this round.

        Never derived from h_hat or from the hybrid (see the class
        docstring). Default (RANDOM_SKIP_TARGET_N=None): recomputed each
        round purely from the global cutoff (SKIP_FACTOR) and the real
        pool size sampled THIS round (n_valid -- available to any method
        before any estimate is made, not an h_hat measurement; that's why
        hybrid/no_estimate/random_skip always see the same n_valid per
        round).

        RANDOM_SKIP_TARGET_N != None: manual override, a fixed constant
        (exploratory/debugging use -- ignores the round's real pool size).
        RANDOM_SKIP_MODE == "paired": kept only for completeness/historical
        comparison -- uses the hybrid's real n_participating. Not a fair
        comparison for the reason above; do not use it to draw conclusions.
        """
        if config.RANDOM_SKIP_MODE == "paired":
            return self._paired_n_participating.get(
                server_round, round(n_valid * (1 - config.SKIP_FACTOR))
            )
        if config.RANDOM_SKIP_TARGET_N is not None:
            return config.RANDOM_SKIP_TARGET_N
        return round(n_valid * (1 - config.SKIP_FACTOR))


REGISTRY = {
    "no_estimate":   BaselineNoEstimate,
    "temporal_only": BaselineTemporal,
    "spatial_only":  BaselineSpatial,
    "velocity_only": BaselineVelocityOnly,
    "mode_only":     BaselineModal,
    "random_skip":   BaselineRandomSkip,
    "hybrid":        HybridEstimator,
}


def create_estimator(name: str, **kwargs):
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown estimator '{name}'. "
                         f"Options: {list(REGISTRY)}")
    return cls(**kwargs)


def compute_gradient_divergence(
    local_w:  list[np.ndarray],
    global_w: list[np.ndarray],
) -> float:
    return sum(float(np.linalg.norm(l - g))
               for l, g in zip(local_w, global_w))
