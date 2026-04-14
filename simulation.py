# simulation.py — Synthetic data generation and frozen NN scoring.
# Logic matches aggressive_regional_trust.py exactly.

import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

LATENT_DIM       = 8
N_BASE_CLASSES   = 5
N_SAMPLES        = 200       # total samples per week (base)
N_WEEKS          = 20
SEED             = 42
TEMPERATURE      = 0.8
SPREAD           = 0.6       # Gaussian noise std (noise= in original)
DRIFT_RATE       = 0.15      # units per week


# ─── Centroid setup (circular, deterministic) ─────────────────────────────────

def _circle_center(i: int, n: int, spread: float = 3.0) -> np.ndarray:
    a = 2 * np.pi * i / n
    c = np.zeros(LATENT_DIM)
    c[0] = spread * np.cos(a)
    c[1] = spread * np.sin(a)
    return c


def make_centroids(rng=None) -> dict:
    """
    5-class circular layout, spread=3.0.  Fully deterministic — rng is ignored.
    Returns {cls: center_array}.
    """
    return {i: _circle_center(i, N_BASE_CLASSES) for i in range(N_BASE_CLASSES)}


def make_class5_centroid(rng=None) -> np.ndarray:
    """Class 5 from the 6-class circular layout.  Deterministic — rng is ignored."""
    return _circle_center(5, 6)


def make_drift_vectors() -> dict:
    """
    Uniform drift direction d = ones/sqrt(8), magnitude DRIFT_RATE per step.
    Applied per week (all classes shift by the same vector).
    """
    d = np.ones(LATENT_DIM) / np.sqrt(LATENT_DIM) * DRIFT_RATE
    return {c: d for c in range(N_BASE_CLASSES)}


# ─── Sample generation ────────────────────────────────────────────────────────

def _randn(shape, rng=None) -> np.ndarray:
    if rng is not None:
        return rng.standard_normal(shape)
    return np.random.randn(*shape) if isinstance(shape, tuple) else np.random.randn(*shape)


def generate_week_data(
    live_centroids: dict,
    shift_activated: bool,
    scenario_cfg: dict,
    rng,
    n_classes: int,
) -> tuple:
    """
    Generate one week of synthetic samples and return an eval/feedback split.

    Matches the scenario data functions from aggressive_regional_trust.py:
      - emergence: classes 0-4 before week >= shift_week, 0-5 after
      - imbalance: class 0 gets 10x more samples after shift
      - drift: same distribution, but centroids have already been drifted

    Returns (x_eval, y_eval, x_fb, y_fb) with a 60/40 split.
    """
    shift_type = scenario_cfg["shift_type"]

    # Build (X, y) matching the original scenario data functions
    if shift_type == "imbalance" and shift_activated:
        # imbal_data: per = N_SAMPLES // (N_CLASSES*2) = 20
        per = N_SAMPLES // (N_BASE_CLASSES * 2)
        X_l, y_l = [], []
        pts0 = live_centroids[0] + _randn((per * 10, LATENT_DIM), rng) * SPREAD
        X_l.append(pts0)
        y_l.extend([0] * len(pts0))
        for c in range(1, N_BASE_CLASSES):
            pts = live_centroids[c] + _randn((per, LATENT_DIM), rng) * SPREAD
            X_l.append(pts)
            y_l.extend([c] * len(pts))
        X = np.vstack(X_l)
        y = np.array(y_l)
    else:
        # emergence / drift / pre-shift imbalance: equal samples per active class
        classes = list(range(n_classes))
        per = N_SAMPLES // len(classes)
        X_l, y_l = [], []
        for c in classes:
            center = live_centroids.get(c, np.zeros(LATENT_DIM))
            pts = center + _randn((per, LATENT_DIM), rng) * SPREAD
            X_l.append(pts)
            y_l.extend([c] * per)
        X = np.vstack(X_l)
        y = np.array(y_l)

    # Shuffle
    idx = np.arange(len(X))
    if rng is not None:
        rng.shuffle(idx)
    else:
        np.random.shuffle(idx)
    X, y = X[idx], y[idx]

    # 60 / 40 eval / feedback split
    split = int(len(X) * 0.6)
    return X[:split], y[:split], X[split:], y[split:]


# ─── Drift ────────────────────────────────────────────────────────────────────

def apply_drift(live_centroids: dict, drift_vectors: dict) -> dict:
    """Move each centroid one step along its drift vector."""
    new = {}
    for c in live_centroids:
        if c in drift_vectors:
            new[c] = live_centroids[c] + drift_vectors[c]
        else:
            new[c] = live_centroids[c].copy()
    return new


# ─── Frozen NN ────────────────────────────────────────────────────────────────

def nn_proba(X: np.ndarray, centers: np.ndarray, temp: float = TEMPERATURE) -> np.ndarray:
    """
    Batch nearest-centroid softmax.
    X shape: (N, LATENT_DIM)  |  centers shape: (n_classes, LATENT_DIM)
    Returns shape: (N, n_classes)
    """
    diffs  = X[:, None, :] - centers[None, :, :]
    dists  = np.linalg.norm(diffs, axis=2)
    logits = -dists / temp
    logits -= logits.max(axis=1, keepdims=True)
    exp    = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def frozen_nn_predict(x: np.ndarray, frozen_centroids: dict) -> np.ndarray:
    """
    Single-sample wrapper for app.py compatibility.
    frozen_centroids: {0: arr, 1: arr, …}  — only original classes.
    Returns probability vector of length len(frozen_centroids).
    """
    n = len(frozen_centroids)
    centers = np.array([frozen_centroids[k] for k in range(n)])
    return nn_proba(x[None, :], centers)[0]
