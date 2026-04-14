# basins.py — Evolutionary memory layer.
# Logic matches aggressive_regional_trust.py exactly.

import numpy as np
from collections import defaultdict
from simulation import LATENT_DIM

# ─── Constants ────────────────────────────────────────────────────────────────

BASINS_PER_CLASS = 6
N_GENERATIONS    = 6
ELITE_FRACTION   = 0.3
MUTATION_SCALE   = 0.15
FORGETTING_RATE  = 0.004


# ─── Initialisation ───────────────────────────────────────────────────────────

def _new_basin(center: np.ndarray, cls: int, rng=None) -> dict:
    if rng is not None:
        c = center + rng.standard_normal(LATENT_DIM) * 0.4
        r = 1.0 + float(rng.random()) * 0.5
        w = 0.5 + float(rng.random()) * 0.5
    else:
        c = center + np.random.randn(LATENT_DIM) * 0.4
        r = 1.0 + np.random.rand() * 0.5
        w = 0.5 + np.random.rand() * 0.5
    return {'c': c, 'r': r, 'label': cls, 'w': w}


def initialize_basins(centroids: dict, rng=None) -> list:
    """Create BASINS_PER_CLASS basins per class around each centroid."""
    basins = []
    for cls in range(len(centroids)):
        for _ in range(BASINS_PER_CLASS):
            basins.append(_new_basin(centroids[cls], cls, rng))
    return basins


def add_basins_for_class(k: int, seed_pts: np.ndarray, rng=None) -> list:
    """Bootstrap BASINS_PER_CLASS basins for a new class from seed points."""
    center = seed_pts.mean(axis=0)
    return [_new_basin(center, k, rng) for _ in range(BASINS_PER_CLASS)]


# ─── Scoring (batch) ──────────────────────────────────────────────────────────

def basin_acts_batch(X: np.ndarray, basins: list) -> np.ndarray:
    """
    Vectorised Gaussian basin activations.
    Returns shape (len(X), len(basins)).
    """
    C  = np.array([b['c'] for b in basins])
    R  = np.array([b['r'] for b in basins])
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-d2 / (2 * R ** 2 + 1e-8))


def evo_proba_batch(X: np.ndarray, basins: list, n_classes: int) -> np.ndarray:
    """
    Batch evo predictions via weighted basin voting.
    Returns shape (len(X), n_classes).
    """
    acts = basin_acts_batch(X, basins)
    W    = np.array([b['w'] for b in basins])
    L    = np.array([b['label'] for b in basins])
    S    = np.zeros((len(X), n_classes))
    for cls in range(n_classes):
        m = L == cls
        if m.any():
            S[:, cls] = (acts[:, m] * W[m]).sum(axis=1)
    S += 1e-8
    return S / S.sum(axis=1, keepdims=True)


def nearest_labels_batch(X: np.ndarray, basins: list) -> np.ndarray:
    """Nearest basin label for each sample in X."""
    L = np.array([b['label'] for b in basins])
    return L[basin_acts_batch(X, basins).argmax(axis=1)]


def score_sample(x: np.ndarray, basins: list, n_classes: int) -> tuple:
    """
    Single-sample wrapper for app.py compatibility.
    Returns (probs shape (n_classes,), per-basin activations shape (len(basins),)).
    """
    acts = basin_acts_batch(x[None, :], basins)[0]   # (len(basins),)
    W    = np.array([b['w'] for b in basins])
    L    = np.array([b['label'] for b in basins])
    S    = np.zeros(n_classes)
    for cls in range(n_classes):
        m = L == cls
        if m.any():
            S[cls] = (acts[m] * W[m]).sum()
    S += 1e-8
    return S / S.sum(), acts


# ─── Fitness ──────────────────────────────────────────────────────────────────

def _compute_fitness(basins: list, X: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Per-basin fitness = activation-weighted fraction of correctly classified samples.
    Matches aggressive_regional_trust.py compute_fitness exactly.
    """
    acts    = basin_acts_batch(X, basins)
    correct = (evo_proba_batch(X, basins, n_classes).argmax(axis=1) == y).astype(float)
    fitness = np.zeros(len(basins))
    for i in range(len(basins)):
        a = acts[:, i]
        fitness[i] = (a * correct).sum() / (a.sum() + 1e-8)
    return fitness


# ─── Evolution ────────────────────────────────────────────────────────────────

def evolve(
    basins: list,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_gen: int = N_GENERATIONS,
    rng=None,
) -> list:
    """
    Class-stratified evolution matching aggressive_regional_trust.py evolve():
      - Top ELITE_FRACTION survive unchanged.
      - Remainder mutated from elite parents (mutation scale 0.15, radius 0.1, weight 0.05).
      - Weight decay FORGETTING_RATE applied to ALL basins after each generation.
    """
    def _ri(n):
        return int(rng.integers(0, n)) if rng is not None else int(np.random.randint(n))

    def _rn(shape):
        return rng.standard_normal(shape) if rng is not None else np.random.randn(*shape)

    def _rn1():
        return float(rng.standard_normal()) if rng is not None else float(np.random.randn())

    for _ in range(n_gen):
        by_cls  = defaultdict(list)
        for i, b in enumerate(basins):
            by_cls[b['label']].append(i)

        fitness    = _compute_fitness(basins, X, y, n_classes)
        new_basins = []

        for cls, idxs in by_cls.items():
            f     = np.array([fitness[i] for i in idxs])
            order = np.argsort(f)[::-1]
            n_el  = max(1, int(len(idxs) * ELITE_FRACTION))

            for rank, si in enumerate(order):
                b = {k: v.copy() if isinstance(v, np.ndarray) else v
                     for k, v in basins[idxs[si]].items()}
                if rank < n_el:
                    new_basins.append(b)
                else:
                    p = basins[idxs[order[_ri(n_el)]]]
                    new_basins.append({
                        'c':     p['c'] + _rn((LATENT_DIM,)) * MUTATION_SCALE,
                        'r':     max(0.1, p['r'] + _rn1() * 0.1),
                        'label': cls,
                        'w':     float(np.clip(p['w'] + _rn1() * 0.05, 0.1, 2.0)),
                    })

        for b in new_basins:
            b['w'] *= (1 - FORGETTING_RATE)

        basins = new_basins

    return basins


def evolve_basins(basins: list, feedback: list, rng=None) -> list:
    """
    App.py wrapper: feedback is list of (x, y) pairs.
    Unpacks to arrays and calls evolve().
    """
    if not feedback:
        return basins
    X = np.array([p[0] for p in feedback])
    y = np.array([int(p[1]) for p in feedback])
    n_classes = max(b['label'] for b in basins) + 1
    return evolve(basins, X, y, n_classes, N_GENERATIONS, rng)


# ─── Top contributing basins (audit) ─────────────────────────────────────────

def top_contributing_basins(basins: list, activations: np.ndarray, n: int = 3) -> list:
    """Return top-n basins by activation for the audit panel."""
    top_idx = np.argsort(activations)[::-1][:n]
    return [
        {
            "class":      basins[i]['label'],
            "radius":     float(basins[i]['r']),
            "weight":     float(basins[i]['w']),
            "activation": float(activations[i]),
        }
        for i in top_idx
    ]
