#!/usr/bin/env python3
"""
verify.py — Headless accuracy verification for all three EAM scenarios.

Replicates aggressive_regional_trust.py run_scenario() for the C2 variant:
  - np.random.seed(42) at start of each scenario (global numpy state, not rng)
  - Evolve on ALL weekly data, evaluate on the SAME data (no eval/feedback split)
  - Trust update uses nearest-basin labels (regional, not ground-truth class)
  - C2 params: pen=0.5/0.5, reward=0.5/0.5, 3-week rolling window

Target ranges (C2 variant, seed 42):
  Emergence : hybrid ~86%,   NN ~89%,   evo ~79%
  Imbalance : hybrid ~99.5%, NN ~99.7%, evo ~95%
  Drift     : hybrid ~97%,   NN ~96%,   evo ~93.5%

Tolerance: ±3 pp
"""

import numpy as np
from collections import deque

from simulation import LATENT_DIM, N_BASE_CLASSES, N_SAMPLES, N_WEEKS, TEMPERATURE, SPREAD, DRIFT_RATE
from basins import initialize_basins, evolve, evo_proba_batch, nearest_labels_batch, BASINS_PER_CLASS
from trust import initialize_trust, make_history, update_trust, blend_batch
from simulation import nn_proba

TARGETS = {
    "emergence": {"hybrid": 0.86,  "nn": 0.89,  "evo": 0.79},
    "imbalance": {"hybrid": 0.995, "nn": 0.997, "evo": 0.95},
    "drift":     {"hybrid": 0.97,  "nn": 0.96,  "evo": 0.935},
}
TOLERANCE = 0.03


# ─── Centroid helpers (mirrors make_centers in original) ──────────────────────

def _make_centers(n: int, spread: float = 3.0) -> np.ndarray:
    """Circular layout, returns (n, LATENT_DIM) array."""
    c = np.zeros((n, LATENT_DIM))
    for i in range(n):
        a = 2 * np.pi * i / n
        c[i, 0] = spread * np.cos(a)
        c[i, 1] = spread * np.sin(a)
    return c


# ─── Data functions (mirrors aggressive_regional_trust.py exactly) ────────────

def _sample_data(centers: np.ndarray, n: int = N_SAMPLES, classes: list = None) -> tuple:
    if classes is None:
        classes = list(range(len(centers)))
    per = n // len(classes)
    X_l, y_l = [], []
    for cls in classes:
        X_l.append(centers[cls] + np.random.randn(per, LATENT_DIM) * SPREAD)
        y_l.extend([cls] * per)
    return np.vstack(X_l), np.array(y_l)


def _emrg_data(wk: int, centers: np.ndarray) -> tuple:
    """Emergence: class 5 added at wk >= 8 (0-indexed)."""
    cls = list(range(5)) if wk < 8 else list(range(6))
    X, y = _sample_data(centers, classes=cls)
    return X, y, centers


def _imbal_data(wk: int, centers: np.ndarray) -> tuple:
    """Imbalance: class 0 gets 10x at wk >= 5 (0-indexed)."""
    if wk < 5:
        X, y = _sample_data(centers)
        return X, y, centers
    per = N_SAMPLES // (N_BASE_CLASSES * 2)
    X_l, y_l = [], []
    X_l.append(centers[0] + np.random.randn(per * 10, LATENT_DIM) * SPREAD)
    y_l.extend([0] * (per * 10))
    for cls in range(1, N_BASE_CLASSES):
        X_l.append(centers[cls] + np.random.randn(per, LATENT_DIM) * SPREAD)
        y_l.extend([cls] * per)
    return np.vstack(X_l), np.array(y_l), centers


def _drift_data(wk: int, centers: np.ndarray) -> tuple:
    """Drift: shift centroids DRIFT_RATE per week, then sample."""
    d = np.ones(LATENT_DIM) / np.sqrt(LATENT_DIM)
    centers = centers + d * DRIFT_RATE
    X, y = _sample_data(centers)
    return X, y, centers


# ─── Scenario runner ──────────────────────────────────────────────────────────

def run_scenario(scenario_key: str) -> dict:
    """
    Replicates aggressive_regional_trust.py run_scenario() for C2 variant.
    Uses global numpy random state — caller must seed before calling.
    """
    if scenario_key == "emergence":
        c6       = _make_centers(6)
        centers  = c6.copy()               # 6-row array (original uses _c6)
        nn_cents = c6[:5].copy()           # frozen NN sees only 5 classes
        data_fn  = _emrg_data
        # Basins initialised for 5 classes only — matches init_basins(centers, N_CLASSES=5)
        centers_dict = {i: centers[i] for i in range(N_BASE_CLASSES)}

    elif scenario_key == "imbalance":
        centers  = _make_centers(5)
        nn_cents = centers.copy()
        data_fn  = _imbal_data
        centers_dict = {i: centers[i] for i in range(N_BASE_CLASSES)}

    else:  # drift
        centers  = _make_centers(5)
        nn_cents = centers.copy()
        data_fn  = _drift_data
        centers_dict = {i: centers[i] for i in range(N_BASE_CLASSES)}

    basins   = initialize_basins(centers_dict)   # uses global np.random state
    regional = initialize_trust(N_BASE_CLASSES)  # {0:0.5, …, 4:0.5}
    history  = make_history(N_BASE_CLASSES)
    GLOBAL_TRUST = 0.8

    # n_classes is ALWAYS N_BASE_CLASSES=5 throughout all scenarios.
    # For emergence: class 5 data appears at wk>=8 but models always output 5 classes.
    # Class 5 samples are never correctly classified — this matches the original.
    n_classes = N_BASE_CLASSES

    totals = {"nn": 0, "evo": 0, "hybrid": 0, "n": 0}

    for wk in range(N_WEEKS):
        X, y, centers = data_fn(wk, centers)
        centers_dict = {i: centers[i] for i in range(N_BASE_CLASSES)}

        # Evolve on ALL data (no split — matches original)
        basins = evolve(basins, X, y, n_classes)

        # Predict
        nn_p   = nn_proba(X, nn_cents)           # shape (N, 5)
        if nn_p.shape[1] < n_classes:
            pad   = np.full((len(X), n_classes - nn_p.shape[1]), 1e-8)
            nn_p  = np.hstack([nn_p, pad])
            nn_p /= nn_p.sum(axis=1, keepdims=True)

        evo_p  = evo_proba_batch(X, basins, n_classes)   # (N, n_classes)
        nl     = nearest_labels_batch(X, basins)          # (N,)

        blended = blend_batch(nn_p, evo_p, GLOBAL_TRUST, regional, nl)

        nn_preds     = nn_p.argmax(axis=1)
        evo_preds    = evo_p.argmax(axis=1)
        hybrid_preds = blended.argmax(axis=1)

        totals["nn"]     += int((nn_preds     == y).sum())
        totals["evo"]    += int((evo_preds    == y).sum())
        totals["hybrid"] += int((hybrid_preds == y).sum())
        totals["n"]      += len(y)

        # Trust update — C2 regional
        nn_correct  = nn_preds  == y
        evo_correct = evo_preds == y
        regional = update_trust(
            regional,
            history,
            nl,
            nn_correct,
            evo_correct,
            mode="legacy",
        )

    n = totals["n"]
    return {k: totals[k] / n for k in ["nn", "evo", "hybrid"]}


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("EAM Verification — seed 42, 20 weeks, 3 scenarios")
    print("=" * 62)

    all_pass = True
    for key in ["emergence", "imbalance", "drift"]:
        np.random.seed(42)   # reset global state per scenario (mirrors original)
        from scenarios import SCENARIOS
        print(f"\n[{SCENARIOS[key]['name']}]")
        result = run_scenario(key)
        tgts   = TARGETS[key]
        for label in ["hybrid", "nn", "evo"]:
            got  = result[label]
            tgt  = tgts[label]
            diff = abs(got - tgt)
            flag = "OK  " if diff <= TOLERANCE else "WARN"
            print(f"  {label:6s}: got {got:.1%}  target ~{tgt:.1%}  "
                  f"(Δ {diff*100:.1f}pp)  [{flag} ≤{TOLERANCE*100:.0f}pp]")
            if diff > TOLERANCE:
                all_pass = False

    print("\n" + "=" * 62)
    if all_pass:
        print("All results within ±3 pp of targets. ✓")
    else:
        print("WARNING: One or more results outside ±3 pp tolerance.")
    print("=" * 62)
