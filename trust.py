# trust.py — Regional trust updates and prediction blending.
# Includes both the legacy C2 updater and a comparative updater that
# responds to NN-vs-memory performance instead of smoothing NN accuracy alone.

import numpy as np
from collections import deque

# ─── Constants ────────────────────────────────────────────────────────────────

# Legacy C2: pen_old=0.5, pen_new=0.5, rew_old=0.5, rew_new=0.5
LEGACY_TRUST_OLD_WEIGHT = 0.5
LEGACY_TRUST_NEW_WEIGHT = 0.5

# Comparative trust:
# - only reduce NN trust when evo beats it by a meaningful margin
# - otherwise stay close to the stronger legacy C2 behavior
COMPARE_MARGIN            = 0.05
COMPARE_PENALTY_STRENGTH  = 0.60
COMPARE_DECAY_OLD_WEIGHT  = 0.55
COMPARE_DECAY_NEW_WEIGHT  = 0.45
COMPARE_RISE_OLD_WEIGHT   = 0.60
COMPARE_RISE_NEW_WEIGHT   = 0.40

TRUST_INITIAL_KNOWN  = 0.5
TRUST_INITIAL_NEW    = 0.0
ROLLING_WINDOW       = 3


# ─── Initialisation ───────────────────────────────────────────────────────────

def initialize_trust(n_classes: int = 5) -> dict:
    """Return {k: 0.5} for each base class."""
    return {k: TRUST_INITIAL_KNOWN for k in range(n_classes)}


def make_history(n_classes: int) -> dict:
    """Create rolling-window history deques for C2 trust update."""
    return {k: deque(maxlen=ROLLING_WINDOW) for k in range(n_classes)}


def add_new_class(trust: dict, class_id: int) -> None:
    """Register a newly discovered class with trust = 0.0 (in-place)."""
    trust[class_id] = TRUST_INITIAL_NEW


# ─── Trust update (C2 regional) ───────────────────────────────────────────────

def _relative_nn_target(recent_nn: float, recent_evo: float) -> float:
    """
    Convert recent regional performance into a trust target for the frozen NN.

    The baseline target is recent NN accuracy. If evo only barely wins, we do
    not overreact. If evo wins by more than COMPARE_MARGIN, reduce the target
    in proportion to that gap. This keeps the blend stable while still letting
    new or drifting regions move toward the adaptive memory.
    """
    gap = max(0.0, recent_evo - recent_nn - COMPARE_MARGIN)
    target = recent_nn * (1.0 - COMPARE_PENALTY_STRENGTH * gap)
    return float(np.clip(target, 0.0, 1.0))


def update_trust(
    trust: dict,
    history: dict,
    nl: np.ndarray,
    nn_correct: np.ndarray,
    evo_correct: np.ndarray,
    mode: str = "comparative",
) -> dict:
    """
    Regional trust update.

    Modes
    -----
    legacy:
        Original C2 updater from aggressive_regional_trust.py.
        Trust is just an EMA of recent NN regional accuracy.

    comparative:
        Trust starts from recent NN regional accuracy and only drops when evo
        beats the NN by more than a small margin. The update is smoothed
        asymmetrically so shifts away from stale NN behavior remain controlled.

    Mutates history in-place; returns a new trust dict.

    Parameters
    ----------
    trust    : {cls: float} current trust values
    history  : {cls: deque} rolling (nn_acc, evo_acc) pairs — mutated in-place
    nl       : (N,) nearest-basin class label per sample
    nn_correct  : (N,) bool — was frozen NN correct for each sample?
    evo_correct : (N,) bool — was evo memory correct for each sample?
    """
    new_trust = dict(trust)
    for cls in trust:
        mask = nl == cls
        if not mask.any():
            continue
        nn_acc  = float(nn_correct[mask].mean())
        evo_acc = float(evo_correct[mask].mean())
        history[cls].append((nn_acc, evo_acc))

        recent_nn  = float(np.mean([h[0] for h in history[cls]]))
        recent_evo = float(np.mean([h[1] for h in history[cls]]))

        t = trust[cls]
        if mode == "legacy":
            new_val = (
                LEGACY_TRUST_OLD_WEIGHT * t
                + LEGACY_TRUST_NEW_WEIGHT * recent_nn
            )
        elif mode == "comparative":
            target = _relative_nn_target(recent_nn, recent_evo)
            if target < t:
                old_w = COMPARE_DECAY_OLD_WEIGHT
                new_w = COMPARE_DECAY_NEW_WEIGHT
            else:
                old_w = COMPARE_RISE_OLD_WEIGHT
                new_w = COMPARE_RISE_NEW_WEIGHT
            new_val = old_w * t + new_w * target
        else:
            raise ValueError(f"Unknown trust mode: {mode}")

        new_trust[cls] = float(np.clip(new_val, 0.0, 1.0))

    return new_trust


# ─── Prediction blending ──────────────────────────────────────────────────────

def blend_predictions(
    p_nn: np.ndarray,
    p_evo: np.ndarray,
    trust: dict,
) -> np.ndarray:
    """
    Single-sample per-class blend for app.py:
        p_final[k] = tau_k² × p_nn[k] + (1 − tau_k²) × p_evo[k]

    p_nn covers original classes only; p_evo covers all active classes.
    Result renormalised to sum to 1.
    """
    n_classes = len(p_evo)
    p_final = np.zeros(n_classes)
    for k in range(n_classes):
        tau    = trust.get(k, 0.0)
        w      = tau ** 2
        p_nn_k = float(p_nn[k]) if k < len(p_nn) else 0.0
        p_final[k] = w * p_nn_k + (1.0 - w) * float(p_evo[k])
    total = p_final.sum()
    if total > 1e-12:
        p_final /= total
    return p_final


def blend_batch(
    nn_p: np.ndarray,
    evo_p: np.ndarray,
    global_trust: float,
    regional: dict,
    nl: np.ndarray,
) -> np.ndarray:
    """
    Batch blend for verify.py — matches aggressive_regional_trust.py blend():
        w[i]  = regional[nl[i]]²
        out   = w[:,None]*nn_p + (1-w[:,None])*evo_p
    """
    reg = np.array([regional.get(int(l), global_trust) for l in nl])
    w   = reg ** 2
    return w[:, None] * nn_p + (1 - w[:, None]) * evo_p


def effective_blend_weights(trust: dict, n_classes: int) -> tuple:
    """Return (nn_weights, evo_weights) arrays for visualisation."""
    nn_w = np.array([trust.get(k, 0.0) ** 2 for k in range(n_classes)])
    return nn_w, 1.0 - nn_w
