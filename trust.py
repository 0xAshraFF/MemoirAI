# trust.py — C2 regional trust: fast penalty, fast reward (0.5/0.5 each).
# Matches aggressive_regional_trust.py TRUST_PARAMS['C2'] = (0.50, 0.50, 0.50, 0.50).

import numpy as np
from collections import deque

# ─── Constants ────────────────────────────────────────────────────────────────

# C2: pen_old=0.5, pen_new=0.5, rew_old=0.5, rew_new=0.5
# Both reward and penalty: tau = 0.5*tau + 0.5*recent_nn_acc
TRUST_OLD_WEIGHT     = 0.5
TRUST_NEW_WEIGHT     = 0.5
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

def update_trust(
    trust: dict,
    history: dict,
    nl: np.ndarray,
    nn_correct: np.ndarray,
    evo_correct: np.ndarray,
) -> dict:
    """
    C2 symmetric regional trust update:

      For each class region (identified by nearest-basin label):
        nn_acc  = mean(nn_correct  where nl == cls)
        evo_acc = mean(evo_correct where nl == cls)
        append (nn_acc, evo_acc) to rolling history

        recent_nn  = rolling mean of nn_acc over last ROLLING_WINDOW weeks
        recent_evo = rolling mean of evo_acc over last ROLLING_WINDOW weeks

        if recent_nn >= recent_evo:   # reward
            tau = 0.5 * tau + 0.5 * recent_nn
        else:                          # penalty (same formula for C2)
            tau = 0.5 * tau + 0.5 * recent_nn

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
        # C2: same formula for reward and penalty
        new_val = TRUST_OLD_WEIGHT * t + TRUST_NEW_WEIGHT * recent_nn
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
