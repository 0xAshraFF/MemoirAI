# audit.py — Build the audit record for a single prediction.
# The audit panel is the primary tool for reviewer inspection.

import numpy as np
from basins import top_contributing_basins


def generate_audit(
    x: np.ndarray,
    true_class: int,
    p_nn: np.ndarray,
    p_evo: np.ndarray,
    trust: dict,
    p_hybrid: np.ndarray,
    basins: list,
    activations: np.ndarray,
    week: int,
) -> dict:
    """
    Produce a serialisable audit record for one prediction.

    Fields returned:
        week, true_class
        p_nn, p_evo, p_hybrid   — probability vectors (lists)
        nn_pred, evo_pred, hybrid_pred
        tau_used                — trust value for the hybrid-predicted class
        nn_weight, evo_weight   — effective blend weights for that class
        correct_nn, correct_evo, correct_hybrid
        top_basins              — top-3 contributing basins
    """
    nn_pred = int(np.argmax(p_nn))
    evo_pred = int(np.argmax(p_evo))
    hybrid_pred = int(np.argmax(p_hybrid))

    tau = trust.get(hybrid_pred, 0.0)
    tau_sq = tau ** 2

    return {
        "week": int(week),
        "true_class": int(true_class),
        # Probability vectors (padded to the same length for display)
        "p_nn": p_nn.tolist(),
        "p_evo": p_evo.tolist(),
        "p_hybrid": p_hybrid.tolist(),
        # Predictions
        "nn_pred": nn_pred,
        "evo_pred": evo_pred,
        "hybrid_pred": hybrid_pred,
        # Trust used for the hybrid-predicted class
        "tau_used": float(tau),
        "nn_weight": float(tau_sq),
        "evo_weight": float(1.0 - tau_sq),
        # Correctness
        "correct_nn": nn_pred == int(true_class),
        "correct_evo": evo_pred == int(true_class),
        "correct_hybrid": hybrid_pred == int(true_class),
        # Top contributing basins
        "top_basins": top_contributing_basins(basins, activations, n=3),
    }
