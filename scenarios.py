# scenarios.py — Scenario definitions.
# Timings match aggressive_regional_trust.py exactly (0-indexed week thresholds):
#   emergence:  class 5 appears at wk >= 8  (0-indexed)  → 1-indexed week 9
#   imbalance:  class 0 10x at wk >= 5  (0-indexed)  → 1-indexed week 6
#   drift:      all centroids drift from week 1 (shift_week=1, always active)

SCENARIOS = {
    "emergence": {
        "name": "Class Emergence",
        "description": (
            "A new class (Class 5) appears at week 9. "
            "The frozen NN has only 5 output neurons and assigns near-zero probability to Class 5. "
            "The evolutionary memory must discover and cover Class 5 from scratch."
        ),
        "shift_week": 9,          # 1-indexed; matches wk<8 (0-indexed) threshold in original
        "shift_type": "emergence",
        "new_class": 5,
        "imbalanced_class": None,
        "imbalance_factor": 1,
        "drift_rate": 0.0,
    },
    "imbalance": {
        "name": "Class Imbalance",
        "description": (
            "Class 0 becomes 10× more frequent at week 6. "
            "The frozen NN geometry is unchanged; trust adapts to per-class performance shifts."
        ),
        "shift_week": 6,          # 1-indexed; matches wk<5 (0-indexed) threshold in original
        "shift_type": "imbalance",
        "new_class": None,
        "imbalanced_class": 0,
        "imbalance_factor": 10,
        "drift_rate": 0.0,
    },
    "drift": {
        "name": "Distributional Drift",
        "description": (
            "All class centroids migrate 0.15 units per week. "
            "The frozen NN uses fixed centroids so its accuracy degrades over time. "
            "The evolutionary memory adapts its basins to track the shift."
        ),
        "shift_week": 1,          # drift starts from week 1 (always active)
        "shift_type": "drift",
        "new_class": None,
        "imbalanced_class": None,
        "imbalance_factor": 1,
        "drift_rate": 0.15,
    },
}
