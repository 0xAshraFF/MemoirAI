#!/usr/bin/env python3
"""
compare_versions.py — Compare the original MemoirAI benchmark against
fairer online variants and an improved comparative-trust variant.

Variants
--------
legacy_paper:
    Original verify.py protocol. Optimistic because it evolves on all data
    and evaluates on the same samples. Emergence stays effectively 5-class.

online_c2:
    Fairer online protocol. Evaluates on a held-out split each week, evolves
    only on feedback data, and supports a true 6th class at emergence time.
    Trust uses the original C2 updater.

online_compare:
    Same fair online protocol, but uses comparative trust that reacts to the
    NN-vs-evo performance gap.
"""

import numpy as np
from collections import deque

from basins import (
    add_basins_for_class,
    evolve_basins,
    evo_proba_batch,
    initialize_basins,
    nearest_labels_batch,
)
from scenarios import SCENARIOS
from simulation import (
    N_BASE_CLASSES,
    N_WEEKS,
    SEED,
    SPREAD,
    apply_drift,
    generate_week_data,
    make_centroids,
    make_class5_centroid,
    make_drift_vectors,
    nn_proba,
)
from trust import add_new_class, initialize_trust, make_history, update_trust, blend_batch
from verify import run_scenario as run_legacy_scenario


SEEDS = [42, 43, 44, 45, 46]
SCENARIO_KEYS = ["emergence", "imbalance", "drift"]


def _activate_shift(state: dict) -> None:
    cfg = state["cfg"]
    state["shift_activated"] = True

    if cfg["shift_type"] == "emergence":
        k = cfg["new_class"]
        c5 = make_class5_centroid(state["rng"])
        state["live_centroids"][k] = c5
        seed_pts = c5 + state["rng"].standard_normal((10, 8)) * SPREAD
        state["basins"].extend(add_basins_for_class(k, seed_pts, state["rng"]))
        add_new_class(state["trust"], k)
        state["history"][k] = state["history"][0].__class__(maxlen=state["history"][0].maxlen)
        state["n_classes"] = 6


def run_online_scenario(
    scenario_key: str,
    seed: int,
    trust_mode: str,
    replay_weeks: int = 1,
) -> dict:
    rng = np.random.default_rng(seed)
    frozen_centroids = make_centroids(rng)
    live_centroids = {k: v.copy() for k, v in frozen_centroids.items()}
    drift_vectors = make_drift_vectors()
    cfg = SCENARIOS[scenario_key]

    state = dict(
        cfg=cfg,
        rng=rng,
        basins=initialize_basins(frozen_centroids, rng),
        trust=initialize_trust(N_BASE_CLASSES),
        history=make_history(N_BASE_CLASSES),
        frozen_centroids=frozen_centroids,
        live_centroids=live_centroids,
        drift_vectors=drift_vectors,
        n_classes=N_BASE_CLASSES,
        shift_activated=(cfg["shift_type"] == "drift"),
    )

    totals = {"nn": 0, "evo": 0, "hybrid": 0, "n": 0}
    replay = deque(maxlen=replay_weeks)

    for week_idx in range(N_WEEKS):
        week_num = week_idx + 1

        if (
            not state["shift_activated"]
            and week_num >= cfg["shift_week"]
        ):
            _activate_shift(state)

        if cfg["shift_type"] == "drift" and state["shift_activated"]:
            state["live_centroids"] = apply_drift(
                state["live_centroids"],
                state["drift_vectors"],
            )

        x_eval, y_eval, x_fb, y_fb = generate_week_data(
            live_centroids=state["live_centroids"],
            shift_activated=state["shift_activated"],
            scenario_cfg=cfg,
            rng=state["rng"],
            n_classes=state["n_classes"],
        )

        nn_cents_arr = np.array(
            [state["frozen_centroids"][k] for k in range(len(state["frozen_centroids"]))]
        )
        nn_p = nn_proba(x_eval, nn_cents_arr)
        evo_p = evo_proba_batch(x_eval, state["basins"], state["n_classes"])
        nl = nearest_labels_batch(x_eval, state["basins"])

        if nn_p.shape[1] < state["n_classes"]:
            pad = np.full((len(x_eval), state["n_classes"] - nn_p.shape[1]), 1e-8)
            nn_p = np.hstack([nn_p, pad])
            nn_p /= nn_p.sum(axis=1, keepdims=True)

        blended = blend_batch(nn_p, evo_p, global_trust=0.8, regional=state["trust"], nl=nl)

        nn_preds = nn_p.argmax(axis=1)
        evo_preds = evo_p.argmax(axis=1)
        hybrid_preds = blended.argmax(axis=1)

        totals["nn"] += int((nn_preds == y_eval).sum())
        totals["evo"] += int((evo_preds == y_eval).sum())
        totals["hybrid"] += int((hybrid_preds == y_eval).sum())
        totals["n"] += len(y_eval)

        state["trust"] = update_trust(
            state["trust"],
            state["history"],
            nl,
            nn_preds == y_eval,
            evo_preds == y_eval,
            mode=trust_mode,
        )

        if len(x_fb):
            replay.append(list(zip(x_fb, y_fb.tolist())))
            feedback = []
            for chunk in replay:
                feedback.extend(chunk)
            state["basins"] = evolve_basins(state["basins"], feedback, state["rng"])

    n = totals["n"]
    return {k: totals[k] / n for k in ["nn", "evo", "hybrid"]}


def collect_results() -> dict:
    variants = {
        "legacy_paper": lambda key, seed: _run_legacy(key, seed),
        "online_c2": lambda key, seed: run_online_scenario(key, seed, "legacy", replay_weeks=1),
        "online_compare": lambda key, seed: run_online_scenario(
            key,
            seed,
            "comparative",
            replay_weeks=3,
        ),
    }

    results = {variant: {key: [] for key in SCENARIO_KEYS} for variant in variants}
    for variant, runner in variants.items():
        for seed in SEEDS:
            for key in SCENARIO_KEYS:
                results[variant][key].append(runner(key, seed))
    return results


def _run_legacy(scenario_key: str, seed: int) -> dict:
    np.random.seed(seed)
    return run_legacy_scenario(scenario_key)


def _summary(values: list, metric: str) -> tuple:
    arr = np.array([v[metric] for v in values], dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def _format_pct(mean: float, std: float) -> str:
    return f"{mean * 100:.1f}% ± {std * 100:.1f}"


def print_tables(results: dict) -> None:
    print("=" * 86)
    print("MemoirAI Comparison — 5 seeds, mean accuracy ± std-dev")
    print("=" * 86)

    for key in SCENARIO_KEYS:
        print(f"\n[{SCENARIOS[key]['name']}]")
        print(f"{'Variant':<18} {'Hybrid':<16} {'NN':<16} {'Evo':<16} {'Δ vs online_c2':<14}")
        print("-" * 86)

        baseline_mean = _summary(results["online_c2"][key], "hybrid")[0]
        for variant in ["legacy_paper", "online_c2", "online_compare"]:
            hybrid_mean, hybrid_std = _summary(results[variant][key], "hybrid")
            nn_mean, nn_std = _summary(results[variant][key], "nn")
            evo_mean, evo_std = _summary(results[variant][key], "evo")
            delta = hybrid_mean - baseline_mean
            delta_text = "baseline" if variant == "online_c2" else f"{delta * 100:+.1f}pp"
            print(
                f"{variant:<18} "
                f"{_format_pct(hybrid_mean, hybrid_std):<16} "
                f"{_format_pct(nn_mean, nn_std):<16} "
                f"{_format_pct(evo_mean, evo_std):<16} "
                f"{delta_text:<14}"
            )

    print("\n" + "=" * 86)
    print("Overall Hybrid Mean Across Scenarios")
    print("=" * 86)
    print(f"{'Variant':<18} {'Hybrid Mean':<16}")
    print("-" * 40)
    for variant in ["legacy_paper", "online_c2", "online_compare"]:
        scenario_means = [
            _summary(results[variant][key], "hybrid")[0]
            for key in SCENARIO_KEYS
        ]
        arr = np.array(scenario_means, dtype=float)
        print(f"{variant:<18} {arr.mean() * 100:.1f}%")


if __name__ == "__main__":
    print(f"Using seeds: {SEEDS} (baseline seed is {SEED})")
    print_tables(collect_results())
