# app.py — Streamlit UI for the Evolutionary Attention Memory research demo.
# Synthetic-data-based. Designed for live reviewer walkthroughs.

import streamlit as st
import numpy as np

from scenarios import SCENARIOS
from simulation import (
    make_centroids, make_drift_vectors, make_class5_centroid,
    generate_week_data, apply_drift, frozen_nn_predict,
    N_BASE_CLASSES, SEED, N_WEEKS, SPREAD,
)
from basins import (
    initialize_basins, score_sample, evolve_basins, add_basins_for_class,
    nearest_labels_batch,
)
from trust import (
    initialize_trust, make_history, add_new_class, update_trust,
    blend_predictions, effective_blend_weights,
)
from audit import generate_audit
from charts import (
    trust_bars_chart, blend_weight_chart,
    accuracy_chart, prob_comparison_chart,
)

st.set_page_config(
    page_title="EAM Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session-state helpers ────────────────────────────────────────────────────

def _init_state(scenario_key: str = "emergence") -> None:
    """Initialise (or fully reset) all session state for the chosen scenario."""
    rng = np.random.default_rng(SEED)
    centroids = make_centroids(rng)
    drift_vecs = make_drift_vectors()   # deterministic, no rng needed

    st.session_state.update(
        scenario_key=scenario_key,
        week=0,
        frozen_centroids=centroids,
        live_centroids={k: v.copy() for k, v in centroids.items()},
        drift_vectors=drift_vecs,
        basins=initialize_basins(centroids, rng),
        trust=initialize_trust(N_BASE_CLASSES),
        trust_history=make_history(N_BASE_CLASSES),
        feedback_buffer=[],     # (x, y) pairs from most recent Feed Sample
        feedback_replay=[],     # rolling list of recent feedback buffers
        history=[],
        audit=None,
        shift_activated=False,
        n_classes=N_BASE_CLASSES,
        rng=rng,
        evo_ran=False,
    )


# ─── Action functions ─────────────────────────────────────────────────────────

def _feed_week() -> None:
    """
    Generate one week of data, evaluate 60% of it, update trust.
    The remaining 40% is stored for the next Simulate Evolution call.
    """
    ss = st.session_state
    if ss.week >= N_WEEKS:
        return

    cfg = SCENARIOS[ss.scenario_key]
    week = ss.week + 1
    n_classes = ss.n_classes

    # Drift is applied BEFORE sample generation (matches original)
    if cfg["shift_type"] == "drift" and ss.shift_activated:
        ss.live_centroids = apply_drift(ss.live_centroids, ss.drift_vectors)

    # Generate eval + feedback split
    x_eval, y_eval, x_fb, y_fb = generate_week_data(
        live_centroids=ss.live_centroids,
        shift_activated=ss.shift_activated,
        scenario_cfg=cfg,
        rng=ss.rng,
        n_classes=n_classes,
    )

    # Store this week's feedback for evolution and short replay
    ss.feedback_buffer = list(zip(x_fb, y_fb.tolist())) if len(x_fb) else []
    if ss.feedback_buffer:
        ss.feedback_replay = (ss.feedback_replay + [ss.feedback_buffer])[-3:]

    # ── Evaluate eval subset (batch) ─────────────────────────────────────────

    import numpy as _np
    from simulation import nn_proba
    from basins import evo_proba_batch

    # Batch predictions for the eval subset
    nn_cents_arr = _np.array([ss.frozen_centroids[k] for k in range(len(ss.frozen_centroids))])
    nn_p_batch   = nn_proba(x_eval, nn_cents_arr)   # (N, 5)
    evo_p_batch  = evo_proba_batch(x_eval, ss.basins, n_classes)  # (N, n_classes)
    nl_batch     = nearest_labels_batch(x_eval, ss.basins)        # (N,)

    # Pad NN output to n_classes if new class appeared
    if nn_p_batch.shape[1] < n_classes:
        pad = _np.full((len(x_eval), n_classes - nn_p_batch.shape[1]), 1e-8)
        nn_p_batch = _np.hstack([nn_p_batch, pad])
        nn_p_batch /= nn_p_batch.sum(axis=1, keepdims=True)

    nn_preds  = nn_p_batch.argmax(axis=1)
    evo_preds = evo_p_batch.argmax(axis=1)

    # Hybrid predictions (per-sample, using trust)
    nn_correct_arr  = (nn_preds  == y_eval)
    evo_correct_arr = (evo_preds == y_eval)

    # Build hybrid preds for accuracy chart
    hyb_preds = _np.zeros(len(x_eval), dtype=int)
    for i in range(len(x_eval)):
        p_h = blend_predictions(nn_p_batch[i], evo_p_batch[i], ss.trust)
        hyb_preds[i] = int(_np.argmax(p_h))

    # Audit record for the last sample
    last_idx = len(x_eval) - 1
    _, last_activations = score_sample(x_eval[last_idx], ss.basins, n_classes)
    last_audit = generate_audit(
        x=x_eval[last_idx], true_class=int(y_eval[last_idx]),
        p_nn=nn_p_batch[last_idx], p_evo=evo_p_batch[last_idx],
        trust=ss.trust,
        p_hybrid=blend_predictions(nn_p_batch[last_idx], evo_p_batch[last_idx], ss.trust),
        basins=ss.basins, activations=last_activations,
        week=week,
    )

    # ── Trust update (comparative regional) ───────────────────────────────────

    ss.trust = update_trust(
        ss.trust,
        ss.trust_history,
        nl_batch,
        nn_correct_arr,
        evo_correct_arr,
        mode="comparative",
    )

    # ── Week-level accuracy summary ───────────────────────────────────────────

    nn_tot  = len(y_eval)
    nn_c    = int(nn_correct_arr.sum())
    nn_e    = int(evo_correct_arr.sum())
    nn_h    = int((hyb_preds == y_eval).sum())

    ss.history.append(dict(
        week=week,
        nn_accuracy=nn_c / nn_tot if nn_tot else 0.0,
        evo_accuracy=nn_e / nn_tot if nn_tot else 0.0,
        hybrid_accuracy=nn_h / nn_tot if nn_tot else 0.0,
        trust=dict(ss.trust),
    ))

    ss.audit = last_audit
    ss.evo_ran = False
    ss.week = week


def _activate_shift() -> None:
    """Trigger the scenario-specific domain shift."""
    ss = st.session_state
    cfg = SCENARIOS[ss.scenario_key]
    ss.shift_activated = True

    if cfg["shift_type"] == "emergence":
        from collections import deque
        from trust import ROLLING_WINDOW
        k = cfg["new_class"]                # 5
        c5 = make_class5_centroid(ss.rng)
        ss.live_centroids[k] = c5
        # Bootstrap 10 seed points around the new centroid to prime evolution
        seed_pts = c5 + ss.rng.standard_normal((10, 8)) * SPREAD
        ss.basins.extend(add_basins_for_class(k, seed_pts, ss.rng))
        add_new_class(ss.trust, k)
        ss.trust_history[k] = deque(maxlen=ROLLING_WINDOW)
        ss.n_classes = 6


def _simulate_evolution() -> None:
    """Run one offline evolutionary update on a short replay of feedback samples."""
    ss = st.session_state
    if not ss.feedback_buffer:
        return
    replay_feedback = []
    for chunk in ss.feedback_replay:
        replay_feedback.extend(chunk)
    ss.basins = evolve_basins(ss.basins, replay_feedback, ss.rng)
    ss.evo_ran = True


# ─── App layout ───────────────────────────────────────────────────────────────

if "scenario_key" not in st.session_state:
    _init_state("emergence")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("EAM Demo")
    st.caption("Synthetic-data-based · For thesis review")

    selected = st.selectbox(
        "Scenario",
        options=list(SCENARIOS.keys()),
        format_func=lambda k: SCENARIOS[k]["name"],
    )
    if selected != st.session_state.scenario_key:
        _init_state(selected)
        st.rerun()

    st.divider()

    cfg = SCENARIOS[st.session_state.scenario_key]
    week = st.session_state.week

    # Feed sample
    if st.button(
        f"Feed Sample  (week {week} → {week + 1})",
        disabled=week >= N_WEEKS,
        use_container_width=True,
        type="primary",
        help="Generate one week of synthetic samples and update predictions + trust.",
    ):
        _feed_week()
        st.rerun()

    # Activate shift
    shift_week = cfg["shift_week"]
    if cfg["shift_type"] == "drift":
        shift_label = "Activate Shift (start drift)"
    elif cfg["shift_type"] == "emergence":
        shift_label = f"Activate Shift (Class 5 at wk {shift_week})"
    else:
        shift_label = f"Activate Shift (imbalance wk {shift_week})"

    shift_available = not st.session_state.shift_activated and week >= max(0, shift_week - 1)
    if st.button(
        shift_label,
        disabled=not shift_available,
        use_container_width=True,
        help=f"Trigger the scenario event. Recommended at week {shift_week}.",
    ):
        _activate_shift()
        st.rerun()

    # Simulate evolution
    if st.button(
        "Simulate Evolution",
        disabled=not st.session_state.feedback_buffer,
        use_container_width=True,
        help="Run 5 generations of basin evolution on this week's feedback data.",
    ):
        _simulate_evolution()
        st.rerun()

    st.divider()

    if st.button("Reset Session", use_container_width=True):
        _init_state(st.session_state.scenario_key)
        st.rerun()

    st.divider()

    st.metric("Week", f"{week} / {N_WEEKS}")
    st.metric("Active Classes", st.session_state.n_classes)
    st.metric("Total Basins", len(st.session_state.basins))
    if st.session_state.shift_activated:
        st.success("Shift active")
    else:
        st.info(f"Shift at week {shift_week} — not yet activated")
    if st.session_state.evo_ran:
        st.success("Evolution ran this step")


# ── Main content ─────────────────────────────────────────────────────────────

st.title("Evolutionary Attention Memory — Research Demo")
st.caption(
    "**Synthetic-data-based demo.** All results use seed 42. "
    "For thesis reviewer use only."
)

scenario_cfg = SCENARIOS[st.session_state.scenario_key]
st.info(f"**{scenario_cfg['name']}:** {scenario_cfg['description']}")

# ── Prediction cards ─────────────────────────────────────────────────────────

audit = st.session_state.audit
col_nn, col_hybrid = st.columns(2)


def _pred_card(container, title, pred_class, confidence, correct, true_class, color, border):
    bg = "#f0fff4" if correct else "#fff5f5"
    icon = "✓" if correct else "✗"
    container.markdown(
        f"""<div style="border:2px solid {border};border-radius:10px;padding:16px;background:{bg}">
  <p style="margin:0;font-size:0.85em;color:#555;">{title}</p>
  <h2 style="margin:4px 0;color:{color};">Class {pred_class}</h2>
  <p style="margin:0;">Confidence: <b>{confidence:.1%}</b> &nbsp; {icon} {"Correct" if correct else "Wrong"}</p>
  <p style="margin:0;font-size:0.85em;color:#777;">True class: {true_class}</p>
</div>""",
        unsafe_allow_html=True,
    )


if audit:
    nn_conf = audit["p_nn"][audit["nn_pred"]] if audit["nn_pred"] < len(audit["p_nn"]) else 0.0
    hyb_conf = audit["p_hybrid"][audit["hybrid_pred"]] if audit["hybrid_pred"] < len(audit["p_hybrid"]) else 0.0

    with col_nn:
        _pred_card(col_nn, "Frozen NN Prediction",
                   audit["nn_pred"], nn_conf, audit["correct_nn"],
                   audit["true_class"], "#4C72B0", "#4C72B0")
    with col_hybrid:
        _pred_card(col_hybrid, "Hybrid Prediction (EAM)",
                   audit["hybrid_pred"], hyb_conf, audit["correct_hybrid"],
                   audit["true_class"], "#55A868", "#55A868")

    if audit["nn_pred"] != audit["hybrid_pred"]:
        st.warning(
            f"Predictions diverge: Frozen NN → Class {audit['nn_pred']} | "
            f"Hybrid → Class {audit['hybrid_pred']}  (true: Class {audit['true_class']})"
        )
else:
    col_nn.markdown(
        """<div style="border:2px solid #4C72B0;border-radius:10px;padding:16px;background:#f8f9fa">
<p style="color:#888;">Frozen NN — feed a sample to populate</p></div>""",
        unsafe_allow_html=True)
    col_hybrid.markdown(
        """<div style="border:2px solid #55A868;border-radius:10px;padding:16px;background:#f8f9fa">
<p style="color:#888;">Hybrid (EAM) — feed a sample to populate</p></div>""",
        unsafe_allow_html=True)

st.write("")

# ── Charts row 1 ─────────────────────────────────────────────────────────────

col_acc, col_prob = st.columns([3, 2])
with col_acc:
    st.plotly_chart(accuracy_chart(st.session_state.history), use_container_width=True)
with col_prob:
    if audit:
        st.plotly_chart(
            prob_comparison_chart(
                audit["p_nn"], audit["p_evo"], audit["p_hybrid"],
                st.session_state.n_classes,
            ),
            use_container_width=True,
        )
    else:
        st.caption("Probability vectors appear here after first sample.")

# ── Charts row 2 ─────────────────────────────────────────────────────────────

col_trust, col_blend = st.columns(2)
with col_trust:
    st.plotly_chart(trust_bars_chart(st.session_state.trust), use_container_width=True)
with col_blend:
    st.plotly_chart(
        blend_weight_chart(st.session_state.trust, st.session_state.n_classes),
        use_container_width=True,
    )

# ── Audit panel ───────────────────────────────────────────────────────────────

st.divider()
st.subheader("Audit Panel — Last Prediction")

if audit:
    a = audit
    n_c = st.session_state.n_classes

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Class", f"Class {a['true_class']}")
    c2.metric("Hybrid Prediction", f"Class {a['hybrid_pred']}",
              delta="✓ Correct" if a["correct_hybrid"] else "✗ Wrong")
    c3.metric("Trust Used (τ)", f"{a['tau_used']:.3f}")
    c4.metric("Week", a["week"])

    with st.expander("Probability vectors", expanded=True):
        ca, cb, cc = st.columns(3)
        p_nn_full = list(a["p_nn"]) + [0.0] * max(0, n_c - len(a["p_nn"]))
        with ca:
            st.markdown("**Frozen NN  (p_nn)**")
            for k in range(n_c):
                bar = "█" * int(p_nn_full[k] * 20)
                st.text(f"C{k}: {p_nn_full[k]:.4f}  {bar}")
        with cb:
            st.markdown("**Evo Memory  (p_evo)**")
            for k in range(n_c):
                bar = "█" * int(a["p_evo"][k] * 20)
                st.text(f"C{k}: {a['p_evo'][k]:.4f}  {bar}")
        with cc:
            st.markdown("**Hybrid  (p_hybrid)**")
            for k in range(n_c):
                bar = "█" * int(a["p_hybrid"][k] * 20)
                st.text(f"C{k}: {a['p_hybrid'][k]:.4f}  {bar}")

    with st.expander("Blend weights & trust", expanded=True):
        nn_w, evo_w = effective_blend_weights(st.session_state.trust, n_c)
        cols = st.columns(n_c)
        for k, col in enumerate(cols):
            col.metric(
                f"Class {k}",
                f"τ = {st.session_state.trust.get(k, 0.0):.3f}",
                help=f"NN weight: {nn_w[k]:.3f}  |  Evo weight: {evo_w[k]:.3f}",
            )
        st.caption(
            f"Hybrid-predicted class {a['hybrid_pred']}: "
            f"NN weight τ² = {a['nn_weight']:.3f}, "
            f"Evo weight 1−τ² = {a['evo_weight']:.3f}"
        )

    with st.expander("Top-3 contributing basins", expanded=True):
        if a["top_basins"]:
            cols = st.columns(len(a["top_basins"]))
            for col, b in zip(cols, a["top_basins"]):
                col.markdown(f"**Class {b['class']}**")
                col.text(f"Activation: {b['activation']:.4f}")
                col.text(f"Weight:     {b['weight']:.4f}")
                col.text(f"Radius:     {b['radius']:.4f}")
        else:
            st.caption("No basin activations recorded.")

    with st.expander("Correctness flags"):
        st.write({
            "Frozen NN correct": a["correct_nn"],
            "Evo memory correct": a["correct_evo"],
            "Hybrid correct": a["correct_hybrid"],
            "True class": a["true_class"],
            "Week": a["week"],
        })

else:
    st.caption("Feed at least one sample to populate the audit panel.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "EAM Demo · Synthetic data only · Seed 42 · "
    "Trust update: τ_k = 0.5·τ_k + 0.5·rolling_nn_acc_k (3-week window) · "
    "Blend: p_hybrid[k] = τ_k²·p_nn[k] + (1−τ_k²)·p_evo[k]"
)
