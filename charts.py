# charts.py — Plotly figure builders for the EAM demo UI.

import numpy as np
import plotly.graph_objects as go

# Consistent colour palette across all charts
_NN_COLOR = "#4C72B0"       # steel-blue  → frozen NN
_EVO_COLOR = "#DD8452"      # warm-orange → evolutionary memory
_HYBRID_COLOR = "#55A868"   # green       → hybrid prediction

# Per-class colours (up to 7 classes)
_CLASS_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3",
]

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
    margin=dict(l=10, r=10, t=36, b=10),
)


# ─── Trust bars ───────────────────────────────────────────────────────────────

def trust_bars_chart(trust: dict) -> go.Figure:
    """Horizontal bar chart of per-class trust values (τ_k)."""
    classes = sorted(trust.keys())
    labels = [f"Class {k}" for k in classes]
    values = [trust[k] for k in classes]
    colors = [_CLASS_COLORS[k % len(_CLASS_COLORS)] for k in classes]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="auto",
        hovertemplate="Class %{y}: τ = %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Per-Class Trust (τ_k)",
        xaxis=dict(range=[0, 1], title="Trust value", gridcolor="#e0e0e0"),
        yaxis=dict(title=""),
        height=max(160, 40 * len(classes) + 60),
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ─── Blend weight chart ───────────────────────────────────────────────────────

def blend_weight_chart(trust: dict, n_classes: int) -> go.Figure:
    """
    Stacked bar showing the NN weight (τ²) versus evo weight (1−τ²) per class.
    Visualises the effective contribution of each component to the hybrid prediction.
    """
    classes = list(range(n_classes))
    labels = [f"C{k}" for k in classes]
    nn_w = [trust.get(k, 0.0) ** 2 for k in classes]
    evo_w = [1.0 - w for w in nn_w]

    fig = go.Figure([
        go.Bar(
            name="Frozen NN  (τ²)",
            x=labels, y=nn_w,
            marker_color=_NN_COLOR,
            hovertemplate="C%{x}: NN weight = %{y:.3f}<extra></extra>",
        ),
        go.Bar(
            name="Evo Memory  (1−τ²)",
            x=labels, y=evo_w,
            marker_color=_EVO_COLOR,
            hovertemplate="C%{x}: Evo weight = %{y:.3f}<extra></extra>",
        ),
    ])
    fig.update_layout(
        barmode="stack",
        title="Blend Weights per Class",
        yaxis=dict(range=[0, 1], title="Weight", gridcolor="#e0e0e0"),
        xaxis=dict(title="Class"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=240,
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ─── Accuracy over time ───────────────────────────────────────────────────────

def accuracy_chart(history: list) -> go.Figure:
    """
    Line chart of per-week accuracy for all three predictors.
    The hybrid line is drawn thicker to make its trajectory visually distinct.
    """
    if not history:
        fig = go.Figure()
        fig.update_layout(
            title="Accuracy by Week (feed samples to populate)",
            height=280, **_LAYOUT_DEFAULTS,
        )
        return fig

    weeks = [h["week"] for h in history]
    nn_acc = [h["nn_accuracy"] for h in history]
    evo_acc = [h["evo_accuracy"] for h in history]
    hybrid_acc = [h["hybrid_accuracy"] for h in history]

    fig = go.Figure([
        go.Scatter(
            x=weeks, y=nn_acc, name="Frozen NN",
            line=dict(color=_NN_COLOR, dash="dash", width=1.5),
            hovertemplate="Week %{x}: NN = %{y:.1%}<extra></extra>",
        ),
        go.Scatter(
            x=weeks, y=evo_acc, name="Evo Memory",
            line=dict(color=_EVO_COLOR, dash="dot", width=1.5),
            hovertemplate="Week %{x}: Evo = %{y:.1%}<extra></extra>",
        ),
        go.Scatter(
            x=weeks, y=hybrid_acc, name="Hybrid (EAM)",
            line=dict(color=_HYBRID_COLOR, width=3),
            hovertemplate="Week %{x}: Hybrid = %{y:.1%}<extra></extra>",
        ),
    ])
    fig.update_layout(
        title="Rolling Accuracy by Week",
        xaxis=dict(title="Week", gridcolor="#e0e0e0"),
        yaxis=dict(
            range=[max(0, min(nn_acc + evo_acc + hybrid_acc) - 0.05), 1.02],
            title="Accuracy",
            tickformat=".0%",
            gridcolor="#e0e0e0",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=280,
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ─── Probability vector comparison ───────────────────────────────────────────

def prob_comparison_chart(
    p_nn: list,
    p_evo: list,
    p_hybrid: list,
    n_classes: int,
) -> go.Figure:
    """
    Grouped bar chart comparing probability vectors of all three predictors
    for the last processed sample.
    """
    labels = [f"C{k}" for k in range(n_classes)]

    # Pad p_nn to n_classes (frozen NN only has 5 outputs)
    p_nn_padded = list(p_nn) + [0.0] * max(0, n_classes - len(p_nn))

    fig = go.Figure([
        go.Bar(
            name="Frozen NN",
            x=labels, y=p_nn_padded[:n_classes],
            marker_color=_NN_COLOR,
            hovertemplate="C%{x}: %{y:.3f}<extra>Frozen NN</extra>",
        ),
        go.Bar(
            name="Evo Memory",
            x=labels, y=list(p_evo)[:n_classes],
            marker_color=_EVO_COLOR,
            hovertemplate="C%{x}: %{y:.3f}<extra>Evo Memory</extra>",
        ),
        go.Bar(
            name="Hybrid (EAM)",
            x=labels, y=list(p_hybrid)[:n_classes],
            marker_color=_HYBRID_COLOR,
            hovertemplate="C%{x}: %{y:.3f}<extra>Hybrid</extra>",
        ),
    ])
    fig.update_layout(
        barmode="group",
        title="Probability Vectors — Last Sample",
        xaxis=dict(title="Class", gridcolor="#e0e0e0"),
        yaxis=dict(range=[0, 1.05], title="Probability", gridcolor="#e0e0e0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=280,
        **_LAYOUT_DEFAULTS,
    )
    return fig
