# EAM Demo

Local-first research demo for **Evolutionary Attention Memory (EAM)**.  
Synthetic-data-based. Designed for thesis reviewer walkthroughs.

## What it shows

| Component | Description |
|---|---|
| Frozen NN | Nearest-centroid classifier (T=0.8), 5 fixed output classes |
| Evolutionary memory | 30 Gaussian-activation basins, evolved offline |
| Trust mechanism | Legacy C2 trust plus a comparative regional trust mode |
| Hybrid prediction | `p = τ²·p_nn + (1−τ²)·p_evo` — normalised per class |

Three scenarios demonstrate the mechanism under domain shift:

- **Class Emergence** — a 6th class appears at week 8; the frozen NN assigns it zero probability
- **Class Imbalance** — class 0 becomes 10× more frequent at week 5
- **Distributional Drift** — all class centroids migrate 0.15 units/week for 20 weeks

## Setup

```bash
pip install streamlit numpy plotly
```

Tested with Python 3.10+, NumPy 1.26, Streamlit 1.32, Plotly 5.20.

## Run

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.  
No internet connection required after install.

## Verify accuracy targets

```bash
python verify.py
```

Expected output (seed 42, C2 variant):

```
[Class Emergence]  hybrid ~86.1%  NN ~88.8%  evo ~78.8%
[Class Imbalance]  hybrid ~99.5%  NN ~99.7%  evo ~95.1%
[Distributional Drift]  hybrid ~96.9%  NN ~95.8%  evo ~93.5%
```

All results within ±3 pp of these targets (see paper Table 1).

## Compare baseline vs improved version

```bash
python compare_versions.py
```

This runs a 5-seed comparison across three variants:

- `legacy_paper` — original optimistic paper-style benchmark
- `online_c2` — fairer online split using the original C2 trust
- `online_compare` — fairer online split with comparative trust and 3-week replay evolution

Current 5-seed summary:

| Scenario | `online_c2` hybrid | `online_compare` hybrid | Change |
|---|---:|---:|---:|
| Class Emergence | 86.6% ± 2.4 | 86.6% ± 0.6 | +0.1 pp |
| Class Imbalance | 99.6% ± 0.1 | 99.6% ± 0.1 | -0.1 pp |
| Distributional Drift | 96.4% ± 0.5 | 96.6% ± 0.4 | +0.1 pp |
| Overall mean | 94.2% | 94.3% | +0.1 pp |

Additional stability gain:

- Drift evo memory improved from `80.0% ± 17.2` to `87.5% ± 8.8` under the fair online benchmark.

## Walkthrough (≈2 minutes)

1. Select **Class Emergence** in the sidebar.
2. Click **Feed Sample** 7 times to build baseline trust.  
   Observe: both predictions agree, trust bars sit at ~0.5 and rising.
3. Click **Activate Shift** — Class 5 is injected.  
   Observe: Class 5 trust initialises at 0.0; blend weight shifts to evo for Class 5.
4. Feed a few more weeks. Observe the hybrid diverging from the frozen NN on Class 5 samples.
5. Click **Simulate Evolution**.  
   Observe: evo accuracy rises as basins cover Class 5.
6. Inspect the **Audit Panel** to see probability vectors, trust, and top-3 contributing basins.
7. Repeat with **Distributional Drift** to see trust decay as NN centroids go stale.

## File structure

```
app.py          Streamlit UI
simulation.py   Synthetic data generation and frozen NN
basins.py       Basin representation, scoring, evolution
trust.py        Trust initialisation, EMA update, prediction blending
audit.py        Audit-record builder
charts.py       Plotly figure builders
scenarios.py    Scenario definitions
verify.py       Headless accuracy verification script
compare_versions.py  Multi-seed baseline vs improved comparison
requirements.txt
```

## Parameters (match paper, C2 variant)

| Parameter | Value |
|---|---|
| Latent dimension | 8 |
| Base classes | 5 (labels 0–4) |
| Samples / week | 200 |
| Simulation duration | 20 weeks |
| Basins | 6 / class = 30 total |
| Evolution generations | 6 |
| Elite preservation | top 30% per class |
| Weight decay | 0.4% / generation |
| Temperature T | 0.8 |
| Trust λ | 0.5 (symmetric EMA) |
| Trust initial (known) | 0.5 |
| Trust initial (new) | 0.0 |
| Rolling window | 3 weeks |
| Random seed | 42 |

## Limitations

- All results are on synthetic Gaussian clusters. Real-data validation is future work.
- Trust region assignment uses class label; more formal analysis is listed as future work in the paper.
- Evolution runs in pure Python (O(n) basin scan); KD-tree indexing would reduce latency.
- The original paper-aligned verification is still single-seed and optimistic by design.
- The new multi-seed benchmark is fairer, but still uses synthetic data rather than real deployment traces.
