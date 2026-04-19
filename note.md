# MemoirAI Journal

## Project Summary

MemoirAI is currently a research demo for an adaptive AI architecture:

- A frozen classifier handles the patterns it already knows.
- An evolutionary memory layer adapts through basin updates.
- A trust layer blends frozen-model and memory predictions per class.
- The whole system is tested under three synthetic shift scenarios:
  class emergence, class imbalance, and distributional drift.

This is best understood as a proof-of-concept for "frozen intelligence + adaptive memory + trust-gated blending under shift."

## Initial Analysis Notes

Plain-English interpretation of the project:

- The frozen model is the stable long-term memory.
- The basin memory is the flexible short-term adaptation layer.
- Trust decides when to believe the stable model and when to lean on the adaptive layer.
- The app is designed for reviewer walkthroughs, not production deployment.

Main strengths identified:

- Clear research story and architecture.
- Reproducible baseline numbers.
- Good reviewer/demo framing with audit visibility.
- Reasonable motivation for adaptive systems under shift.

Main weaknesses identified:

- Synthetic data only.
- Original benchmark used a single seed.
- The original C2 trust rule mostly tracked NN accuracy rather than truly comparing NN vs evo performance.
- The original paper-style benchmark is optimistic because it evolves on all data and evaluates on the same data.
- Emergence handling differs between the original experiment and the live app.
- Evo memory showed instability under drift in fairer online evaluation.

## Baseline Numbers Confirmed

The original verification script was run and matched the documented baseline targets:

- Emergence: hybrid 86.2%, NN 88.9%, evo 78.8%
- Imbalance: hybrid 99.6%, NN 99.7%, evo 95.3%
- Drift: hybrid 96.9%, NN 95.8%, evo 93.5%

Interpretation:

- The baseline repo is internally consistent.
- The numbers make sense for a synthetic simulation.
- They should not be treated as evidence of real-world performance yet.

## Changes Made In This Session

### 1. Preserved the original benchmark path

Reason:

- Keep the paper/demo baseline reproducible.
- Avoid breaking the documented seed-42 results.

What changed:

- `verify.py` now explicitly calls `update_trust(..., mode="legacy")`.

Outcome:

- The original accuracy targets still reproduce exactly.

### 2. Added a comparative trust mode

Reason:

- Fix the weakness where trust was effectively just smoothing NN accuracy.

What changed:

- Added `legacy` and `comparative` trust modes in `trust.py`.
- The new comparative mode:
  - starts from recent NN regional accuracy
  - only reduces trust when evo beats NN by more than a small margin
  - uses controlled asymmetric smoothing

Outcome:

- Trust now has a more meaningful interpretation.
- It reacts to relative NN vs evo performance instead of ignoring evo.

### 3. Added a short replay buffer for evolution

Reason:

- Evo memory was too unstable under online drift when trained only on one week of feedback.

What changed:

- `app.py` now stores a rolling 3-week replay of feedback.
- Evolution in the app now uses this short replay instead of only the latest feedback batch.

Outcome:

- More stable adaptation behavior.
- Better drift robustness in the fairer comparison setup.

### 4. Updated the app to use comparative trust

Reason:

- The live demo should reflect the more defensible trust story.

What changed:

- `app.py` now updates trust using `mode="comparative"`.

Outcome:

- The app behavior is closer to the research claim that trust compares the stable and adaptive paths.

### 5. Added a new comparison benchmark

Reason:

- Needed a fair before/after table instead of only the optimistic paper-style benchmark.

What changed:

- Added `compare_versions.py`.
- It compares three variants:
  - `legacy_paper`
  - `online_c2`
  - `online_compare`
- It runs 5 seeds: 42, 43, 44, 45, 46.

Outcome:

- We now have a more honest benchmark path.
- We can compare the original and improved variants side by side.

## Comparison Results From This Session

### Fairer Multi-Seed Comparison

#### Class Emergence

- `legacy_paper`: hybrid 87.2% ± 0.6
- `online_c2`: hybrid 86.6% ± 2.4
- `online_compare`: hybrid 86.6% ± 0.6

Takeaway:

- Improved trust did not raise mean accuracy much here.
- It reduced variance substantially relative to the plain online C2 setup.

#### Class Imbalance

- `legacy_paper`: hybrid 99.1% ± 0.4
- `online_c2`: hybrid 99.6% ± 0.1
- `online_compare`: hybrid 99.6% ± 0.1

Takeaway:

- This scenario is already easy for the frozen model.
- Improvements do not matter much because performance is near ceiling.

#### Distributional Drift

- `legacy_paper`: hybrid 96.6% ± 0.2
- `online_c2`: hybrid 96.4% ± 0.5
- `online_compare`: hybrid 96.6% ± 0.4

Takeaway:

- The improved version slightly improved hybrid drift performance.
- More importantly, evo drift stability improved noticeably.

### Evo Stability Under Drift

Observed during comparison:

- `online_c2` evo drift: 80.0% ± 17.2
- `online_compare` evo drift: 87.5% ± 8.8

Interpretation:

- The replay buffer was a meaningful fix.
- Even where hybrid gains are small, the adaptive memory is much less erratic.

## Files Changed In This Session

- `trust.py`
- `verify.py`
- `app.py`
- `compare_versions.py`
- `note.md`

## Suggested Next Steps

### Immediate research next steps

- Add the before/after comparison table to `README.md`.
- Run more seeds, such as 20+ seeds.
- Add confidence intervals, not just means.
- Benchmark against simpler baselines:
  recalibration, retrieval baseline, online prototypes, periodic fine-tuning.

### Product next steps

- Pick one narrow commercial use case.
- Replace synthetic data with a real dataset or logged traffic.
- Add calibration, abstention, and review routing.
- Measure business KPIs, not just classification accuracy.

### Dataset guidance

Chest X-rays are possible, but only if medical AI is your actual target.

Why to be careful:

- Medical use is high stakes.
- It requires stronger validation, calibration, safety review, and likely expert involvement.
- A weak result there can be harder to interpret.

Better default path:

- Start with a lower-risk public dataset first.
- Good first choices are datasets with clear domain shift or class imbalance.
- Move to chest X-rays only if your thesis or product is specifically about medical adaptation.

## GitHub Notes

If you want to publish these changes from the working copy, the normal flow is:

```bash
cd "/Users/ash/Downloads/files (3)/MemoirAI_work"
git status
git add app.py trust.py verify.py compare_versions.py note.md
git commit -m "Add comparative trust, replay evolution, and benchmark comparison"
git remote -v
git push origin main
```

If this working copy is not the repo you want to publish from, copy the same changes into the original repo first or point this copy at your GitHub remote.

## Journal Rule Going Forward

From this point on, `note.md` should be updated whenever one of these happens:

- algorithm changes
- benchmark changes
- result changes
- product decisions
- dataset decisions
- deployment decisions

That will keep the file useful as both a research journal and a build log.

## Real-Data Harness Added

The repo now has a separate real-eval path under `real_eval/`.

Why this was needed:

- The main app remains a synthetic reviewer demo.
- We needed a clean path for LongBench-style testing without rewriting the demo.
- The first target is a smaller local model so the pipeline can be validated end to end.

What was added:

- focused LongBench subset loader
- lightweight metrics for QA, summarization, and code tasks
- conservative MemoirAI prompt-compression proxy
- optional TurboQuant backend detection and memory-proxy stacking
- model adapter layer with `dummy` and `transformers` backends
- report writer for side-by-side variant output

Important caveat:

- The new benchmark path is honest about the current state: MemoirAI is still
  approximated at prompt/prefill level until a real KV-cache integration is
  wired into the serving runtime.
