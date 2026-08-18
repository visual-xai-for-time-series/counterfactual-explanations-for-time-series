# CoDec — Implementation Workplan

**CoDec** (Counterfactual Decomposition) generalizes IMFACT — a single-decomposition (EMD),
single-reference (NUN), greedy index-matched counterfactual method for time series
classifiers — into a modular framework where reference selection, decomposition,
matching, and perturbation are all swappable and empirically validated.

This document is a build plan for a coding agent. It defines module contracts first,
then a phased implementation order, then acceptance criteria per phase. Follow the
phases in order — later phases assume earlier interfaces are stable.

---

## 0. Goal and non-goals

**Goal:** a Python library that, given a fitted black-box time series classifier and a
query series, produces a valid, sparse, plausible counterfactual by decomposing the
series into components, swapping one or more components with a matched reference's
components, and recomposing.

**Non-goals (do not build unless explicitly asked):**
- No gradient-based / classifier-white-box perturbation (CoDec is classifier-agnostic
  by design — see Motivation slide, "Classifier scope").
- No new decomposition math research — wrap existing, well-tested implementations.
- No UI. This is a research/experimentation library with a CLI and a results pipeline.

---

## 1. Repository structure

```
codec/
├── codec/
│   ├── __init__.py
│   ├── pipeline.py            # CoDecPipeline orchestrator (the search loop)
│   ├── references/
│   │   ├── base.py            # ReferenceSelector ABC
│   │   ├── nun.py             # single-NUN (IMFACT baseline, implement first)
│   │   ├── composite.py       # per-component composite reference (favored)
│   │   ├── barycenter.py      # DTW/average barycenter reference
│   │   └── typicality.py      # density-anchored reference (LOF / isolation forest)
│   ├── decompositions/
│   │   ├── base.py            # Decomposer ABC (decompose / reconstruct)
│   │   ├── emd.py             # EMD / IMFs (IMFACT baseline, implement first)
│   │   ├── wavelet.py         # Wavelets / EWT
│   │   ├── fourier.py         # STFT / Fourier
│   │   ├── stl.py             # STL (trend/seasonal/residual)
│   │   ├── eigen.py           # PCA / SSA
│   │   ├── shapelet.py        # Shapelet-based "components"
│   │   ├── changepoint.py     # Changepoint segmentation
│   │   └── quantile.py        # Quantile transform / state-space
│   ├── matching/
│   │   ├── base.py            # Matcher ABC
│   │   ├── hungarian.py       # scipy.optimize.linear_sum_assignment, implement first
│   │   └── index.py           # naive index-aligned fallback
│   ├── perturbation/
│   │   ├── base.py            # Perturber ABC
│   │   ├── replace.py         # direct component replacement (implement first)
│   │   ├── interpolate.py     # gradual interpolation
│   │   └── multi_reference.py # blend across k references
│   ├── selection/
│   │   └── component_selection.py  # which component(s) to touch, retry widening
│   ├── evaluation/
│   │   ├── metrics.py         # proximity, plausibility, sparsity(by component), robustness
│   │   ├── baselines.py       # wrappers around existing TSC-CF methods
│   │   └── datasets.py        # UCR (128) + UEA (30) loaders, dataset registry
│   ├── classifier.py          # ClassifierWrapper (black-box f(x) -> class, batched)
│   └── config.py              # run-config schema (see §7)
├── experiments/
│   ├── run_single.py          # CLI: run CoDec on one dataset/config
│   ├── run_archive.py         # CLI: run across full UCR+UEA archive
│   └── configs/               # YAML configs per experiment
├── tests/
│   └── ...                    # unit tests mirroring codec/ structure
└── notebooks/
    └── decomposition_heuristic_sanity_checks.ipynb
```

---

## 2. Core interfaces

Define these first. Every later phase implements against these contracts — do not let
concrete implementations leak module-specific assumptions into the pipeline.

```python
# codec/decompositions/base.py
from abc import ABC, abstractmethod
import numpy as np

class Decomposer(ABC):
    """Splits a 1D series into an ordered list of components that sum to the
    original series, and can reconstruct a series from a (possibly modified)
    component list."""

    @abstractmethod
    def decompose(self, x: np.ndarray) -> list[np.ndarray]:
        """Return components [c_1, ..., c_k, r] such that sum(components) == x
        (up to numerical tolerance). Order must be stable and comparable across
        calls on different series of the same dataset (needed by Matcher)."""

    @abstractmethod
    def reconstruct(self, components: list[np.ndarray]) -> np.ndarray:
        """Inverse of decompose: sum/recombine components back into a series."""
```

```python
# codec/matching/base.py
from abc import ABC, abstractmethod
import numpy as np

class Matcher(ABC):
    @abstractmethod
    def match(
        self, query_components: list[np.ndarray], ref_components: list[np.ndarray]
    ) -> list[tuple[int, int]]:
        """Return a list of (query_idx, ref_idx) pairs — a one-to-one assignment.
        Must handle len(query_components) != len(ref_components) (pad/drop by cost)."""
```

```python
# codec/references/base.py
from abc import ABC, abstractmethod
import numpy as np

class ReferenceSelector(ABC):
    @abstractmethod
    def select(
        self, x: np.ndarray, target_class, candidates: np.ndarray, k: int = 1
    ) -> np.ndarray | list[np.ndarray]:
        """Return either a single reference series, or (for composite strategies)
        a per-component list of donor series. Concrete return type must match
        what the configured ReferenceSelector.strategy declares — see §7."""
```

```python
# codec/perturbation/base.py
from abc import ABC, abstractmethod
import numpy as np

class Perturber(ABC):
    @abstractmethod
    def perturb(
        self, query_component: np.ndarray, ref_component: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Return a new component to substitute in place of query_component."""
```

```python
# codec/pipeline.py
from dataclasses import dataclass
import numpy as np

@dataclass
class CoDecResult:
    x_cf: np.ndarray | None       # None if no valid CF found within budget
    valid: bool
    n_substitutions: int
    substituted_indices: list[int]
    proximity: float
    sparsity: int                 # count of components changed, NOT raw time points
    history: list[dict]           # one entry per search iteration, for debugging/plots

class CoDecPipeline:
    def __init__(self, decomposer, reference_selector, matcher, perturber,
                 component_selector, classifier, max_iter: int = 20):
        ...

    def run(self, x: np.ndarray, y: int, y_target: int, candidates: np.ndarray) -> CoDecResult:
        """Implements the retry loop from the CoDec Approach figure:
        select reference(s) -> decompose -> match -> select component(s) to try ->
        perturb -> reconstruct -> query classifier -> if invalid, widen component
        set or try next reference, up to max_iter -> return best valid candidate
        (or best invalid attempt + valid=False) with full CoDecResult."""
```

**Acceptance for this section:** interfaces merged, `mypy --strict` (or pyright) clean
on the `base.py` files, and a trivial `IdentityDecomposer` + `NearestNeighborReference`
+ `HungarianMatcher` + `ReplacePerturber` combination runs end-to-end in
`CoDecPipeline.run()` on a synthetic sine-wave dataset (no real data needed yet).

---

## 3. Phased implementation order

Each phase lists: what to build, which conversation decision it implements, and its
own acceptance test. Do not start phase *N+1* until phase *N*'s acceptance test passes.

### Phase 0 — Project setup & data access
- Add dependencies (see §6).
- Implement `codec/evaluation/datasets.py`: load any of the full **128 UCR** +
  **30 UEA** datasets by name via `aeon.datasets` (`load_classification`), returning
  `(X_train, y_train, X_test, y_test)` in a consistent shape (`n_instances, n_channels,
  n_timepoints`). Cache downloads locally.
- Register the five characteristic-spotlight datasets explicitly (used throughout
  experiments, not just decoration):

  | Dataset | Characteristic |
  |---|---|
  | CBF | shape-driven, low frequency reliance |
  | ECG5000 | morphology-critical |
  | FordA | frequency-heavy, noisy |
  | Yoga | subtle amplitude / shape |
  | TwoPatterns | synthetic, controlled |

- **Acceptance:** `load_dataset("CBF")` and `load_dataset("FordA")` both return
  correctly shaped arrays; `list_full_archive()` returns 158 dataset names.

### Phase 1 — Decomposition module (generalizes IMFACT's "Dual Decomposition")
Implement decomposers in this order, matching the heuristic table (see the
"Choosing a Decomposition" slide) so each is testable against the dataset it's meant for:

1. `EMDDecomposer` (wraps `PyEMD.EMD`) — the IMFACT baseline. Must exactly reproduce
   IMFACT's IMF ordering/behavior for regression testing against the original method.
2. `WaveletDecomposer` (PyWavelets, multi-level DWT or EWT approx) — multi-scale,
   non-stationary oscillations.
3. `FourierSTLDecomposer` — trend + seasonal structure. Prefer `statsmodels`' `STL`
   when the series has a defined period; fall back to STFT (`scipy.signal.stft`)
   otherwise.
4. `EigenDecomposer` — PCA/SSA for correlated multivariate signals (needed once UEA
   multivariate data enters the pipeline).
5. `ShapeletDecomposer` — discriminative local shape. Components here are not
   required to be strictly additive; document the reconstruction approximation used.
6. `ChangepointDecomposer` (via `ruptures`) — abrupt regime shifts / level changes.
7. `QuantileDecomposer` — heavy noise, few clean repeating patterns.

- **Acceptance:** for each decomposer, `reconstruct(decompose(x))` reconstructs `x`
  within a documented tolerance (exact for additive methods like EMD/wavelets/STL;
  approximate and explicitly flagged for shapelet/changepoint). Unit test each
  decomposer on its matching spotlight dataset from the table above.

### Phase 2 — Reference selection (generalizes IMFACT's "NUN Retrieval")
1. `NUNReferenceSelector` — exact IMFACT behavior (nearest unlike neighbor by raw
   distance). Implement first; this is the regression baseline.
2. `CompositeReferenceSelector` **(favored — prioritize this one)** — per component
   slot, search the full target-class candidate pool for whichever instance's matched
   component minimizes the matching cost; return a per-component donor list instead
   of one whole reference series. Requires `Matcher` (Phase 3) to already exist, since
   the composite search needs per-component cost — implement `HungarianMatcher`
   before finishing this class.
3. `BarycenterReferenceSelector` — DTW-barycenter (or simple average) over k-NUN
   components into a synthetic prototype.
4. `TypicalityReferenceSelector` — pick by class density (`sklearn.neighbors.LocalOutlierFactor`
   or `IsolationForest`) rather than distance to `x`.
- Do **not** implement a generative reference selector this cycle — it's explicitly
  flagged as a later-stage idea, higher risk, lower priority.

- **Acceptance:** all four selectors satisfy the `ReferenceSelector` contract;
  `CompositeReferenceSelector` has a unit test proving it can select components from
  *different* donor series for the same query (i.e., it is not silently degenerating
  into single-NUN).

### Phase 3 — Matching (generalizes IMFACT's implicit index alignment)
1. `IndexMatcher` — naive fallback, pairs `query[i]` with `ref[i]`.
2. `HungarianMatcher` **(prioritize)** — build cost matrix `C[i, j]` from a
   configurable cost function (dominant-frequency distance, spectral similarity,
   energy difference — make this pluggable, not hardcoded), solve with
   `scipy.optimize.linear_sum_assignment`. Must handle `len(query) != len(ref)` by
   padding the smaller side with a high-cost dummy row/column.

- **Acceptance:** unit test with a hand-constructed cost matrix where the optimal
  assignment is known; `HungarianMatcher` recovers it. Property test: matching is
  invariant to shuffling `ref_components`' order (i.e., it's actually solving an
  assignment problem, not accidentally relying on input order).

### Phase 4 — Perturbation + component selection + search loop
1. `ReplacePerturber` (direct substitution) — implement first, matches IMFACT.
2. `InterpolatePerturber` (gradual, step-parameterized) — for corner-case detection;
   flag as computationally heavier.
3. `MultiReferencePerturber` — blend across k references' matched components.
4. `component_selection.py`: strategy for choosing which component(s) to try each
   iteration, and how to **widen** the set on retry (this is the "retry loop" from
   the CoDec Approach figure — mirrors IMFACT's `query f(x^(s)); class flips?` loop,
   generalized to: on failure, either grow the substituted-component set or advance
   to the next candidate reference, up to `max_iter`).
5. Wire all of the above into `CoDecPipeline.run()`.

- **Acceptance:** end-to-end run on CBF with `EMDDecomposer` + `NUNReferenceSelector`
  + `HungarianMatcher` + `ReplacePerturber` reproduces IMFACT's published behavior
  (same or better validity rate on a held-out sample) — this is the critical
  regression checkpoint proving the generalized framework hasn't regressed the
  original method.

### Phase 5 — Evaluation harness
1. `metrics.py`: proximity (distance to original), plausibility (e.g., local outlier
   score against the target class), **sparsity computed as count of components
   changed, not raw time points changed** — this was an explicit reviewer-driven
   correction, do not regress to point-level sparsity.
2. `baselines.py`: thin wrappers around a small, well-justified subset of existing
   time-series CF methods (not an exhaustive table — pick recent, citable baselines;
   confirm the exact list with the team before hardcoding it).
3. `run_archive.py`: runs a given CoDec config across all 128 UCR + 30 UEA datasets,
   writes per-dataset results to disk, resumable (don't recompute finished datasets).

- **Acceptance:** `run_archive.py --config configs/default.yaml --dry-run` enumerates
  all 158 datasets without error; a real run on the 5 spotlight datasets completes
  and produces a results table with all metrics populated (no NaNs from silent
  failures).

### Phase 6 — Decomposition-heuristic experiments
Run the actual empirical validation of the heuristic table: for each spotlight
dataset, run CoDec with (a) the heuristic-recommended decomposition and (b) at least
one mismatched decomposition, and confirm the recommended one wins on
validity/plausibility/sparsity. This is the experiment that justifies the "Choosing a
Decomposition" slide with data rather than just intuition.

- **Acceptance:** a results table/plot showing recommended-vs-mismatched decomposition
  performance per spotlight dataset, saved to `experiments/results/`.

---

## 4. Pre-September-meeting checklist (priority order)

These map directly to the agreed next steps — treat them as the sprint backlog, in
this order:

1. **Draft the Hungarian-matching implementation** (Phase 3, `HungarianMatcher`) —
   unblocks the composite reference selector.
2. **Prototype the composite reference-selection module** (Phase 2,
   `CompositeReferenceSelector`) — the favored arbitrary-NUN generalization.
3. **Stand up the full UCR + UEA archive evaluation pipeline** (Phase 0 + Phase 5's
   `run_archive.py`) — doesn't need every decomposer finished, just needs to run
   end-to-end on at least the EMD baseline across the full archive.
4. **Finalize the small, well-justified baseline CF algorithm subset** (Phase 5,
   `baselines.py`) — this is a decision item as much as a coding task; needs team
   input on which methods to cite/wrap, don't pick unilaterally.

---

## 5. Config schema (sketch)

Every experiment run should be fully specified by one YAML file, so results are
reproducible and diffable:

```yaml
# experiments/configs/default.yaml
dataset: CBF
decomposition:
  method: emd            # emd | wavelet | fourier_stl | eigen | shapelet | changepoint | quantile
reference_selection:
  strategy: composite     # nun | composite | barycenter | typicality
  k: 5
matching:
  method: hungarian        # hungarian | index
  cost_fn: dominant_frequency  # dominant_frequency | spectral_similarity | energy
perturbation:
  method: replace           # replace | interpolate | multi_reference
search:
  max_iter: 20
  widen_on_failure: true
classifier:
  name: rocket               # any aeon/sktime classifier, treated strictly black-box
```

---

## 6. Suggested dependencies

- `aeon` (UCR/UEA dataset loaders, baseline classifiers, some existing CF/XAI tooling)
- `PyEMD` — EMD/IMF decomposition
- `PyWavelets` — wavelet decomposition
- `statsmodels` — STL, potentially state-space models
- `scipy` — `optimize.linear_sum_assignment` (Hungarian), `signal.stft`, `fft`
- `scikit-learn` — PCA, `LocalOutlierFactor`, `IsolationForest`
- `ruptures` — changepoint segmentation
- `numpy` — everywhere

Pin versions in `pyproject.toml`; this is a research codebase but should still be
reproducible.

---

## 7. Testing conventions

- Every module under `codec/` gets a matching file under `tests/`.
- Any new `Decomposer`, `ReferenceSelector`, `Matcher`, or `Perturber` must include a
  contract test that checks it satisfies its ABC's documented invariants (not just
  "runs without crashing").
- Regression test against IMFACT behavior (Phase 4's acceptance test) must remain in
  the suite permanently — it's the framework's core non-regression guarantee.

---

## 8. Open decisions for the team (do not resolve unilaterally in code)

- Exact baseline CF algorithm subset (Phase 5).
- Whether `ShapeletDecomposer` and `ChangepointDecomposer`'s non-additive
  reconstruction approximation is acceptable for the paper, or needs a stricter
  formulation.
- Compute budget for the full 158-dataset archive run (affects how aggressively
  `max_iter` and k can be tuned before Phase 6's experiments).
