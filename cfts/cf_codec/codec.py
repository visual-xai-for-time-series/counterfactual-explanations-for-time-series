"""
cf_codec.codec - CoDec: modular decomposition-based counterfactual search.

Generalizes IMFACT (single decomposition, single reference, greedy
index-matched substitution - see ``cfts/cf_imfact/imfact.py``) into a
framework where reference selection, decomposition, matching, and
perturbation are all swappable, per ``CoDec_workplan.md`` and
``CoDec_presentation.pdf``.

Two entry points:

- :class:`CoDecPipeline` - the encode -> intervene -> decode search loop
  (``CoDec_presentation.pdf`` slide 6, "The CoDec Approach") as a standalone,
  classifier-framework-agnostic object. It only needs a
  ``predict_fn(np.ndarray (C, L)) -> np.ndarray (n_classes,)`` callable, in
  keeping with the "Classifier scope: black-box, no gradients" slide.
- :func:`codec_cf` - the public function every other method in this
  repository exposes (see ``cfts/cf__abstract/abstract.py``'s ``CFMethod``
  contract). It's a thin ``torch.nn.Module`` adapter: normalise the sample
  and dataset, wire up the requested strategies from
  :mod:`cfts.cf_codec.decompositions` / ``.matching`` / ``.references`` /
  ``.perturbation``, and hand off to :class:`CoDecPipeline`.

Scope of this pass
-------------------
This module implements the *algorithm* (workplan §2's core interfaces plus
Phases 1-4: decomposition, reference selection, matching, perturbation, and
the retry/search loop) with all eight decomposers from the "Choosing a
Decomposition" heuristic table (:mod:`cfts.cf_codec.decompositions`), two
reference selectors, two matchers and two perturbers - deliberately a *way
smaller* file layout than ``CoDec_workplan.md`` §1's proposed nested
``codec/`` package tree (no ``experiments/`` tests directory beyond
``test_codec.py``, no YAML config schema). Explicitly out of scope here, per
the workplan's own phasing and its §8 "do not resolve unilaterally" list:

- Phase 5/6's evaluation harness - dataset registry, baseline-method
  wrappers, and the full 128 UCR + 30 UEA archive runner (a smaller
  per-dataset version lives in ``experiments/compare_ucr.py``). Wrapping
  baselines is flagged in the workplan itself as a team decision, not a
  coding task.
- Barycenter / typicality-anchored / generative reference selectors, and
  ``multi_reference`` perturbation - each is a registry entry away (see
  ``references.REFERENCE_SELECTORS`` / ``perturbation.PERTURBERS``) but
  needs materially more design work (the generative selector is also
  explicitly flagged in the workplan as a later-stage, do-not-build-yet
  idea).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np
import torch

from cfts.cf__abstract.abstract import (
    detach_to_numpy,
    numpy_to_torch,
    ensure_ncl,
    revert_orientation,
    subsample_dataset,
)
from cfts.cf_codec.decompositions import make_decomposer
from cfts.cf_codec.matching import make_matcher
from cfts.cf_codec.perturbation import make_perturber
from cfts.cf_codec.references import CompositeReferenceSelector, make_reference_selector


@dataclass
class CoDecResult:
    """Search-trace output of :meth:`CoDecPipeline.run`.

    Mirrors ``CoDec_workplan.md`` §2's ``CoDecResult`` dataclass, with one
    addition (``scores``) needed to satisfy this repository's ``<name>_cf``
    contract, which always returns model scores alongside the counterfactual.
    """

    x_cf: np.ndarray
    scores: np.ndarray
    valid: bool
    n_substitutions: int          # count of components changed (workplan §5: NOT raw time points)
    sparsity: int                 # same count, kept as a separate field for parity with the workplan spec
    substituted_components: list  # [(channel, component_idx), ...] for the returned candidate
    proximity: float              # L2 distance to the original query
    reference_index: int          # pool position that donated the winning candidate; -1 for composite/none
    history: list = field(default_factory=list)  # one entry per search iteration, for debugging/plots


class CoDecPipeline:
    """Orchestrates the retry loop from the CoDec Approach figure: select
    reference(s) -> decompose -> match -> select component(s) to try ->
    perturb -> reconstruct -> query classifier -> if invalid, widen the
    substituted-component set or advance to the next reference, up to
    ``max_iter`` -> return the best valid candidate found (or the best
    invalid attempt, with ``valid=False``).

    Framework-agnostic by design: ``predict_fn`` is a plain callable, not a
    ``torch.nn.Module`` - see :func:`codec_cf` for the PyTorch adapter used
    by the rest of this repository.
    """

    def __init__(
        self,
        decomposer,
        reference_selector,
        matcher,
        perturber,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        max_iter: int = 20,
        k: int = 5,
        widen_on_failure: bool = True,
        component_order: str = "cost_asc",
    ):
        if component_order not in ("cost_asc", "cost_desc"):
            raise ValueError("component_order must be 'cost_asc' or 'cost_desc'")
        self.decomposer = decomposer
        self.reference_selector = reference_selector
        self.matcher = matcher
        self.perturber = perturber
        self.predict_fn = predict_fn
        self.max_iter = max_iter
        self.k = k
        self.widen_on_failure = widen_on_failure
        self.component_order = component_order
        self.is_composite = isinstance(reference_selector, CompositeReferenceSelector)

    def run(
        self,
        x: np.ndarray,
        candidates: np.ndarray,
        label_orig: int,
        target_class: int | None,
        scores_orig: np.ndarray,
        verbose: bool = False,
    ) -> CoDecResult:
        """
        Parameters
        ----------
        x:
            Query series, shape ``(C, L)``.
        candidates:
            Target-class (or any-unlike-``label_orig``) donor pool, shape
            ``(N, C, L)``. Must be non-empty.
        label_orig, scores_orig:
            The query's original predicted label and score vector.
        target_class:
            Desired class, or ``None`` to accept any flip away from
            ``label_orig``.
        """
        C, L = x.shape
        best = {
            "candidate": x.copy(),
            "scores": scores_orig.copy(),
            "key": self._rank_key(scores_orig, label_orig, target_class),
            "n_active": 0,
            "active_components": [],
            "ref_pos": -1,
        }
        history: list = []
        valid_found = False
        iterations = 0

        query_components = [self.decomposer.decompose(x[c]) for c in range(C)]

        pool = self.reference_selector.select(x, candidates, k=min(self.k, len(candidates)))
        references_to_try = [None] if self.is_composite else list(pool)

        for ref_pos, ref in enumerate(references_to_try):
            if valid_found or iterations >= self.max_iter:
                break

            slots = self._build_slots(query_components, ref, pool, C)
            if not slots:
                continue
            order = np.argsort([s["cost"] for s in slots])
            if self.component_order == "cost_desc":
                order = order[::-1]

            active_idx: list = []
            for pos in order:
                if iterations >= self.max_iter:
                    break
                if self.widen_on_failure:
                    active_idx.append(int(pos))
                    current_active = list(active_idx)
                else:
                    current_active = [int(pos)]

                candidate = self._apply(query_components, slots, current_active, C)
                scores_cand = self.predict_fn(candidate)
                label_cand = int(np.argmax(scores_cand))
                iterations += 1

                is_valid = (
                    label_cand == target_class if target_class is not None else label_cand != label_orig
                )
                rank_key = self._rank_key(scores_cand, label_orig, target_class)
                active_components = [(slots[i]["channel"], slots[i]["comp_idx"]) for i in current_active]

                history.append(
                    {
                        "iteration": iterations,
                        "reference": "composite" if self.is_composite else ref_pos,
                        "n_active": len(current_active),
                        "active_components": active_components,
                        "predicted_label": label_cand,
                        "valid": is_valid,
                    }
                )
                if verbose:
                    ref_tag = "composite" if self.is_composite else f"ref={ref_pos}"
                    print(
                        f"[CoDecPipeline] iter {iterations:3d} {ref_tag} "
                        f"n_active={len(current_active):2d} predicted={label_cand} valid={is_valid}"
                    )

                if is_valid or rank_key > best["key"]:
                    best = {
                        "candidate": candidate,
                        "scores": scores_cand,
                        "key": rank_key,
                        "n_active": len(current_active),
                        "active_components": active_components,
                        "ref_pos": -1 if self.is_composite else ref_pos,
                    }
                if is_valid:
                    valid_found = True
                    break

        return CoDecResult(
            x_cf=best["candidate"],
            scores=best["scores"],
            valid=valid_found,
            n_substitutions=best["n_active"],
            sparsity=best["n_active"],
            substituted_components=best["active_components"],
            proximity=float(np.linalg.norm(best["candidate"] - x)),
            reference_index=best["ref_pos"],
            history=history,
        )

    # -- internals ------------------------------------------------------

    @staticmethod
    def _rank_key(scores: np.ndarray, label_orig: int, target_class: int | None) -> float:
        """Best-effort search heuristic: higher is "closer to done". Used
        only to pick a fallback candidate when no valid flip is found within
        ``max_iter`` - not a substitute for the proximity/plausibility
        metrics the workplan's Phase 5 evaluation harness would report."""
        return float(scores[target_class]) if target_class is not None else float(-scores[label_orig])

    def _build_slots(self, query_components: list, ref, pool: np.ndarray, C: int) -> list:
        """Per-channel, per-component candidate substitutions for one
        reference (or, for composite selection, for the whole pool at once).
        Each slot is ``{"channel", "comp_idx", "ref_component", "cost"}``."""
        slots: list = []
        for c in range(C):
            q_comp_c = query_components[c]
            if self.is_composite:
                donor_comp, _donor_src, donor_cost = self.reference_selector.select_components(
                    q_comp_c, pool[:, c, :]
                )
                for qi in range(len(q_comp_c)):
                    slots.append(
                        {"channel": c, "comp_idx": qi, "ref_component": donor_comp[qi], "cost": float(donor_cost[qi])}
                    )
            else:
                ref_components_c = self.decomposer.decompose(ref[c])
                pairs, costs = self.matcher.match(q_comp_c, ref_components_c, return_costs=True)
                for (qi, ri), cost in zip(pairs, costs):
                    ref_comp = (
                        ref_components_c[ri]
                        if 0 <= ri < len(ref_components_c)
                        else np.zeros_like(q_comp_c[qi])
                    )
                    slots.append({"channel": c, "comp_idx": qi, "ref_component": ref_comp, "cost": float(cost)})
        return slots

    def _apply(self, query_components: list, slots: list, active_idx: list, C: int) -> np.ndarray:
        """Reconstruct a candidate series with the active slots perturbed."""
        cand_components = [comp.copy() for comp in query_components]
        for idx in active_idx:
            s = slots[idx]
            new_comp = self.perturber.perturb(query_components[s["channel"]][s["comp_idx"]], s["ref_component"])
            cand_components[s["channel"]][s["comp_idx"]] = new_comp
        return np.stack(
            [self.decomposer.reconstruct(cand_components[c]) for c in range(C)], axis=0
        ).astype(np.float32)


####
# CoDec: Counterfactual Decomposition (2026, in progress)
#
# Paper: N/A yet - this generalizes IMFACT (Schlegel et al., XKDD @ ECML-PKDD
#        2026, see cfts/cf_imfact/imfact.py) into a modular framework per
#        CoDec_workplan.md / CoDec_presentation.pdf.
#
# Strategy: encode -> intervene -> decode. Pick reference(s) from the target
# class (nearest-unlike-neighbor, or a per-component composite stitched from
# several candidates), decompose query and reference(s) into components with
# a swappable decomposer, align components across series via a swappable
# matcher, substitute the cheapest-to-swap components first and widen the
# substituted set (or advance to the next reference) until the classifier
# flips or the iteration budget runs out.
####

def codec_cf(
    sample: np.ndarray | list,
    model: torch.nn.Module,
    target_class: int | None = None,
    dataset: list | np.ndarray = None,
    decomposition: str = "emd",
    reference_selection: str = "composite",
    matching: str = "hungarian",
    cost_fn: str = "dominant_frequency",
    perturbation: str = "replace",
    interpolate_step: float = 0.5,
    component_order: str = "cost_asc",
    k: int = 5,
    max_iter: int = 20,
    max_imfs: int = 6,
    decomposer_kwargs: dict | None = None,
    widen_on_failure: bool = True,
    max_samples: int | None = None,
    verbose: bool = False,
    return_result: bool = False,
    *args,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """CoDec counterfactual explanation for time series classification.

    Follows the same signature pattern as every other CF method in this
    repository (``imfact_cf``, ``wachter_genetic_cf``, ...) so it plugs
    straight into the existing evaluation and example scripts.

    Parameters
    ----------
    sample:
        Query time series; 1-D ``(L,)``, ``(C, L)`` or ``(L, C)``.
    model:
        PyTorch classifier with signature ``forward(B, C, L) -> (B, n_classes)``.
    target_class:
        Desired class. When ``None``, any class different from the query's
        predicted class is accepted (matches ``imfact_cf``'s behaviour).
    dataset:
        Training set as a sequence of ``(x, y)`` pairs, or an
        ``(N, C, L)``-shaped array with a matching label array - used as the
        donor pool for reference selection.
    decomposition:
        ``"emd"`` (IMFACT baseline, wraps ``emd.sift.sift``), ``"wavelet"``
        (multi-level DWT via PyWavelets), ``"fourier"`` (trend + STFT
        frequency bands), ``"stl"`` (trend/seasonal/residual, falls back to
        ``"fourier"`` when no period is detected), ``"eigen"`` (per-channel
        SSA), ``"shapelet"`` (localized high-variance windows), ``"changepoint"``
        (piecewise-constant regime segmentation via ``ruptures``), or
        ``"quantile"`` (robust rolling-median trend + spike/noise split).
        See :mod:`cfts.cf_codec.decompositions` and the "Choosing a
        Decomposition" heuristic table in ``CoDec_presentation.pdf``.
    reference_selection:
        ``"composite"`` (favored - per-component donor stitched from the
        ``k`` nearest candidates) or ``"nun"`` (single nearest-unlike-neighbor,
        exact IMFACT behaviour). See :mod:`cfts.cf_codec.references`.
    matching:
        ``"hungarian"`` (favored - optimal assignment via ``cost_fn``) or
        ``"index"`` (naive positional pairing). See :mod:`cfts.cf_codec.matching`.
    cost_fn:
        Feature used by the Hungarian matcher's cost matrix: ``"dominant_frequency"``,
        ``"energy"``, or ``"spectral_similarity"``. Ignored when ``matching="index"``.
    perturbation:
        ``"replace"`` (direct substitution, IMFACT baseline) or ``"interpolate"``
        (blend ``interpolate_step`` of the way toward the donor component).
        See :mod:`cfts.cf_codec.perturbation`.
    component_order:
        Order in which components are added to the substituted set as the
        search widens: ``"cost_asc"`` (cheapest/most-similar first, default)
        or ``"cost_desc"``.
    k:
        Size of the candidate reference pool (number of NUN candidates
        considered, or the composite selector's donor pool size).
    max_iter:
        Maximum number of classifier queries before returning the best
        candidate found.
    max_imfs:
        Maximum IMFs per channel. Only used when ``decomposition="emd"``.
    decomposer_kwargs:
        Extra keyword arguments forwarded to the chosen decomposer's
        constructor (merged after ``max_imfs`` for ``"emd"``), e.g.
        ``{"wavelet": "sym4"}`` for ``"wavelet"``, ``{"period": 24}`` for
        ``"stl"``, ``{"n_components": 6}`` for ``"eigen"``,
        ``{"shapelet_length": 15}`` for ``"shapelet"``, ``{"penalty": 5.0}``
        for ``"changepoint"``, or ``{"low_q": 0.05, "high_q": 0.95}`` for
        ``"quantile"``. See each class's constructor in
        :mod:`cfts.cf_codec.decompositions` for its full parameter set.
    widen_on_failure:
        If ``True`` (default), grow the substituted-component set by one
        each failed attempt (cumulative). If ``False``, try each component
        individually instead of accumulating.
    max_samples:
        Optional cap on how many ``dataset`` items to consider (stratified
        subsample), for large training sets.
    verbose:
        Print per-iteration diagnostics when ``True``.
    return_result:
        When ``True``, also return the full :class:`CoDecResult` (validity,
        sparsity-by-components, substituted component indices, search
        history) alongside ``(counterfactual, scores)``.

    Returns
    -------
    counterfactual : np.ndarray
        In the same shape / orientation as ``sample``.
    scores : np.ndarray, shape (n_classes,)
        Model output scores for the counterfactual.
    result : CoDecResult
        Only returned when ``return_result=True``.

    Example
    -------
    >>> cf, scores = codec_cf(sample, model, dataset=dataset, target_class=1)
    >>> cf, scores, result = codec_cf(sample, model, dataset=dataset, verbose=True, return_result=True)
    >>> result.valid, result.sparsity
    """
    device = next(model.parameters()).device

    if max_samples is not None:
        dataset = subsample_dataset(dataset, max_samples)
    sample_cl, ts, ori = ensure_ncl(sample, dataset)
    C, L = sample_cl.shape

    raw_labels = np.array([item[1] for item in dataset])
    labels = np.argmax(raw_labels, axis=1).astype(int) if raw_labels.ndim > 1 else raw_labels.astype(int)

    def _predict(x_cl: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return detach_to_numpy(model(numpy_to_torch(x_cl.reshape(1, C, L), device))).reshape(-1)

    scores_orig = _predict(sample_cl)
    label_orig = int(np.argmax(scores_orig))

    mask = (labels == target_class) if target_class is not None else (labels != label_orig)
    candidates = ts[mask]

    if len(candidates) == 0:
        # No usable donor pool: conservative fallback, mirrors imfact_cf.
        x_cf = revert_orientation(sample_cl, ori)
        if not return_result:
            return x_cf, scores_orig
        return x_cf, scores_orig, CoDecResult(
            x_cf=x_cf, scores=scores_orig, valid=False, n_substitutions=0, sparsity=0,
            substituted_components=[], proximity=0.0, reference_index=-1, history=[],
        )

    dec_kwargs = {"max_imfs": max_imfs} if decomposition == "emd" else {}
    dec_kwargs.update(decomposer_kwargs or {})
    decomposer = make_decomposer(decomposition, **dec_kwargs)
    matcher = make_matcher(matching, **({"cost_fn": cost_fn} if matching == "hungarian" else {}))
    perturber = make_perturber(perturbation, **({"step": interpolate_step} if perturbation == "interpolate" else {}))
    if reference_selection == "composite":
        reference_selector = make_reference_selector("composite", decomposer=decomposer, matcher=matcher, pool_k=k)
    else:
        reference_selector = make_reference_selector(reference_selection)

    pipeline = CoDecPipeline(
        decomposer=decomposer,
        reference_selector=reference_selector,
        matcher=matcher,
        perturber=perturber,
        predict_fn=_predict,
        max_iter=max_iter,
        k=k,
        widen_on_failure=widen_on_failure,
        component_order=component_order,
    )
    result = pipeline.run(
        x=sample_cl,
        candidates=candidates,
        label_orig=label_orig,
        target_class=target_class,
        scores_orig=scores_orig,
        verbose=verbose,
    )

    x_cf = revert_orientation(result.x_cf, ori)
    result.x_cf = x_cf
    if return_result:
        return x_cf, result.scores, result
    return x_cf, result.scores
