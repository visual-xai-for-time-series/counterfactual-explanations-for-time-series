"""
cf_codec.perturbation - component-level substitution strategies.

CoDec's "Component Perturbation" stage (``CoDec_presentation.pdf`` slide 6).
Given a matched ``(query_component, ref_component)`` pair, a
:class:`Perturber` returns the new component to substitute into the
candidate before reconstruction.

Implemented here
-----------------
- :class:`ReplacePerturber` - direct substitution, matches IMFACT's greedy
  IMF swap. Implemented first per workplan Phase 4.
- :class:`InterpolatePerturber` - gradual, step-parameterized blend toward
  the reference component, for corner-case detection (flagged in the
  workplan as computationally heavier - it returns one point on the
  interpolation path per call, driven by ``step``).

``multi_reference`` blending (workplan Phase 4 item 3) is left unimplemented:
it needs the pipeline to hold k live references at once rather than one
retry-loop reference, which is a bigger orchestration change than this pass's
"way smaller" scope covers. :data:`PERTURBERS` is the extension point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Perturber(ABC):
    @abstractmethod
    def perturb(self, query_component: np.ndarray, ref_component: np.ndarray, **kwargs) -> np.ndarray:
        """Return a new component to substitute in place of ``query_component``."""
        raise NotImplementedError


class ReplacePerturber(Perturber):
    """Direct component substitution - the IMFACT baseline."""

    def perturb(self, query_component: np.ndarray, ref_component: np.ndarray, **kwargs) -> np.ndarray:
        return np.asarray(ref_component, dtype=query_component.dtype).copy()


class InterpolatePerturber(Perturber):
    """Linear interpolation toward the reference component: ``(1-t) * query
    + t * ref``. ``t`` defaults to ``self.step`` and can be overridden
    per-call via the ``t`` kwarg (used by the search loop to walk the path
    gradually instead of jumping straight to full replacement).
    """

    def __init__(self, step: float = 0.25):
        self.step = step

    def perturb(self, query_component: np.ndarray, ref_component: np.ndarray, t: float | None = None, **kwargs) -> np.ndarray:
        t = self.step if t is None else t
        t = float(np.clip(t, 0.0, 1.0))
        ref_component = np.asarray(ref_component, dtype=query_component.dtype)
        return (1.0 - t) * query_component + t * ref_component


PERTURBERS: dict[str, type[Perturber]] = {
    "replace": ReplacePerturber,
    "interpolate": InterpolatePerturber,
}


def make_perturber(name: str, **kwargs) -> Perturber:
    """Instantiate a registered :class:`Perturber` by name (see :data:`PERTURBERS`)."""
    if name not in PERTURBERS:
        raise ValueError(f"Unknown perturbation method '{name}'. Available: {sorted(PERTURBERS)}")
    return PERTURBERS[name](**kwargs)
