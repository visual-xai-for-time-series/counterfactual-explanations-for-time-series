"""
Smoke tests for cf_codec - not a pytest suite, a runnable script (same style
as cfts/cf_mg_cf/test_stumpy_mg_cf.py). Exercises the acceptance criteria
CoDec_workplan.md lays out for §2 (core interfaces) and Phases 1-4
(decomposition reconstruction tolerance, Hungarian optimality/shuffle
invariance, composite selection pulling from multiple donors, and an
end-to-end class flip) against synthetic data, so no UCR/UEA download is
needed to check the pipeline is wired correctly.

Run directly: `python -m cfts.cf_codec.test_codec`
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cfts.cf_codec.decompositions import DECOMPOSERS, FourierBandDecomposer, make_decomposer
from cfts.cf_codec.matching import HungarianMatcher, IndexMatcher
from cfts.cf_codec.references import CompositeReferenceSelector, NUNReferenceSelector
from cfts.cf_codec.perturbation import InterpolatePerturber, ReplacePerturber
from cfts.cf_codec.codec import codec_cf


# ---------------------------------------------------------------------------
# Phase 1: decomposition reconstruction tolerance - every registered
# decomposer (workplan's full "Choosing a Decomposition" heuristic table).
# ---------------------------------------------------------------------------

def test_decomposer_reconstruction():
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0, 8 * np.pi, 200)) + 0.5 * np.sin(np.linspace(0, 40 * np.pi, 200))
    x += rng.normal(0, 0.05, size=x.shape)
    x = x.astype(np.float32)

    tol = 1e-2  # generous: some decomposers (STL) run in float64 with their own smoothing tolerances
    for name in DECOMPOSERS:
        decomposer = make_decomposer(name)
        comps = decomposer.decompose(x)
        recon = decomposer.reconstruct(comps)
        err = float(np.max(np.abs(recon - x)))
        assert err < tol, f"{name} reconstruction error {err} exceeds {tol}"
        print(f"  {name:12s}: {comps.shape[0]} components, max recon error {err:.2e}")
    print("test_decomposer_reconstruction: OK")


def test_decomposer_edge_cases():
    """Short, constant, and periodic inputs shouldn't crash any decomposer,
    and every one must still reconstruct exactly (or near-exactly)."""
    rng = np.random.default_rng(2)
    cases = {
        "short": np.linspace(0, 1, 10).astype(np.float32),
        "constant": np.full(50, 3.0, dtype=np.float32),
        "periodic": (np.sin(2 * np.pi * np.arange(300) / 20) + 0.05 * rng.normal(size=300)).astype(np.float32),
    }
    for case_name, x in cases.items():
        for name in DECOMPOSERS:
            decomposer = make_decomposer(name)
            comps = decomposer.decompose(x)
            recon = decomposer.reconstruct(comps)
            err = float(np.max(np.abs(recon - x)))
            assert err < 1e-2, f"{name} on '{case_name}' reconstruction error {err}"
    print("test_decomposer_edge_cases: OK")


# ---------------------------------------------------------------------------
# Phase 3: Hungarian matcher recovers a known-optimal assignment, and is
# invariant to shuffling the reference components' order.
# ---------------------------------------------------------------------------

def test_hungarian_matcher_optimal_and_shuffle_invariant():
    L = 64
    t = np.linspace(0, 1, L)
    # Three query components at increasing frequency.
    q = [np.sin(2 * np.pi * f * t) for f in (2, 8, 20)]
    # Three reference components at the *same* frequencies but shuffled and
    # phase/amplitude-shifted, so "dominant_frequency" cost should still
    # recover the frequency-matched pairing regardless of storage order.
    ref_freqs = [20, 2, 8]  # deliberately out of order vs q
    r = [0.7 * np.sin(2 * np.pi * f * t + 0.3) for f in ref_freqs]

    matcher = HungarianMatcher(cost_fn="dominant_frequency")
    pairs = dict(matcher.match(np.array(q), np.array(r)))
    expected = {0: 1, 1: 2, 2: 0}  # q[i] should pair with the ref index sharing its frequency
    assert pairs == expected, f"Hungarian assignment {pairs} != expected {expected}"

    # Shuffle invariance: permute the reference array, re-match, and check
    # the recovered *frequency* pairing (not raw index) is identical.
    rng = np.random.default_rng(1)
    perm = rng.permutation(3)
    r_shuffled = [r[i] for i in perm]
    pairs_shuffled = dict(matcher.match(np.array(q), np.array(r_shuffled)))
    recovered_freqs = {qi: ref_freqs[perm[ri]] for qi, ri in pairs_shuffled.items()}
    original_freqs = {0: 2, 1: 8, 2: 20}
    assert recovered_freqs == original_freqs, f"shuffled match {recovered_freqs} != {original_freqs}"
    print("test_hungarian_matcher_optimal_and_shuffle_invariant: OK")


def test_index_matcher_pads_shorter_side():
    q = np.zeros((3, 10))
    r = np.zeros((2, 10))
    pairs = IndexMatcher().match(q, r)
    assert pairs == [(0, 0), (1, 1), (2, 1)]
    print("test_index_matcher_pads_shorter_side: OK")


# ---------------------------------------------------------------------------
# Phase 2: CompositeReferenceSelector must pull components from *different*
# donors for the same query, not silently degenerate into single-NUN.
# ---------------------------------------------------------------------------

def test_composite_reference_selector_uses_multiple_donors():
    L = 64
    t = np.linspace(0, 1, L)
    comp_a = np.sin(2 * np.pi * 3 * t)
    comp_b = np.sin(2 * np.pi * 15 * t)
    query_components = np.stack([comp_a, comp_b], axis=0).astype(np.float32)

    # donor_1 matches comp_a well, is a poor match for comp_b.
    donor_1 = comp_a + 0.6 * np.sin(2 * np.pi * 31 * t)
    # donor_2 matches comp_b well, is a poor match for comp_a.
    donor_2 = 0.05 * np.sin(2 * np.pi * 1 * t) + comp_b
    pool = np.stack([donor_1, donor_2], axis=0).astype(np.float32)

    decomposer = FourierBandDecomposer(n_bands=2)
    matcher = HungarianMatcher(cost_fn="dominant_frequency")
    selector = CompositeReferenceSelector(decomposer=decomposer, matcher=matcher, pool_k=2)

    donor_components, donor_source, donor_cost = selector.select_components(query_components, pool)
    assert donor_components.shape == query_components.shape
    assert set(donor_source.tolist()) == {0, 1}, (
        f"expected both donors used, got sources {donor_source.tolist()}"
    )
    print(f"  donor_source={donor_source.tolist()} donor_cost={np.round(donor_cost, 4).tolist()}")
    print("test_composite_reference_selector_uses_multiple_donors: OK")


# ---------------------------------------------------------------------------
# §2 acceptance: a trivial combination runs end-to-end (NUN + index matcher +
# replace perturber), then the full default stack (composite + hungarian).
# ---------------------------------------------------------------------------

class _TinyCNN(nn.Module):
    def __init__(self, seq_length: int, n_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def _make_frequency_dataset(n_per_class: int = 60, L: int = 128, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, L)
    X, y = [], []
    for _ in range(n_per_class):
        low = np.sin(2 * np.pi * 3 * t + rng.uniform(0, 2 * np.pi)) + rng.normal(0, 0.05, L)
        X.append(low.astype(np.float32)[None, :])
        y.append(0)
        high = np.sin(2 * np.pi * 18 * t + rng.uniform(0, 2 * np.pi)) + rng.normal(0, 0.05, L)
        X.append(high.astype(np.float32)[None, :])
        y.append(1)
    return X, np.array(y)


def _train_tiny_cnn(X, y, seq_length, epochs=60):
    torch.manual_seed(0)  # keep runs deterministic - some decomposers only flip an easy-but-random init
    model = _TinyCNN(seq_length)
    Xt = torch.tensor(np.stack(X), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(Xt)
        loss = loss_fn(out, yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xt).argmax(1) == yt).float().mean().item()
    return model, acc


def test_codec_cf_end_to_end():
    L = 128
    X, y = _make_frequency_dataset(n_per_class=60, L=L)
    model, acc = _train_tiny_cnn(X, y, seq_length=L)
    print(f"  tiny CNN train accuracy: {acc:.3f}")
    assert acc > 0.9, "classifier didn't learn the toy frequency task; test setup is broken"

    dataset = list(zip(X, y))
    sample = X[0]  # a class-0 (low frequency) sample
    assert y[0] == 0

    # Trivial stack: NUN + index matcher + replace, single-reference greedy.
    cf, scores = codec_cf(
        sample, model, target_class=1, dataset=dataset,
        decomposition="fourier", reference_selection="nun", matching="index",
        perturbation="replace", k=3, max_iter=15,
    )
    assert cf.shape == sample.shape
    print(f"  [nun/index/replace] predicted={int(np.argmax(scores))} (target=1)")

    # Favored stack: composite + hungarian + replace.
    cf2, scores2, result = codec_cf(
        sample, model, target_class=1, dataset=dataset,
        decomposition="emd", reference_selection="composite", matching="hungarian",
        cost_fn="dominant_frequency", perturbation="replace", k=5, max_iter=25,
        return_result=True, verbose=False,
    )
    assert cf2.shape == sample.shape
    print(
        f"  [composite/hungarian/replace] predicted={int(np.argmax(scores2))} (target=1) "
        f"valid={result.valid} sparsity={result.sparsity} proximity={result.proximity:.3f}"
    )
    assert result.valid, "expected the favored strategy stack to flip this easy synthetic case"
    assert int(np.argmax(scores2)) == 1

    # Interpolate perturber + Fourier decomposition, just needs to run cleanly.
    cf3, scores3 = codec_cf(
        sample, model, target_class=1, dataset=dataset,
        decomposition="fourier", reference_selection="nun", matching="hungarian",
        perturbation="interpolate", interpolate_step=1.0, k=3, max_iter=15,
    )
    assert cf3.shape == sample.shape
    print("test_codec_cf_end_to_end: OK")


def test_codec_cf_all_decomposers():
    """Every registered decomposition wired end-to-end through codec_cf (not
    just emd/fourier) - each must run cleanly and return a same-shaped
    candidate; validity isn't asserted for all of them since some (e.g.
    changepoint, quantile) aren't a natural fit for this smooth sine-wave
    toy task - see decompositions.py's heuristic-table docstring for what
    each is actually suited to."""
    L = 128
    X, y = _make_frequency_dataset(n_per_class=60, L=L)
    model, acc = _train_tiny_cnn(X, y, seq_length=L)
    assert acc > 0.9, "classifier didn't learn the toy frequency task; test setup is broken"
    dataset = list(zip(X, y))
    sample = X[0]

    for name in DECOMPOSERS:
        cf, scores, result = codec_cf(
            sample, model, target_class=1, dataset=dataset,
            decomposition=name, reference_selection="composite", matching="hungarian",
            perturbation="replace", k=5, max_iter=25, return_result=True,
        )
        assert cf.shape == sample.shape
        print(f"  decomposition={name:12s} predicted={int(np.argmax(scores))} valid={result.valid}")
    print("test_codec_cf_all_decomposers: OK")


if __name__ == "__main__":
    test_decomposer_reconstruction()
    test_decomposer_edge_cases()
    test_hungarian_matcher_optimal_and_shuffle_invariant()
    test_index_matcher_pads_shorter_side()
    test_composite_reference_selector_uses_multiple_donors()
    test_codec_cf_end_to_end()
    test_codec_cf_all_decomposers()
    print("\nAll cf_codec smoke tests passed.")
