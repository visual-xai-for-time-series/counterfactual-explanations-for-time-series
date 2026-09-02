"""
Comprehensive Counterfactual Metrics Evaluation Example

This example demonstrates how to evaluate counterfactual explanation generation
algorithms using the comprehensive metrics suite. It integrates with the existing
counterfactual methods (Native Guide, COMTE, COMTE-TS, SETS, MOC, Wachter, GLACIER,
Multi-SpaCE, Sub-SpaCE, TSEvo, LASTS, TSCF, FASTPACE, TIME-CF, SG-CF, MG-CF,
Latent-CF, DiSCoX, CELS, FFT-CF, TERCE, AB-CF, CFWOT, CGM, COUNTS, SPARCE,
CEM-PN, Abstract-CF, TS-Tweaking-kNN, TS-Tweaking-Irrev, TS-Tweaking-Rev, CFE4MTS,
CONFETTI, MASCOTS, IMFACT, TimeX, TimeX++) from the cfts package and evaluates
them on the FordA dataset, plus additional variants of several of them:
8 GLACIER bake-off variants (Glacier-{AE,NoAE}-{Unc,Loc,Glob,Unif}), InfoCELS,
NG-DBA, COMTE-Distractor, COMTE-Advanced-Gradient, Multi-SpaCE-Canonical,
SPARCE-GAN, and CGM-Simple — see the "Additional variants" comment in
create_algorithm_wrappers() for what each adds over its already-listed
counterpart and why a few other unwired variants (M-CELS, CEM-PP,
confetti_package_cf, moc_cf_diverse, the *_fast reimplementations, and the
11-strong FFT-CF variant family) were deliberately left out.

Note: Sub-SpaCE is designed primarily for multivariate time series and may not work
with univariate datasets like FordA. It will be skipped if incompatible.

On "TimeX" and "TimeX++" here: both names collide with unrelated saliency/
attribution papers that require a separately pre-trained explainer model this
repository has no training routine for (see `cfts/cf_timex/timex.py` and the
module docstring of `cfts/cf_timex_plus_plus/timex_plus_plus.py`). The
algorithms evaluated below are instead this repository's own counterfactual
reinterpretations that are trained inline, per call, like every other trained
method here: `cfts.cf_timex.timex_cf.timex_cf` (a Wachter-style optimiser with
a DTW class-prototype term, matching the "TimeX" method from the
TS-Counterfactual-Explanation-Bake-off benchmark) and
`cfts.cf_timex_plus_plus.timex_plus_plus.timexplusplus_cf` (TimeX++'s
explanation-extractor + conditioner architecture, retargeted from label
preservation to `target_class` — see that module's docstring for the full
reasoning). Both produce an actual counterfactual time series, not an
attribution map.

All 53 algorithms are evaluated, but to keep it readable, metrics_combined.png
(and the individual metrics_*.png / keane_*.png panels that feed into it) only
plot the top 10, ranked by a composite score of validity, proximity, and realism.
The complete, unfiltered results for every algorithm — including rank and
whether it made the top 10 — are always written to metrics_full_results.csv.

Features:
- Real counterfactual algorithms evaluation (53 methods)
- Comprehensive metrics across all categories
- Keane et al. (2021) evaluation metrics (validity, proximity, compactness)
- The single-function cfts.metrics.evaluate.evaluate_counterfactual() suite
  (the same one used by the cf_imfact comparison scripts), which adds a
  MASCOTS-style z-score-normalised distance and DTW distance
- Per-algorithm wall-clock runtime (Runtime_Mean_Seconds / Runtime_Std_Seconds /
  Runtime_N_Attempts in metrics_full_results.csv), timed around each algorithm
  call and recorded for every attempt, including failures and timeouts
- Algorithm benchmarking and comparison
- Top-10 visualization plus a full-results CSV for every algorithm
- Professional visualization of results
- Time series comparison plots (original vs counterfactuals)
- Statistical analysis of performance

Evaluated instances are drawn from the training split (dataset_train) rather
than the test split by default — see EVAL_SPLIT in main() to switch back to
'test'. Algorithms use the same split as their NUN/reference pool.

The instance selection itself (np.random.choice in main()) is seeded
(np.random.seed(13), set once near the top of this module) so the same
query instances are evaluated on every run — matching every other
example_*.py script in this directory, each of which seeds its own
single-instance np.random.randint selection the same way.

Checkpointing: each instance's result is pickled to metrics_checkpoint.pkl
right after it finishes, so a run killed partway through (e.g. by an
external timeout) can resume from the last completed instance on its next
invocation instead of redoing the whole ~3h13m-3h20m, 10-instance sweep from
scratch. See CHECKPOINT_PATH / RESUME_FROM_CHECKPOINT below. The checkpoint
file is removed once every instance completes; set METRICS_EVAL_RESUME=0 to
disable resuming and always start fresh.
"""

import os
import sys
import signal
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}/../')

# ---------------------------------------------------------------------------
# Logging – tee stdout/stderr to a file alongside terminal output
# ---------------------------------------------------------------------------
class _Tee:
    """Mirror writes to both the original stream and a log file."""
    def __init__(self, stream, logfile):
        self._stream = stream
        self._log = open(logfile, 'w', buffering=1)
    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
    def flush(self):
        self._stream.flush()
        self._log.flush()
    def __getattr__(self, name):
        return getattr(self._stream, name)

_log_dir = os.path.join(script_path, 'logs')
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, 'example_metrics_evaluation.log')
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)
print(f'Logging to: {_log_file}')
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# Import base modules
import base.model as bm
import base.data as bd

# Import counterfactual methods
import cfts.cf_native_guide.native_guide as ng
import cfts.cf_wachter.wachter as w
import cfts.cf_comte.comte as comte
import cfts.cf_sets.sets as sets
import cfts.cf_dandl.dandl as dandl
import cfts.cf_glacier.glacier as glacier
import cfts.cf_glacier.glacier_reimp as glacier_reimp
from cfts.cf_glacier.glacier_autoencoder import train_glacier_autoencoder, make_autoencoder_fns
import cfts.cf_multispace.multispace as ms
import cfts.cf_subspace.subspace as subspace
import cfts.cf_tsevo.tsevo as tsevo
import cfts.cf_lasts.lasts as lasts
import cfts.cf_tscf.tscf as tscf
import cfts.cf_fastpace.fastpace as fastpace
import cfts.cf_time_cf.time_cf as time_cf
import cfts.cf_sg_cf.sg_cf as sg_cf
from cfts.cf_mg_cf import mg_cf_generate_stumpy
import cfts.cf_latent_cf.latent_cf as latent_cf
import cfts.cf_discox.discox as discox
import cfts.cf_cels.cels as cels
from cfts.cf_fft_cf.fft_cf import fft_nn_cf
import cfts.cf_terce.terce as terce
import cfts.cf_ab_cf.ab_cf as ab_cf
import cfts.cf_cfwot.cfwot as cfwot
import cfts.cf_cgm.cgm as cgm
import cfts.cf_counts.counts as counts
import cfts.cf_sparce.sparce as sparce
import cfts.cf_cem.cem as cem_mod
import cfts.cf__abstract.abstract as abstract_mod
import cfts.cf_ts_tweaking.ts_tweaking as ts_tweaking
import cfts.cf_cfe4mts.cfe4mts as cfe4mts
import cfts.cf_confetti.confetti as confetti
import cfts.cf_mascots.mascots as mascots
import cfts.cf_imfact.imfact as imfact
from cfts.cf_timex.timex_cf import timex_cf
from cfts.cf_timex_plus_plus.timex_plus_plus import timexplusplus_cf

# Import metrics
from cfts.metrics import (
    CounterfactualEvaluator, benchmark_algorithms, create_metric_suite,
    l2_distance, prediction_change, percentage_changed_points,
    temporal_consistency, pairwise_distance, algorithmic_stability
)

# Import Keane et al. (2021) metrics
from cfts.metrics.keane import keane_validity, keane_proximity, keane_compactness, keane_evaluate_metrics

# Import the single-function evaluate.py metric suite (validity, proximity,
# sparsity, realism computed together) — the same function used by the
# cf_imfact comparison scripts (e.g. cf_imfact/experiments/compare_ucr.py).
from cfts.metrics.evaluate import evaluate_counterfactual

# Set up plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Fixed seed so the query instances selected below (np.random.choice) are
# the same on every run, across all example_*.py scripts, rather than a
# different random sample of instances each time.
np.random.seed(13)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Per-algorithm timeout (seconds). SIGALRM interrupts blocking C/Python code.
PER_ALGO_TIMEOUT = 180

# Runtime handicap for rank_algorithms()'s composite score: the seconds of
# mean per-instance runtime at which an algorithm's score is discounted to
# half. See rank_algorithms()'s docstring for the full formula — a fast
# algorithm keeps ~its full score, a slow one (approaching PER_ALGO_TIMEOUT)
# is discounted hard, so two algorithms with similar validity/proximity/
# realism no longer rank identically regardless of how long one takes.
RUNTIME_HANDICAP_SCALE = 30.0

# ---------------------------------------------------------------------------
# Incremental checkpointing.
#
# The 10-instance x 53-algorithm sweep in main() takes ~19-20 min/instance
# (~3h13m-3h20m total) — long enough that an external kill (e.g. run_all.py's
# per-script timeout, or a machine restart) previously discarded every
# already-completed instance, since results were only written to disk after
# ALL n_instances finished. Each instance's result is now pickled to
# CHECKPOINT_PATH right after it completes; on the next run, any instance
# already present there is loaded instead of recomputed, so the sweep just
# continues where it left off. Set METRICS_EVAL_RESUME=0 in the environment
# (or delete the checkpoint file) to force a clean run from scratch.
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = os.path.join('./', 'metrics_checkpoint.pkl')
RESUME_FROM_CHECKPOINT = os.environ.get('METRICS_EVAL_RESUME', '1') != '0'

# ---------------------------------------------------------------------------
# Fastest-methods, larger-sample sweep.
#
# select_fastest_algorithms() / run_fastest_methods_sweep() (near the bottom
# of this file, just above main()) take just the N quickest-running
# algorithms from a completed main() run's metrics_full_results.csv and
# re-evaluate *only* those across a much larger sample of instances (100 by
# default, vs. main()'s 10) — feasible specifically because the slow
# algorithms that make the full 53-algorithm sweep take ~3h13m-3h20m have
# already been dropped, so 100 instances of just the fast ones costs a small
# fraction of that. Opt-in, not part of the default run: invoke with
# `python example_metrics_evaluation.py --fast-sweep` (see the __main__
# guard at the bottom of this file) after a normal run has produced
# metrics_full_results.csv.
# ---------------------------------------------------------------------------
N_FASTEST_FOR_LARGE_SWEEP = 10
LARGE_SWEEP_N_INSTANCES = 100
FAST_SWEEP_CHECKPOINT_PATH = os.path.join('./', 'metrics_fast_sweep_checkpoint.pkl')
FAST_SWEEP_OUTPUT_DIR = os.path.join('./', 'fast_sweep_outputs')


def _save_checkpoint(path, instance_signature, completed):
    """Atomically persist per-instance results computed so far.

    Args:
        path: checkpoint file path.
        instance_signature: list of plain-int instance indices (the full
            instance_indices selection for this run), used on resume to
            confirm a found checkpoint matches the current seed/n_instances.
        completed: dict mapping position-in-instance_signature -> a dict
            with 'original_ts', 'target_class', and 'result' (the same
            structure evaluate_single_instance() returns).
    """
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(
            {'instance_signature': instance_signature, 'completed': completed}, f
        )
    os.replace(tmp_path, path)  # atomic on POSIX: never leaves a half-written file


def _load_checkpoint(path, instance_signature):
    """Load a checkpoint if present and it matches the current run's instance selection.

    Returns the {position: {...}} 'completed' dict (possibly empty).
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
    except Exception as e:
        print(f"⚠ Could not read checkpoint ({e}); starting fresh")
        return {}

    if checkpoint.get('instance_signature') != instance_signature:
        print("⚠ Checkpoint's instance selection doesn't match this run "
              "(different seed/n_instances?) — ignoring it")
        return {}

    completed = checkpoint.get('completed', {})
    if completed:
        print(f"✓ Resuming from checkpoint: {len(completed)}/{len(instance_signature)} "
              f"instance(s) already evaluated ({path})")
    return completed


def load_forda_data_and_model():
    """Load FordA dataset and trained model."""
    print('Loading FordA dataset...')
    _, dataset_train = bd.get_UCR_UEA_dataloader(split='train')
    _, dataset_test = bd.get_UCR_UEA_dataloader(split='test')
    
    output_classes = dataset_train.y_shape[1]
    model = bm.SimpleCNN(output_channels=output_classes).to(device)
    
    # Load pre-trained model
    models_dir = os.path.abspath(os.path.join(script_path, '..', 'models'))
    model_file = os.path.join(models_dir, f'simple_cnn_forda_{output_classes}.pth')
    
    if os.path.exists(model_file):
        print(f'Loading saved model from {model_file}')
        state = torch.load(model_file, map_location=device)
        model.load_state_dict(state)
        model.eval()
    else:
        raise FileNotFoundError(f"Model not found at {model_file}. Please train the model first using example_forda.py")
    
    return model, dataset_train, dataset_test


def create_algorithm_wrappers(dataset, model):
    """Create wrapper functions for counterfactual algorithms with consistent interface."""
    
    def native_guide_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = ng.native_guide_uni_cf(original_ts, model, dataset=dataset)
        return cf if cf is not None else original_ts
    
    def comte_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = comte.comte_cf_gradient(original_ts, model, target_class=target_class, dataset=dataset)
        return cf if cf is not None else original_ts

    def comte_ts_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = comte.comte_ts_cf_gradient(original_ts, model, target_class=target_class, dataset=dataset)
        return cf if cf is not None else original_ts

    def sets_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = sets.sets_cf(original_ts, model, target_class=target_class, dataset=dataset)
        return cf if cf is not None else original_ts

    def moc_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = dandl.moc_cf(original_ts, model, target_class=target_class, dataset=dataset)
        return cf if cf is not None else original_ts

    def wachter_gradient_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = w.wachter_gradient_cf(original_ts, model, target_class=target_class, dataset=dataset)
        return cf if cf is not None else original_ts

    def wachter_genetic_wrapper(original_ts, target_class=None, **kwargs):
        step_size = np.mean(dataset.std) + 0.2
        cf, _ = w.wachter_genetic_cf(original_ts, model, step_size=step_size, max_steps=100)
        return cf if cf is not None else original_ts

    def glacier_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = glacier.glacier_cf(original_ts, model, target_class=target_class, dataset=dataset)
        return cf if cf is not None else original_ts

    def multispace_wrapper(original_ts, target_class=None, **kwargs):
        # Multi-SpaCE doesn't support explicit target class, it finds the nearest different class
        cf, _ = ms.multispace_fast(original_ts, model, dataset=dataset,
                                  population_size=30,
                                  max_iterations=50,
                                  sparsity_weight=0.3,
                                  validity_weight=0.7,
                                  verbose=False)
        return cf if cf is not None else original_ts

    def subspace_wrapper(original_ts, target_class=None, **kwargs):
        # Sub-SpaCE is designed for multivariate time series
        # Check if data is univariate and skip if so
        ts_array = np.asarray(original_ts)

        # Determine if univariate: if 1D or if 2D with one dimension being 1
        is_univariate = (ts_array.ndim == 1 or
                        (ts_array.ndim == 2 and (ts_array.shape[0] == 1 or ts_array.shape[1] == 1)))

        if is_univariate:
            raise ValueError("Sub-SpaCE not compatible with univariate data (designed for multivariate time series)")

        # For multivariate data, proceed with Sub-SpaCE
        cf, _ = subspace.subspace_cf(original_ts, model,
                                     target_class=target_class,
                                     dataset=dataset,
                                     population_size=50,
                                     max_iter=100,
                                     alpha=0.8,
                                     beta=0.15,
                                     eta=0.05,
                                     invalid_penalization=20,
                                     init_pct=0.4,
                                     reinit=True,
                                     verbose=False)
        return cf if cf is not None else original_ts


    def tsevo_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = tsevo.tsevo_cf(original_ts, model,
                               target_class=target_class,
                               dataset=dataset,
                               population_size=30,
                               generations=30,
                               verbose=False)
        return cf if cf is not None else original_ts

    def lasts_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = lasts.lasts_cf(original_ts, model,
                               target_class=target_class,
                               dataset=dataset,
                               latent_dim=32,
                               n_iterations=200,
                               train_ae_epochs=10,
                               verbose=False)
        return cf if cf is not None else original_ts

    def tscf_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = tscf.tscf_cf(original_ts, model,
                            target_class=target_class,
                            dataset=dataset,
                            lambda_l1=0.01,
                            lambda_l2=0.01,
                            lambda_smooth=0.001,
                            learning_rate=0.1,
                            max_iterations=500,
                            verbose=False)
        return cf if cf is not None else original_ts

    def fastpace_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = fastpace.fastpace_cf(original_ts, model,
                                        target_class=target_class,
                                        dataset=dataset,
                                        n_planning_steps=10,
                                        intervention_step_size=0.3,
                                        lambda_proximity=1.0,
                                        lambda_plausibility=0.5,
                                        max_refinement_iterations=500,
                                        verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def time_cf_wrapper(original_ts, target_class=None, **kwargs):
        try:
            # NOTE: time_cf_generate()'s real kwargs are timegan_epochs/M, not
            # n_epochs/n_synthetic — those silently no-op'd as **unused** kwargs
            # were never accepted before and raised TypeError on every call,
            # which the except below swallowed as a plain "failed" result.
            cf, _ = time_cf.time_cf_generate(original_ts, model,
                                            target_class=target_class,
                                            dataset=dataset,
                                            timegan_epochs=5,
                                            M=50,
                                            verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def sg_cf_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = sg_cf.sg_cf(original_ts, model,
                               target_class=target_class,
                               dataset=dataset,
                               max_iter=200,
                               verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def mg_cf_wrapper(original_ts, target_class=None, **kwargs):
        try:
            # MG-CF with STUMPY optimization for faster motif mining
            from torch.utils.data import Subset
            subset_size = min(100, len(dataset))
            dataset_subset = Subset(dataset, range(subset_size))
            cf, _ = mg_cf_generate_stumpy(original_ts, model, target_class=target_class, dataset=dataset_subset,
                                         top_k=5,
                                         verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def latent_cf_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = latent_cf.latent_cf_generate(original_ts, model,
                                                target_class=target_class,
                                                dataset=dataset,
                                                latent_dim=8,
                                                max_iter=100,
                                                verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def discox_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = discox.discox_cf(original_ts, model,
                                             target_class=target_class,
                                             dataset=dataset,
                                             window_size=20,
                                             verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def cels_wrapper(original_ts, target_class=None, **kwargs):
        try:
            # CELS requires training data for nearest unlike neighbor
            X_train = np.array([dataset[i][0] for i in range(min(100, len(dataset)))])
            y_train = np.array([dataset[i][1] for i in range(min(100, len(dataset)))])
            cf, _ = cels.cels_generate(original_ts, model, X_train, y_train,
                                      target_class=target_class,
                                      max_iter=100,
                                      verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def fft_cf_wrapper(original_ts, target_class=None, **kwargs):
        try:
            # Using nearest neighbor FFT blending approach
            cf, _ = fft_nn_cf(original_ts, model,
                            target_class=target_class,
                            dataset=dataset,
                            k=5,
                            blend_ratio=0.5,
                            frequency_bands="all",
                            verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def terce_wrapper(original_ts, target_class=None, **kwargs):
        try:
            # TERCE requires training data for nearest unlike neighbor and rule mining
            X_train = np.array([dataset[i][0] for i in range(min(100, len(dataset)))])
            y_train = np.array([dataset[i][1] for i in range(min(100, len(dataset)))])
            cf, _ = terce.terce_generate(original_ts, model, X_train, y_train,
                                        target_class=target_class,
                                        n_regions=5,
                                        window_size_ratio=0.1,
                                        verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def ab_cf_wrapper(original_ts, target_class=None, **kwargs):
        try:
            # AB-CF requires training data for nearest unlike neighbor retrieval
            X_train = np.array([dataset[i][0] for i in range(min(100, len(dataset)))])
            y_train = np.array([dataset[i][1] for i in range(min(100, len(dataset)))])
            cf, _ = ab_cf.ab_cf_generate(original_ts, model, X_train, y_train,
                                        target_class=target_class,
                                        n_segments=10,
                                        window_size_ratio=0.1,
                                        verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def cfwot_wrapper(original_ts, target_class=None, **kwargs):
        try:
            # cfwot() expects (K, D) = (timesteps, features); the rest of this
            # harness uses (channels, length) — transpose in and back out, or
            # every univariate series gets silently read as D=length "features"
            # of a single timestep and crashes at the model's conv1d call.
            cf, _ = cfwot.cfwot(original_ts.T, model,
                               target_class=target_class,
                               M_E=20,
                               M_T=20,
                               verbose=False)
            return cf.T if cf is not None else original_ts
        except Exception:
            return original_ts

    def cgm_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = cgm.cgm_generate(original_ts, model,
                                    target_class=target_class,
                                    dataset=dataset,
                                    latent_dim=16,
                                    max_iter=100,
                                    verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def counts_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = counts.counts_cf_with_pretrained_model(original_ts, model,
                                                          target_class=target_class,
                                                          dataset=dataset,
                                                          latent_dim=16,
                                                          max_iter=100,
                                                          verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def sparce_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = sparce.sparce_gradient_cf(original_ts, model,
                                             target_class=target_class,
                                             max_iter=100,
                                             verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def cem_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = cem_mod.cem_cf(
                original_ts, model,
                mode='PN',
                autoencoder=None,
                kappa=0.5,
                beta=0.1,
                gamma=0.2,
                c_init=10.0,
                c_steps=3,
                max_iterations=200,
                learning_rate=1e-2,
                verbose=False,
            )
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def abstract_cf_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = abstract_mod.abstract_cf(
                original_ts, model, target_class=target_class, dataset=dataset,
                max_iter=100,
                noise_scale=0.05,
                escalate_every=10,
                verbose=False,
            )
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def ts_tweaking_knn_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = ts_tweaking.ts_tweaking_knn_cf(
                original_ts, model, target_class=target_class, dataset=dataset, verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def ts_tweaking_irrev_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = ts_tweaking.ts_tweaking_irreversible_cf(
                original_ts, model, target_class=target_class, dataset=dataset, verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def ts_tweaking_rev_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = ts_tweaking.ts_tweaking_reversible_cf(
                original_ts, model, target_class=target_class, dataset=dataset, verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def cfe4mts_wrapper(original_ts, target_class=None, **kwargs):
        try:
            # CFE4MTS trains a noiser/discriminator per call to match this
            # harness's single-call signature; kept small to stay within budget.
            cf, _ = cfe4mts.cfe4mts_cf(original_ts, dataset, model,
                                      target_class=target_class,
                                      epochs=20,
                                      max_train_samples=100,
                                      verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def confetti_wrapper(original_ts, target_class=None, **kwargs):
        try:
            # confetti_nsga_cf mirrors the official CONFETTI algorithm more
            # closely than confetti_genetic_cf (NUN search + NSGA-II window search).
            cf, _ = confetti.confetti_nsga_cf(original_ts, model,
                                             target_class=target_class,
                                             dataset=dataset,
                                             population_size=30,
                                             max_generations=20,
                                             max_samples=100,
                                             seed=42,
                                             verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def mascots_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = mascots.mascots_cf(original_ts, model,
                                      target_class=target_class,
                                      dataset=dataset,
                                      max_samples=100,
                                      verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def imfact_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = imfact.imfact_cf(original_ts, model,
                                    target_class=target_class,
                                    dataset=dataset,
                                    max_samples=100,
                                    verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def timex_wrapper(original_ts, target_class=None, **kwargs):
        try:
            # Bake-off-style TimeX-CF (Wachter loss + DTW class-prototype term),
            # not Harvard's saliency-based "TimeX" — see module docstring of
            # cfts/cf_timex/timex_cf.py for the naming collision.
            cf, _ = timex_cf(original_ts, model,
                            target_class=target_class,
                            dataset=dataset,
                            max_samples=100,
                            verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def timex_plus_plus_wrapper(original_ts, target_class=None, **kwargs):
        try:
            # IB explanation extractor + conditioner, retargeted from label
            # preservation to target_class — see module docstring of
            # cfts/cf_timex_plus_plus/timex_plus_plus.py for the reasoning.
            cf, _ = timexplusplus_cf(original_ts, model,
                                    target_class=target_class,
                                    dataset=dataset,
                                    max_train_samples=100,
                                    verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    # -----------------------------------------------------------------------
    # Additional variants of already-represented methods.
    #
    # Each method above wires in exactly one algorithm/implementation; several
    # of the underlying cfts modules ship more than one — a genuinely
    # different technique or fidelity/speed tradeoff, not a legacy duplicate
    # or a pure performance twin of one already wired above. Wired in here:
    #
    #   - GLACIER: the 8 bake-off-named variants (AE/NoAE crossed with 4
    #     step-weight modes) from glacier_reimp.GLACIER_VARIANTS, alongside
    #     the already-wired 'GLACIER' (a separate, older implementation in
    #     glacier.py). The autoencoder and global step weights these need are
    #     query-independent, so they're trained/computed once here rather
    #     than per instance.
    #   - InfoCELS (soft-mask CELS variant) alongside 'CELS'.
    #   - NG-DBA (DTW-barycenter blending) alongside 'Native Guide' (NG-CAM).
    #   - COMTE-Distractor (the real distractor-swap algorithm) and
    #     COMTE-Advanced-Gradient (extra distance/constraint options)
    #     alongside 'COMTE'/'COMTE-TS' (both gradient-based).
    #   - Multi-SpaCE-Canonical (the paper's own two-stage NSGA-II search)
    #     alongside 'Multi-SpaCE' (a speed-adapted approximation of it).
    #   - SPARCE-GAN (adversarial architecture) alongside 'SPARCE'
    #     (gradient-based).
    #   - CGM-Simple (direct optimisation, no VAE) alongside 'CGM' (VAE-based).
    #
    # Deliberately NOT wired in:
    #   - M-CELS: requires multivariate data; FordA is univariate (same
    #     reason Sub-SpaCE's wrapper raises/skips above).
    #   - CEM-PP: by cem_cf's own docstring, PP mode keeps the *original*
    #     prediction and returns a delta, not x0+delta — it isn't a
    #     counterfactual under this harness's validity-based evaluation.
    #   - confetti_package_cf: wraps the official, external "confetti" pip
    #     package, which isn't a dependency of this repo (see
    #     confetti.py's own comment on why confetti_nsga_cf was picked as
    #     the default over it).
    #   - moc_cf_diverse: returns a list of diverse counterfactuals, not the
    #     single (cf, scores) pair this harness evaluates per instance.
    #   - subspace_fast / sg_cf_fast / time_cf_generate_fast / counts_cf_fast:
    #     each is documented as a faster, (near-)equivalent reimplementation
    #     of an already-wired method, not a different algorithm — wiring in
    #     both would evaluate the same method twice.
    #   - The FFT-CF family (11 more variants in cf_fft_cf/fft_cf.py) is
    #     intentionally excluded — see example_metrics_evaluation.py's
    #     module docstring.
    # -----------------------------------------------------------------------

    # --- GLACIER bake-off variants: shared, query-independent setup, done
    # once here rather than per instance/per call. ---
    glacier_autoencoder_fns = None
    glacier_global_weights = None
    try:
        glacier_device = next(model.parameters()).device
        X_glacier = np.asarray(dataset.X, dtype=np.float32)
        # train_glacier_autoencoder's own default device (CUDA if available)
        # can silently differ from `model`'s — force them to match, or the
        # "-AE-" variants below crash with a cross-device tensor mismatch.
        ae_model = train_glacier_autoencoder(
            X_glacier, n_features=X_glacier.shape[1], n_epochs=30,
            device=glacier_device, verbose=False,
        )
        glacier_autoencoder_fns = make_autoencoder_fns(ae_model)

        def _glacier_predict_label(x_1d):
            with torch.no_grad():
                out = model(torch.from_numpy(np.asarray(x_1d, dtype=np.float32))
                             .reshape(1, 1, -1).to(glacier_device))
            return int(torch.argmax(out, dim=-1).item())

        y_glacier = np.argmax(dataset.y, axis=1) if np.ndim(dataset.y) > 1 else np.asarray(dataset.y)
        glacier_global_weights = glacier_reimp.compute_global_step_weights(
            X_glacier.reshape(X_glacier.shape[0], -1), y_glacier,
            _glacier_predict_label, X_glacier.shape[-1], max_samples=100,
        )
    except Exception as e:
        print(f"  Warning: GLACIER variant setup (autoencoder/global weights) failed: {e}")

    def _make_glacier_variant_wrapper(variant_name):
        def _wrapper(original_ts, target_class=None, **kwargs):
            try:
                spec = glacier_reimp.GLACIER_VARIANTS[variant_name]
                if spec['use_ae']:
                    if glacier_autoencoder_fns is None:
                        return original_ts  # AE setup failed above; skip rather than crash.
                    ae_arg = glacier_autoencoder_fns
                else:
                    ae_arg = None
                extra = {}
                if spec['step_weights'] == 'global' and glacier_global_weights is not None:
                    extra['precomputed_global_weights'] = glacier_global_weights
                cf, _ = glacier_reimp.glacier_variant(
                    variant_name, original_ts, model, autoencoder=ae_arg,
                    dataset=dataset, target_label=target_class, max_iter=100,
                    **extra,
                )
                if cf is None:
                    return original_ts
                # glacier_reimp always returns a flat (L,) array regardless
                # of original_ts's own orientation/channel dim (unlike every
                # other <name>_cf in this repo) — reshape back to match, or
                # downstream model/metric calls silently misinterpret the
                # channel axis as a batch axis.
                return np.asarray(cf, dtype=np.float32).reshape(np.asarray(original_ts).shape)
            except Exception:
                return original_ts
        return _wrapper

    glacier_variant_wrappers = {
        name: _make_glacier_variant_wrapper(name) for name in glacier_reimp.GLACIER_VARIANTS
    }

    def infocels_wrapper(original_ts, target_class=None, **kwargs):
        try:
            X_train = np.array([dataset[i][0] for i in range(min(100, len(dataset)))])
            y_train = np.array([dataset[i][1] for i in range(min(100, len(dataset)))])
            cf, _ = cels.infocels_generate(original_ts, model, X_train, y_train,
                                          target_class=target_class,
                                          max_iter=100,
                                          verbose=False)
            if cf is None:
                return original_ts
            # infocels_generate returns a batched (1, C, L) array, unlike
            # original_ts's plain (C, L)/(L,) orientation — reshape back.
            return np.asarray(cf, dtype=np.float32).reshape(np.asarray(original_ts).shape)
        except Exception:
            return original_ts

    def ng_dba_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = ng.native_guide_dba_cf(original_ts, model,
                                          target_class=target_class,
                                          dataset=dataset,
                                          max_samples=100,
                                          verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def comte_distractor_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = comte.comte_cf(original_ts, model,
                                  target_class=target_class,
                                  dataset=dataset,
                                  max_samples=100,
                                  verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def comte_advanced_gradient_wrapper(original_ts, target_class=None, **kwargs):
        try:
            # comte_cf_advanced_gradient() calls model.to(device) internally
            # and, unlike every other method here, defaults device to
            # 'cuda' whenever it's available rather than deriving it from
            # `model` — silently moving the shared model out from under
            # every other algorithm if it happened to be kept on a
            # different device. Pin it explicitly to model's own device.
            cf, _ = comte.comte_cf_advanced_gradient(original_ts, model,
                                                     target_class=target_class,
                                                     dataset=dataset,
                                                     device=next(model.parameters()).device)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def multispace_canonical_wrapper(original_ts, target_class=None, **kwargs):
        try:
            # Paper's own two-stage NSGA-II search (see multispace_cf's
            # docstring), scaled down from its defaults (population_size=100,
            # grouped_iter=75, pruning_iter=25) to fit this harness's budget.
            cf, _ = ms.multispace_cf(original_ts, model,
                                    target_class=target_class,
                                    dataset=dataset,
                                    population_size=30,
                                    grouped_iter=30,
                                    pruning_iter=15,
                                    verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def sparce_gan_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = sparce.sparce_gan_cf(original_ts, model,
                                        target_class=target_class,
                                        dataset=dataset,
                                        num_epochs=50,
                                        verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    def cgm_simple_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = cgm.cgm_generate_simple(original_ts, model,
                                           target_class=target_class,
                                           dataset=dataset,
                                           max_iter=100,
                                           verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts

    return {
        'Native Guide': native_guide_wrapper,
        'COMTE': comte_wrapper,
        'COMTE-TS': comte_ts_wrapper,
        'SETS': sets_wrapper,
        'MOC': moc_wrapper,
        'Wachter Gradient': wachter_gradient_wrapper,
        'Wachter Genetic': wachter_genetic_wrapper,
        'GLACIER': glacier_wrapper,
        'Multi-SpaCE': multispace_wrapper,
        'Sub-SpaCE': subspace_wrapper,
        'TSEvo': tsevo_wrapper,
        'LASTS': lasts_wrapper,
        'TSCF': tscf_wrapper,
        'FASTPACE': fastpace_wrapper,
        'TIME-CF': time_cf_wrapper,
        'SG-CF': sg_cf_wrapper,
        'MG-CF': mg_cf_wrapper,
        'Latent-CF': latent_cf_wrapper,
        'DiSCoX': discox_wrapper,
        'CELS': cels_wrapper,
        'FFT-CF': fft_cf_wrapper,
        'TERCE': terce_wrapper,
        'AB-CF': ab_cf_wrapper,
        'CFWOT': cfwot_wrapper,
        'CGM': cgm_wrapper,
        'COUNTS': counts_wrapper,
        'SPARCE': sparce_wrapper,
        'CEM-PN': cem_wrapper,
        'Abstract-CF': abstract_cf_wrapper,
        'TS-Tweaking-kNN': ts_tweaking_knn_wrapper,
        'TS-Tweaking-Irrev': ts_tweaking_irrev_wrapper,
        'TS-Tweaking-Rev': ts_tweaking_rev_wrapper,
        'CFE4MTS': cfe4mts_wrapper,
        'CONFETTI': confetti_wrapper,
        'MASCOTS': mascots_wrapper,
        'IMFACT': imfact_wrapper,
        'TimeX': timex_wrapper,
        'TimeX++': timex_plus_plus_wrapper,
        **glacier_variant_wrappers,
        'InfoCELS': infocels_wrapper,
        'NG-DBA': ng_dba_wrapper,
        'COMTE-Distractor': comte_distractor_wrapper,
        'COMTE-Advanced-Gradient': comte_advanced_gradient_wrapper,
        'Multi-SpaCE-Canonical': multispace_canonical_wrapper,
        'SPARCE-GAN': sparce_gan_wrapper,
        'CGM-Simple': cgm_simple_wrapper,
    }


def pytorch_model_wrapper(model):
    """Create a wrapper for PyTorch model to work with metrics."""
    def model_wrapper(ts):
        if isinstance(ts, np.ndarray):
            ts_tensor = torch.from_numpy(ts).float().to(device)
            if ts_tensor.dim() == 1:
                ts_tensor = ts_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            elif ts_tensor.dim() == 2:
                ts_tensor = ts_tensor.unsqueeze(0)  # Add batch dim
        else:
            ts_tensor = ts
        
        with torch.no_grad():
            output = model(ts_tensor)
            return torch.softmax(output, dim=-1).squeeze().cpu().numpy()
    
    return model_wrapper


def evaluate_single_instance(original_ts, label, model_wrapper, algorithms, dataset):
    """Evaluate all algorithms on a single time series instance."""
    print(f"\nEvaluating instance with original class: {label}")
    
    # Determine target class (flip to opposite class)
    target_class = 1 - label  # For binary classification
    
    # Generate counterfactuals with each algorithm
    counterfactuals = {}
    successful_algorithms = []
    # Wall-clock runtime of the algorithm call itself (seconds), recorded for
    # every attempt — including timeouts and errors — not just successes.
    runtimes = {}

    def _sigalrm_handler(signum, frame):
        raise TimeoutError(f"Algorithm timed out after {PER_ALGO_TIMEOUT}s")

    for name, algorithm in tqdm(algorithms.items(), desc="  Generating CFs", leave=False):
        start_time = time.perf_counter()
        try:
            print(f"  Generating counterfactual with {name}...")
            signal.signal(signal.SIGALRM, _sigalrm_handler)
            signal.alarm(PER_ALGO_TIMEOUT)
            try:
                cf = algorithm(original_ts, target_class=target_class)
            finally:
                signal.alarm(0)  # Cancel alarm regardless of outcome
            runtimes[name] = time.perf_counter() - start_time

            # Check if prediction actually changed
            original_pred = model_wrapper(original_ts)
            cf_pred = model_wrapper(cf)

            original_class = np.argmax(original_pred)
            cf_class = np.argmax(cf_pred)

            if cf_class == target_class:
                counterfactuals[name] = cf
                successful_algorithms.append(name)
                print(f"    ✓ Success: {original_class} → {cf_class} ({runtimes[name]:.2f}s)")
            else:
                print(f"    ✗ Failed: {original_class} → {cf_class} (target: {target_class}, {runtimes[name]:.2f}s)")

        except TimeoutError as e:
            runtimes[name] = time.perf_counter() - start_time
            print(f"    ✗ Timeout: {name} — {str(e)}")
        except Exception as e:
            runtimes[name] = time.perf_counter() - start_time
            print(f"    ✗ Error with {name}: {str(e)}")

    if not counterfactuals:
        print("  No successful counterfactuals generated!")
        return None
    
    # Initialize evaluator with reference data
    reference_data = np.array([dataset.X[i] for i in range(min(100, len(dataset.X)))])
    evaluator = CounterfactualEvaluator(reference_data=reference_data)
    
    # Evaluate each counterfactual
    results = {}
    for name, cf in counterfactuals.items():
        try:
            result = evaluator.evaluate_single(
                original_ts=original_ts,
                counterfactual_ts=cf,
                model=model_wrapper,
                target_class=target_class
            )
            results[name] = result
            print(f"  Evaluated {name}: {len(result)} metrics computed")
        except Exception as e:
            print(f"  Error evaluating {name}: {str(e)}")
    
    # Evaluate diversity if multiple counterfactuals
    if len(counterfactuals) >= 2:
        try:
            diversity_results = evaluator.evaluate_multiple(
                original_ts=original_ts,
                counterfactuals=list(counterfactuals.values()),
                model=model_wrapper,
                target_class=target_class
            )
            print(f"  Diversity evaluation: {len(diversity_results)} metrics computed")
        except Exception as e:
            print(f"  Error in diversity evaluation: {str(e)}")
            diversity_results = {}
    else:
        diversity_results = {}
    
    return {
        'counterfactuals': counterfactuals,
        'individual_results': results,
        'diversity_results': diversity_results,
        'successful_algorithms': successful_algorithms,
        'runtimes': runtimes
    }


def compute_runtime_dataframe(all_results):
    """Collect per-instance algorithm runtimes (seconds) into a long-format DataFrame.

    Unlike compute_metrics_dataframe(), this includes every algorithm *attempt*
    (successes, failures, and timeouts alike) — runtime is measured around the
    algorithm call itself, before the success/failure check.
    """
    all_runtime_data = []
    for instance_idx, instance_results in enumerate(all_results):
        if instance_results is None:
            continue

        for algorithm, elapsed in instance_results.get('runtimes', {}).items():
            all_runtime_data.append({
                'Instance': instance_idx,
                'Algorithm': algorithm,
                'Runtime': elapsed
            })

    if not all_runtime_data:
        return None

    return pd.DataFrame(all_runtime_data)


def compute_metrics_dataframe(all_results):
    """Collect all individual per-instance metrics into one long-format DataFrame."""
    all_metrics_data = []
    for instance_idx, instance_results in enumerate(all_results):
        if instance_results is None:
            continue

        for algorithm, metrics in instance_results['individual_results'].items():
            for metric_name, value in metrics.items():
                all_metrics_data.append({
                    'Instance': instance_idx,
                    'Algorithm': algorithm,
                    'Metric': metric_name,
                    'Value': value
                })

    if not all_metrics_data:
        return None

    return pd.DataFrame(all_metrics_data)


def rank_algorithms(summary_stats, algorithm_names,
                     key_metrics=('prediction_change', 'normalized_distance', 'temporal_consistency'),
                     runtime_by_algorithm=None, runtime_scale=RUNTIME_HANDICAP_SCALE):
    """
    Rank algorithms by a composite score combining validity (prediction_change),
    proximity (inverted normalized_distance, since lower is better), and realism
    (temporal_consistency).

    Args:
        summary_stats, algorithm_names, key_metrics: as before — the
            (Algorithm, Metric) -> mean/std/median table and the metrics to
            blend for the base validity/proximity/realism score.
        runtime_by_algorithm: optional {algorithm_name: mean_runtime_seconds}
            dict (e.g. df_runtime.groupby('Algorithm')['Runtime'].mean(),
            from compute_runtime_dataframe() — every attempt, not just
            successes). When given, the base score is multiplied by a
            runtime handicap: 1 / (1 + mean_runtime / runtime_scale). A
            near-instant algorithm keeps almost its full score (handicap≈1);
            one averaging runtime_scale seconds is discounted to ~0.5; one
            averaging PER_ALGO_TIMEOUT (i.e. it mostly timed out) is
            discounted hard (~0.14 at the default scale). An algorithm
            absent from runtime_by_algorithm (no recorded attempts) gets no
            handicap (factor 1.0) rather than being penalized for a data gap.
        runtime_scale: seconds at which the handicap reaches 0.5 (default
            RUNTIME_HANDICAP_SCALE).

    Returns a list of (algorithm_name, score) tuples, best (highest score) first.
    """
    algorithm_scores = {}
    for algorithm in algorithm_names:
        scores = []
        for metric in key_metrics:
            try:
                if metric == 'normalized_distance':
                    # Lower is better - invert for ranking
                    score = 1 / (1 + summary_stats.loc[(algorithm, metric), 'mean'])
                else:
                    # prediction_change / temporal_consistency: higher is better
                    score = summary_stats.loc[(algorithm, metric), 'mean']
                scores.append(score)
            except KeyError:
                continue

        if scores:
            composite = np.mean(scores)
            if runtime_by_algorithm is not None and algorithm in runtime_by_algorithm:
                mean_runtime = runtime_by_algorithm[algorithm]
                runtime_handicap = 1.0 / (1.0 + mean_runtime / runtime_scale)
                composite *= runtime_handicap
            algorithm_scores[algorithm] = composite

    return sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=True)


def visualize_composite_scores(ranked_algorithms, output_dir='./', top_n=10):
    """
    Bar chart of the composite score (validity/proximity/realism blend,
    discounted by a runtime handicap, from rank_algorithms()) for the
    top-ranked algorithms — the single "who won overall" panel that
    metrics_*.png / keane_*.png / evalpy_*.png (each one metric at a time)
    don't show on their own.

    Args:
        ranked_algorithms: list of (algorithm_name, score) tuples, best
            (highest score) first — the output of rank_algorithms().
        output_dir: Directory to save the plot.
        top_n: How many top-ranked algorithms to show.

    Returns:
        List containing the path to the saved plot (empty if there was
        nothing to plot).
    """
    if not ranked_algorithms:
        print("No ranked algorithms to visualize composite scores for!")
        return []

    top = ranked_algorithms[:top_n]
    names = [name for name, _ in top]
    scores = [score for _, score in top]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(names))))
    bars = ax.barh(names, scores, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Composite Score', fontweight='bold', fontsize=11)
    ax.set_title(f'Composite Score — Top {len(names)} Algorithms\n'
                 '(validity + proximity + realism blend, runtime-handicapped, Higher is Better)',
                 fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')

    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    for i, (bar, val) in enumerate(zip(bars, scores)):
        ax.text(val + 0.02 * x_range, i, f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    fpath = os.path.join(output_dir, 'composite_scores.png')
    plt.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nComposite score plot saved as: {fpath}")
    return [fpath]


def create_results_visualization(df, output_dir='./', top_algorithms=None):
    """
    Create one plot per metric category and save each as a separate PNG.

    Args:
        df: long-format metrics DataFrame from compute_metrics_dataframe().
        output_dir: directory to save the plots.
        top_algorithms: optional ordered list of algorithm names to restrict
            the plots to (e.g. the best N by composite score). Summary
            statistics are still computed over the *full* df regardless, so
            callers can export complete results even when the plots are
            capped.
    """
    if df is None or df.empty:
        print("No results to visualize!")
        return [], None

    plot_df = df if top_algorithms is None else df[df['Algorithm'].isin(top_algorithms)]

    # One figure per metric category
    metric_categories = {
        'Validity':  ['prediction_change', 'class_confidence', 'boundary_distance'],
        'Proximity': ['l2_distance', 'manhattan_distance', 'normalized_distance'],
        'Sparsity':  ['l0_norm', 'percentage_changed', 'segment_sparsity'],
        'Realism':   ['temporal_consistency', 'range_validity',
                      'autocorr_preservation', 'statistical_similarity'],
    }

    # Preserve rank order (best first) on the x-axis when a top-N list is given,
    # otherwise fall back to the previous alphabetical ordering.
    algo_order = list(top_algorithms) if top_algorithms is not None else None

    output_filenames = []
    for category_name, metric_list in metric_categories.items():
        cat_data = plot_df[plot_df['Metric'].isin(metric_list)]
        if cat_data.empty:
            continue

        # Keep only metrics that actually appear in the data
        present_metrics = [m for m in metric_list if m in cat_data['Metric'].unique()]
        n_metrics = len(present_metrics)
        if n_metrics == 0:
            continue

        n_algos = len(cat_data['Algorithm'].unique())
        subplot_w = max(10, n_algos * 0.5)
        subplot_h = 4

        fig, axes = plt.subplots(
            n_metrics, 1,
            figsize=(subplot_w, subplot_h * n_metrics),
            squeeze=False,
        )

        title_suffix = f' — Top {len(algo_order)}' if algo_order is not None else ''
        fig.suptitle(
            f'{category_name} Metrics — Counterfactual Algorithm Comparison{title_suffix}',
            fontweight='bold', fontsize=15, y=1.01,
        )

        for row, metric_name in enumerate(present_metrics):
            ax = axes[row, 0]
            metric_data = cat_data[cat_data['Metric'] == metric_name]
            present_algos = set(metric_data['Algorithm'].unique())
            order = ([a for a in algo_order if a in present_algos] if algo_order is not None
                      else sorted(present_algos))
            sns.boxplot(data=metric_data, x='Algorithm', y='Value', ax=ax, order=order)
            ax.set_title(metric_name.replace('_', ' ').title(), fontweight='bold', fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel('Value', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        fname = os.path.join(output_dir, f'metrics_{category_name.lower()}.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"\n{category_name} metrics plot saved as: {fname}")
        output_filenames.append(fname)

    # Summary statistics table — computed over the FULL (unfiltered) df so
    # nothing is lost even when the plots above are capped to top_algorithms.
    summary_stats = df.groupby(['Algorithm', 'Metric'])['Value'].agg(
        ['mean', 'std', 'median']).round(3)
    print("\n=== Summary Statistics ===")
    print(summary_stats)

    return output_filenames, summary_stats


def visualize_counterfactuals(all_results, output_dir='./'):
    """Create line plots comparing original time series with generated counterfactuals."""
    
    # Filter valid results
    valid_results = [r for r in all_results if r is not None and r['counterfactuals']]
    
    if not valid_results:
        print("No counterfactuals to visualize!")
        return None
    
    # Determine grid size
    n_instances = len(valid_results)
    n_cols = min(3, n_instances)
    n_rows = (n_instances + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_instances == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_instances > 1 else axes
    
    fig.suptitle('Original vs Counterfactual Time Series Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Color palette for different algorithms
    colors = plt.cm.Set2(np.linspace(0, 1, 10))
    
    for idx, (ax, result) in enumerate(zip(axes[:n_instances], valid_results)):
        counterfactuals = result['counterfactuals']
        
        if not counterfactuals:
            ax.axis('off')
            continue
        
        # Get original time series from first counterfactual (they all share same original)
        # We'll need to extract this from context or pass it separately
        # For now, we'll plot the counterfactuals
        
        # Plot each counterfactual
        legend_labels = []
        for cf_idx, (algorithm_name, cf_ts) in enumerate(counterfactuals.items()):
            # Flatten if multi-dimensional
            if cf_ts.ndim > 1:
                cf_flat = cf_ts.flatten() if cf_ts.shape[1] == 1 else cf_ts.mean(axis=1)
            else:
                cf_flat = cf_ts
            
            ax.plot(cf_flat, label=algorithm_name, 
                   color=colors[cf_idx % len(colors)], 
                   linewidth=2, alpha=0.7)
            legend_labels.append(algorithm_name)
        
        ax.set_title(f'Instance {idx + 1}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_instances, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = os.path.join(output_dir, 'counterfactual_timeseries_comparison.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nCounterfactual time series comparison saved as: {output_filename}")
    plt.close()
    
    return output_filename


def evaluate_keane_metrics_batch(original_ts_list, all_results, model_wrapper, target_classes_list):
    """
    Evaluate Keane et al. (2021) metrics across all algorithms.
    
    Args:
        original_ts_list: List of original time series
        all_results: List of evaluation results with counterfactuals
        model_wrapper: Model wrapper function
        target_classes_list: List of target classes for each instance
    
    Returns:
        DataFrame with Keane metrics for each algorithm
    """
    print("\n=== Evaluating Keane et al. (2021) Metrics ===")
    print("Reference: Keane, M. T., et al. (2021). If only we had better counterfactual")
    print("explanations. IJCAI 2021.\n")
    
    # Collect counterfactuals by algorithm
    algorithm_counterfactuals = {}
    algorithm_originals = {}
    algorithm_targets = {}
    
    # Group counterfactuals by algorithm
    for orig_ts, result, target_class in zip(original_ts_list, all_results, target_classes_list):
        if result is None or not result.get('counterfactuals'):
            continue
            
        for algorithm_name, cf_ts in result['counterfactuals'].items():
            if algorithm_name not in algorithm_counterfactuals:
                algorithm_counterfactuals[algorithm_name] = []
                algorithm_originals[algorithm_name] = []
                algorithm_targets[algorithm_name] = []
            
            algorithm_counterfactuals[algorithm_name].append(cf_ts)
            algorithm_originals[algorithm_name].append(orig_ts)
            algorithm_targets[algorithm_name].append(target_class)
    
    if not algorithm_counterfactuals:
        print("No counterfactuals available for Keane metrics evaluation!")
        return None
    
    # Evaluate each algorithm
    keane_results = []
    
    for algorithm_name in sorted(algorithm_counterfactuals.keys()):
        originals = algorithm_originals[algorithm_name]
        counterfactuals = algorithm_counterfactuals[algorithm_name]
        targets = algorithm_targets[algorithm_name]
        
        print(f"\nEvaluating {algorithm_name}:")
        print(f"  Number of counterfactuals: {len(counterfactuals)}")
        
        # Calculate Keane metrics
        try:
            # 1. Validity
            val_score = keane_validity(originals, counterfactuals, model_wrapper, target_classes=targets)
            print(f"  ✓ Validity: {val_score:.2%} (fraction achieving target class)")

            # 2. Proximity
            prox_score = keane_proximity(originals, counterfactuals)
            print(f"  ✓ Proximity: {prox_score:.4f} (average L2 distance)")

            # 3. Compactness
            comp_score = keane_compactness(originals, counterfactuals, tolerance=0.01)
            print(f"  ✓ Compactness: {comp_score:.2%} (fraction unchanged)")
            
            keane_results.append({
                'Algorithm': algorithm_name,
                'Validity': val_score,
                'Proximity': prox_score,
                'Compactness': comp_score,
                'N_Samples': len(counterfactuals)
            })
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create DataFrame
    if keane_results:
        df_keane = pd.DataFrame(keane_results)
        df_keane = df_keane.sort_values('Validity', ascending=False)
        
        print("\n" + "="*70)
        print("Keane et al. (2021) Metrics Summary")
        print("="*70)
        print(df_keane.to_string(index=False))
        print("="*70)
        print("\nMetric Interpretation:")
        print("  - Validity: Higher is better (1.0 = 100% successful)")
        print("  - Proximity: Lower is better (smaller distance to original)")
        print("  - Compactness: Higher is better (more values unchanged)")
        print("="*70)
        
        return df_keane
    
    return None


def visualize_keane_metrics(df_keane, output_dir='./', top_algorithms=None):
    """
    Create one plot per Keane et al. (2021) metric, each saved as a separate PNG.

    Args:
        df_keane: DataFrame with Keane metrics
        output_dir: Directory to save the plots
        top_algorithms: optional list of algorithm names to restrict the bars
            to (e.g. the best N by composite score). df_keane itself is left
            untouched so callers can still export the full table elsewhere.

    Returns:
        List of paths to saved plots
    """
    if df_keane is None or df_keane.empty:
        print("No Keane metrics to visualize!")
        return []

    plot_keane = (df_keane if top_algorithms is None
                  else df_keane[df_keane['Algorithm'].isin(top_algorithms)])
    if plot_keane.empty:
        print("No Keane metrics left to visualize after filtering to top_algorithms!")
        return []

    algorithms = plot_keane['Algorithm'].tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
    output_filenames = []
    title_suffix = f' — Top {len(algorithms)}' if top_algorithms is not None else ''

    metric_specs = [
        ('Validity',    'Validity Score',        'keane_validity.png',    True),
        ('Proximity',   'Proximity (L2 Distance)', 'keane_proximity.png',  False),
        ('Compactness', 'Compactness Score',      'keane_compactness.png', True),
    ]

    for col, xlabel, filename, higher_better in metric_specs:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(algorithms))))
        bars = ax.barh(algorithms, plot_keane[col], color=colors, alpha=0.8, edgecolor='black')
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=11)
        direction = '(Higher is Better)' if higher_better else '(Lower is Better)'
        ax.set_title(f'Keane et al. (2021) — {col}{title_suffix}\n{direction}',
                     fontweight='bold', fontsize=13)
        if higher_better:
            ax.set_xlim(0, 1)
            ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')

        # Value labels
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        for i, (bar, val) in enumerate(zip(bars, plot_keane[col])):
            fmt = f'{val:.1%}' if higher_better else f'{val:.3f}'
            ax.text(val + 0.02 * x_range, i, fmt, va='center', fontsize=9)

        plt.tight_layout()
        fpath = os.path.join(output_dir, filename)
        plt.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"\nKeane {col} plot saved as: {fpath}")
        output_filenames.append(fpath)

    return output_filenames


def evaluate_full_metrics_batch(original_ts_list, all_results, model_wrapper,
                                 target_classes_list, reference_data):
    """
    Evaluate the cfts.metrics.evaluate.evaluate_counterfactual() metric suite
    across all algorithms.

    This is the same single-call function used by the cf_imfact comparison
    scripts (e.g. cf_imfact/experiments/compare_ucr.py). It reports validity,
    proximity, sparsity, and realism together, and includes a few metrics
    evaluate_keane_metrics_batch() does not: a MASCOTS-style z-score
    normalised distance, DTW distance, and Keane-naming aliases.

    Args:
        original_ts_list: List of original time series (one per instance).
        all_results: List of per-instance evaluation results (from
            evaluate_single_instance()), aligned with original_ts_list.
        model_wrapper: Model wrapper function.
        target_classes_list: List of target classes, aligned with
            original_ts_list.
        reference_data: Reference dataset used for the range_validity metric.

    Returns:
        DataFrame with one row per algorithm (mean of every
        evaluate_counterfactual() metric across instances), or None if no
        counterfactuals were available.
    """
    print("\n=== Evaluating evaluate.py Metrics (evaluate_counterfactual) ===")
    print("cfts/metrics/evaluate.py — same metric suite used by the cf_imfact")
    print("comparison scripts (validity, proximity, sparsity, realism).\n")

    # Group (original, counterfactual, target_class) triples by algorithm.
    algorithm_pairs = {}
    for orig_ts, result, target_class in zip(original_ts_list, all_results, target_classes_list):
        if result is None or not result.get('counterfactuals'):
            continue
        for algorithm_name, cf_ts in result['counterfactuals'].items():
            algorithm_pairs.setdefault(algorithm_name, []).append((orig_ts, cf_ts, target_class))

    if not algorithm_pairs:
        print("No counterfactuals available for evaluate_counterfactual metrics evaluation!")
        return None

    per_instance_rows = []
    for algorithm_name in sorted(algorithm_pairs.keys()):
        pairs = algorithm_pairs[algorithm_name]
        print(f"\nEvaluating {algorithm_name}: {len(pairs)} counterfactual(s)")
        for orig_ts, cf_ts, target_class in pairs:
            try:
                metrics = evaluate_counterfactual(
                    orig_ts, cf_ts, model=model_wrapper,
                    target_class=int(target_class),
                    reference_data=reference_data,
                )
                metrics['Algorithm'] = algorithm_name
                per_instance_rows.append(metrics)
            except Exception as e:
                print(f"  ✗ Error evaluating {algorithm_name}: {str(e)}")

    if not per_instance_rows:
        print("No metrics computed via evaluate_counterfactual!")
        return None

    df_per_instance = pd.DataFrame(per_instance_rows)
    n_samples = df_per_instance.groupby('Algorithm').size().rename('N_Samples')
    df_full = df_per_instance.groupby('Algorithm').mean(numeric_only=True).round(4)
    df_full = df_full.join(n_samples).reset_index()
    if 'validity' in df_full.columns:
        df_full = df_full.sort_values('validity', ascending=False, ignore_index=True)

    print("\n" + "="*70)
    print("evaluate_counterfactual() Metrics Summary (mean per algorithm)")
    print("="*70)
    print(df_full.to_string(index=False))
    print("="*70)

    return df_full


def visualize_full_metrics(df_full, output_dir='./', top_algorithms=None):
    """
    Bar-chart panels for the evaluate_counterfactual() metrics that aren't
    already covered by metrics_*.png / keane_*.png: the MASCOTS-style
    z-score-normalised distance and DTW distance.

    Args:
        df_full: DataFrame from evaluate_full_metrics_batch().
        output_dir: Directory to save the plots.
        top_algorithms: optional list of algorithm names to restrict the
            bars to (e.g. the best N by composite score).

    Returns:
        List of paths to saved plots.
    """
    if df_full is None or df_full.empty:
        print("No evaluate_counterfactual metrics to visualize!")
        return []

    plot_df = (df_full if top_algorithms is None
               else df_full[df_full['Algorithm'].isin(top_algorithms)])
    if plot_df.empty:
        print("No evaluate_counterfactual metrics left to visualize after filtering to top_algorithms!")
        return []

    algorithms = plot_df['Algorithm'].tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
    output_filenames = []
    title_suffix = f' — Top {len(algorithms)}' if top_algorithms is not None else ''

    metric_specs = [
        ('euclidean_dist_zscore', 'Z-score Normalised L2 Distance (MASCOTS-style)', 'evalpy_euclidean_zscore.png'),
        ('dtw_distance', 'DTW Distance', 'evalpy_dtw_distance.png'),
    ]

    for col, xlabel, filename in metric_specs:
        if col not in plot_df.columns or plot_df[col].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(algorithms))))
        bars = ax.barh(algorithms, plot_df[col], color=colors, alpha=0.8, edgecolor='black')
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=11)
        ax.set_title(f'evaluate_counterfactual() — {xlabel}{title_suffix}\n(Lower is Better)',
                     fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='x')

        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        for i, (bar, val) in enumerate(zip(bars, plot_df[col])):
            ax.text(val + 0.02 * x_range, i, f'{val:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        fpath = os.path.join(output_dir, filename)
        plt.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"\nevaluate_counterfactual {xlabel} plot saved as: {fpath}")
        output_filenames.append(fpath)

    return output_filenames


def combine_png_files(file_list, output_path, delete_individual=True, header_lines=None):
    """
    Stack a list of PNG files vertically into a single combined image.

    Each source image is kept at its original width; narrower images are
    padded with white on the right so all panels line up.  A thin grey
    separator line is drawn between panels for readability.

    Args:
        file_list         : ordered list of PNG paths to combine
        output_path       : path for the output PNG
        delete_individual : if True (default), delete source files after
                            the combined image is successfully written
        header_lines      : optional list of strings to render as a banner
                            above all panels (e.g. dataset/model/sample-size
                            provenance) — one line per string, centered.

    Returns:
        output_path if successful, else None
    """
    from PIL import Image, ImageDraw, ImageFont

    existing = [p for p in file_list if p and os.path.exists(p)]
    if not existing:
        print("combine_png_files: no source files found, skipping.")
        return None

    images = [Image.open(p).convert('RGB') for p in existing]
    max_w   = max(img.width  for img in images)
    sep     = 6   # pixels between panels
    total_h = sum(img.height for img in images) + sep * (len(images) - 1)

    # Optional provenance banner (dataset / model / sample counts) rendered
    # above the stacked panels.
    header_h = 0
    font = None
    line_h = 0
    if header_lines:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 72)
        except OSError:
            try:
                font = ImageFont.load_default(size=72)  # Pillow >= 9.2
            except TypeError:
                font = ImageFont.load_default()  # older Pillow: fixed small bitmap font
        line_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 24
        header_h = line_h * len(header_lines) + 40  # 20px top/bottom padding
        # Widen the canvas if the header text is wider than every panel —
        # otherwise long provenance lines get clipped at the right edge.
        text_w = max(font.getbbox(line)[2] for line in header_lines)
        max_w = max(max_w, text_w + 64)

    canvas = Image.new('RGB', (max_w, header_h + total_h), color=(255, 255, 255))
    if header_lines:
        draw = ImageDraw.Draw(canvas)
        for i, line in enumerate(header_lines):
            line_w = font.getbbox(line)[2]
            x = max(0, (max_w - line_w) // 2)
            draw.text((x, 20 + i * line_h), line, fill=(30, 30, 30), font=font)
        draw.line([(0, header_h - 1), (max_w, header_h - 1)], fill=(180, 180, 180), width=2)

    y_off  = header_h
    for i, img in enumerate(images):
        # Pad narrower images with white on the right
        if img.width < max_w:
            padded = Image.new('RGB', (max_w, img.height), (255, 255, 255))
            padded.paste(img, (0, 0))
            img = padded
        canvas.paste(img, (0, y_off))
        y_off += img.height
        if i < len(images) - 1:
            # Draw separator
            for row in range(sep):
                for col in range(max_w):
                    canvas.putpixel((col, y_off + row), (200, 200, 200))
            y_off += sep

    canvas.save(output_path, dpi=(300, 300))
    size_kb = os.path.getsize(output_path) // 1024
    print(f"\nCombined plot saved as: {output_path} ({size_kb} KB, {len(existing)} panels)")

    if delete_individual:
        for p in existing:
            try:
                os.remove(p)
                print(f"  Deleted individual file: {os.path.basename(p)}")
            except OSError as e:
                print(f"  Warning: could not delete {p}: {e}")

    return output_path


def export_full_metrics_csv(summary_stats, df_keane, ranked_algorithms, top_n, output_path,
                             all_algorithm_names=None, df_evalpy=None, df_runtime=None):
    """
    Write one CSV row per algorithm with its aggregate metrics, Keane scores,
    evaluate.py (evaluate_counterfactual) scores, runtime, and composite rank —
    the complete results table, including algorithms that fall outside the
    top_n shown in metrics_combined.png.

    Algorithms that were prepared (present in ``all_algorithm_names``) but never
    produced a single successful counterfactual across any evaluated instance
    never enter ``summary_stats`` / ``df_keane`` / ``df_evalpy`` / ``ranked_algorithms``
    — they still get a row here, with ``CompositeScore=0``, no Rank, and blank
    metric columns (there is no data to report, only the fact that they failed).
    Runtime is the exception: it's recorded for every attempt (see df_runtime),
    so even an algorithm that never succeeded can still get a runtime figure.

    Args:
        summary_stats: full (Algorithm, Metric) -> mean/std/median DataFrame
            from create_results_visualization(), computed over ALL algorithms.
        df_keane: full Keane et al. (2021) metrics DataFrame (all algorithms).
        ranked_algorithms: list of (algorithm_name, composite_score) tuples,
            best first, from rank_algorithms().
        top_n: how many top-ranked algorithms are shown in metrics_combined.png;
            used to fill the ShownInCombinedPlot column.
        output_path: path to write the CSV to.
        all_algorithm_names: every algorithm name that was prepared for
            evaluation (e.g. ``algorithms.keys()``), including ones that
            failed on every instance and so are absent from the other
            arguments. When given, those algorithms still get a CSV row.
        df_evalpy: full evaluate.py (evaluate_counterfactual) metrics DataFrame
            from evaluate_full_metrics_batch() (all algorithms), or None to
            skip these columns.
        df_runtime: long-format (Instance, Algorithm, Runtime) DataFrame from
            compute_runtime_dataframe() (all algorithms, every attempt —
            successes, failures, and timeouts), or None to skip these columns.

    Returns:
        output_path
    """
    rank_map = {name: i + 1 for i, (name, _) in enumerate(ranked_algorithms)}
    score_map = dict(ranked_algorithms)

    metrics_wide = None
    if summary_stats is not None and not summary_stats.empty:
        metrics_wide = summary_stats['mean'].unstack('Metric')

    keane_by_algo = {} if df_keane is None else df_keane.set_index('Algorithm').to_dict('index')
    evalpy_by_algo = {} if df_evalpy is None else df_evalpy.set_index('Algorithm').to_dict('index')

    runtime_by_algo = {}
    if df_runtime is not None and not df_runtime.empty:
        runtime_stats = df_runtime.groupby('Algorithm')['Runtime'].agg(['mean', 'std', 'count'])
        runtime_by_algo = runtime_stats.to_dict('index')

    all_algorithms = set(rank_map)
    if metrics_wide is not None:
        all_algorithms |= set(metrics_wide.index)
    all_algorithms |= set(keane_by_algo)
    all_algorithms |= set(evalpy_by_algo)
    all_algorithms |= set(runtime_by_algo)
    if all_algorithm_names is not None:
        all_algorithms |= set(all_algorithm_names)

    rows = []
    for algorithm in all_algorithms:
        row = {
            'Algorithm': algorithm,
            'Rank': rank_map.get(algorithm),
            # Algorithms with no ranked score never succeeded on any instance —
            # score 0 (rather than blank) so they still sort/compare cleanly.
            'CompositeScore': score_map.get(algorithm, 0.0),
            'ShownInCombinedPlot': rank_map.get(algorithm, top_n + 1) <= top_n,
        }

        keane_row = keane_by_algo.get(algorithm)
        if keane_row:
            row['Keane_Validity'] = keane_row.get('Validity')
            row['Keane_Proximity'] = keane_row.get('Proximity')
            row['Keane_Compactness'] = keane_row.get('Compactness')
            row['N_Samples'] = keane_row.get('N_Samples')

        if metrics_wide is not None and algorithm in metrics_wide.index:
            for metric_name, value in metrics_wide.loc[algorithm].items():
                row[f'mean_{metric_name}'] = value

        evalpy_row = evalpy_by_algo.get(algorithm)
        if evalpy_row:
            for metric_name, value in evalpy_row.items():
                if metric_name == 'N_Samples':
                    row['evalpy_N_Samples'] = value
                else:
                    row[f'evalpy_{metric_name}'] = value

        runtime_row = runtime_by_algo.get(algorithm)
        if runtime_row:
            row['Runtime_Mean_Seconds'] = runtime_row.get('mean')
            row['Runtime_Std_Seconds'] = runtime_row.get('std')
            row['Runtime_N_Attempts'] = runtime_row.get('count')

        rows.append(row)

    df_full = pd.DataFrame(rows).sort_values(
        by='Rank', na_position='last', ignore_index=True
    )
    df_full.to_csv(output_path, index=False)
    print(f"\nFull results for all {len(df_full)} algorithms saved as: {output_path}")
    print(f"  (metrics_combined.png shows only the top {top_n} by composite score)")
    return output_path


def select_fastest_algorithms(csv_path='./metrics_full_results.csv',
                               n_fastest=N_FASTEST_FOR_LARGE_SWEEP,
                               min_keane_validity=0.0):
    """
    Pick the n_fastest quickest-running algorithms out of a prior main()
    run's metrics_full_results.csv.

    Only algorithms that actually produced at least one successful
    counterfactual (i.e. have a Rank — see export_full_metrics_csv()) are
    eligible: an algorithm that merely errors out instantly (e.g. Sub-SpaCE
    on univariate FordA data, ~0s runtime) is fast but useless, and would
    otherwise crowd out algorithms that are fast *and* actually work.

    Args:
        csv_path: path to a metrics_full_results.csv produced by main().
        n_fastest: how many algorithms to select (default
            N_FASTEST_FOR_LARGE_SWEEP).
        min_keane_validity: optional extra quality floor (0-1) on Keane
            validity — set e.g. 0.5 to also require the algorithm to
            succeed on at least half its attempts, not just be fast when it
            does succeed. Default 0.0 imposes no extra floor beyond "ranked
            at all".

    Returns:
        List of algorithm names, fastest (lowest mean runtime) first.
    """
    df = pd.read_csv(csv_path)

    eligible = df[df['Rank'].notna()].dropna(subset=['Runtime_Mean_Seconds']).copy()
    if min_keane_validity > 0 and 'Keane_Validity' in eligible.columns:
        eligible = eligible[eligible['Keane_Validity'] >= min_keane_validity]

    eligible = eligible.sort_values('Runtime_Mean_Seconds', ascending=True)
    fastest = eligible['Algorithm'].head(n_fastest).tolist()

    print(f"Selected {len(fastest)} fastest algorithm(s) from {csv_path} "
          f"(of {len(eligible)} eligible, ranked, successful ones):")
    for name, runtime in zip(fastest, eligible['Runtime_Mean_Seconds'].head(n_fastest)):
        print(f"  {name}: {runtime:.3f}s mean runtime")

    return fastest


def run_fastest_methods_sweep(fastest_names=None, n_instances=LARGE_SWEEP_N_INSTANCES,
                               csv_path='./metrics_full_results.csv',
                               output_dir=FAST_SWEEP_OUTPUT_DIR,
                               checkpoint_path=FAST_SWEEP_CHECKPOINT_PATH,
                               resume=True):
    """
    Re-evaluate just the fastest previously-ranked algorithms across a much
    larger sample of instances than main()'s own 10 — see the module-level
    comment above N_FASTEST_FOR_LARGE_SWEEP for why this is only feasible
    once the slow algorithms have been dropped.

    Mirrors main()'s own per-instance loop (same evaluate_single_instance(),
    same incremental checkpointing pattern — see CHECKPOINT_PATH/_save_
    checkpoint()/_load_checkpoint() above, but under FAST_SWEEP_CHECKPOINT_PATH
    so it can't collide with a main() run's own checkpoint) and reuses all of
    main()'s aggregation/plotting/export helpers, just scoped to fewer
    algorithms and written under output_dir instead of the repo root so
    nothing here overwrites main()'s own metrics_*.png / metrics_full_results.csv.

    Args:
        fastest_names: explicit list of algorithm names to run, or None to
            select them via select_fastest_algorithms(csv_path=csv_path).
        n_instances: how many instances to evaluate (default
            LARGE_SWEEP_N_INSTANCES = 100).
        csv_path: metrics_full_results.csv to read fastest_names from, when
            fastest_names isn't given explicitly. Ignored otherwise.
        output_dir: directory for this sweep's own plots/CSV (created if
            missing).
        checkpoint_path: checkpoint file for this sweep specifically.
        resume: whether to resume from checkpoint_path if present (set
            False to always start fresh, mirroring RESUME_FROM_CHECKPOINT).

    Returns:
        The path to the exported CSV (metrics_full_results_fast_sweep.csv
        under output_dir), or None if no instance produced any successful
        counterfactual.
    """
    print("=== Fastest-Methods, Large-Sample Sweep ===\n")
    os.makedirs(output_dir, exist_ok=True)

    model, dataset_train, dataset_test = load_forda_data_and_model()
    model_wrapper = pytorch_model_wrapper(model)

    EVAL_SPLIT = 'train'
    eval_dataset = dataset_train if EVAL_SPLIT == 'train' else dataset_test

    if fastest_names is None:
        fastest_names = select_fastest_algorithms(csv_path=csv_path)
    if not fastest_names:
        print("❌ No fastest algorithms selected — nothing to run!")
        return None

    all_algorithms = create_algorithm_wrappers(eval_dataset, model)
    algorithms = {name: all_algorithms[name] for name in fastest_names if name in all_algorithms}
    missing = [name for name in fastest_names if name not in all_algorithms]
    if missing:
        print(f"⚠ {len(missing)} selected algorithm(s) no longer exist in "
              f"create_algorithm_wrappers(), skipping: {missing}")
    print(f"✓ Running {len(algorithms)} algorithm(s) on {n_instances} instances "
          f"of the '{EVAL_SPLIT}' split")

    n_instances = min(n_instances, len(eval_dataset.X))
    instance_indices = np.random.choice(len(eval_dataset.X), n_instances, replace=False)
    instance_signature = [int(idx) for idx in instance_indices]

    completed = _load_checkpoint(checkpoint_path, instance_signature) if resume else {}

    all_results = []
    original_ts_list = []
    target_classes_list = []

    for i, idx in tqdm(enumerate(instance_indices), total=n_instances, desc="Fast-sweep instances"):
        if i in completed:
            cached = completed[i]
            original_ts_list.append(cached['original_ts'])
            target_classes_list.append(cached['target_class'])
            all_results.append(cached['result'])
            continue

        original_ts = eval_dataset.X[idx]
        label = np.argmax(eval_dataset.y[idx])
        target_class = 1 - label

        original_ts_list.append(original_ts)
        target_classes_list.append(target_class)

        result = evaluate_single_instance(original_ts, label, model_wrapper, algorithms, eval_dataset)
        all_results.append(result)

        completed[i] = {'original_ts': original_ts, 'target_class': target_class, 'result': result}
        _save_checkpoint(checkpoint_path, instance_signature, completed)

    if resume and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    valid_results = [r for r in all_results if r is not None]
    if not valid_results:
        print("❌ No successful evaluations in fast sweep!")
        return None
    print(f"\n✓ Successfully evaluated {len(valid_results)} instances")

    # Collect runtimes up front so rank_algorithms() can apply its handicap,
    # same as main() does.
    df_runtime = compute_runtime_dataframe(all_results)
    runtime_by_algorithm = (df_runtime.groupby('Algorithm')['Runtime'].mean().to_dict()
                             if df_runtime is not None else None)

    df = compute_metrics_dataframe(valid_results)
    ranked_algorithms = []
    if df is not None and not df.empty:
        summary_stats_for_ranking = df.groupby(['Algorithm', 'Metric'])['Value'].agg(
            ['mean', 'std', 'median']).round(3)
        ranked_algorithms = rank_algorithms(summary_stats_for_ranking, algorithms.keys(),
                                             runtime_by_algorithm=runtime_by_algorithm)

    # No top_algorithms truncation needed anywhere below — every algorithm
    # here is already one of the pre-selected fastest ones.
    composite_filenames = visualize_composite_scores(ranked_algorithms, output_dir=output_dir,
                                                      top_n=len(algorithms))
    output_filenames, summary_stats = create_results_visualization(df, output_dir=output_dir)

    df_keane = evaluate_keane_metrics_batch(original_ts_list, all_results, model_wrapper, target_classes_list)
    keane_filenames = visualize_keane_metrics(df_keane, output_dir=output_dir) if df_keane is not None else []

    reference_data = np.array([eval_dataset.X[i] for i in range(min(100, len(eval_dataset.X)))])
    df_evalpy = evaluate_full_metrics_batch(original_ts_list, all_results, model_wrapper,
                                             target_classes_list, reference_data)
    evalpy_filenames = visualize_full_metrics(df_evalpy, output_dir=output_dir) if df_evalpy is not None else []

    all_plot_files = (list(composite_filenames) + list(output_filenames)
                      + list(keane_filenames) + list(evalpy_filenames))
    header_lines = [
        f"Fastest-methods sweep — {len(algorithms)} algorithm(s), {n_instances} instances "
        f"({EVAL_SPLIT} split)",
        f"Selected by lowest mean runtime from: {csv_path}",
    ]
    if ranked_algorithms:
        top_name, top_score = ranked_algorithms[0]
        header_lines.append(f"Top algorithm: {top_name}  (composite score: {top_score:.3f})")
    combine_png_files(all_plot_files, os.path.join(output_dir, 'metrics_combined_fast_sweep.png'),
                       header_lines=header_lines)

    output_csv = os.path.join(output_dir, 'metrics_full_results_fast_sweep.csv')
    export_full_metrics_csv(
        summary_stats, df_keane, ranked_algorithms, len(algorithms), output_csv,
        all_algorithm_names=algorithms.keys(), df_evalpy=df_evalpy, df_runtime=df_runtime,
    )

    print("\n=== Fastest-Methods Sweep Complete ===")
    print(f"  - {output_dir}/metrics_combined_fast_sweep.png")
    print(f"  - {output_csv}")
    return output_csv


def main():
    """Main execution function."""
    print("=== Comprehensive Counterfactual Metrics Evaluation ===\n")
    
    # Load data and model
    try:
        model, dataset_train, dataset_test = load_forda_data_and_model()
        model_wrapper = pytorch_model_wrapper(model)
        print("✓ Model and data loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model or data: {e}")
        return

    # Which split to draw instances from, and which split algorithms use as
    # their NUN/reference pool (dataset= kwarg in create_algorithm_wrappers).
    # Set to 'test' to restore the original behaviour.
    EVAL_SPLIT = 'train'
    eval_dataset = dataset_train if EVAL_SPLIT == 'train' else dataset_test
    print(f"✓ Evaluating on the '{EVAL_SPLIT}' split ({len(eval_dataset.X)} instances available)")

    # Create algorithm wrappers
    algorithms = create_algorithm_wrappers(eval_dataset, model)
    print(f"✓ {len(algorithms)} algorithms prepared")

    # Select instances (diverse examples) from the chosen split
    n_instances = 10  # Evaluate on 10 instances
    instance_indices = np.random.choice(len(eval_dataset.X), n_instances, replace=False)
    instance_signature = [int(idx) for idx in instance_indices]

    print(f"\n=== Evaluating {n_instances} {EVAL_SPLIT} instances ===")

    completed = (_load_checkpoint(CHECKPOINT_PATH, instance_signature)
                 if RESUME_FROM_CHECKPOINT else {})

    all_results = []
    original_ts_list = []  # Store original time series for visualization
    target_classes_list = []  # Store target classes for Keane/evaluate.py metrics

    for i, idx in tqdm(enumerate(instance_indices), total=n_instances, desc="Evaluating instances"):
        if i in completed:
            cached = completed[i]
            original_ts_list.append(cached['original_ts'])
            target_classes_list.append(cached['target_class'])
            all_results.append(cached['result'])
            print(f"\n--- Instance {i+1}/{n_instances} (Index: {idx}) — loaded from checkpoint ---")
            continue

        original_ts = eval_dataset.X[idx]
        label = np.argmax(eval_dataset.y[idx])
        target_class = 1 - label  # Binary classification

        original_ts_list.append(original_ts)  # Save for visualization
        target_classes_list.append(target_class)  # Save for Keane/evaluate.py metrics

        print(f"\n--- Instance {i+1}/{n_instances} (Index: {idx}) ---")
        result = evaluate_single_instance(original_ts, label, model_wrapper, algorithms, eval_dataset)
        all_results.append(result)

        # Checkpoint immediately so a kill/timeout after this point doesn't
        # lose this (and every prior) instance's ~19-20 min of work.
        completed[i] = {'original_ts': original_ts, 'target_class': target_class, 'result': result}
        _save_checkpoint(CHECKPOINT_PATH, instance_signature, completed)

    # The expensive per-instance sweep is now fully done — remove the
    # checkpoint so a future fresh run doesn't resume stale results.
    if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    # Filter out None results
    valid_results = [r for r in all_results if r is not None]

    if not valid_results:
        print("❌ No successful evaluations!")
        return

    print(f"\n✓ Successfully evaluated {len(valid_results)} instances")
    
    # Cap how many algorithms get plotted into metrics_combined.png; the
    # complete, unfiltered results for every algorithm always go to CSV
    # (see export_full_metrics_csv below).
    TOP_N = 10

    # Create visualizations and summary
    try:
        # Collect algorithm runtimes (all algorithms, every attempt — successes,
        # failures, and timeouts alike) up front, so rank_algorithms() below can
        # apply its runtime handicap to the composite score.
        df_runtime = compute_runtime_dataframe(all_results)
        runtime_by_algorithm = (df_runtime.groupby('Algorithm')['Runtime'].mean().to_dict()
                                 if df_runtime is not None else None)

        # Build the full per-instance metrics table and rank algorithms on it
        # up front, so we know which ones to plot before any plotting happens.
        df = compute_metrics_dataframe(valid_results)
        ranked_algorithms = []
        top_algorithms = None
        if df is not None and not df.empty:
            summary_stats_for_ranking = df.groupby(['Algorithm', 'Metric'])['Value'].agg(
                ['mean', 'std', 'median']).round(3)
            ranked_algorithms = rank_algorithms(summary_stats_for_ranking, algorithms.keys(),
                                                 runtime_by_algorithm=runtime_by_algorithm)
            top_algorithms = [name for name, _ in ranked_algorithms[:TOP_N]]
            print(f"\n✓ Ranked {len(ranked_algorithms)} algorithms (runtime-handicapped; "
                  f"scale={RUNTIME_HANDICAP_SCALE}s); "
                  f"plotting the top {len(top_algorithms)} in metrics_combined.png")

        # Composite-score bar chart — the overall "who won" ranking, shown
        # before the per-metric-category breakdowns below.
        composite_filenames = visualize_composite_scores(ranked_algorithms, top_n=TOP_N)

        # Create metrics visualization (only top_algorithms are plotted, but
        # summary_stats is still computed over ALL algorithms internally)
        output_filenames, summary_stats = create_results_visualization(df, top_algorithms=top_algorithms)

        # Evaluate Keane et al. (2021) metrics (all algorithms)
        df_keane = evaluate_keane_metrics_batch(original_ts_list, all_results,
                                                model_wrapper, target_classes_list)

        # Visualize Keane metrics (only top_algorithms are plotted)
        keane_filenames = []
        if df_keane is not None:
            keane_filenames = visualize_keane_metrics(df_keane, top_algorithms=top_algorithms)

        # Evaluate the evaluate.py (evaluate_counterfactual) metric suite (all algorithms)
        reference_data = np.array([eval_dataset.X[i] for i in range(min(100, len(eval_dataset.X)))])
        df_evalpy = evaluate_full_metrics_batch(original_ts_list, all_results,
                                                 model_wrapper, target_classes_list,
                                                 reference_data)

        # Visualize evaluate.py metrics not already covered above (only top_algorithms plotted)
        evalpy_filenames = []
        if df_evalpy is not None:
            evalpy_filenames = visualize_full_metrics(df_evalpy, top_algorithms=top_algorithms)

        # Combine all metric plots into one image, with a provenance banner
        # (dataset, model, and sample-count context) at the top.
        all_plot_files = (list(composite_filenames) + list(output_filenames)
                          + list(keane_filenames) + list(evalpy_filenames))
        header_lines = [
            f"Dataset: {getattr(eval_dataset, 'name', 'unknown')}  "
            f"({EVAL_SPLIT} split, {len(eval_dataset.X)} instances available)",
            f"Model: {type(model).__name__}",
            f"Query instances evaluated: {len(valid_results)}  |  "
            f"Reference/training samples used: {len(reference_data)}",
        ]
        if ranked_algorithms:
            top_name, top_score = ranked_algorithms[0]
            header_lines.append(f"Top algorithm: {top_name}  (composite score: {top_score:.3f})")
        combine_png_files(all_plot_files, os.path.join('./', 'metrics_combined.png'),
                           header_lines=header_lines)

        # (df_runtime was already collected above, before ranking, so
        # rank_algorithms() could apply its runtime handicap.)

        # Export the complete, unfiltered results (every algorithm, not just
        # the top_algorithms shown above) to CSV.
        export_full_metrics_csv(
            summary_stats, df_keane, ranked_algorithms, TOP_N,
            os.path.join('./', 'metrics_full_results.csv'),
            all_algorithm_names=algorithms.keys(),
            df_evalpy=df_evalpy,
            df_runtime=df_runtime,
        )

        # Calculate algorithm success rates
        print("\n=== Algorithm Success Rates ===")
        algorithm_success = {}
        for result in valid_results:
            for algorithm in result['successful_algorithms']:
                algorithm_success[algorithm] = algorithm_success.get(algorithm, 0) + 1

        for algorithm, successes in algorithm_success.items():
            success_rate = successes / len(valid_results) * 100
            print(f"{algorithm}: {successes}/{len(valid_results)} ({success_rate:.1f}%)")

        # Overall performance summary
        print("\n=== Overall Performance Summary ===")
        if ranked_algorithms:
            print("Algorithm Rankings (based on validity, proximity, and realism; runtime-handicapped):")
            for i, (algorithm, score) in enumerate(ranked_algorithms, 1):
                shown = " [shown in metrics_combined.png]" if algorithm in (top_algorithms or []) else ""
                print(f"{i}. {algorithm}: {score:.3f}{shown}")

    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Evaluation Complete ===")
    print(f"Evaluated on the '{EVAL_SPLIT}' split.")
    print("Generated outputs:")
    print(f"  - metrics_combined.png  (top {TOP_N} algorithms' composite score, metrics, Keane, and evaluate.py panels in one file)")
    print("  - composite_scores.png (composite score ranking, top algorithms)")
    print("  - metrics_validity.png (validity metric comparisons, top algorithms)")
    print("  - metrics_proximity.png (proximity metric comparisons, top algorithms)")
    print("  - metrics_sparsity.png (sparsity metric comparisons, top algorithms)")
    print("  - metrics_realism.png (realism metric comparisons, top algorithms)")
    print("  - keane_validity.png (Keane validity, top algorithms)")
    print("  - keane_proximity.png (Keane proximity, top algorithms)")
    print("  - keane_compactness.png (Keane compactness, top algorithms)")
    print("  - evalpy_euclidean_zscore.png (evaluate.py z-score-normalised distance, top algorithms)")
    print("  - evalpy_dtw_distance.png (evaluate.py DTW distance, top algorithms)")
    print("  - metrics_full_results.csv (full ranked results for ALL algorithms, incl. evalpy_* and Runtime_* columns)")
    print("\nKeane et al. (2021) Reference:")
    print("  Keane, M. T., Kenny, E. M., Delaney, E., & Smyth, B. (2021).")
    print("  If only we had better counterfactual explanations: Five key deficits")
    print("  to rectify in the evaluation of counterfactual XAI techniques.")
    print("  In IJCAI (Vol. 21, pp. 4466-4474).")


if __name__ == "__main__":
    # --fast-sweep: run just the N fastest previously-ranked algorithms (see
    # select_fastest_algorithms()) across LARGE_SWEEP_N_INSTANCES (100)
    # instances instead of the default main() sweep — see the module-level
    # comment above N_FASTEST_FOR_LARGE_SWEEP. Requires metrics_full_results.csv
    # from a prior main() run to already exist (to know which algorithms are
    # fastest); default behavior (no flag) is unchanged.
    if '--fast-sweep' in sys.argv:
        run_fastest_methods_sweep()
    else:
        main()