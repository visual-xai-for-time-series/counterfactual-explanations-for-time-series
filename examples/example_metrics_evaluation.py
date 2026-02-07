"""
Comprehensive Counterfactual Metrics Evaluation Example

This example demonstrates how to evaluate counterfactual explanation generation
algorithms using the comprehensive metrics suite. It integrates with the existing
counterfactual methods (Native Guide, COMTE, COMTE-TS, SETS, MOC, Wachter, GLACIER,
Multi-SpaCE, Sub-SpaCE, TSEvo, LASTS, TSCF, FASTPACE, TIME-CF, SG-CF, MG-CF, 
Latent-CF, DiSCoX, CELS, FFT-CF, TERCE, AB-CF, CFWOT, CGM, COUNTS, SPARCE) from the 
cfts package and evaluates them on the FordA dataset.

Note: Sub-SpaCE is designed primarily for multivariate time series and may not work
with univariate datasets like FordA. It will be skipped if incompatible.

Features:
- Real counterfactual algorithms evaluation (27 methods)
- Comprehensive metrics across all categories
- Keane et al. (2021) evaluation metrics (validity, proximity, compactness)
- Algorithm benchmarking and comparison
- Professional visualization of results
- Time series comparison plots (original vs counterfactuals)
- Statistical analysis of performance
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}/../')

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

# Import metrics
from cfts.metrics import (
    CounterfactualEvaluator, benchmark_algorithms, create_metric_suite,
    l2_distance, prediction_change, percentage_changed_points,
    temporal_consistency, pairwise_distance, algorithmic_stability
)

# Import Keane et al. (2021) metrics
from cfts.metrics.keane import validity, proximity, compactness, evaluate_keane_metrics

# Set up plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_forda_data_and_model():
    """Load FordA dataset and trained model."""
    print('Loading FordA dataset...')
    _, dataset_train = bd.get_UCR_UEA_dataloader(split='train')
    _, dataset_test = bd.get_UCR_UEA_dataloader(split='test')
    
    output_classes = dataset_train.y_shape[1]
    model = bm.SimpleCNN(output_channels=output_classes).to(device)
    
    # Load pre-trained model
    models_dir = os.path.abspath(os.path.join(script_path, '..', 'models'))
    model_file = os.path.join(models_dir, f'simple_cnn_{output_classes}.pth')
    
    if os.path.exists(model_file):
        print(f'Loading saved model from {model_file}')
        state = torch.load(model_file, map_location=device)
        model.load_state_dict(state)
        model.eval()
    else:
        raise FileNotFoundError(f"Model not found at {model_file}. Please train the model first using example_forda.py")
    
    return model, dataset_train, dataset_test


def create_algorithm_wrappers(dataset_test, model):
    """Create wrapper functions for counterfactual algorithms with consistent interface."""
    
    def native_guide_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = ng.native_guide_uni_cf(original_ts, dataset_test, model)
        return cf if cf is not None else original_ts
    
    def comte_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = comte.comte_cf(original_ts, dataset_test, model)
        return cf if cf is not None else original_ts
    
    def comte_ts_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = comte.comte_ts_cf(original_ts, dataset_test, model, target_class=target_class)
        return cf if cf is not None else original_ts
    
    def sets_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = sets.sets_cf(original_ts, dataset_test, model)
        return cf if cf is not None else original_ts
    
    def moc_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = dandl.moc_cf(original_ts, dataset_test, model)
        return cf if cf is not None else original_ts
    
    def wachter_gradient_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = w.wachter_gradient_cf(original_ts, dataset_test, model)
        return cf if cf is not None else original_ts
    
    def wachter_genetic_wrapper(original_ts, target_class=None, **kwargs):
        step_size = np.mean(dataset_test.std) + 0.2
        cf, _ = w.wachter_genetic_cf(original_ts, model, step_size=step_size, max_steps=100)
        return cf if cf is not None else original_ts
    
    def glacier_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = glacier.glacier_cf(original_ts, dataset_test, model)
        return cf if cf is not None else original_ts
    
    def multispace_wrapper(original_ts, target_class=None, **kwargs):
        # Multi-SpaCE doesn't support explicit target class, it finds the nearest different class
        cf, _ = ms.multi_space_cf(original_ts, dataset_test, model, 
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
        cf, _ = subspace.subspace_cf(original_ts, dataset_test, model,
                                     desired_class=target_class,
                                     population_size=100,
                                     max_iter=200,
                                     alpha=0.8,
                                     beta=0.15,
                                     eta=0.05,
                                     invalid_penalization=20,
                                     init_pct=0.4,
                                     reinit=True,
                                     verbose=False)
        return cf if cf is not None else original_ts
    
    
    def tsevo_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = tsevo.tsevo_cf(original_ts, dataset_test, model, 
                               target_class=target_class,
                               population_size=50,
                               generations=100,
                               verbose=False)
        return cf if cf is not None else original_ts
    
    def lasts_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = lasts.lasts_cf(original_ts, dataset_test, model, 
                               target_class=target_class,
                               latent_dim=32,
                               max_iterations=1000,
                               train_ae_epochs=50,
                               verbose=False)
        return cf if cf is not None else original_ts
    
    def tscf_wrapper(original_ts, target_class=None, **kwargs):
        cf, _ = tscf.tscf_cf(original_ts, dataset_test, model, 
                            target_class=target_class,
                            lambda_l1=0.01,
                            lambda_l2=0.01,
                            lambda_smooth=0.001,
                            learning_rate=0.1,
                            max_iterations=2000,
                            verbose=False)
        return cf if cf is not None else original_ts
    
    def fastpace_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = fastpace.fastpace_cf(original_ts, dataset_test, model, 
                                        target=target_class,
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
            # TIME-CF works better with training data
            X_train_subset = np.array([dataset_test[i][0] for i in range(min(50, len(dataset_test)))])
            y_train_subset = np.array([np.argmax(dataset_test[i][1]) if hasattr(dataset_test[i][1], 'shape') and len(dataset_test[i][1].shape) > 0 else dataset_test[i][1] for i in range(min(50, len(dataset_test)))])
            cf, _ = time_cf.time_cf_generate(original_ts, model, 
                                            X_train=X_train_subset,
                                            y_train=y_train_subset,
                                            target=target_class,
                                            n_epochs=20,
                                            n_synthetic=50,
                                            verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts
    
    def sg_cf_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = sg_cf.sg_cf(original_ts, model, 
                               target=target_class,
                               max_iter=1000,
                               verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts
    
    def mg_cf_wrapper(original_ts, target_class=None, **kwargs):
        try:
            # MG-CF with STUMPY optimization for faster motif mining
            from torch.utils.data import Subset
            subset_size = min(100, len(dataset_test))
            dataset_subset = Subset(dataset_test, range(subset_size))
            cf, _ = mg_cf_generate_stumpy(original_ts, dataset_subset, model, 
                                         target=target_class,
                                         top_k=5,
                                         verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts
    
    def latent_cf_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = latent_cf.latent_cf_generate(original_ts, dataset_test, model, 
                                                target=target_class,
                                                latent_dim=8,
                                                max_iter=100,
                                                verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts
    
    def discox_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = discox.discox_generate_cf(original_ts, model, 
                                             target=target_class,
                                             window_size=20,
                                             max_attempts=50,
                                             modification_factor=1.5,
                                             verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts
    
    def cels_wrapper(original_ts, target_class=None, **kwargs):
        try:
            # CELS requires training data for nearest unlike neighbor
            X_train = np.array([dataset_test[i][0] for i in range(min(100, len(dataset_test)))])
            y_train = np.array([dataset_test[i][1] for i in range(min(100, len(dataset_test)))])
            cf, _ = cels.cels_generate(original_ts, model, X_train, y_train,
                                      target=target_class,
                                      max_iter=100,
                                      verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts
    
    def fft_cf_wrapper(original_ts, target_class=None, **kwargs):
        try:
            # Using nearest neighbor FFT blending approach
            cf, _ = fft_nn_cf(original_ts, dataset_test, model, 
                            target_class=target_class,
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
            X_train = np.array([dataset_test[i][0] for i in range(min(100, len(dataset_test)))])
            y_train = np.array([dataset_test[i][1] for i in range(min(100, len(dataset_test)))])
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
            X_train = np.array([dataset_test[i][0] for i in range(min(100, len(dataset_test)))])
            y_train = np.array([dataset_test[i][1] for i in range(min(100, len(dataset_test)))])
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
            cf, _ = cfwot.cfwot(original_ts, model,
                               target=target_class,
                               M_E=50,
                               M_T=50,
                               verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts
    
    def cgm_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = cgm.cgm_generate(original_ts, dataset_test, model,
                                    target=target_class,
                                    latent_dim=16,
                                    max_iter=100,
                                    verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts
    
    def counts_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = counts.counts_cf_with_pretrained_model(original_ts, dataset_test, model,
                                                          target=target_class,
                                                          latent_dim=16,
                                                          max_iter=100,
                                                          verbose=False)
            return cf if cf is not None else original_ts
        except Exception:
            return original_ts
    
    def sparce_wrapper(original_ts, target_class=None, **kwargs):
        try:
            cf, _ = sparce.sparce_gradient_cf(original_ts, model,
                                             target=target_class,
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
        'SPARCE': sparce_wrapper
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


def evaluate_single_instance(original_ts, label, model_wrapper, algorithms, dataset_test):
    """Evaluate all algorithms on a single time series instance."""
    print(f"\nEvaluating instance with original class: {label}")
    
    # Determine target class (flip to opposite class)
    target_class = 1 - label  # For binary classification
    
    # Generate counterfactuals with each algorithm
    counterfactuals = {}
    successful_algorithms = []
    
    for name, algorithm in tqdm(algorithms.items(), desc="  Generating CFs", leave=False):
        try:
            print(f"  Generating counterfactual with {name}...")
            cf = algorithm(original_ts, target_class=target_class)
            
            # Check if prediction actually changed
            original_pred = model_wrapper(original_ts)
            cf_pred = model_wrapper(cf)
            
            original_class = np.argmax(original_pred)
            cf_class = np.argmax(cf_pred)
            
            if cf_class == target_class:
                counterfactuals[name] = cf
                successful_algorithms.append(name)
                print(f"    ✓ Success: {original_class} → {cf_class}")
            else:
                print(f"    ✗ Failed: {original_class} → {cf_class} (target: {target_class})")
                
        except Exception as e:
            print(f"    ✗ Error with {name}: {str(e)}")
    
    if not counterfactuals:
        print("  No successful counterfactuals generated!")
        return None
    
    # Initialize evaluator with reference data
    reference_data = np.array([dataset_test.X[i] for i in range(min(100, len(dataset_test.X)))])
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
        'successful_algorithms': successful_algorithms
    }


def create_results_visualization(all_results, output_dir='./'):
    """Create comprehensive visualization of evaluation results."""
    
    # Collect all individual metrics
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
        print("No results to visualize!")
        return
    
    df = pd.DataFrame(all_metrics_data)
    
    # Create metric category groupings
    validity_metrics = ['prediction_change', 'class_confidence', 'boundary_distance']
    proximity_metrics = ['l2_distance', 'manhattan_distance', 'normalized_distance']
    sparsity_metrics = ['l0_norm', 'percentage_changed', 'segment_sparsity']
    realism_metrics = ['temporal_consistency', 'range_validity', 'autocorr_preservation', 'statistical_similarity']
    
    # Create subplots for different metric categories
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Counterfactual Algorithm Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot validity metrics
    validity_data = df[df['Metric'].isin(validity_metrics)]
    if not validity_data.empty:
        sns.boxplot(data=validity_data, x='Algorithm', y='Value', hue='Metric', ax=axes[0,0])
        axes[0,0].set_title('Validity Metrics', fontweight='bold')
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45, ha='right')
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot proximity metrics
    proximity_data = df[df['Metric'].isin(proximity_metrics)]
    if not proximity_data.empty:
        sns.boxplot(data=proximity_data, x='Algorithm', y='Value', hue='Metric', ax=axes[0,1])
        axes[0,1].set_title('Proximity Metrics', fontweight='bold')
        axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45, ha='right')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot sparsity metrics
    sparsity_data = df[df['Metric'].isin(sparsity_metrics)]
    if not sparsity_data.empty:
        sns.boxplot(data=sparsity_data, x='Algorithm', y='Value', hue='Metric', ax=axes[1,0])
        axes[1,0].set_title('Sparsity Metrics', fontweight='bold')
        axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45, ha='right')
        axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot realism metrics
    realism_data = df[df['Metric'].isin(realism_metrics)]
    if not realism_data.empty:
        sns.boxplot(data=realism_data, x='Algorithm', y='Value', hue='Metric', ax=axes[1,1])
        axes[1,1].set_title('Realism Metrics', fontweight='bold')
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45, ha='right')
        axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    output_filename = os.path.join(output_dir, 'counterfactual_metrics_evaluation.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\nComprehensive metrics visualization saved as: {output_filename}")
    
    # Create summary statistics table
    summary_stats = df.groupby(['Algorithm', 'Metric'])['Value'].agg(['mean', 'std', 'median']).round(3)
    print("\n=== Summary Statistics ===")
    print(summary_stats)
    
    return output_filename, summary_stats


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
            val_score = validity(originals, counterfactuals, model_wrapper, target_classes=targets)
            print(f"  ✓ Validity: {val_score:.2%} (fraction achieving target class)")
            
            # 2. Proximity
            prox_score = proximity(originals, counterfactuals)
            print(f"  ✓ Proximity: {prox_score:.4f} (average L2 distance)")
            
            # 3. Compactness
            comp_score = compactness(originals, counterfactuals, tolerance=0.01)
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


def visualize_keane_metrics(df_keane, output_dir='./'):
    """
    Create visualization of Keane et al. (2021) metrics.
    
    Args:
        df_keane: DataFrame with Keane metrics
        output_dir: Directory to save the plot
    
    Returns:
        Path to saved plot
    """
    if df_keane is None or df_keane.empty:
        print("No Keane metrics to visualize!")
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    algorithms = df_keane['Algorithm'].tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
    
    # 1. Validity (higher is better)
    ax1 = axes[0]
    bars1 = ax1.barh(algorithms, df_keane['Validity'], color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Validity Score', fontweight='bold', fontsize=11)
    ax1.set_title('Validity\n(Higher is Better)', fontweight='bold', fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, df_keane['Validity'])):
        ax1.text(val + 0.02, i, f'{val:.1%}', va='center', fontsize=9)
    
    # 2. Proximity (lower is better)
    ax2 = axes[1]
    bars2 = ax2.barh(algorithms, df_keane['Proximity'], color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Proximity Score (L2 Distance)', fontweight='bold', fontsize=11)
    ax2.set_title('Proximity\n(Lower is Better)', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, df_keane['Proximity'])):
        ax2.text(val + 0.02 * ax2.get_xlim()[1], i, f'{val:.2f}', va='center', fontsize=9)
    
    # 3. Compactness (higher is better)
    ax3 = axes[2]
    bars3 = ax3.barh(algorithms, df_keane['Compactness'], color=colors, alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Compactness Score', fontweight='bold', fontsize=11)
    ax3.set_title('Compactness\n(Higher is Better)', fontweight='bold', fontsize=12)
    ax3.set_xlim(0, 1)
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars3, df_keane['Compactness'])):
        ax3.text(val + 0.02, i, f'{val:.1%}', va='center', fontsize=9)
    
    plt.suptitle('Keane et al. (2021) Evaluation Metrics Comparison', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the plot
    output_filename = os.path.join(output_dir, 'keane_metrics_comparison.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nKeane metrics comparison saved as: {output_filename}")
    
    return output_filename


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
    
    # Create algorithm wrappers
    algorithms = create_algorithm_wrappers(dataset_test, model)
    print(f"✓ {len(algorithms)} algorithms prepared")
    
    # Select test instances (diverse examples)
    n_instances = 5  # Evaluate on 5 instances
    test_indices = np.random.choice(len(dataset_test.X), n_instances, replace=False)
    
    print(f"\n=== Evaluating {n_instances} test instances ===")
    
    all_results = []
    original_ts_list = []  # Store original time series for visualization
    target_classes_list = []  # Store target classes for Keane metrics
    
    for i, idx in tqdm(enumerate(test_indices), total=n_instances, desc="Evaluating instances"):
        original_ts = dataset_test.X[idx]
        label = np.argmax(dataset_test.y[idx])
        target_class = 1 - label  # Binary classification
        
        original_ts_list.append(original_ts)  # Save for visualization
        target_classes_list.append(target_class)  # Save for Keane metrics
        
        print(f"\n--- Instance {i+1}/{n_instances} (Index: {idx}) ---")
        result = evaluate_single_instance(original_ts, label, model_wrapper, algorithms, dataset_test)
        all_results.append(result)
    
    # Filter out None results
    valid_results = [r for r in all_results if r is not None]
    
    if not valid_results:
        print("❌ No successful evaluations!")
        return
    
    print(f"\n✓ Successfully evaluated {len(valid_results)} instances")
    
    # Create visualizations and summary
    try:
        # Create metrics visualization
        output_filename, summary_stats = create_results_visualization(valid_results)
        
        # Evaluate Keane et al. (2021) metrics
        df_keane = evaluate_keane_metrics_batch(original_ts_list, all_results, 
                                                model_wrapper, target_classes_list)
        
        # Visualize Keane metrics
        if df_keane is not None:
            visualize_keane_metrics(df_keane)
        
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
        if summary_stats is not None and not summary_stats.empty:
            # Focus on key metrics for ranking
            key_metrics = ['prediction_change', 'normalized_distance', 'temporal_consistency']
            
            algorithm_scores = {}
            for algorithm in algorithms.keys():
                scores = []
                for metric in key_metrics:
                    try:
                        if metric == 'prediction_change':
                            # Higher is better
                            score = summary_stats.loc[(algorithm, metric), 'mean']
                            scores.append(score)
                        elif metric == 'normalized_distance':
                            # Lower is better - invert for ranking
                            score = 1 / (1 + summary_stats.loc[(algorithm, metric), 'mean'])
                            scores.append(score)
                        elif metric == 'temporal_consistency':
                            # Higher is better
                            score = summary_stats.loc[(algorithm, metric), 'mean']
                            scores.append(score)
                    except KeyError:
                        continue
                
                if scores:
                    algorithm_scores[algorithm] = np.mean(scores)
            
            # Rank algorithms
            ranked_algorithms = sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=True)
            
            print("Algorithm Rankings (based on validity, proximity, and realism):")
            for i, (algorithm, score) in enumerate(ranked_algorithms, 1):
                print(f"{i}. {algorithm}: {score:.3f}")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Evaluation Complete ===")
    print("Generated outputs:")
    print("  - counterfactual_metrics_evaluation.png (metric comparisons)")
    print("  - keane_metrics_comparison.png (Keane et al. 2021 metrics)")
    print("\nKeane et al. (2021) Reference:")
    print("  Keane, M. T., Kenny, E. M., Delaney, E., & Smyth, B. (2021).")
    print("  If only we had better counterfactual explanations: Five key deficits")
    print("  to rectify in the evaluation of counterfactual XAI techniques.")
    print("  In IJCAI (Vol. 21, pp. 4466-4474).")


if __name__ == "__main__":
    main()