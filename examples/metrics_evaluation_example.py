"""
Comprehensive Counterfactual Metrics Evaluation Example

This example demonstrates how to evaluate counterfactual explanation generation
algorithms using the comprehensive metrics suite. It integrates with the existing
counterfactual methods (Native Guide, COMTE, SETS, MOC, Wachter, GLACIER) from
the cfts package and evaluates them on the FordA dataset.

Features:
- Real counterfactual algorithms evaluation
- Comprehensive metrics across all categories
- Algorithm benchmarking and comparison
- Professional visualization of results
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

# Import metrics
from cfts.metrics import (
    CounterfactualEvaluator, benchmark_algorithms, create_metric_suite,
    l2_distance, prediction_change, percentage_changed_points,
    temporal_consistency, pairwise_distance, algorithmic_stability
)

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
    
    return {
        'Native Guide': native_guide_wrapper,
        'COMTE': comte_wrapper,
        'SETS': sets_wrapper,
        'MOC': moc_wrapper,
        'Wachter Gradient': wachter_gradient_wrapper,
        'Wachter Genetic': wachter_genetic_wrapper,
        'GLACIER': glacier_wrapper
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
    
    for name, algorithm in algorithms.items():
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
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot proximity metrics
    proximity_data = df[df['Metric'].isin(proximity_metrics)]
    if not proximity_data.empty:
        sns.boxplot(data=proximity_data, x='Algorithm', y='Value', hue='Metric', ax=axes[0,1])
        axes[0,1].set_title('Proximity Metrics', fontweight='bold')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot sparsity metrics
    sparsity_data = df[df['Metric'].isin(sparsity_metrics)]
    if not sparsity_data.empty:
        sns.boxplot(data=sparsity_data, x='Algorithm', y='Value', hue='Metric', ax=axes[1,0])
        axes[1,0].set_title('Sparsity Metrics', fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot realism metrics
    realism_data = df[df['Metric'].isin(realism_metrics)]
    if not realism_data.empty:
        sns.boxplot(data=realism_data, x='Algorithm', y='Value', hue='Metric', ax=axes[1,1])
        axes[1,1].set_title('Realism Metrics', fontweight='bold')
        axes[1,1].tick_params(axis='x', rotation=45)
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
    for i, idx in enumerate(test_indices):
        original_ts = dataset_test.X[idx]
        label = np.argmax(dataset_test.y[idx])
        
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
        output_filename, summary_stats = create_results_visualization(valid_results)
        
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
    
    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    main()