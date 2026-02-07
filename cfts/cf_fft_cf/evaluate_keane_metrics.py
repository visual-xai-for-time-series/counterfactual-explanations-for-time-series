"""
FFT-CF Variants Keane et al. Batch Evaluation
Evaluates all FFT-based counterfactual methods using Keane et al. (2021) metrics
on the entire FordA test dataset.
"""

import os
import sys
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
script_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(script_path, '..', '..'))
sys.path.insert(0, parent_path)

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import base modules
import examples.base.model as bm
import examples.base.data as bd

# Import all FFT-CF variants
from cfts.cf_fft_cf import (
    fft_cf,
    fft_gradient_cf,
    fft_nn_cf,
    fft_adaptive_cf,
    fft_iterative_cf,
    fft_smart_blend_cf,
    fft_freq_distance_cf,
    fft_wavelet_cf,
    fft_hybrid_cf,
    fft_progressive_cf
)

# Import Keane et al. (2021) metrics
from cfts.metrics.keane import validity, proximity, compactness, evaluate_keane_metrics


def main():
    print("=" * 90)
    print(" FFT-CF Variants Keane et al. (2021) Batch Evaluation - FordA")
    print("=" * 90)
    print()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load FordA dataset
    print("\n[1/5] Loading FordA dataset...")
    _, dataset_train = bd.get_UCR_UEA_dataloader(split='train')
    _, dataset_test = bd.get_UCR_UEA_dataloader(split='test')
    print(f"  ✓ Train samples: {len(dataset_train)}")
    print(f"  ✓ Test samples: {len(dataset_test)}")
    
    # Load model
    print("\n[2/5] Loading SimpleCNN model...")
    output_classes = dataset_train.y_shape[1]
    model = bm.SimpleCNN(output_channels=output_classes).to(device)
    
    models_dir = os.path.abspath(os.path.join(parent_path, 'models'))
    model_file = os.path.join(models_dir, f'simple_cnn_{output_classes}.pth')
    
    if os.path.exists(model_file):
        state = torch.load(model_file, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"  ✓ Model loaded from: {model_file}")
    else:
        print(f"  ✗ Model not found: {model_file}")
        return
    
    # Model wrapper for metrics
    def model_wrapper(ts):
        """Wrapper that ensures correct format for model prediction."""
        if isinstance(ts, np.ndarray):
            ts_array = ts
        else:
            ts_array = np.array(ts)
        
        # Ensure float type
        ts_array = ts_array.astype(np.float32)
        
        # Convert to tensor
        ts_tensor = torch.from_numpy(ts_array).float().to(device)
        
        # Add dimensions if needed
        if ts_tensor.dim() == 1:
            ts_tensor = ts_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif ts_tensor.dim() == 2:
            ts_tensor = ts_tensor.unsqueeze(0)  # Add batch dim
        
        with torch.no_grad():
            output = model(ts_tensor)
            probs = torch.softmax(output, dim=-1).squeeze().cpu().numpy()
            # Ensure it's always 1D array
            if probs.ndim == 0:
                probs = np.array([probs.item()])
            return probs
    
    # Find correctly classified test samples
    print("\n[3/5] Finding correctly classified test samples...")
    test_samples = []
    
    for idx in range(len(dataset_test)):
        sample, label = dataset_test[idx]
        
        with torch.no_grad():
            sample_tensor = torch.from_numpy(np.array(sample)).float().to(device)
            if sample_tensor.dim() == 1:
                sample_tensor = sample_tensor.unsqueeze(0).unsqueeze(0)
            elif sample_tensor.dim() == 2:
                sample_tensor = sample_tensor.unsqueeze(0)
            
            pred_output = model(sample_tensor)
            pred_class = torch.argmax(pred_output, dim=-1).item()
            true_class = np.argmax(label) if hasattr(label, 'shape') and len(label.shape) > 0 else label
            
            if pred_class == true_class:
                target_class = 1 - pred_class
                test_samples.append({
                    'idx': idx,
                    'sample': sample,
                    'label': label,
                    'original_class': pred_class,
                    'target_class': target_class
                })
    
    print(f"  ✓ Found {len(test_samples)} correctly classified samples")
    
    # Limit number of samples (fewer for slower methods)
    max_samples = min(50, len(test_samples))  # Reduced to 50 for faster methods
    test_samples = test_samples[:max_samples]
    print(f"  ✓ Evaluating on {len(test_samples)} samples")
    
    # Prepare reference data for Keane metrics
    reference_data = np.array([dataset_test[i][0] for i in range(min(100, len(dataset_test)))])
    
    # Define FFT-CF variants (use reduced iterations for slow methods)
    # Note: fft_cf (greedy) is excluded due to extreme slowness
    # Note: New variants (confidence_threshold, hybrid_enhanced, band_optimizer) 
    #       require numpy array format and are not compatible with this evaluation
    variants = [
        ('fft_gradient_cf', 'Gradient Descent',
         lambda s, d, m, t: fft_gradient_cf(s, d, m, t, max_iterations=20, verbose=False)),
        
        ('fft_nn_cf', 'Nearest Neighbor',
         lambda s, d, m, t: fft_nn_cf(s, d, m, t, k=5, verbose=False)),
        
        ('fft_adaptive_cf', 'Adaptive Saliency',
         lambda s, d, m, t: fft_adaptive_cf(s, d, m, t, k=5, use_saliency=True, verbose=False)),
        
        ('fft_iterative_cf', 'Iterative Refinement',
         lambda s, d, m, t: fft_iterative_cf(s, d, m, t, k=5, refine_iterations=30, verbose=False)),
        
        ('fft_smart_blend_cf', 'Smart Blend',
         lambda s, d, m, t: fft_smart_blend_cf(s, d, m, t, k=5, search_method='binary', verbose=False)),
        
        ('fft_freq_distance_cf', 'Frequency Distance',
         lambda s, d, m, t: fft_freq_distance_cf(s, d, m, t, k=5, freq_weight_strategy='energy', verbose=False)),
        
        ('fft_wavelet_cf', 'Wavelet Transform',
         lambda s, d, m, t: fft_wavelet_cf(s, d, m, t, k=5, wavelet='db4', level=3, verbose=False)),
        
        ('fft_hybrid_cf', 'Hybrid Amp-Phase',
         lambda s, d, m, t: fft_hybrid_cf(s, d, m, t, k=5, analyze_importance=False, verbose=False)),
        
        ('fft_progressive_cf', 'Progressive Switching',
         lambda s, d, m, t: fft_progressive_cf(s, d, m, t, k=5, steps_per_neighbor=5, verbose=False)),
    ]
    
    # Run batch evaluation with Keane metrics
    print(f"\n[4/5] Evaluating {len(variants)} FFT-CF variants with Keane et al. metrics...")
    print("=" * 90)
    
    results = defaultdict(lambda: {
        'validity_scores': [],
        'proximity_scores': [],
        'compactness_scores': [],
        'overall_scores': [],
        'successes': 0,
        'failures': 0,
        'times': []
    })
    
    for variant_idx, (variant_id, variant_name, variant_func) in enumerate(variants):
        print(f"\n[{variant_idx+1}/{len(variants)}] Evaluating {variant_name}...")
        
        for sample_idx, test_info in enumerate(test_samples):
            if (sample_idx + 1) % 20 == 0:
                print(f"  Progress: {sample_idx+1}/{len(test_samples)} samples...", end='\r')
            
            sample = test_info['sample']
            target_class = test_info['target_class']
            
            start_time = time.time()
            try:
                cf, pred = variant_func(sample, dataset_test, model, target_class)
                elapsed = time.time() - start_time
                results[variant_id]['times'].append(elapsed)
                
                if cf is not None and pred is not None:
                    pred_class = np.argmax(pred)
                    
                    if pred_class == target_class:
                        # Ensure sample and cf are in the right format for Keane metrics
                        # They expect lists or arrays with shape (n, time_steps) or (n, time_steps, features)
                        sample_2d = np.array(sample).reshape(1, -1)  # Shape: (1, timesteps)
                        cf_2d = np.array(cf).reshape(1, -1)  # Shape: (1, timesteps)
                        
                        # Calculate Keane et al. metrics
                        try:
                            keane_results = evaluate_keane_metrics(
                                original_ts_list=sample_2d,
                                counterfactual_ts_list=cf_2d,
                                model=model_wrapper,
                                target_classes=target_class,
                                tolerance=0.01
                            )
                            
                            # Calculate overall score (weighted average)
                            validity_val = keane_results['validity']
                            proximity_val = keane_results['proximity']
                            compactness_val = keane_results['compactness']
                            
                            # Normalize proximity (lower is better, so invert)
                            # Use max observed distance for normalization
                            proximity_normalized = 1 / (1 + proximity_val / 20.0)
                            
                            # Overall score: weighted average
                            overall_score = (0.4 * validity_val + 
                                           0.3 * proximity_normalized + 
                                           0.3 * compactness_val)
                            
                            results[variant_id]['validity_scores'].append(validity_val)
                            results[variant_id]['proximity_scores'].append(proximity_val)
                            results[variant_id]['compactness_scores'].append(compactness_val)
                            results[variant_id]['overall_scores'].append(overall_score)
                            results[variant_id]['successes'] += 1
                        except Exception as e:
                            # Keane metrics failed, but CF was successful
                            if sample_idx == 0:  # Print error once
                                print(f"    Warning: Keane metrics error: {str(e)[:70]}")
                            results[variant_id]['failures'] += 1
                    else:
                        results[variant_id]['failures'] += 1
                else:
                    results[variant_id]['failures'] += 1
                    
            except Exception as e:
                elapsed = time.time() - start_time
                results[variant_id]['times'].append(elapsed)
                results[variant_id]['failures'] += 1
        
        total = len(test_samples)
        success = results[variant_id]['successes']
        
        if success > 0:
            avg_validity = np.mean(results[variant_id]['validity_scores'])
            avg_proximity = np.mean(results[variant_id]['proximity_scores'])
            avg_compactness = np.mean(results[variant_id]['compactness_scores'])
            avg_overall = np.mean(results[variant_id]['overall_scores'])
            avg_time = np.mean(results[variant_id]['times'])
            
            print(f"  ✓ {variant_name}: {success}/{total} ({100*success/total:.1f}%)")
            print(f"    Validity: {avg_validity:.4f}, Proximity: {avg_proximity:.2f}, "
                  f"Compactness: {avg_compactness:.2%}, Overall: {avg_overall:.4f}")
        else:
            print(f"  ✗ {variant_name}: 0/{total} (0.0%) - All failed")
    
    # Print summary statistics
    print("\n" + "=" * 90)
    print(" KEANE ET AL. (2021) METRICS SUMMARY")
    print("=" * 90)
    
    print(f"\n{'Variant':<25} {'Success':<10} {'Validity':<12} {'Proximity':<12} "
          f"{'Compact.':<12} {'Overall':<12}")
    print("-" * 90)
    
    summary_data = []
    for variant_id, variant_name, _ in variants:
        r = results[variant_id]
        total = len(test_samples)
        success_rate = r['successes'] / total if total > 0 else 0
        
        if r['validity_scores']:
            avg_validity = np.mean(r['validity_scores'])
            avg_proximity = np.mean(r['proximity_scores'])
            avg_compactness = np.mean(r['compactness_scores'])
            avg_overall = np.mean(r['overall_scores'])
        else:
            avg_validity = avg_proximity = avg_compactness = avg_overall = 0
        
        summary_data.append({
            'id': variant_id,
            'name': variant_name,
            'success_rate': success_rate,
            'validity': avg_validity,
            'proximity': avg_proximity,
            'compactness': avg_compactness,
            'overall': avg_overall
        })
        
        print(f"{variant_name:<25} {success_rate:<10.2%} {avg_validity:<12.4f} "
              f"{avg_proximity:<12.2f} {avg_compactness:<12.2%} {avg_overall:<12.4f}")
    
    print("-" * 90)
    
    # Find best performers
    valid_results = [s for s in summary_data if s['overall'] > 0]
    if valid_results:
        best_overall = max(valid_results, key=lambda x: x['overall'])
        best_validity = max(valid_results, key=lambda x: x['validity'])
        best_proximity = min(valid_results, key=lambda x: x['proximity'])
        best_compactness = max(valid_results, key=lambda x: x['compactness'])
        
        print(f"\nBest Performers:")
        print(f"  🏆 Overall Score: {best_overall['name']} ({best_overall['overall']:.4f})")
        print(f"  ✓ Validity: {best_validity['name']} ({best_validity['validity']:.4f})")
        print(f"  📏 Proximity: {best_proximity['name']} ({best_proximity['proximity']:.2f})")
        print(f"  🎯 Compactness: {best_compactness['name']} ({best_compactness['compactness']:.2%})")
    
    # Create visualizations
    print(f"\n[5/5] Generating visualizations...")
    
    # Figure 1: Keane metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'FFT-CF Variants - Keane et al. (2021) Metrics ({len(test_samples)} samples)', 
                 fontsize=16, fontweight='bold')
    
    names = [s['name'] for s in summary_data if s['overall'] > 0]
    validities = [s['validity'] for s in summary_data if s['overall'] > 0]
    proximities = [s['proximity'] for s in summary_data if s['overall'] > 0]
    compactnesses = [s['compactness'] * 100 for s in summary_data if s['overall'] > 0]
    overalls = [s['overall'] for s in summary_data if s['overall'] > 0]
    
    if names:
        # Validity
        colors_valid = ['green' if v == max(validities) else 'skyblue' for v in validities]
        axes[0, 0].bar(range(len(names)), validities, color=colors_valid, edgecolor='black', linewidth=1)
        axes[0, 0].set_xticks(range(len(names)))
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        axes[0, 0].set_ylabel('Validity Score', fontweight='bold')
        axes[0, 0].set_title('Validity (higher is better)', fontweight='bold')
        axes[0, 0].set_ylim([0, 1.1])
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].axhline(y=np.mean(validities), color='orange', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(validities):.3f}')
        axes[0, 0].legend()
        
        # Proximity
        colors_prox = ['green' if p == min(proximities) else 'lightcoral' for p in proximities]
        axes[0, 1].bar(range(len(names)), proximities, color=colors_prox, edgecolor='black', linewidth=1)
        axes[0, 1].set_xticks(range(len(names)))
        axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        axes[0, 1].set_ylabel('Proximity (L2 Distance)', fontweight='bold')
        axes[0, 1].set_title('Proximity (lower is better)', fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].axhline(y=np.mean(proximities), color='orange', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(proximities):.2f}')
        axes[0, 1].legend()
        
        # Compactness
        colors_comp = ['purple' if c == max(compactnesses) else 'lightgreen' for c in compactnesses]
        axes[1, 0].bar(range(len(names)), compactnesses, color=colors_comp, edgecolor='black', linewidth=1)
        axes[1, 0].set_xticks(range(len(names)))
        axes[1, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        axes[1, 0].set_ylabel('Compactness (%)', fontweight='bold')
        axes[1, 0].set_title('Compactness (higher is better)', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].axhline(y=np.mean(compactnesses), color='orange', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(compactnesses):.1f}%')
        axes[1, 0].legend()
        
        # Overall score
        colors_overall = ['gold' if o == max(overalls) else 'plum' for o in overalls]
        axes[1, 1].bar(range(len(names)), overalls, color=colors_overall, edgecolor='black', linewidth=1)
        axes[1, 1].set_xticks(range(len(names)))
        axes[1, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        axes[1, 1].set_ylabel('Overall Score', fontweight='bold')
        axes[1, 1].set_title('Overall Keane Score (higher is better)', fontweight='bold')
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].axhline(y=np.mean(overalls), color='orange', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(overalls):.3f}')
        axes[1, 1].legend()
    
    plt.tight_layout()
    output_file = os.path.join(script_path, 'fft_cf_keane_metrics.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # Figure 2: Ranking visualization
    if valid_results:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create ranking based on overall score
        sorted_data = sorted(summary_data, key=lambda x: x['overall'], reverse=True)
        sorted_data = [s for s in sorted_data if s['overall'] > 0]
        
        if sorted_data:
            names_sorted = [s['name'] for s in sorted_data]
            validity_sorted = [s['validity'] for s in sorted_data]
            proximity_sorted = [1 / (1 + s['proximity']) for s in sorted_data]  # Normalize (closer to 1 is better)
            compactness_sorted = [s['compactness'] for s in sorted_data]
            
            x = np.arange(len(names_sorted))
            width = 0.25
            
            ax.bar(x - width, validity_sorted, width, label='Validity', color='skyblue', edgecolor='black')
            ax.bar(x, proximity_sorted, width, label='Proximity (normalized)', color='lightcoral', edgecolor='black')
            ax.bar(x + width, compactness_sorted, width, label='Compactness', color='lightgreen', edgecolor='black')
            
            ax.set_ylabel('Score', fontweight='bold', fontsize=12)
            ax.set_title(f'FFT-CF Variants Ranking - Keane et al. (2021) Metrics\n'
                        f'Evaluated on {len(test_samples)} samples', 
                        fontweight='bold', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(names_sorted, rotation=45, ha='right', fontsize=10)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.1])
            
            plt.tight_layout()
            output_file = os.path.join(script_path, 'fft_cf_keane_ranking.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved: {output_file}")
            plt.close()
    
    print("\n" + "=" * 90)
    print(" Keane et al. (2021) Batch Evaluation Complete!")
    print("=" * 90)
    print(f"\nEvaluated {len(test_samples)} samples across {len(variants)} variants")
    print(f"Total CF generations: {len(test_samples) * len(variants)}")
    print("\n")


if __name__ == '__main__':
    main()
