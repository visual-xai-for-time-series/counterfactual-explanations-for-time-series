"""
FFT-CF Variants Comparison
Compares all FFT-based counterfactual methods using FordA dataset and SimpleCNN model.
"""

import os
import sys
import time

# Add parent directories to path
script_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(script_path, '..', '..'))
sys.path.insert(0, parent_path)

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Import base modules (same as example_univariate.py)
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
    fft_progressive_cf,
    fft_confidence_threshold_cf,
    fft_hybrid_enhanced_cf,
    fft_band_optimizer_cf
)


def main():
    print("=" * 90)
    print(" FFT-CF Variants Comparison - FordA Dataset")
    print("=" * 90)
    print()

    # Setup (same as example_univariate.py)
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
        print("  Please run example_univariate.py first to train the model.")
        return
    
    # Select test sample
    print("\n[3/5] Selecting correctly classified test sample...")
    max_attempts = 100
    sample = None
    
    for attempt in range(max_attempts):
        random_idx = np.random.randint(0, len(dataset_test))
        candidate_sample, candidate_label = dataset_test[random_idx]
        
        with torch.no_grad():
            sample_tensor = torch.from_numpy(np.array(candidate_sample)).float().to(device)
            if sample_tensor.dim() == 1:
                sample_tensor = sample_tensor.unsqueeze(0).unsqueeze(0)
            elif sample_tensor.dim() == 2:
                sample_tensor = sample_tensor.unsqueeze(0)
            
            pred_output = model(sample_tensor)
            pred_class = torch.argmax(pred_output, dim=-1).item()
            true_class = np.argmax(candidate_label)
            
            if pred_class == true_class:
                sample = candidate_sample
                label = candidate_label
                original_class = pred_class
                confidence = pred_output[0, pred_class].item()
                print(f"  ✓ Found sample at index {random_idx}")
                print(f"    - Original class: {original_class}")
                print(f"    - Confidence: {confidence:.4f}")
                break
    
    if sample is None:
        print("  ✗ Could not find correctly classified sample")
        return
    
    target_class = 1 - original_class
    print(f"    - Target class: {target_class}")
    
    # Define FFT-CF variants to test
    # Note: New variants (confidence_threshold, hybrid_enhanced, band_optimizer) 
    #       require numpy array format and are not compatible with this script
    variants = [
        ('fft_cf', 'Original Greedy', 
         lambda: fft_cf(sample, dataset_test, model, target_class, max_iterations=50, verbose=False)),
        
        ('fft_gradient_cf', 'Gradient Descent',
         lambda: fft_gradient_cf(sample, dataset_test, model, target_class, max_iterations=50, verbose=False)),
        
        ('fft_nn_cf', 'Nearest Neighbor',
         lambda: fft_nn_cf(sample, dataset_test, model, target_class, k=5, verbose=False)),
        
        ('fft_adaptive_cf', 'Adaptive Saliency',
         lambda: fft_adaptive_cf(sample, dataset_test, model, target_class, k=5, use_saliency=True, verbose=False)),
        
        ('fft_iterative_cf', 'Iterative Refinement',
         lambda: fft_iterative_cf(sample, dataset_test, model, target_class, k=5, refine_iterations=50, verbose=False)),
        
        ('fft_smart_blend_cf', 'Smart Blend (Binary)',
         lambda: fft_smart_blend_cf(sample, dataset_test, model, target_class, k=5, search_method='binary', verbose=False)),
        
        ('fft_freq_distance_cf', 'Frequency Distance',
         lambda: fft_freq_distance_cf(sample, dataset_test, model, target_class, k=5, freq_weight_strategy='energy', verbose=False)),
        
        ('fft_wavelet_cf', 'Wavelet Transform',
         lambda: fft_wavelet_cf(sample, dataset_test, model, target_class, k=5, wavelet='db4', level=3, verbose=False)),
        
        ('fft_hybrid_cf', 'Hybrid Amp-Phase',
         lambda: fft_hybrid_cf(sample, dataset_test, model, target_class, k=5, analyze_importance=True, verbose=False)),
        
        ('fft_progressive_cf', 'Progressive Switching',
         lambda: fft_progressive_cf(sample, dataset_test, model, target_class, k=5, steps_per_neighbor=5, verbose=False)),
    ]
    
    # Run all variants
    print(f"\n[4/5] Testing {len(variants)} FFT-CF variants...")
    print("-" * 90)
    print(f"{'Variant':<30} {'Status':<12} {'Time (s)':<12} {'Distance':<12} {'Confidence':<12}")
    print("-" * 90)
    
    results = []
    
    for variant_id, variant_name, variant_func in variants:
        start_time = time.time()
        try:
            cf, pred = variant_func()
            elapsed = time.time() - start_time
            
            if cf is not None and pred is not None:
                pred_class = np.argmax(pred)
                confidence = np.max(pred)
                distance = np.linalg.norm(cf - sample)
                
                success = pred_class == target_class
                status = "✓ SUCCESS" if success else "✗ WRONG CLASS"
                
                # Calculate additional metrics
                # Sparsity: percentage of values with minimal change (< 1% of signal range)
                signal_range = np.max(sample) - np.min(sample)
                threshold = 0.01 * signal_range if signal_range > 0 else 0.01
                sparsity = np.sum(np.abs(cf - sample) < threshold) / sample.size
                max_change = np.max(np.abs(cf - sample))
                
                results.append({
                    'id': variant_id,
                    'name': variant_name,
                    'success': success,
                    'time': elapsed,
                    'cf': cf,
                    'pred': pred,
                    'pred_class': pred_class,
                    'confidence': confidence,
                    'distance': distance,
                    'sparsity': sparsity * 100,
                    'max_change': max_change
                })
                
                print(f"{variant_name:<30} {status:<12} {elapsed:<12.3f} {distance:<12.2f} {confidence:<12.4f}")
            else:
                results.append({
                    'id': variant_id,
                    'name': variant_name,
                    'success': False,
                    'time': elapsed,
                    'cf': None
                })
                print(f"{variant_name:<30} {'✗ FAILED':<12} {elapsed:<12.3f} {'-':<12} {'-':<12}")
                
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)[:30]
            results.append({
                'id': variant_id,
                'name': variant_name,
                'success': False,
                'time': elapsed,
                'cf': None,
                'error': error_msg
            })
            print(f"{variant_name:<30} {'✗ ERROR':<12} {elapsed:<12.3f} {error_msg:<12}")
    
    print("-" * 90)
    
    # Summary statistics
    successful = [r for r in results if r['success']]
    print(f"\n{'='*90}")
    print(" SUMMARY STATISTICS")
    print(f"{'='*90}")
    print(f"Success Rate: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)")
    
    if successful:
        avg_time = np.mean([r['time'] for r in successful])
        avg_distance = np.mean([r['distance'] for r in successful])
        avg_confidence = np.mean([r['confidence'] for r in successful])
        avg_sparsity = np.mean([r['sparsity'] for r in successful])
        
        print(f"\nAverages (successful methods only):")
        print(f"  - Time: {avg_time:.3f}s")
        print(f"  - Distance: {avg_distance:.2f}")
        print(f"  - Confidence: {avg_confidence:.4f}")
        print(f"  - Sparsity: {avg_sparsity:.2f}%")
        
        # Find best performers
        fastest = min(successful, key=lambda x: x['time'])
        best_dist = min(successful, key=lambda x: x['distance'])
        best_conf = max(successful, key=lambda x: x['confidence'])
        
        print(f"\nBest Performers:")
        print(f"  ⚡ Fastest: {fastest['name']} ({fastest['time']:.3f}s)")
        print(f"  📏 Closest: {best_dist['name']} (distance: {best_dist['distance']:.2f})")
        print(f"  🎯 Most Confident: {best_conf['name']} (confidence: {best_conf['confidence']:.4f})")
    
    # Create visualizations (same format as example_univariate.py)
    print(f"\n[5/5] Generating visualizations...")
    
    # Normalize sample for plotting
    sample_pl = sample.reshape(-1) if len(sample.shape) > 1 else sample
    
    # Helper function for plotting (same as example_univariate.py)
    def plot_cf(ax, arr, title):
        """Plot a single time series."""
        ax.plot(arr, linewidth=1.5, color='black', alpha=0.8)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def overlay_cf(ax, original, cf, title):
        """Plot counterfactual overlaid on original (same style as example_univariate.py)."""
        ax.plot(original, linestyle='--', color='blue', linewidth=1.5, alpha=0.6, label='Original')
        ax.plot(cf, linewidth=1.5, color='black', alpha=0.9, label='CF')
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Figure 1: Vertical layout like example_univariate.py
    # 1 original + N individual CFs + N overlays
    n_methods = len(results)
    if n_methods > 0:
        n_rows = 1 + n_methods + n_methods  # original + individuals + overlays
        
        fig, axs = plt.subplots(n_rows, figsize=(10, 1.75 * n_rows))
        fig.suptitle('FFT-CF Variants Comparison - FordA', y=0.998, fontsize=14)
        
        i = 0
        
        # Original sample
        pred_str = f"Class {original_class} (conf: {confidence:.4f})"
        axs[i].plot(sample_pl, linewidth=2, color='blue', alpha=0.8)
        axs[i].set_title(f'Original sample — true: Class {original_class}, pred: {pred_str}', fontsize=9)
        axs[i].grid(True, alpha=0.3)
        i += 1
        
        # Individual counterfactual plots (all methods)
        for result in results:
            if result['success']:
                pred_str = f"Class {result['pred_class']} (conf: {result['confidence']:.4f})"
                cf_pl = result['cf'].reshape(-1) if len(result['cf'].shape) > 1 else result['cf']
                title = f"{result['name']} [✓] — pred: {pred_str}"
                plot_cf(axs[i], cf_pl, title)
            else:
                axs[i].set_title(f"{result['name']} [✗ FAILED]", fontsize=9)
            i += 1
        
        # Overlay plots: counterfactual vs original (all methods)
        for result in results:
            if result['success']:
                pred_str = f"Class {result['pred_class']} (conf: {result['confidence']:.4f})"
                cf_pl = result['cf'].reshape(-1) if len(result['cf'].shape) > 1 else result['cf']
                title = f"{result['name']} vs Original [✓] — pred: {pred_str}"
                overlay_cf(axs[i], sample_pl, cf_pl, title)
            else:
                axs[i].set_title(f"{result['name']} vs Original [✗ FAILED]", fontsize=9)
            i += 1
        
        plt.tight_layout(rect=[0, 0.01, 1, 0.999])
        output_file = os.path.join(script_path, 'fft_cf_comparison.png')
        plt.savefig(output_file, dpi=300)
        print(f"  ✓ Saved: {output_file}")
        plt.close()
    
    # Figure 2: Performance metrics
    if len(successful) >= 3:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('FFT-CF Variants - Performance Metrics', 
                     fontsize=16, fontweight='bold')
        
        names = [r['name'] for r in successful]
        times = [r['time'] for r in successful]
        distances = [r['distance'] for r in successful]
        confidences = [r['confidence'] for r in successful]
        sparsities = [r['sparsity'] for r in successful]
        
        # Execution time
        colors_time = ['red' if t == min(times) else 'skyblue' for t in times]
        axes[0, 0].bar(range(len(names)), times, color=colors_time, edgecolor='black', linewidth=1)
        axes[0, 0].set_xticks(range(len(names)))
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        axes[0, 0].set_ylabel('Time (seconds)', fontweight='bold')
        axes[0, 0].set_title('Execution Time (lower is better)', fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].axhline(y=np.mean(times), color='orange', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(times):.3f}s')
        axes[0, 0].legend()
        
        # Distance to original
        colors_dist = ['green' if d == min(distances) else 'lightcoral' for d in distances]
        axes[0, 1].bar(range(len(names)), distances, color=colors_dist, edgecolor='black', linewidth=1)
        axes[0, 1].set_xticks(range(len(names)))
        axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        axes[0, 1].set_ylabel('L2 Distance', fontweight='bold')
        axes[0, 1].set_title('Proximity (lower is better)', fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].axhline(y=np.mean(distances), color='orange', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(distances):.2f}')
        axes[0, 1].legend()
        
        # Confidence
        colors_conf = ['purple' if c == max(confidences) else 'lightgreen' for c in confidences]
        axes[1, 0].bar(range(len(names)), confidences, color=colors_conf, edgecolor='black', linewidth=1)
        axes[1, 0].set_xticks(range(len(names)))
        axes[1, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        axes[1, 0].set_ylabel('Confidence', fontweight='bold')
        axes[1, 0].set_title('Target Class Confidence (higher is better)', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].axhline(y=np.mean(confidences), color='orange', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
        axes[1, 0].legend()
        
        # Sparsity
        axes[1, 1].bar(range(len(names)), sparsities, color='plum', edgecolor='black', linewidth=1)
        axes[1, 1].set_xticks(range(len(names)))
        axes[1, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        axes[1, 1].set_ylabel('Sparsity (%)', fontweight='bold')
        axes[1, 1].set_title('Percentage of Unchanged Values (higher is better)', fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].axhline(y=np.mean(sparsities), color='orange', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(sparsities):.1f}%')
        axes[1, 1].legend()
        
        plt.tight_layout()
        output_file = os.path.join(script_path, 'fft_cf_metrics.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
    
    print(f"\n{'='*90}")
    print(" Comparison complete!")
    print(f"{'='*90}\n")


if __name__ == '__main__':
    main()
