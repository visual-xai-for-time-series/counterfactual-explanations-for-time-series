"""
FFT-CF Variants Batch Evaluation
Evaluates all FFT-based counterfactual methods on the entire FordA test dataset.
"""

import os
import sys
import time
from collections import defaultdict

# Add parent directories to path
script_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(script_path, '..', '..'))
sys.path.insert(0, parent_path)

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

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
    fft_progressive_cf,
    fft_confidence_threshold_cf,
    fft_hybrid_enhanced_cf,
    fft_band_optimizer_cf
)


def main():
    print("=" * 90)
    print(" FFT-CF Variants Batch Evaluation - FordA Test Dataset")
    print("=" * 90)
    print()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load FordA dataset
    print("\n[1/6] Loading FordA dataset...")
    _, dataset_train = bd.get_UCR_UEA_dataloader(split='train')
    _, dataset_test = bd.get_UCR_UEA_dataloader(split='test')
    print(f"  ✓ Train samples: {len(dataset_train)}")
    print(f"  ✓ Test samples: {len(dataset_test)}")
    
    # Load model
    print("\n[2/6] Loading SimpleCNN model...")
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
    
    # Find correctly classified test samples
    print("\n[3/6] Finding correctly classified test samples...")
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
    
    # Limit number of samples for faster evaluation
    max_samples = min(100, len(test_samples))  # Evaluate on 100 samples
    test_samples = test_samples[:max_samples]
    print(f"  ✓ Evaluating on {len(test_samples)} samples")
    
    # Define FFT-CF variants (using faster parameters)
    variants = [
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
        
        ('fft_confidence_threshold_cf', 'Confidence Threshold',
         lambda s, d, m, t: fft_confidence_threshold_cf(s, d, m, t, k=5, confidence_threshold=0.85, verbose=False)),
        
        ('fft_hybrid_enhanced_cf', 'Hybrid Enhanced',
         lambda s, d, m, t: fft_hybrid_enhanced_cf(s, d, m, t, k=5, analyze_importance=True, fallback_on_failure=True, verbose=False)),
        
        ('fft_band_optimizer_cf', 'Band Optimizer',
         lambda s, d, m, t: fft_band_optimizer_cf(s, d, m, t, k=5, num_bands=3, use_saliency=True, verbose=False)),
    ]
    
    # Run batch evaluation
    print(f"\n[4/6] Evaluating {len(variants)} FFT-CF variants on {len(test_samples)} samples...")
    print("=" * 90)
    
    results = defaultdict(lambda: {
        'successes': 0,
        'failures': 0,
        'errors': 0,
        'distances': [],
        'confidences': [],
        'sparsities': [],
        'times': []
    })
    
    for variant_idx, (variant_id, variant_name, variant_func) in enumerate(variants):
        print(f"\n[{variant_idx+1}/{len(variants)}] Testing {variant_name}...")
        
        for sample_idx, test_info in enumerate(test_samples):
            if (sample_idx + 1) % 20 == 0:
                print(f"  Progress: {sample_idx+1}/{len(test_samples)} samples...", end='\r')
            
            sample = test_info['sample']
            target_class = test_info['target_class']
            
            start_time = time.time()
            try:
                cf, pred = variant_func(sample, dataset_test, model, target_class)
                elapsed = time.time() - start_time
                
                if cf is not None and pred is not None:
                    pred_class = np.argmax(pred)
                    confidence = np.max(pred)
                    distance = np.linalg.norm(cf - sample)
                    
                    # Calculate sparsity
                    signal_range = np.max(sample) - np.min(sample)
                    threshold = 0.01 * signal_range if signal_range > 0 else 0.01
                    sparsity = np.sum(np.abs(cf - sample) < threshold) / sample.size
                    
                    if pred_class == target_class:
                        results[variant_id]['successes'] += 1
                        results[variant_id]['distances'].append(distance)
                        results[variant_id]['confidences'].append(confidence)
                        results[variant_id]['sparsities'].append(sparsity * 100)
                    else:
                        results[variant_id]['failures'] += 1
                    
                    results[variant_id]['times'].append(elapsed)
                else:
                    results[variant_id]['failures'] += 1
                    results[variant_id]['times'].append(elapsed)
                    
            except Exception as e:
                elapsed = time.time() - start_time
                results[variant_id]['errors'] += 1
                results[variant_id]['times'].append(elapsed)
        
        total = len(test_samples)
        success = results[variant_id]['successes']
        failure = results[variant_id]['failures']
        error = results[variant_id]['errors']
        avg_time = np.mean(results[variant_id]['times'])
        
        print(f"  ✓ {variant_name}: {success}/{total} success ({100*success/total:.1f}%), "
              f"{failure} failures, {error} errors, avg time: {avg_time:.3f}s")
    
    # Print summary statistics
    print("\n" + "=" * 90)
    print(" SUMMARY STATISTICS")
    print("=" * 90)
    
    print(f"\n{'Variant':<25} {'Success':<10} {'Avg Dist':<12} {'Avg Conf':<12} {'Avg Sparse':<12} {'Avg Time':<12}")
    print("-" * 90)
    
    summary_data = []
    for variant_id, variant_name, _ in variants:
        r = results[variant_id]
        total = len(test_samples)
        success_rate = r['successes'] / total if total > 0 else 0
        
        if r['distances']:
            avg_dist = np.mean(r['distances'])
            avg_conf = np.mean(r['confidences'])
            avg_sparse = np.mean(r['sparsities'])
        else:
            avg_dist = avg_conf = avg_sparse = 0
        
        avg_time = np.mean(r['times']) if r['times'] else 0
        
        summary_data.append({
            'id': variant_id,
            'name': variant_name,
            'success_rate': success_rate,
            'avg_dist': avg_dist,
            'avg_conf': avg_conf,
            'avg_sparse': avg_sparse,
            'avg_time': avg_time
        })
        
        print(f"{variant_name:<25} {success_rate:<10.2%} {avg_dist:<12.2f} {avg_conf:<12.4f} "
              f"{avg_sparse:<12.2f}% {avg_time:<12.3f}s")
    
    print("-" * 90)
    
    # Find best performers
    best_success = max(summary_data, key=lambda x: x['success_rate'])
    best_dist = min([s for s in summary_data if s['avg_dist'] > 0], key=lambda x: x['avg_dist'])
    best_conf = max([s for s in summary_data if s['avg_conf'] > 0], key=lambda x: x['avg_conf'])
    best_time = min(summary_data, key=lambda x: x['avg_time'])
    
    print(f"\nBest Performers:")
    print(f"  ✓ Success Rate: {best_success['name']} ({best_success['success_rate']:.1%})")
    print(f"  📏 Closest: {best_dist['name']} (avg dist: {best_dist['avg_dist']:.2f})")
    print(f"  🎯 Most Confident: {best_conf['name']} (avg conf: {best_conf['avg_conf']:.4f})")
    print(f"  ⚡ Fastest: {best_time['name']} (avg time: {best_time['avg_time']:.3f}s)")
    
    # Create visualizations
    print(f"\n[5/6] Generating visualizations...")
    
    # Figure 1: Success rates
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'FFT-CF Variants Evaluation ({len(test_samples)} samples)', 
                 fontsize=16, fontweight='bold')
    
    names = [s['name'] for s in summary_data]
    success_rates = [s['success_rate'] * 100 for s in summary_data]
    avg_dists = [s['avg_dist'] for s in summary_data]
    avg_confs = [s['avg_conf'] for s in summary_data]
    avg_times = [s['avg_time'] for s in summary_data]
    
    # Success rates
    colors_success = ['green' if sr == max(success_rates) else 'skyblue' for sr in success_rates]
    axes[0, 0].bar(range(len(names)), success_rates, color=colors_success, edgecolor='black', linewidth=1)
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[0, 0].set_ylabel('Success Rate (%)', fontweight='bold')
    axes[0, 0].set_title('Success Rate (higher is better)', fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].axhline(y=np.mean(success_rates), color='orange', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(success_rates):.1f}%')
    axes[0, 0].legend()
    
    # Average distances (only for successful CFs)
    valid_dists = [d for d in avg_dists if d > 0]
    valid_names_dist = [names[i] for i, d in enumerate(avg_dists) if d > 0]
    if valid_dists:
        colors_dist = ['green' if d == min(valid_dists) else 'lightcoral' for d in valid_dists]
        axes[0, 1].bar(range(len(valid_names_dist)), valid_dists, color=colors_dist, 
                       edgecolor='black', linewidth=1)
        axes[0, 1].set_xticks(range(len(valid_names_dist)))
        axes[0, 1].set_xticklabels(valid_names_dist, rotation=45, ha='right', fontsize=9)
        axes[0, 1].set_ylabel('Average L2 Distance', fontweight='bold')
        axes[0, 1].set_title('Proximity (lower is better)', fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].axhline(y=np.mean(valid_dists), color='orange', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(valid_dists):.2f}')
        axes[0, 1].legend()
    
    # Average confidences
    valid_confs = [c for c in avg_confs if c > 0]
    valid_names_conf = [names[i] for i, c in enumerate(avg_confs) if c > 0]
    if valid_confs:
        colors_conf = ['purple' if c == max(valid_confs) else 'lightgreen' for c in valid_confs]
        axes[1, 0].bar(range(len(valid_names_conf)), valid_confs, color=colors_conf, 
                       edgecolor='black', linewidth=1)
        axes[1, 0].set_xticks(range(len(valid_names_conf)))
        axes[1, 0].set_xticklabels(valid_names_conf, rotation=45, ha='right', fontsize=9)
        axes[1, 0].set_ylabel('Average Confidence', fontweight='bold')
        axes[1, 0].set_title('Target Class Confidence (higher is better)', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].axhline(y=np.mean(valid_confs), color='orange', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(valid_confs):.3f}')
        axes[1, 0].legend()
    
    # Average times
    colors_time = ['red' if t == min(avg_times) else 'plum' for t in avg_times]
    axes[1, 1].bar(range(len(names)), avg_times, color=colors_time, edgecolor='black', linewidth=1)
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_ylabel('Average Time (seconds)', fontweight='bold')
    axes[1, 1].set_title('Execution Time (lower is better)', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].axhline(y=np.mean(avg_times), color='orange', linestyle='--',
                      linewidth=2, label=f'Mean: {np.mean(avg_times):.3f}s')
    axes[1, 1].legend()
    
    plt.tight_layout()
    output_file = os.path.join(script_path, 'fft_cf_batch_evaluation.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # Figure 2: Box plots for distribution analysis
    print(f"[6/6] Generating distribution plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'FFT-CF Variants Distribution Analysis ({len(test_samples)} samples)', 
                 fontsize=16, fontweight='bold')
    
    # Distance distributions
    dist_data = [results[s['id']]['distances'] for s in summary_data if results[s['id']]['distances']]
    dist_labels = [s['name'] for s in summary_data if results[s['id']]['distances']]
    if dist_data:
        bp1 = axes[0].boxplot(dist_data, labels=dist_labels, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
        axes[0].set_ylabel('L2 Distance', fontweight='bold')
        axes[0].set_title('Distance Distribution', fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        axes[0].grid(axis='y', alpha=0.3)
    
    # Confidence distributions
    conf_data = [results[s['id']]['confidences'] for s in summary_data if results[s['id']]['confidences']]
    conf_labels = [s['name'] for s in summary_data if results[s['id']]['confidences']]
    if conf_data:
        bp2 = axes[1].boxplot(conf_data, labels=conf_labels, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightgreen')
        axes[1].set_ylabel('Confidence', fontweight='bold')
        axes[1].set_title('Confidence Distribution', fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        axes[1].grid(axis='y', alpha=0.3)
    
    # Time distributions
    time_data = [results[s['id']]['times'] for s in summary_data]
    time_labels = [s['name'] for s in summary_data]
    bp3 = axes[2].boxplot(time_data, labels=time_labels, patch_artist=True)
    for patch in bp3['boxes']:
        patch.set_facecolor('lightyellow')
    axes[2].set_ylabel('Time (seconds)', fontweight='bold')
    axes[2].set_title('Execution Time Distribution', fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(script_path, 'fft_cf_distributions.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    print("\n" + "=" * 90)
    print(" Batch Evaluation Complete!")
    print("=" * 90)
    print(f"\nEvaluated {len(test_samples)} samples across {len(variants)} variants")
    print(f"Total CF generations: {len(test_samples) * len(variants)}")
    print("\n")


if __name__ == '__main__':
    main()
