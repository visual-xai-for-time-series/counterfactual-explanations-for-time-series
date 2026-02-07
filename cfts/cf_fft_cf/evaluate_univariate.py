"""
FFT-CF Evaluation for Univariate Time Series
Evaluates all FFT-based counterfactual methods on the FordA dataset (univariate).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from aeon.datasets import load_classification
from examples.base.model import SimpleCNN

from cfts.cf_fft_cf import (
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


class NumpyDatasetWrapper:
    """Wrapper to make numpy array compatible with FFT-CF methods that expect iterables."""
    def __init__(self, data):
        """
        Args:
            data: numpy array where last column is the label
        """
        self.data = data
        self.features = data[:, :-1]
        self.labels = data[:, -1].astype(int)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Support both integer indexing and boolean/array indexing
        if isinstance(idx, (int, np.integer)):
            return self.features[idx], self.labels[idx]
        else:
            # For array indexing (used by new variants), return the underlying data
            return self.data[idx]
    
    # For new variants that access array directly
    def __array__(self):
        return self.data


def main():
    print("=" * 80)
    print("FFT-CF UNIVARIATE EVALUATION - FordA Dataset")
    print("=" * 80)
    
    device = torch.device('cpu')
    
    # Load FordA dataset (univariate)
    print("\n[1/5] Loading FordA dataset (univariate)...")
    X_train, y_train = load_classification("FordA", split="train")
    X_test, y_test = load_classification("FordA", split="test")
    
    X_train = X_train.squeeze()
    X_test = X_test.squeeze()
    y_train = (y_train == '1').astype(int)
    y_test = (y_test == '1').astype(int)
    
    train_data_array = np.column_stack([X_train, y_train])
    train_data = NumpyDatasetWrapper(train_data_array)
    
    print(f"  Dataset shape: {X_train.shape}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Time steps: {X_train.shape[1]}")
    print(f"  Channels: 1 (univariate)")
    
    # Load model
    print("\n[2/5] Loading SimpleCNN model...")
    model_path = "../../models/simple_cnn_2.pth"
    model = SimpleCNN(output_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"  Model loaded: {model_path}")
    
    # Find correctly classified samples
    print("\n[3/5] Finding correctly classified samples...")
    correct_indices = []
    for i, (x, y) in enumerate(zip(X_test, y_test)):
        x_tensor = torch.FloatTensor(x).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_tensor).cpu().numpy()[0]
        if np.argmax(pred) == y:
            correct_indices.append(i)
    
    print(f"  Found {len(correct_indices)} correctly classified samples")
    
    # Evaluate on subset
    max_samples = min(100, len(correct_indices))
    test_indices = correct_indices[:max_samples]
    
    print(f"\n[4/5] Evaluating FFT-CF variants on {max_samples} samples...")
    print("=" * 80)
    
    # Define all FFT-CF variants
    variants = [
        ('Nearest Neighbor', 
         lambda s, d, m, t: fft_nn_cf(s, d, m, t, k=5, verbose=False)),
        
        ('Adaptive Saliency', 
         lambda s, d, m, t: fft_adaptive_cf(s, d, m, t, k=5, use_saliency=True, verbose=False)),
        
        ('Iterative Refinement', 
         lambda s, d, m, t: fft_iterative_cf(s, d, m, t, k=5, refine_iterations=30, verbose=False)),
        
        ('Smart Blend', 
         lambda s, d, m, t: fft_smart_blend_cf(s, d, m, t, k=5, search_method='binary', verbose=False)),
        
        ('Frequency Distance', 
         lambda s, d, m, t: fft_freq_distance_cf(s, d, m, t, k=5, freq_weight_strategy='energy', verbose=False)),
        
        ('Wavelet Transform', 
         lambda s, d, m, t: fft_wavelet_cf(s, d, m, t, k=5, wavelet='db4', level=3, verbose=False)),
        
        ('Hybrid Amp-Phase', 
         lambda s, d, m, t: fft_hybrid_cf(s, d, m, t, k=5, analyze_importance=False, verbose=False)),
        
        ('Progressive Switching', 
         lambda s, d, m, t: fft_progressive_cf(s, d, m, t, k=5, steps_per_neighbor=5, verbose=False)),
        
        ('Confidence Threshold', 
         lambda s, d, m, t: fft_confidence_threshold_cf(s, d, m, t, k=5, confidence_threshold=0.85, verbose=False)),
        
        ('Hybrid Enhanced', 
         lambda s, d, m, t: fft_hybrid_enhanced_cf(s, d, m, t, k=5, analyze_importance=True, fallback_on_failure=True, verbose=False)),
        
        ('Band Optimizer', 
         lambda s, d, m, t: fft_band_optimizer_cf(s, d, m, t, k=5, num_bands=3, use_saliency=True, verbose=False)),
    ]
    
    results = {}
    
    for variant_name, variant_func in variants:
        print(f"\nEvaluating {variant_name}...")
        
        successes = 0
        confidences = []
        distances = []
        sparsities = []
        times = []
        
        for idx in test_indices:
            sample = X_test[idx]
            true_label = y_test[idx]
            target_class = 1 - true_label
            
            start_time = time.time()
            try:
                cf, pred = variant_func(sample, train_data, model, target_class)
                elapsed = time.time() - start_time
                
                if cf is not None and pred is not None and np.argmax(pred) == target_class:
                    successes += 1
                    confidences.append(pred[target_class])
                    distance = np.linalg.norm(cf.flatten() - sample.flatten())
                    distances.append(distance)
                    
                    # Compute sparsity
                    diff = np.abs(cf.flatten() - sample.flatten())
                    threshold = 0.01 * np.ptp(sample)
                    sparsity = np.sum(diff > threshold) / len(sample.flatten())
                    sparsities.append(sparsity * 100)
                    
                    times.append(elapsed)
            except Exception as e:
                pass
        
        success_rate = successes / max_samples * 100
        
        results[variant_name] = {
            'success_rate': success_rate,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_distance': np.mean(distances) if distances else 0,
            'avg_sparsity': np.mean(sparsities) if sparsities else 0,
            'avg_time': np.mean(times) if times else 0,
            'confidences': confidences,
            'distances': distances,
            'times': times
        }
        
        print(f"  Success: {successes}/{max_samples} ({success_rate:.1f}%)", end="")
        if confidences:
            print(f" | Conf: {np.mean(confidences):.3f} | Dist: {np.mean(distances):.1f} | Time: {np.mean(times):.3f}s")
        else:
            print()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - UNIVARIATE (FordA)")
    print("=" * 80)
    print(f"{'Variant':<25} {'Success':<10} {'Confidence':<12} {'Distance':<12} {'Sparsity':<12} {'Time (s)':<10}")
    print("-" * 80)
    
    for name, res in results.items():
        print(f"{name:<25} {res['success_rate']:<10.1f}% {res['avg_confidence']:<12.4f} "
              f"{res['avg_distance']:<12.2f} {res['avg_sparsity']:<12.2f}% {res['avg_time']:<10.4f}")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    
    # Find best performers (only among successful variants)
    successful_results = {k: v for k, v in results.items() if v['success_rate'] > 0}
    
    if successful_results:
        best_success = max(successful_results.items(), key=lambda x: x[1]['success_rate'])
        best_confidence = max(successful_results.items(), key=lambda x: x[1]['avg_confidence'])
        best_distance = min(successful_results.items(), key=lambda x: x[1]['avg_distance'])
        best_sparsity = max(successful_results.items(), key=lambda x: x[1]['avg_sparsity'])
        best_time = min(successful_results.items(), key=lambda x: x[1]['avg_time'])
        
        print(f"  🏆 Best Success Rate: {best_success[0]} ({best_success[1]['success_rate']:.1f}%)")
        print(f"  🎯 Best Confidence: {best_confidence[0]} ({best_confidence[1]['avg_confidence']:.4f})")
        print(f"  📏 Best Distance (closest): {best_distance[0]} ({best_distance[1]['avg_distance']:.2f})")
        print(f"  ✂️  Best Sparsity (most sparse): {best_sparsity[0]} ({best_sparsity[1]['avg_sparsity']:.2f}%)")
        print(f"  ⚡ Fastest: {best_time[0]} ({best_time[1]['avg_time']:.4f}s)")
    
    # Visualization
    print("\n[5/5] Generating visualizations...")
    
    # Filter out failed variants for plotting
    plot_results = {k: v for k, v in results.items() if v['success_rate'] > 0}
    
    if plot_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('FFT-CF Univariate Evaluation Results - FordA Dataset', fontsize=14, fontweight='bold')
        
        names = list(plot_results.keys())
        
        # Success rates
        success_rates = [plot_results[n]['success_rate'] for n in names]
        axes[0, 0].barh(names, success_rates, color='steelblue')
        axes[0, 0].set_xlabel('Success Rate (%)')
        axes[0, 0].set_title('Success Rate by Variant')
        axes[0, 0].set_xlim([0, 105])
        for i, v in enumerate(success_rates):
            axes[0, 0].text(v + 1, i, f'{v:.1f}%', va='center')
        
        # Confidence
        confidences = [plot_results[n]['avg_confidence'] for n in names]
        axes[0, 1].barh(names, confidences, color='green')
        axes[0, 1].set_xlabel('Average Confidence')
        axes[0, 1].set_title('Average Confidence by Variant')
        axes[0, 1].set_xlim([0, 1.0])
        for i, v in enumerate(confidences):
            axes[0, 1].text(v + 0.02, i, f'{v:.3f}', va='center')
        
        # Distance
        distances = [plot_results[n]['avg_distance'] for n in names]
        axes[1, 0].barh(names, distances, color='coral')
        axes[1, 0].set_xlabel('Average Distance')
        axes[1, 0].set_title('Average Distance by Variant')
        for i, v in enumerate(distances):
            axes[1, 0].text(v + 0.5, i, f'{v:.1f}', va='center')
        
        # Execution time
        times = [plot_results[n]['avg_time'] for n in names]
        axes[1, 1].barh(names, times, color='purple')
        axes[1, 1].set_xlabel('Average Time (seconds)')
        axes[1, 1].set_title('Average Execution Time by Variant')
        for i, v in enumerate(times):
            axes[1, 1].text(v + 0.001, i, f'{v:.3f}s', va='center')
        
        plt.tight_layout()
        output_path = "fft_cf_univariate_evaluation.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()
    
    print("\n" + "=" * 80)
    print("Univariate Evaluation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
