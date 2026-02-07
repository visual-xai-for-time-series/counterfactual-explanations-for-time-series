"""
FFT-CF Evaluation for Multivariate Time Series
Evaluates FFT-based counterfactual methods on the SpokenArabicDigits dataset (multivariate).
Note: FFT methods work channel-wise for multivariate data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from aeon.datasets import load_classification

# Import the actual model used in examples
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'examples'))
from base.model import SimpleCNNMulti

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
    """Wrapper to make numpy array compatible with FFT-CF methods."""
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


def apply_fft_cf_multivariate(sample, dataset, model, target_class, variant_func, device):
    """
    Apply FFT-CF method to multivariate time series.
    
    Note: This is a simplified approach that treats the flattened multivariate  
    time series as a single long univariate series for FFT processing.
    This may not capture inter-channel dependencies optimally.
    
    Args:
        sample: Shape (channels, timesteps)
        dataset: Training dataset (wrapper around flattened data)
        model: Classification model
        target_class: Target class for CF
        variant_func: FFT-CF variant function
        device: Torch device
    
    Returns:
        cf: Counterfactual time series (channels, timesteps)
        pred: Model prediction
    """
    n_channels, n_timesteps = sample.shape
    
    # Flatten the multivariate sample to treat as univariate
    flat_sample = sample.flatten()
    
    # Apply FFT-CF to flattened representation
    try:
        cf_flat, pred = variant_func(flat_sample, dataset, model, target_class)
        
        if cf_flat is not None and pred is not None:
            # Reshape back to multivariate format
            cf = cf_flat.reshape(n_channels, n_timesteps)
            return cf, pred
        else:
            return None, None
    except Exception as e:
        return None, None


def main():
    print("=" * 80)
    print("FFT-CF MULTIVARIATE EVALUATION - SpokenArabicDigits Dataset")
    print("=" * 80)
    
    device = torch.device('cpu')
    
    # Load SpokenArabicDigits dataset (multivariate)
    print("\n[1/5] Loading SpokenArabicDigits dataset (multivariate)...")
    X_train, y_train = load_classification("SpokenArabicDigits", split="train")
    X_test, y_test = load_classification("SpokenArabicDigits", split="test")
    
    # Dataset comes as (samples, channels, timesteps) - already in correct format!
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    print(f"  Dataset shape: {X_train.shape}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Channels: {X_train.shape[1]} (multivariate)")
    print(f"  Time steps: {X_train.shape[2]}")
    
    # Create dataset in format expected by FFT methods (flatten channels for neighbor search)
    # Shape: (samples, channels*timesteps + 1 label)
    train_data = []
    for i in range(len(X_train)):
        flat_sample = X_train[i].flatten()
        train_data.append(np.concatenate([flat_sample, [y_train[i]]]))
    train_data_array = np.array(train_data)
    train_data = NumpyDatasetWrapper(train_data_array)
    
    # Load model
    print("\n[2/5] Loading Multivariate CNN model...")
    model_path = os.path.join(os.path.dirname(__file__), "../../models/cnn_multi_arabicdigits_10ch.pth")
    model = SimpleCNNMulti(input_channels=13, output_channels=10, sequence_length=65).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"  Model loaded: {model_path}")
    except Exception as e:
        print(f"  ✗ Model not found or error loading: {model_path}")
        print(f"  Error: {e}")
        print("  Please train the model first using example_multivariate.py")
        return
    
    # Find correctly classified samples
    print("\n[3/5] Finding correctly classified samples...")
    correct_indices = []
    for i, (x, y) in enumerate(zip(X_test, y_test)):
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_tensor).cpu().numpy()[0]
        if np.argmax(pred) == y:
            correct_indices.append(i)
    
    print(f"  Found {len(correct_indices)} correctly classified samples")
    
    if len(correct_indices) == 0:
        print("  No correctly classified samples found. Please check the model.")
        return
    
    # Evaluate on subset (smaller for multivariate due to computational cost)
    max_samples = min(30, len(correct_indices))
    test_indices = correct_indices[:max_samples]
    
    n_channels = X_test.shape[1]
    n_timesteps = X_test.shape[2]
    
    print(f"\n[4/5] Evaluating FFT-CF variants on {max_samples} samples...")
    print(f"  Note: FFT-CF treats flattened multivariate series as univariate ({n_channels}x{n_timesteps}={n_channels*n_timesteps} points)")
    print("=" * 80)
    
    # Define FFT-CF variants (fewer for multivariate due to computational cost)
    variants = [
        ('Nearest Neighbor', 
         lambda s, d, m, t: fft_nn_cf(s, d, m, t, k=3, verbose=False)),
        
        ('Iterative Refinement', 
         lambda s, d, m, t: fft_iterative_cf(s, d, m, t, k=3, refine_iterations=20, verbose=False)),
        
        ('Hybrid Enhanced', 
         lambda s, d, m, t: fft_hybrid_enhanced_cf(s, d, m, t, k=3, analyze_importance=True, fallback_on_failure=True, verbose=False)),
    ]
    
    results = {}
    
    for variant_name, variant_func in variants:
        print(f"\nEvaluating {variant_name}...")
        
        successes = 0
        confidences = []
        distances = []
        times = []
        errors = 0
        
        for idx in test_indices:
            sample = X_test[idx]  # Shape: (channels, timesteps)
            true_label = y_test[idx]
            
            # Choose a different target class
            target_class = (true_label + 1) % 10
            
            start_time = time.time()
            try:
                cf, pred = apply_fft_cf_multivariate(sample, train_data, model, target_class, variant_func, device)
                elapsed = time.time() - start_time
                
                if cf is not None and pred is not None:
                    pred_class = np.argmax(pred)
                    if pred_class == target_class:
                        successes += 1
                        confidences.append(pred[target_class])
                        distance = np.linalg.norm(cf.flatten() - sample.flatten())
                        distances.append(distance)
                        times.append(elapsed)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                if errors <= 5:  # Print first 5 errors for debugging
                    print(f"  Error on sample {idx}: {type(e).__name__}: {str(e)[:150]}")
        
        success_rate = successes / max_samples * 100
        
        results[variant_name] = {
            'success_rate': success_rate,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_distance': np.mean(distances) if distances else 0,
            'avg_time': np.mean(times) if times else 0,
            'confidences': confidences,
            'distances': distances,
            'times': times,
            'errors': errors
        }
        
        print(f"  Success: {successes}/{max_samples} ({success_rate:.1f}%) | Errors: {errors}", end="")
        if confidences:
            print(f" | Conf: {np.mean(confidences):.3f} | Dist: {np.mean(distances):.1f} | Time: {np.mean(times):.3f}s")
        else:
            print()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - MULTIVARIATE (SpokenArabicDigits)")
    print("=" * 80)
    print(f"{'Variant':<25} {'Success':<10} {'Confidence':<12} {'Distance':<12} {'Time (s)':<10}")
    print("-" * 80)
    
    for name, res in results.items():
        print(f"{name:<25} {res['success_rate']:<10.1f}% {res['avg_confidence']:<12.4f} "
              f"{res['avg_distance']:<12.2f} {res['avg_time']:<10.4f}")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    
    # Find best performers
    successful_results = {k: v for k, v in results.items() if v['success_rate'] > 0}
    
    if successful_results:
        best_success = max(successful_results.items(), key=lambda x: x[1]['success_rate'])
        best_confidence = max(successful_results.items(), key=lambda x: x[1]['avg_confidence'])
        best_distance = min(successful_results.items(), key=lambda x: x[1]['avg_distance'])
        best_time = min(successful_results.items(), key=lambda x: x[1]['avg_time'])
        
        print(f"  🏆 Best Success Rate: {best_success[0]} ({best_success[1]['success_rate']:.1f}%)")
        print(f"  🎯 Best Confidence: {best_confidence[0]} ({best_confidence[1]['avg_confidence']:.4f})")
        print(f"  📏 Best Distance (closest): {best_distance[0]} ({best_distance[1]['avg_distance']:.2f})")
        print(f"  ⚡ Fastest: {best_time[0]} ({best_time[1]['avg_time']:.4f}s)")
        
        print(f"\n  Note: Multivariate processing requires {X_train.shape[1]}x more computation")
        print(f"        (one FFT-CF per channel)")
    
    # Visualization
    print("\n[5/5] Generating visualizations...")
    
    plot_results = {k: v for k, v in results.items() if v['success_rate'] > 0}
    
    if plot_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('FFT-CF Multivariate Evaluation Results - SpokenArabicDigits Dataset', 
                     fontsize=14, fontweight='bold')
        
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
        axes[1, 0].set_title('Average Distance by Variant (13 channels)')
        for i, v in enumerate(distances):
            axes[1, 0].text(v + 1, i, f'{v:.1f}', va='center')
        
        # Execution time
        times = [plot_results[n]['avg_time'] for n in names]
        axes[1, 1].barh(names, times, color='purple')
        axes[1, 1].set_xlabel('Average Time (seconds)')
        axes[1, 1].set_title('Average Execution Time by Variant')
        for i, v in enumerate(times):
            axes[1, 1].text(v + 0.01, i, f'{v:.3f}s', va='center')
        
        plt.tight_layout()
        output_path = "fft_cf_multivariate_evaluation.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()
    
    print("\n" + "=" * 80)
    print("Multivariate Evaluation Complete!")
    print("=" * 80)
    print("\nNote: FFT methods process each channel independently,")
    print("      which may not capture inter-channel dependencies.")
    print("      Consider using methods specifically designed for multivariate data")
    print("      (e.g., COMTE, Multi-SpaCE) for better performance.")


if __name__ == "__main__":
    main()
