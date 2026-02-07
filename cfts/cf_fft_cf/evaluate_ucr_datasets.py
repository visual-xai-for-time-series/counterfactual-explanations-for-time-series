"""
FFT-CF Evaluation on 40 UCR Datasets
Evaluates FFT-based counterfactual methods on the same 40 UCR datasets used in GLACIER paper.

Reference: Wang et al. (2024) - GLACIER: Guided Locally Constrained Counterfactual 
           Explanations for Time Series Classification
           https://github.com/zhendong3wang/learning-time-series-counterfactuals
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn as nn
import time
import pandas as pd
from collections import defaultdict
from aeon.datasets import load_classification
from examples.base.model import SimpleCNN

from cfts.cf_fft_cf import (
    fft_nn_cf,
    fft_iterative_cf,
    fft_freq_distance_cf,
    fft_hybrid_cf,
    fft_progressive_cf,
    fft_hybrid_enhanced_cf,
)


# 40 UCR datasets from GLACIER paper
UCR_DATASETS = [
    # Datasets with size larger than 1000
    'TwoLeadECG',
    'ItalyPowerDemand',
    'MoteStrain',
    'Wafer',
    'PhalangesOutlinesCorrect',
    'FordA',
    'FordB',
    'HandOutlines',
    'FreezerRegularTrain',
    'FreezerSmallTrain',
    'Yoga',
    
    # Datasets with size between 500 and 1000
    'Strawberry',
    'SonyAIBORobotSurface2',
    'SemgHandGenderCh2',
    'MiddlePhalanxOutlineCorrect',
    'ProximalPhalanxOutlineCorrect',
    'ECGFiveDays',
    'DistalPhalanxOutlineCorrect',
    'SonyAIBORobotSurface1',
    'Computers',
    
    # Datasets with size lower than 500
    'Earthquakes',
    'GunPointAgeSpan',
    'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung',
    'Chinatown',
    'PowerCons',
    'ToeSegmentation1',
    'WormsTwoClass',
    'Ham',
    'ECG200',
    'GunPoint',
    'ShapeletSim',
    'ToeSegmentation2',
    'HouseTwenty',
    'Herring',
    'Lightning2',
    'Wine',
    'Coffee',
    'BeetleFly',
    'BirdChicken',
]


class NumpyDatasetWrapper:
    """Wrapper to make numpy array compatible with FFT-CF methods."""
    def __init__(self, data):
        self.data = data
        self.features = data[:, :-1]
        self.labels = data[:, -1].astype(int)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self.features[idx], self.labels[idx]
        else:
            return self.data[idx]
    
    def __array__(self):
        return self.data


def train_simple_model(X_train, y_train, X_test, y_test, n_classes, max_epochs=50, patience=5):
    """Train a simple CNN model on the dataset."""
    device = torch.device('cpu')
    model = SimpleCNN(output_channels=n_classes).to(device)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
    y_test_tensor = torch.LongTensor(y_test)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    patience_counter = 0
    
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            acc = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    model.eval()
    return model, best_acc


def evaluate_dataset(dataset_name, max_samples=50, max_train_time=300):
    """Evaluate FFT-CF variants on a single dataset."""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print('='*80)
    
    device = torch.device('cpu')
    
    try:
        # Load dataset
        print(f"[1/4] Loading {dataset_name}...")
        X_train, y_train = load_classification(dataset_name, split="train")
        X_test, y_test = load_classification(dataset_name, split="test")
        
        # Handle different label formats
        if isinstance(y_train[0], str):
            unique_labels = sorted(set(y_train))
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            y_train = np.array([label_map[label] for label in y_train])
            y_test = np.array([label_map[label] for label in y_test])
        else:
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
        
        # Ensure univariate
        if len(X_train.shape) == 3:
            if X_train.shape[1] > 1:
                print(f"  ⚠ Multivariate dataset ({X_train.shape[1]} channels), skipping...")
                return None
            X_train = X_train.squeeze()
            X_test = X_test.squeeze()
        
        # Ensure labels are consecutive integers starting from 0
        unique_labels = np.unique(y_train)
        if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
            # Remap to 0, 1, 2, ...
            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            y_train = np.array([label_map[label] for label in y_train])
            y_test = np.array([label_map[label] for label in y_test])
        
        n_classes = len(unique_labels)
        
        print(f"  Shape: {X_train.shape}")
        print(f"  Classes: {n_classes}")
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Create dataset wrapper
        train_data_array = np.column_stack([X_train, y_train])
        train_data = NumpyDatasetWrapper(train_data_array)
        
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        return None
    
    # Train model
    print(f"\n[2/4] Training CNN model...")
    train_start = time.time()
    try:
        model, train_acc = train_simple_model(X_train, y_train, X_test, y_test, n_classes)
        train_time = time.time() - train_start
        
        if train_time > max_train_time:
            print(f"  ⚠ Training took {train_time:.1f}s (>{max_train_time}s), skipping...")
            return None
        
        print(f"  Accuracy: {train_acc:.2%}")
        print(f"  Training time: {train_time:.1f}s")
    except Exception as e:
        print(f"  ✗ Error training model: {e}")
        return None
    
    # Find correctly classified samples
    print(f"\n[3/4] Finding correctly classified samples...")
    correct_indices = []
    for i, (x, y) in enumerate(zip(X_test, y_test)):
        x_tensor = torch.FloatTensor(x).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_tensor).cpu().numpy()[0]
        if np.argmax(pred) == y:
            correct_indices.append(i)
    
    print(f"  Found {len(correct_indices)} correctly classified samples")
    
    if len(correct_indices) < 10:
        print(f"  ⚠ Too few correct samples (<10), skipping...")
        return None
    
    # Evaluate FFT-CF variants
    eval_samples = min(max_samples, len(correct_indices))
    test_indices = np.random.choice(correct_indices, eval_samples, replace=False)
    
    print(f"\n[4/4] Evaluating FFT-CF variants on {eval_samples} samples...")
    
    variants = [
        ('Nearest Neighbor', 
         lambda s, d, m, t: fft_nn_cf(s, d, m, t, k=5, verbose=False)),
        
        ('Iterative Refinement', 
         lambda s, d, m, t: fft_iterative_cf(s, d, m, t, k=5, refine_iterations=30, verbose=False)),
        
        ('Frequency Distance', 
         lambda s, d, m, t: fft_freq_distance_cf(s, d, m, t, k=5, freq_weight_strategy='energy', verbose=False)),
        
        ('Hybrid Amp-Phase', 
         lambda s, d, m, t: fft_hybrid_cf(s, d, m, t, k=5, analyze_importance=False, verbose=False)),
        
        ('Progressive Switching', 
         lambda s, d, m, t: fft_progressive_cf(s, d, m, t, k=5, steps_per_neighbor=5, verbose=False)),
        
        ('Hybrid Enhanced', 
         lambda s, d, m, t: fft_hybrid_enhanced_cf(s, d, m, t, k=5, analyze_importance=True, fallback_on_failure=True, verbose=False)),
    ]
    
    results = {}
    
    for variant_name, variant_func in variants:
        successes = 0
        confidences = []
        distances = []
        times = []
        errors = 0
        
        for idx in test_indices:
            sample = X_test[idx]
            true_label = y_test[idx]
            
            # For multi-class, choose a random different target class
            available_classes = [c for c in range(n_classes) if c != true_label]
            target_class = np.random.choice(available_classes)
            
            start_time = time.time()
            try:
                cf, pred = variant_func(sample, train_data, model, target_class)
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
        
        success_rate = successes / eval_samples * 100
        
        results[variant_name] = {
            'success_rate': success_rate,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_distance': np.mean(distances) if distances else 0,
            'avg_time': np.mean(times) if times else 0,
            'errors': errors
        }
        
        print(f"  {variant_name:<25} Success: {success_rate:>5.1f}%", end="")
        if confidences:
            print(f" | Conf: {np.mean(confidences):.3f} | Time: {np.mean(times):.3f}s")
        else:
            print(f" | Errors: {errors}")
    
    # Prepare summary
    dataset_results = {
        'dataset': dataset_name,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_classes': n_classes,
        'timesteps': X_train.shape[-1],
        'model_accuracy': train_acc,
        'train_time': train_time,
        'eval_samples': eval_samples,
    }
    
    for variant_name, metrics in results.items():
        dataset_results[f'{variant_name}_success'] = metrics['success_rate']
        dataset_results[f'{variant_name}_conf'] = metrics['avg_confidence']
        dataset_results[f'{variant_name}_dist'] = metrics['avg_distance']
        dataset_results[f'{variant_name}_time'] = metrics['avg_time']
    
    return dataset_results


def main():
    print("="*80)
    print("FFT-CF EVALUATION ON 40 UCR DATASETS")
    print("="*80)
    print(f"\nEvaluating {len(UCR_DATASETS)} datasets used in GLACIER paper")
    print("Reference: Wang et al. (2024)")
    print("https://github.com/zhendong3wang/learning-time-series-counterfactuals\n")
    
    all_results = []
    successful_datasets = 0
    skipped_datasets = []
    
    for i, dataset_name in enumerate(UCR_DATASETS, 1):
        print(f"\n[{i}/{len(UCR_DATASETS)}] Processing {dataset_name}...")
        
        try:
            result = evaluate_dataset(dataset_name, max_samples=50, max_train_time=300)
            
            if result is not None:
                all_results.append(result)
                successful_datasets += 1
                print(f"  ✓ Completed")
            else:
                skipped_datasets.append(dataset_name)
                print(f"  ⚠ Skipped")
                
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
            break
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            skipped_datasets.append(dataset_name)
    
    # Save results
    if all_results:
        print(f"\n{'='*80}")
        print("SAVING RESULTS")
        print('='*80)
        
        df = pd.DataFrame(all_results)
        
        # Save full results
        output_file = "fft_cf_ucr_40_datasets_results.csv"
        df.to_csv(output_file, index=False)
        print(f"✓ Saved detailed results: {output_file}")
        
        # Create summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print('='*80)
        print(f"Datasets evaluated: {successful_datasets}/{len(UCR_DATASETS)}")
        print(f"Datasets skipped: {len(skipped_datasets)}")
        
        if skipped_datasets:
            print(f"\nSkipped datasets:")
            for ds in skipped_datasets:
                print(f"  - {ds}")
        
        # Calculate average metrics across all datasets
        print(f"\n{'='*80}")
        print("AVERAGE PERFORMANCE ACROSS DATASETS")
        print('='*80)
        
        variants = ['Nearest Neighbor', 'Iterative Refinement', 'Frequency Distance', 
                   'Hybrid Amp-Phase', 'Progressive Switching', 'Hybrid Enhanced']
        
        summary_data = []
        for variant in variants:
            success_col = f'{variant}_success'
            conf_col = f'{variant}_conf'
            time_col = f'{variant}_time'
            
            if success_col in df.columns:
                avg_success = df[success_col].mean()
                avg_conf = df[conf_col].mean()
                avg_time = df[time_col].mean()
                
                summary_data.append({
                    'Variant': variant,
                    'Avg Success Rate (%)': avg_success,
                    'Avg Confidence': avg_conf,
                    'Avg Time (s)': avg_time,
                    'Best Success Rate (%)': df[success_col].max(),
                    'Worst Success Rate (%)': df[success_col].min()
                })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_file = "fft_cf_ucr_40_datasets_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✓ Saved summary: {summary_file}")
        
        # Identify best variant
        best_variant = summary_df.loc[summary_df['Avg Success Rate (%)'].idxmax(), 'Variant']
        best_success = summary_df['Avg Success Rate (%)'].max()
        print(f"\n🏆 Best Performing Variant: {best_variant} ({best_success:.1f}% avg success)")
        
    else:
        print("\n⚠ No results to save")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print('='*80)


if __name__ == "__main__":
    main()
