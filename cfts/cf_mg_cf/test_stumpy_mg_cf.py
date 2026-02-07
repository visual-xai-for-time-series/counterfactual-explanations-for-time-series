"""
Example demonstrating the STUMPY-based MG-CF implementation.

This compares the original brute-force implementation with the optimized
STUMPY-based version using matrix profiles for faster motif discovery.
"""

import numpy as np
import torch
import torch.nn as nn
import time

# Import both versions
from cfts.cf_mg_cf import (
    mine_motifs, 
    mg_cf_generate,
    mine_motifs_stumpy,
    mg_cf_generate_stumpy
)


# Simple CNN model for testing
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, seq_length=100):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate flattened size
        pooled_length = seq_length // 4
        self.fc1 = nn.Linear(32 * pooled_length, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def generate_synthetic_data(n_samples=100, seq_length=100, n_classes=2):
    """Generate synthetic time series data with distinctive motifs."""
    X = []
    y = []
    
    for i in range(n_samples):
        # Generate base time series
        ts = np.random.randn(seq_length) * 0.3
        
        # Add class-specific motifs
        label = i % n_classes
        
        if label == 0:
            # Class 0: Add sinusoidal motif in the middle
            start = seq_length // 3
            end = start + seq_length // 4
            t = np.linspace(0, 4*np.pi, end - start)
            ts[start:end] = np.sin(t) * 2
        else:
            # Class 1: Add square wave motif
            start = seq_length // 3
            end = start + seq_length // 4
            square_wave = np.where(np.sin(np.linspace(0, 4*np.pi, end - start)) > 0, 1.5, -1.5)
            ts[start:end] = square_wave
        
        X.append(ts)
        y.append(label)
    
    return X, y


def main():
    print("=" * 60)
    print("MG-CF: Comparing Original vs STUMPY-based Implementation")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic dataset
    print("\n1. Generating synthetic dataset...")
    seq_length = 100
    n_samples = 50
    X, y = generate_synthetic_data(n_samples=n_samples, seq_length=seq_length)
    
    # Create dataset in expected format
    dataset = [(torch.tensor(x, dtype=torch.float32).reshape(1, -1), y[i]) 
               for i, x in enumerate(X)]
    
    print(f"   Created {len(dataset)} samples of length {seq_length}")
    
    # Create and train a simple model
    print("\n2. Creating simple CNN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(input_channels=1, num_classes=2, seq_length=seq_length).to(device)
    model.eval()  # Use in eval mode for this demo
    
    print(f"   Model created on device: {device}")
    
    # Select a test sample
    test_sample = dataset[0][0].numpy()
    print(f"\n3. Test sample shape: {test_sample.shape}")
    
    # Compare motif mining times
    print("\n" + "=" * 60)
    print("COMPARISON: Motif Mining")
    print("=" * 60)
    
    # Original implementation
    print("\n[Original Implementation]")
    start_time = time.time()
    motifs_original = mine_motifs(dataset, lengths_ratio=[0.3, 0.5], verbose=True)
    original_time = time.time() - start_time
    print(f"Time taken: {original_time:.3f} seconds")
    
    # STUMPY implementation
    print("\n[STUMPY-based Implementation]")
    start_time = time.time()
    motifs_stumpy = mine_motifs_stumpy(dataset, lengths_ratio=[0.3, 0.5], top_k=5, verbose=True)
    stumpy_time = time.time() - start_time
    print(f"Time taken: {stumpy_time:.3f} seconds")
    
    # Performance comparison
    print("\n" + "-" * 60)
    print(f"Speedup: {original_time / stumpy_time:.2f}x faster with STUMPY")
    print("-" * 60)
    
    # Compare counterfactual generation
    print("\n" + "=" * 60)
    print("COMPARISON: Counterfactual Generation")
    print("=" * 60)
    
    # Original implementation
    print("\n[Original Implementation]")
    start_time = time.time()
    cf_original, pred_original = mg_cf_generate(
        test_sample, dataset, model, motifs=motifs_original, verbose=True
    )
    gen_time_original = time.time() - start_time
    print(f"Time taken: {gen_time_original:.3f} seconds")
    
    # STUMPY implementation
    print("\n[STUMPY-based Implementation]")
    start_time = time.time()
    cf_stumpy, pred_stumpy = mg_cf_generate_stumpy(
        test_sample, dataset, model, motifs=motifs_stumpy, verbose=True
    )
    gen_time_stumpy = time.time() - start_time
    print(f"Time taken: {gen_time_stumpy:.3f} seconds")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original motif mining:      {original_time:.3f}s")
    print(f"STUMPY motif mining:        {stumpy_time:.3f}s")
    print(f"Speedup:                    {original_time / stumpy_time:.2f}x\n")
    
    print(f"Original CF generation:     {gen_time_original:.3f}s")
    print(f"STUMPY CF generation:       {gen_time_stumpy:.3f}s")
    
    if cf_original is not None and cf_stumpy is not None:
        print(f"\nBoth methods successfully generated counterfactuals!")
        print(f"Original prediction: {pred_original}")
        print(f"STUMPY prediction:   {pred_stumpy}")
    
    print("\n" + "=" * 60)
    print("Key Advantages of STUMPY-based Implementation:")
    print("=" * 60)
    print("✓ Faster motif discovery using matrix profiles")
    print("✓ More efficient subsequence search (O(n²) → O(n log n))")
    print("✓ GPU acceleration support (if available)")
    print("✓ Better handling of multivariate time series")
    print("✓ Z-normalized distance calculations for robustness")
    print("=" * 60)


if __name__ == "__main__":
    main()
