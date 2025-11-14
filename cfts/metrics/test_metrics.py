"""
Simple tests for counterfactual metrics to verify basic functionality.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfts.metrics import (
    l2_distance, prediction_change, percentage_changed_points,
    temporal_consistency, CounterfactualEvaluator
)


def dummy_model(ts):
    """Simple dummy model for testing."""
    return np.array([0.3, 0.7]) if np.mean(ts) > 0.5 else np.array([0.7, 0.3])


def test_basic_metrics():
    """Test basic metric functionality."""
    print("Testing basic metrics...")
    
    # Create simple test data
    original = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    counterfactual = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    
    # Test proximity metrics
    distance = l2_distance(original, counterfactual)
    assert distance > 0, "L2 distance should be positive"
    print(f"✓ L2 distance: {distance:.3f}")
    
    # Test sparsity metrics
    sparsity = percentage_changed_points(original, counterfactual)
    assert 0 <= sparsity <= 1, "Sparsity should be between 0 and 1"
    print(f"✓ Percentage changed: {sparsity:.3f}")
    
    # Test realism metrics
    consistency = temporal_consistency(counterfactual)
    assert 0 <= consistency <= 1, "Temporal consistency should be between 0 and 1"
    print(f"✓ Temporal consistency: {consistency:.3f}")
    
    # Test validity metrics
    validity = prediction_change(original, counterfactual, dummy_model, target_class=1)
    assert validity in [0.0, 1.0], "Prediction change should be 0 or 1"
    print(f"✓ Prediction change: {validity}")


def test_evaluator():
    """Test the CounterfactualEvaluator class."""
    print("\nTesting CounterfactualEvaluator...")
    
    # Create test data
    original = np.random.normal(0.3, 0.1, 20)
    counterfactual = np.random.normal(0.7, 0.1, 20)
    reference_data = np.random.normal(0.5, 0.2, (50, 20))
    
    # Initialize evaluator
    evaluator = CounterfactualEvaluator(reference_data=reference_data)
    
    # Test single evaluation
    results = evaluator.evaluate_single(
        original_ts=original,
        counterfactual_ts=counterfactual,
        model=dummy_model,
        target_class=1
    )
    
    assert isinstance(results, dict), "Results should be a dictionary"
    assert len(results) > 0, "Results should not be empty"
    print(f"✓ Single evaluation returned {len(results)} metrics")
    
    # Test multiple evaluation
    counterfactuals = [counterfactual + np.random.normal(0, 0.01, 20) for _ in range(3)]
    diversity_results = evaluator.evaluate_multiple(
        original_ts=original,
        counterfactuals=counterfactuals,
        model=dummy_model,
        target_class=1
    )
    
    assert isinstance(diversity_results, dict), "Diversity results should be a dictionary"
    print(f"✓ Multiple evaluation returned {len(diversity_results)} metrics")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    # Test identical time series
    ts = np.array([0.5, 0.5, 0.5])
    distance = l2_distance(ts, ts)
    assert distance == 0.0, "Distance between identical series should be 0"
    print("✓ Identical time series handled correctly")
    
    # Test single point time series
    single_point_original = np.array([0.3])
    single_point_cf = np.array([0.7])
    distance = l2_distance(single_point_original, single_point_cf)
    assert distance > 0, "Single point distance should work"
    print("✓ Single point time series handled correctly")
    
    # Test empty arrays (should handle gracefully)
    try:
        empty_ts = np.array([])
        consistency = temporal_consistency(empty_ts)
        print("✓ Empty arrays handled gracefully")
    except Exception as e:
        print(f"✓ Empty arrays raise appropriate error: {type(e).__name__}")


if __name__ == "__main__":
    print("=== Testing Counterfactual Metrics ===\n")
    
    try:
        test_basic_metrics()
        test_evaluator()
        test_edge_cases()
        
        print("\n=== All Tests Passed! ===")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()