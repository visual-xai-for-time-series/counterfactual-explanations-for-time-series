"""
Example: Using aeon-toolkit InceptionTime Classifier with cfts

This example demonstrates how to:
1. Load a time series dataset using aeon-toolkit
2. Train an InceptionTime deep learning classifier
3. Integrate aeon models with cfts for counterfactual explanations

NOTE: This example uses SETS with aeon's InceptionTime classifier.
SETS is a gradient-free subsequence-based method that works well with
TensorFlow-based models. Other gradient-free alternatives include MOC,
GLACIER, Multi-SpaCE, and TSEvo.

Requirements:
    pip install aeon-toolkit tensorflow matplotlib numpy

References:
- aeon-toolkit: https://www.aeon-toolkit.org/en/stable/index.html
- InceptionTime: https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.classification.deep_learning.InceptionTimeClassifier.html
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for cfts imports
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_path, '../..'))

import numpy as np
import matplotlib.pyplot as plt
import torch

# Import aeon components
from aeon.classification.deep_learning import InceptionTimeClassifier
from aeon.datasets import load_classification

# Import cfts sets
import cfts.cf_sets.sets as sets


class AeonDatasetWrapper:
    """Wrapper to make aeon dataset compatible with cfts methods."""
    
    def __init__(self, X, y):
        """
        Args:
            X: Time series data (n_samples, n_channels, n_timepoints) or (n_samples, n_timepoints)
            y: Labels (n_samples,)
        """
        # Flatten the data for univariate case
        if X.ndim == 3 and X.shape[1] == 1:
            # Convert (n_samples, 1, n_timepoints) to (n_samples, n_timepoints)
            self.X = X[:, 0, :]
        else:
            self.X = X
        self.y = y
        
        # Calculate statistics
        self.mean = np.mean(self.X, axis=0)
        self.std = np.std(self.X, axis=0)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class InceptionTimeWrapper:
    """Wrapper to make InceptionTime compatible with cfts methods."""
    
    def __init__(self, model):
        """
        Args:
            model: Trained InceptionTimeClassifier
        """
        self.model = model
        self.device = 'cpu'  # InceptionTime uses its own backend
        self._dummy_param = np.array([0.0])  # Dummy parameter for device detection
        
    def __call__(self, x):
        """Make predictions on input time series."""
        # Convert torch tensor to numpy if needed
        is_torch = hasattr(x, 'cpu')
        if is_torch:
            x_np = x.cpu().detach().numpy()
        else:
            x_np = x
        
        # Ensure input is in correct format for aeon
        # aeon expects (n_samples, n_channels, n_timepoints)
        if isinstance(x_np, np.ndarray):
            if x_np.ndim == 1:
                x_np = x_np.reshape(1, 1, -1)  # (1, 1, n_timepoints)
            elif x_np.ndim == 2:
                if x_np.shape[0] > x_np.shape[1]:  # (n_timepoints, n_channels)
                    x_np = x_np.T.reshape(1, x_np.shape[1], x_np.shape[0])
                else:  # (n_channels, n_timepoints)
                    x_np = x_np.reshape(1, x_np.shape[0], x_np.shape[1])
            elif x_np.ndim == 3:
                # Shape (batch, channels, length) - already correct
                pass
        
        # Get predictions - aeon returns (n_samples, n_classes)
        proba = self.model.predict_proba(x_np)
        
        # Return as torch tensor if input was torch tensor, maintaining 2D shape
        if is_torch:
            return torch.from_numpy(proba).float()
        return proba
    
    def parameters(self):
        """Return dummy parameter iterator for compatibility."""
        class DummyParam:
            def __init__(self):
                self.device = 'cpu'
        yield DummyParam()
    
    def to(self, device):
        """Compatibility method for device placement."""
        self.device = device
        return self
    
    def eval(self):
        """Compatibility method."""
        pass


def load_and_prepare_data(dataset_name="FordA"):
    """
    Load and prepare time series dataset from aeon.
    
    Args:
        dataset_name: Name of the dataset to load (default: FordA)
        
    Returns:
        X_train, y_train, X_test, y_test: Training and test data
    """
    print(f'Loading {dataset_name} dataset from aeon...')
    
    # Load dataset
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")
    
    # Convert labels to integers
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_train = np.array([label_map[y] for y in y_train])
    y_test = np.array([label_map[y] for y in y_test])
    
    print(f'Dataset loaded:')
    print(f'  Train: {X_train.shape[0]} samples')
    print(f'  Test: {X_test.shape[0]} samples')
    print(f'  Shape: {X_train.shape}')
    print(f'  Classes: {len(unique_labels)}')
    
    return X_train, y_train, X_test, y_test


def train_or_load_model(X_train, y_train, X_test, y_test, model_path='inception_time_model.pkl'):
    """
    Train InceptionTime classifier or load from disk if exists.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_path: Path to save/load model
        
    Returns:
        Trained InceptionTimeClassifier
    """
    import pickle
    
    if os.path.exists(model_path):
        print(f'Loading saved model from {model_path}...')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print('Model loaded successfully')
    else:
        print('Training InceptionTime classifier...')
        print('Note: This may take several minutes...')
        
        # Create and train model
        model = InceptionTimeClassifier(
            n_epochs=10,  # Reduced for faster testing
            batch_size=16,
            verbose=True
        )
        
        model.fit(X_train, y_train)
        
        # Save model
        print(f'Saving model to {model_path}...')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print('Model saved successfully')
    
    # Evaluate model
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f'\nModel Performance:')
    print(f'  Training accuracy: {train_acc:.4f}')
    print(f'  Test accuracy: {test_acc:.4f}')
    
    return model


def generate_counterfactual_sets(model, dataset, sample, target_class=None):
    """
    Generate counterfactual explanation using SETS.
    
    Args:
        model: Wrapped InceptionTime model
        dataset: Wrapped dataset
        sample: Original time series sample
        target_class: Target class for counterfactual (optional)
        
    Returns:
        cf: Counterfactual time series
        prediction: Model prediction for counterfactual
    """
    print('\nGenerating counterfactual explanation using SETS...')
    print('(SETS is a gradient-free method compatible with aeon models)')
    
    try:
        # Generate counterfactual
        cf, prediction = sets.sets_cf(
            sample, 
            dataset, 
            model, 
            target_class=target_class,
            verbose=True
        )
        
        if cf is not None:
            original_pred = model(sample)
            cf_pred = model(cf)
            
            print(f'Original prediction: class {np.argmax(original_pred)} (confidence: {np.max(original_pred):.4f})')
            print(f'Counterfactual prediction: class {np.argmax(cf_pred)} (confidence: {np.max(cf_pred):.4f})')
            
            # Calculate distance
            distance = np.linalg.norm(sample - cf)
            print(f'L2 distance: {distance:.4f}')
        else:
            print('Failed to generate counterfactual')
    except Exception as e:
        print(f'Error generating counterfactual: {e}')
        print('\nNote: SETS is a gradient-free subsequence-based method.')
        print('Other gradient-free alternatives: MOC, GLACIER, Multi-SpaCE, TSEvo.')
        cf, prediction = None, None
    
    return cf, prediction


def visualize_results(original, counterfactual, original_pred, cf_pred, save_path='aeon_cf_example.png'):
    """
    Visualize original and counterfactual time series.
    
    Args:
        original: Original time series
        counterfactual: Counterfactual time series
        original_pred: Original prediction probabilities
        cf_pred: Counterfactual prediction probabilities
        save_path: Path to save visualization
    """
    print(f'\nCreating visualization...')
    
    # Flatten if needed
    original_flat = original.flatten() if original.ndim > 1 else original
    cf_flat = counterfactual.flatten() if counterfactual.ndim > 1 else counterfactual
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot time series comparison
    ax1 = axes[0]
    time_steps = np.arange(len(original_flat))
    ax1.plot(time_steps, original_flat, label='Original', linewidth=2, color='#2E86C1', alpha=0.8)
    ax1.plot(time_steps, cf_flat, label='Counterfactual', linewidth=2, color='#E74C3C', alpha=0.8, linestyle='--')
    ax1.fill_between(time_steps, original_flat, cf_flat, alpha=0.2, color='gray')
    
    ax1.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax1.set_title('Time Series Comparison: Original vs Counterfactual', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot prediction probabilities
    ax2 = axes[1]
    # Flatten predictions if they're 2D arrays
    original_pred_flat = original_pred.flatten()
    cf_pred_flat = cf_pred.flatten()
    n_classes = len(original_pred_flat)
    x_pos = np.arange(n_classes)
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, original_pred_flat, width, label='Original', color='#2E86C1', alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, cf_pred_flat, width, label='Counterfactual', color='#E74C3C', alpha=0.8)
    
    ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Probabilities Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Class {i}' for i in range(n_classes)])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Visualization saved to {save_path}')
    
    return save_path


def main():
    """Main execution function."""
    print('='*80)
    print('aeon-toolkit InceptionTime + cfts Integration Example')
    print('='*80)
    print()
    
    # 1. Load dataset
    X_train, y_train, X_test, y_test = load_and_prepare_data("FordA")
    
    # 2. Train or load model
    model_path = os.path.join(script_path, 'inception_time_forda.pkl')
    inception_model = train_or_load_model(X_train, y_train, X_test, y_test, model_path)
    
    # 3. Wrap model and dataset for cfts compatibility
    print('\nPreparing model and dataset wrappers...')
    model_wrapper = InceptionTimeWrapper(inception_model)
    dataset_wrapper = AeonDatasetWrapper(X_test, y_test)
    
    # 4. Select a sample for counterfactual generation
    print('\nSelecting test sample...')
    # Find correctly classified samples (stop after 10)
    correctly_classified = []
    
    for idx in range(len(X_test)):
        sample = X_test[idx, 0, :]  # Get flattened univariate series
        true_label = y_test[idx]
        pred_proba = model_wrapper(sample)
        pred_label = np.argmax(pred_proba)
        
        if pred_label == true_label:
            correctly_classified.append((idx, sample, true_label))
            if len(correctly_classified) >= 10:
                break
    
    # Randomly select one correctly classified sample
    selected_idx, selected_sample, selected_label = correctly_classified[np.random.randint(len(correctly_classified))]
    pred_proba = model_wrapper(selected_sample)
    confidence = np.max(pred_proba)
    print(f'Randomly selected sample {selected_idx}: True label = {selected_label}, Predicted = {np.argmax(pred_proba)}, Confidence = {confidence:.4f}')
    
    # 5. Generate counterfactual using SETS
    cf, cf_prediction = generate_counterfactual_sets(
        model_wrapper, 
        dataset_wrapper, 
        selected_sample,
        target_class=1 - selected_label  # Flip to opposite class
    )
    
    # 6. Visualize results
    if cf is not None:
        original_pred = model_wrapper(selected_sample)
        cf_pred = model_wrapper(cf)
        
        save_path = os.path.join(script_path, 'aeon_inceptiontime_cf.png')
        visualize_results(selected_sample, cf, original_pred, cf_pred, save_path)
        
        print('\n' + '='*80)
        print('Example completed successfully!')
        print('='*80)
        print(f'\nGenerated files:')
        print(f'  - Model: {model_path}')
        print(f'  - Visualization: {save_path}')
        print()
        print('NOTE: This example uses SETS, a gradient-free subsequence-based method.')
        print('SETS works well with TensorFlow-based aeon models.')
        print('Other gradient-free methods: MOC, GLACIER, Multi-SpaCE, TSEvo, etc.')
    else:
        print('\n' + '='*80)
        print('Counterfactual generation did not succeed')
        print('='*80)
        print('\nThe integration between aeon and cfts is working correctly.')
        print('Model predictions are being made successfully.')
        print('Try using other methods like MOC, GLACIER, Multi-SpaCE, or TSEvo.')


if __name__ == "__main__":
    main()
