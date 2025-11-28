"""
Arabic Digits Counterfactual Explanations Example

This example demonstrates counterfactual explanation generation for the SpokenArabicDigits dataset
using various methods (Native Guide, COMTE, SETS, MOC, Wachter, GLACIER, Multi-SpaCE) with enhanced visualization.

Features:
- Multi-channel time series support (13 channels, 65 timesteps)
- Professional visualization with separate subplots for each channel
- High-quality PNG output (300 DPI) suitable for publications
- Color-coded methods with clear legends and styling
"""

import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}/../')


import base.model as bm
import base.data as bd


import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch
import numpy as np
import time

# Set up enhanced plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

import cfts.cf_native_guide.native_guide as ng
import cfts.cf_wachter.wachter as w
import cfts.cf_comte.comte as comte
import cfts.cf_sets.sets as sets
import cfts.cf_dandl.dandl as dandl
import cfts.cf_glacier.glacier as glacier
import cfts.cf_multispace.multispace as ms
import cfts.cf_tsevo.tsevo as tsevo
import cfts.cf_lasts.lasts as lasts
import cfts.cf_tscf.tscf as tscf
import cfts.cf_leftist.leftist as leftist
import cfts.cf_glacier.glacier as glacier
import cfts.cf_multispace.multispace as ms



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading SpokenArabicDigits dataset')
dataloader_train, dataset_train = bd.get_UCR_UEA_dataloader('SpokenArabicDigits', split='train')
dataloader_test, dataset_test = bd.get_UCR_UEA_dataloader('SpokenArabicDigits', split='test')

output_classes = dataset_train.y_shape[1]
print(f'Dataset info: {len(dataset_train)} train samples, {len(dataset_test)} test samples, {output_classes} classes')

# Check if dataset is multivariate
sample_shape = dataset_train[0][0].shape
print(f'Sample shape: {sample_shape}')
is_multivariate = len(sample_shape) > 1 and sample_shape[0] > 1

if is_multivariate:
    print(f'Multivariate time series with {sample_shape[0]} channels and {sample_shape[1]} time steps')
    # Use SimpleCNNMulti for multivariate time series
    model = bm.SimpleCNNMulti(
        input_channels=sample_shape[0], 
        output_channels=output_classes,
        sequence_length=sample_shape[1]
    ).to(device)
else:
    print(f'Univariate time series with {sample_shape[-1]} time steps')
    # Use regular SimpleCNN for univariate time series
    model = bm.SimpleCNN(output_channels=output_classes).to(device)

# --- model persistence: load if exists, otherwise train and save ---
models_dir = os.path.abspath(os.path.join(script_path, '..', 'models'))
os.makedirs(models_dir, exist_ok=True)

# Create model filename based on model type
model_type = "multi" if is_multivariate else "uni"
model_file = os.path.join(models_dir, f'cnn_{model_type}_arabicdigits_{output_classes}ch.pth')

model_loaded = False
if os.path.exists(model_file):
    print(f'Loading saved model from {model_file}')
    state = torch.load(model_file, map_location=device)
    model.load_state_dict(state)
    model_loaded = True
else:
    print(f'No saved model at {model_file}; training will run and the model will be saved.')

print('Preparing training components')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()


def trainer(model_, dataloader, criterion_):
    running_loss = 0

    model_.train()

    for _, (inputs, labels) in enumerate(dataloader):
        # Handle multivariate time series input
        if is_multivariate:
            # inputs shape: (batch, channels, length)
            inputs = inputs.float().to(device)
        else:
            # inputs shape: (batch, length) -> (batch, 1, length)
            inputs = inputs.reshape(inputs.shape[0], 1, -1)
            inputs = inputs.float().to(device)
        
        labels = labels.float().to(device)

        optimizer.zero_grad()
        preds = model_(inputs)
        loss_val = criterion_(preds, labels.argmax(dim=-1))
        loss_val.backward()
        optimizer.step()

        running_loss += loss_val.item()

    train_loss = running_loss / len(dataloader)

    return train_loss


def validator(model_, dataloader, criterion_):
    running_loss = 0

    model_.eval()

    for _, (inputs, labels) in enumerate(dataloader):
        # Handle multivariate time series input
        if is_multivariate:
            # inputs shape: (batch, channels, length)
            inputs = inputs.float().to(device)
        else:
            # inputs shape: (batch, length) -> (batch, 1, length)
            inputs = inputs.reshape(inputs.shape[0], 1, -1)
            inputs = inputs.float().to(device)
        
        labels = labels.float().to(device)

        preds = model_(inputs)
        loss_val = criterion_(preds, labels.argmax(dim=-1))

        running_loss += loss_val.item()

    val_loss = running_loss / len(dataloader)

    return val_loss


# only train if we didn't load a saved model
if not model_loaded:
    print('Training model')
    epochs = 100

    for epoch in range(epochs):
        train_loss = trainer(model, dataloader_train, criterion)
        if epoch % 10 == 0:
            val_loss = validator(model, dataloader_test, criterion)
            print(f'Epoch {epoch:3d}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')
        else:
            print(f'Epoch {epoch:3d}: Train loss: {train_loss:.4f}')

    # save trained model state_dict
    torch.save(model.state_dict(), model_file)
    print(f'Model saved to {model_file}')
else:
    print('Using loaded model; skipping training.')

print('Generating counterfactual explanations for SpokenArabicDigits')
# Select a random sample from the test dataset that is correctly classified
model.eval()
sample, label = None, None
original_pred_np, original_class = None, None
attempts = 0
max_attempts = 100

while attempts < max_attempts:
    random_idx = np.random.randint(0, len(dataset_test))
    candidate_sample, candidate_label = dataset_test[random_idx]
    
    # Get prediction for the candidate sample
    with torch.no_grad():
        sample_tensor = torch.tensor(candidate_sample, dtype=torch.float32, device=device)
        if len(sample_tensor.shape) == 1:
            sample_tensor = sample_tensor.reshape(1, 1, -1)
        elif len(sample_tensor.shape) == 2:
            if is_multivariate:
                # For multivariate: (channels, length) -> (1, channels, length)
                sample_tensor = sample_tensor.unsqueeze(0)
            else:
                # For univariate: assume (length, 1) or (1, length) -> (1, 1, length)
                if sample_tensor.shape[0] > sample_tensor.shape[1]:
                    sample_tensor = sample_tensor.T
                sample_tensor = sample_tensor.unsqueeze(0)
        
        pred_output = model(sample_tensor)
        pred_np = torch.softmax(pred_output, dim=-1).squeeze().cpu().numpy()
        pred_class = torch.argmax(pred_output, dim=-1).item()
        
        # Check if prediction matches true label
        true_class = np.argmax(candidate_label)
        if pred_class == true_class:
            sample, label = candidate_sample, candidate_label
            original_pred_np, original_class = pred_np, pred_class
            print(f'Found correctly classified sample {random_idx} after {attempts + 1} attempts')
            break
    
    attempts += 1

if sample is None:
    print(f'Could not find a correctly classified sample after {max_attempts} attempts')
    print('Exiting without generating counterfactuals')
    exit(1)

print(f'Sample shape: {sample.shape}, True label: {label}')
print(f'Original prediction: class {original_class} with confidence {original_pred_np[original_class]:.4f}')

# Select a target class that is different from the predicted class
# Choose the class with the second highest probability
target_class = None
sorted_probs = np.argsort(original_pred_np)[::-1]  # Sort in descending order
for candidate in sorted_probs:
    if candidate != original_class:
        target_class = candidate
        break

if target_class is None:
    # Fallback: just pick any class different from original
    target_class = (original_class + 1) % output_classes

print(f'Target class for all counterfactuals: {target_class}')
print()

# Dictionary to store timing results
timing_results = {}

print('Start with native guide')
start_time = time.time()
# Native Guide doesn't support explicit target class, it finds the nearest different class
cf_ng, prediction_ng = ng.native_guide_uni_cf(sample, dataset_test, model)
timing_results['Native Guide'] = time.time() - start_time
print(f'Native Guide completed in {timing_results["Native Guide"]:.3f} seconds')

print('Start with COMTE')
start_time = time.time()
cf_comte, prediction_comte = comte.comte_cf(sample, dataset_test, model, target_class=target_class)
timing_results['COMTE'] = time.time() - start_time
print(f'COMTE completed in {timing_results["COMTE"]:.3f} seconds')

print('Start with COMTE-TS')
start_time = time.time()
cf_comte_ts, prediction_comte_ts = comte.comte_ts_cf(sample, dataset_test, model, target_class=target_class)
timing_results['COMTE-TS'] = time.time() - start_time
print(f'COMTE-TS completed in {timing_results["COMTE-TS"]:.3f} seconds')

print('Start with SETS')
start_time = time.time()
cf_sets, prediction_sets = sets.sets_cf(sample, dataset_test, model, target_class=target_class)
timing_results['SETS'] = time.time() - start_time
print(f'SETS completed in {timing_results["SETS"]:.3f} seconds')

print('Start with Dandl et al.')
start_time = time.time()
cf_moc, prediction_moc = dandl.moc_cf(sample, dataset_test, model, target_class=target_class)
timing_results['MOC (Dandl)'] = time.time() - start_time
print(f'MOC completed in {timing_results["MOC (Dandl)"]:.3f} seconds')

print('Start with Gradient Wachter et al.')
start_time = time.time()
cf_wg, prediction_wg = w.wachter_gradient_cf(sample, dataset_test, model, target=target_class)
timing_results['Wachter Gradient'] = time.time() - start_time
print(f'Wachter Gradient completed in {timing_results["Wachter Gradient"]:.3f} seconds')

print('Start with Genetic Wachter et al.')
# Adjust step size for multivariate data
if hasattr(dataset_test, 'std'):
    step_size = np.mean(dataset_test.std) + 0.1
else:
    step_size = 0.1
start_time = time.time()
cf_w, prediction_w = w.wachter_genetic_cf(sample, model, target=target_class, step_size=step_size, max_steps=1000)
timing_results['Wachter Genetic'] = time.time() - start_time
print(f'Wachter Genetic completed in {timing_results["Wachter Genetic"]:.3f} seconds')

print('Start with GLACIER')
start_time = time.time()
cf_glacier, prediction_glacier = glacier.glacier_cf(sample, dataset_test, model, target_class=target_class)
timing_results['GLACIER'] = time.time() - start_time
print(f'GLACIER completed in {timing_results["GLACIER"]:.3f} seconds')

print('Start with Multi-SpaCE')
start_time = time.time()
# Multi-SpaCE doesn't support explicit target class, it finds the nearest different class
cf_multispace, prediction_multispace = ms.multi_space_cf(sample, dataset_test, model,
                                                          population_size=30,
                                                          max_iterations=50,
                                                          sparsity_weight=0.3,
                                                          validity_weight=0.7)
timing_results['Multi-SpaCE'] = time.time() - start_time
print(f'Multi-SpaCE completed in {timing_results["Multi-SpaCE"]:.3f} seconds')

print('Start with TSEvo')
start_time = time.time()
cf_tsevo, prediction_tsevo = tsevo.tsevo_cf(sample, dataset_test, model, 
                                            target_class=target_class,
                                            population_size=30,
                                            generations=50)
timing_results['TSEvo'] = time.time() - start_time
print(f'TSEvo completed in {timing_results["TSEvo"]:.3f} seconds')

print('Start with LASTS')
start_time = time.time()
cf_lasts, prediction_lasts = lasts.lasts_cf(sample, dataset_test, model, 
                                            target_class=target_class,
                                            latent_dim=32,
                                            max_iterations=1000,
                                            train_ae_epochs=50,
                                            verbose=False)
timing_results['LASTS'] = time.time() - start_time
print(f'LASTS completed in {timing_results["LASTS"]:.3f} seconds')

print('Start with TSCF')
start_time = time.time()
cf_tscf, prediction_tscf = tscf.tscf_cf(sample, dataset_test, model, 
                                       target_class=target_class,
                                       lambda_l1=0.01,
                                       lambda_l2=0.01,
                                       lambda_smooth=0.001,
                                       learning_rate=0.1,
                                       max_iterations=2000,
                                       verbose=False)
timing_results['TSCF'] = time.time() - start_time
print(f'TSCF completed in {timing_results["TSCF"]:.3f} seconds')

print('Start with LEFTIST')
start_time = time.time()
cf_leftist, prediction_leftist = leftist.leftist_cf(sample, dataset_test, model, 
                                                    target_class=target_class,
                                                    segment_length=5,
                                                    max_iterations=50,
                                                    saliency_threshold=0.1,
                                                    verbose=False)
timing_results['LEFTIST'] = time.time() - start_time
print(f'LEFTIST completed in {timing_results["LEFTIST"]:.3f} seconds')

print()
print('='*80)
print('Combined Results Summary:')
print('='*80)
print(f'Target Class: {target_class}')
print('-'*80)
print(f'{"Method":<20} {"Status":<10} {"Pred Class":<12} {"Confidence":<12} {"Time (s)":>10}')
print('-'*80)

def format_combined_result(name, prediction, elapsed_time):
    if prediction is None:
        return f'{name:<20} {"Failed":<10} {"-":<12} {"-":<12} {elapsed_time:>10.3f}'
    pred_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    return f'{name:<20} {"Success":<10} {pred_class:<12} {confidence:<12.4f} {elapsed_time:>10.3f}'

print(f'{"Original":<20} {"-":<10} {original_class:<12} {original_pred_np[original_class]:<12.4f} {"-":>10}')
print(format_combined_result('Native Guide', prediction_ng, timing_results['Native Guide']))
print(format_combined_result('COMTE', prediction_comte, timing_results['COMTE']))
print(format_combined_result('COMTE-TS', prediction_comte_ts, timing_results['COMTE-TS']))
print(format_combined_result('SETS', prediction_sets, timing_results['SETS']))
print(format_combined_result('MOC (Dandl)', prediction_moc, timing_results['MOC (Dandl)']))
print(format_combined_result('Wachter Gradient', prediction_wg, timing_results['Wachter Gradient']))
print(format_combined_result('Wachter Genetic', prediction_w, timing_results['Wachter Genetic']))
print(format_combined_result('GLACIER', prediction_glacier, timing_results['GLACIER']))
print(format_combined_result('Multi-SpaCE', prediction_multispace, timing_results['Multi-SpaCE']))
print(format_combined_result('TSEvo', prediction_tsevo, timing_results['TSEvo']))
print(format_combined_result('LASTS', prediction_lasts, timing_results['LASTS']))
print(format_combined_result('TSCF', prediction_tscf, timing_results['TSCF']))
print(format_combined_result('LEFTIST', prediction_leftist, timing_results['LEFTIST']))
print('='*80)
print()

# Plotting for multivariate time series
def _to_channel_first(a):
    """Convert array to channel-first format (C, L) for plotting."""
    arr = np.asarray(a)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        r, c = arr.shape
        # For multivariate, assume (channels, length)
        return arr if r <= c else arr.T
    if arr.ndim == 3:
        # take first batch element if present (B, C, L) -> (C, L)
        return arr[0]
    raise ValueError("Unsupported array shape for plotting: %s" % (arr.shape,))

# Enhanced plotting for multivariate time series
def _to_channel_first(a):
    """Convert array to channel-first format (C, L) for plotting."""
    arr = np.asarray(a)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        r, c = arr.shape
        # For multivariate, assume (channels, length)
        return arr if r <= c else arr.T
    if arr.ndim == 3:
        # take first batch element if present (B, C, L) -> (C, L)
        return arr[0]
    raise ValueError("Unsupported array shape for plotting: %s" % (arr.shape,))

def _fmt_pred(pred):
    """Format a model prediction array into 'label (conf)' or 'None'."""
    if pred is None:
        return "None"
    arr = np.asarray(pred).reshape(-1)
    lab = int(np.argmax(arr))
    conf = float(np.max(arr))
    return f"{lab} ({conf:.3f})"

def create_enhanced_visualization(sample, label, original_pred_np, original_class, cf_results, is_multivariate):
    """Create enhanced visualization with professional styling."""
    
    # Convert sample to channel-first format
    sample_pl = _to_channel_first(sample)
    n_channels, n_timesteps = sample_pl.shape
    
    # Define colors for different methods
    method_colors = {
        'Original': '#2E86C1',  # Blue
        'Native Guide': '#E74C3C',  # Red
        'COMTE': '#F39C12',  # Orange
        'COMTE-TS': '#F8B739',  # Light Orange
        'SETS': '#27AE60',  # Green
        'MOC': '#8E44AD',  # Purple
        'Wachter Gradient': '#E67E22',  # Dark Orange
        'Wachter Genetic': '#34495E',  # Dark Gray
        'GLACIER': '#16A085',  # Teal
        'Multi-SpaCE': '#C0392B',  # Dark Red
        'TSEvo': '#D35400',  # Burnt Orange
        'LASTS': '#1ABC9C',  # Turquoise
        'TSCF': '#9B59B6',  # Amethyst
        'LEFTIST': '#E91E63'  # Pink
    }
    
    # Create figure with subplots for each channel
    if is_multivariate and n_channels > 1:
        fig, axes = plt.subplots(n_channels, 1, figsize=(16, 4 * n_channels))
        if n_channels == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(16, 8))
        axes = [axes]
        n_channels = 1
    
    # Format label
    true_label = str(np.argmax(label)) if hasattr(label, 'shape') else str(label)
    original_pred_str = _fmt_pred(original_pred_np)
    
    fig.suptitle(f'Enhanced Counterfactual Explanations - Arabic Digits\n'
                f'Sample Analysis: True Label = {true_label}, Original Prediction = {original_pred_str}', 
                fontsize=16, fontweight='bold')
    
    # Plot each channel
    for ch in range(n_channels):
        ax = axes[ch] if n_channels > 1 else axes[0]
        
        # Plot original time series
        time_steps = np.arange(n_timesteps)
        ax.plot(time_steps, sample_pl[ch], 
               color=method_colors['Original'], 
               linewidth=3, 
               label='Original', 
               alpha=0.9,
               marker='o' if n_timesteps <= 50 else None,
               markersize=3)
        
        # Plot counterfactuals
        for method_name, (cf, pred) in cf_results.items():
            if cf is not None:
                cf_pl = _to_channel_first(cf)
                if ch < cf_pl.shape[0]:  # Make sure channel exists
                    ax.plot(time_steps, cf_pl[ch], 
                           color=method_colors.get(method_name, '#95A5A6'), 
                           linewidth=2, 
                           label=f'{method_name} {_fmt_pred(pred)}',
                           alpha=0.8,
                           linestyle='--')
        
        # Styling
        ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        
        if is_multivariate and n_channels > 1:
            ax.set_title(f'Channel {ch + 1}', fontsize=14, fontweight='bold')
        else:
            ax.set_title('Time Series Analysis', fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Add subtle background
        ax.set_facecolor('#FAFAFA')
        
        # Improve tick formatting
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add border
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#34495E')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    output_filename = 'counterfactuals_arabic_digits.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\nEnhanced visualization saved as: {output_filename}")
    
    return output_filename

# Prepare counterfactual results for visualization
cf_results = {
    'Native Guide': (cf_ng, prediction_ng),
    'COMTE': (cf_comte, prediction_comte),
    'COMTE-TS': (cf_comte_ts, prediction_comte_ts),
    'SETS': (cf_sets, prediction_sets),
    'MOC': (cf_moc, prediction_moc),
    'Wachter Gradient': (cf_wg, prediction_wg),
    'Wachter Genetic': (cf_w, prediction_w),
    'GLACIER': (cf_glacier, prediction_glacier),
    'Multi-SpaCE': (cf_multispace, prediction_multispace),
    'TSEvo': (cf_tsevo, prediction_tsevo),
    'LASTS': (cf_lasts, prediction_lasts),
    'TSCF': (cf_tscf, prediction_tscf),
    'LEFTIST': (cf_leftist, prediction_leftist)
}

# Create enhanced visualization
create_enhanced_visualization(sample, label, original_pred_np, original_class, cf_results, is_multivariate)

# Save the plot but don't show it
# plt.show()  # Disabled to prevent plot display
print("Plot saved. Exiting without displaying.")
