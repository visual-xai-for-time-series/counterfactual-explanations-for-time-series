"""
Arabic Digits Counterfactual Explanations Example

This example demonstrates counterfactual explanation generation for the SpokenArabicDigits dataset
using two representative methods from each category (Optimization-Based: Wachter, COMTE;
Evolutionary: TSEvo, Sub-SpaCE; Instance-Based: Native Guide, CELS; Latent Space: GLACIER,
Latent-CF; Segment-Based: SETS, TS-CEM; Hybrid: TeRCE, MG-CF), plus Soft-DTW-CFE
(Plausibility-Based: gradient optimization in input space with a differentiable Soft-DTW
term pulling the counterfactual toward its nearest target-class training series), with
enhanced visualization.

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

# ---------------------------------------------------------------------------
# Logging – tee stdout/stderr to a file alongside terminal output
# ---------------------------------------------------------------------------
class _Tee:
    """Mirror writes to both the original stream and a log file."""
    def __init__(self, stream, logfile):
        self._stream = stream
        self._log = open(logfile, 'w', buffering=1)
    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
    def flush(self):
        self._stream.flush()
        self._log.flush()
    def __getattr__(self, name):
        return getattr(self._stream, name)

_log_dir = os.path.join(script_path, 'logs')
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, 'example_multivariate.log')
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)
print(f'Logging to: {_log_file}')
# ---------------------------------------------------------------------------


import base.model as bm
import base.data as bd


import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch
import numpy as np
import time
from tqdm import tqdm

# Set up enhanced plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

import cfts.cf_native_guide.native_guide as ng
import cfts.cf_wachter.wachter as w
import cfts.cf_comte.comte as comte
import cfts.cf_sets.sets as sets
import cfts.cf_glacier.glacier as glacier
import cfts.cf_subspace.subspace as subspace
import cfts.cf_tsevo.tsevo as tsevo
from cfts.cf_mg_cf import mg_cf_generate_stumpy
import cfts.cf_latent_cf.latent_cf as latent_cf
import cfts.cf_cels.cels as cels
import cfts.cf_terce.terce as terce
import cfts.cf_cem.cem as cem
from cfts.cf_soft_dtw_cfe.soft_dtw_cfe import soft_dtw_cfe_cf

# Fixed seed so the query instance selected below (np.random.randint /
# np.random.choice) is the same on every run, across all example_*.py
# scripts, rather than a different random instance each time.
np.random.seed(13)

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

# Count total methods to run
methods = [
    'Native Guide', 'COMTE', 'SETS', 'Wachter Genetic', 'GLACIER',
    'Sub-SpaCE', 'TSEvo', 'MG-CF', 'Latent-CF', 'M-CELS', 'TERCE', 'CEM',
    'Soft-DTW-CFE',
]

# Initialize progress bar
progress = tqdm(total=len(methods), desc='Generating Counterfactuals', unit='method')

print('Start with native guide')
start_time = time.time()
cf_ng, prediction_ng = ng.native_guide_uni_cf(sample, model, dataset=dataset_test)
timing_results['Native Guide'] = time.time() - start_time
print(f'Native Guide completed in {timing_results["Native Guide"]:.3f} seconds')
progress.update(1)

print('Start with COMTE')
start_time = time.time()
cf_comte, prediction_comte = comte.comte_cf_gradient(sample, model, target_class=target_class, dataset=dataset_test)
timing_results['COMTE'] = time.time() - start_time
print(f'COMTE completed in {timing_results["COMTE"]:.3f} seconds')
progress.update(1)

print('Start with SETS')
start_time = time.time()
cf_sets, prediction_sets = sets.sets_cf(sample, model, target_class=target_class, dataset=dataset_test)
timing_results['SETS'] = time.time() - start_time
print(f'SETS completed in {timing_results["SETS"]:.3f} seconds')
progress.update(1)

print('Start with Genetic Wachter et al.')
# Adjust step size for multivariate data
if hasattr(dataset_test, 'std'):
    step_size = np.mean(dataset_test.std) + 0.1
else:
    step_size = 0.1
start_time = time.time()
cf_w, prediction_w = w.wachter_genetic_cf(sample, model, target_class=target_class, step_size=step_size, max_steps=1000)
timing_results['Wachter Genetic'] = time.time() - start_time
print(f'Wachter Genetic completed in {timing_results["Wachter Genetic"]:.3f} seconds')
progress.update(1)

print('Start with GLACIER')
start_time = time.time()
cf_glacier, prediction_glacier = glacier.glacier_cf(sample, model, target_class=target_class, dataset=dataset_test)
timing_results['GLACIER'] = time.time() - start_time
print(f'GLACIER completed in {timing_results["GLACIER"]:.3f} seconds')
progress.update(1)

print('Start with Sub-SpaCE')
start_time = time.time()
cf_subspace, prediction_subspace = subspace.subspace_cf(
    sample, model,
    target_class=target_class,
    dataset=dataset_test,
    population_size=100,
    max_iter=200,
    alpha=0.8,
    beta=0.15,
    eta=0.05,
    invalid_penalization=20,
    init_pct=0.4,
    reinit=True,
    verbose=False
)
timing_results['Sub-SpaCE'] = time.time() - start_time
print(f'Sub-SpaCE completed in {timing_results["Sub-SpaCE"]:.3f} seconds')
progress.update(1)

print('Start with TSEvo')
start_time = time.time()
cf_tsevo, prediction_tsevo = tsevo.tsevo_cf(sample, model,
                                            target_class=target_class,
                                            dataset=dataset_test,
                                            population_size=30,
                                            generations=50)
timing_results['TSEvo'] = time.time() - start_time
print(f'TSEvo completed in {timing_results["TSEvo"]:.3f} seconds')
progress.update(1)

print('Start with MG-CF (STUMPY optimized)')
start_time = time.time()
try:
    # MG-CF with STUMPY optimization for faster motif mining
    # Check if STUMPY is available
    if mg_cf_generate_stumpy is None:
        raise ImportError("STUMPY not available for mg_cf_generate_stumpy")
    # Using a smaller subset for faster execution
    subset_size = min(100, len(dataset_test))
    from torch.utils.data import Subset
    dataset_subset = Subset(dataset_test, range(subset_size))
    cf_mg_cf, prediction_mg_cf = mg_cf_generate_stumpy(sample, model, target_class=target_class, dataset=dataset_subset,
                                                        top_k=5,
                                                        verbose=False)
    timing_results['MG-CF'] = time.time() - start_time
    print(f'MG-CF completed in {timing_results["MG-CF"]:.3f} seconds')
except Exception as e:
    cf_mg_cf, prediction_mg_cf = None, None
    timing_results['MG-CF'] = time.time() - start_time
    print(f'MG-CF failed: {type(e).__name__}: {str(e)[:100]}')
progress.update(1)

print('Start with Latent-CF')
start_time = time.time()
try:
    cf_latent_cf, prediction_latent_cf = latent_cf.latent_cf_generate(sample, model,
                                                                      target_class=target_class,
                                                                      dataset=dataset_test,
                                                                      latent_dim=8,
                                                                      max_iter=100,
                                                                      verbose=False)
    timing_results['Latent-CF'] = time.time() - start_time
    print(f'Latent-CF completed in {timing_results["Latent-CF"]:.3f} seconds')
except Exception as e:
    cf_latent_cf, prediction_latent_cf = None, None
    timing_results['Latent-CF'] = time.time() - start_time
    print(f'Latent-CF failed: {type(e).__name__}: {str(e)[:100]}')
progress.update(1)

print('Start with M-CELS (multivariate CELS)')
start_time = time.time()
try:
    # M-CELS requires training data for nearest unlike neighbor
    # Use full dataset to ensure target class samples are available
    subset_size = len(dataset_test)
    X_train = np.array([dataset_test[i][0] for i in range(subset_size)])
    y_train_labels = np.array([np.argmax(dataset_test[i][1]) if hasattr(dataset_test[i][1], 'shape') and len(dataset_test[i][1].shape) > 0 else dataset_test[i][1] for i in range(subset_size)])
    
    # Check if target class exists in training data
    target_samples_count = np.sum(y_train_labels == target_class)
    if target_samples_count < 10:
        print(f'M-CELS: Target class {target_class} has {target_samples_count} samples in training subset')
    
    cf_cels, prediction_cels = cels.m_cels_generate(sample, model, X_train, y_train_labels,
                                                    target_class=target_class,
                                                    max_iter=100,
                                                    verbose=False)
    timing_results['M-CELS'] = time.time() - start_time
    if cf_cels is not None:
        print(f'M-CELS completed successfully in {timing_results["M-CELS"]:.3f} seconds')
    else:
        print(f'M-CELS failed (returned None) in {timing_results["M-CELS"]:.3f} seconds')
except Exception as e:
    cf_cels, prediction_cels = None, None
    timing_results['M-CELS'] = time.time() - start_time
    print(f'M-CELS failed: {type(e).__name__}: {str(e)[:100]}')
progress.update(1)

print('Start with TERCE')
start_time = time.time()
try:
    # TERCE requires training data for nearest unlike neighbor and rule mining
    # Use larger subset to ensure target class samples exist
    subset_size = min(500, len(dataset_test))
    X_train = np.array([dataset_test[i][0] for i in range(subset_size)])
    y_train = np.array([np.argmax(dataset_test[i][1]) if hasattr(dataset_test[i][1], 'shape') and len(dataset_test[i][1].shape) > 0 else dataset_test[i][1] for i in range(subset_size)])
    
    cf_terce, pred_class_terce = terce.terce_generate(sample, model, X_train, y_train,
                                                    target_class=target_class,
                                                    n_regions=5,
                                                    window_size_ratio=0.1,
                                                    verbose=False)
    # TERCE returns an integer class, convert to probability array for consistency
    if pred_class_terce is not None:
        prediction_terce = np.zeros(output_classes)
        prediction_terce[pred_class_terce] = 1.0
    else:
        prediction_terce = None
    timing_results['TERCE'] = time.time() - start_time
    if cf_terce is None:
        print(f'TERCE completed but found no valid counterfactual in {timing_results["TERCE"]:.3f} seconds')
    else:
        print(f'TERCE completed in {timing_results["TERCE"]:.3f} seconds')
except Exception as e:
    cf_terce, prediction_terce = None, None
    timing_results['TERCE'] = time.time() - start_time
    print(f'TERCE failed: {type(e).__name__}: {str(e)[:100]}')
progress.update(1)

print('Start with CEM (Contrastive Explanation Method)')
start_time = time.time()
try:
    cf_cem, prediction_cem = cem.cem_cf(
        sample, model,
        mode='PN',
        autoencoder=None,
        kappa=0.5,
        beta=0.1,
        gamma=0.2,
        c_init=10.0,
        c_steps=5,
        max_iterations=500,
        learning_rate=1e-2,
        verbose=False,
    )
    timing_results['CEM'] = time.time() - start_time
    print(f'CEM completed in {timing_results["CEM"]:.3f} seconds')
except Exception as e:
    cf_cem, prediction_cem = None, None
    timing_results['CEM'] = time.time() - start_time
    print(f'CEM failed: {type(e).__name__}: {str(e)[:100]}')
progress.update(1)

print('Start with Soft-DTW-CFE')
start_time = time.time()
try:
    # steps reduced from the default 500 (paper setting) to keep this demo's
    # runtime comparable to the other iterative methods above.
    cf_soft_dtw_cfe, prediction_soft_dtw_cfe = soft_dtw_cfe_cf(
        sample, model,
        target_class=target_class,
        dataset=dataset_test,
        steps=150,
        verbose=False,
    )
    timing_results['Soft-DTW-CFE'] = time.time() - start_time
    print(f'Soft-DTW-CFE completed in {timing_results["Soft-DTW-CFE"]:.3f} seconds')
except Exception as e:
    cf_soft_dtw_cfe, prediction_soft_dtw_cfe = None, None
    timing_results['Soft-DTW-CFE'] = time.time() - start_time
    print(f'Soft-DTW-CFE failed: {type(e).__name__}: {str(e)[:100]}')
progress.update(1)

# Close the progress bar
progress.close()

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
print(format_combined_result('SETS', prediction_sets, timing_results['SETS']))
print(format_combined_result('Wachter Genetic', prediction_w, timing_results['Wachter Genetic']))
print(format_combined_result('GLACIER', prediction_glacier, timing_results['GLACIER']))
print(format_combined_result('Sub-SpaCE', prediction_subspace, timing_results['Sub-SpaCE']))
print(format_combined_result('TSEvo', prediction_tsevo, timing_results['TSEvo']))
print(format_combined_result('MG-CF', prediction_mg_cf, timing_results['MG-CF']))
print(format_combined_result('Latent-CF', prediction_latent_cf, timing_results['Latent-CF']))
print(format_combined_result('M-CELS', prediction_cels, timing_results['M-CELS']))
print(format_combined_result('TERCE', prediction_terce, timing_results['TERCE']))
print(format_combined_result('CEM', prediction_cem, timing_results['CEM']))
print(format_combined_result('Soft-DTW-CFE', prediction_soft_dtw_cfe, timing_results['Soft-DTW-CFE']))
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
        'Sub-SpaCE': '#E91E63',  # Pink
        'TSEvo': '#D35400',  # Burnt Orange
        'LASTS': '#1ABC9C',  # Turquoise
        'TSCF': '#9B59B6',  # Amethyst
        'FASTPACE': '#3498DB',  # Dodger Blue
        'TIME-CF': '#E8A838',  # Golden
        'SG-CF': '#1F618D',  # Navy Blue
        'MG-CF': '#A569BD',  # Light Purple
        'Latent-CF': '#45B39D',  # Medium Sea Green
        'DiSCoX': '#EC7063',  # Light Red
        'M-CELS': '#5D6D7E',  # Slate Gray
        'FFT-CF': '#7D3C98',  # Deep Purple
        'TERCE': '#F1948A',  # Light Salmon
        'AB-CF': '#AED6F1',  # Light Blue
        'CFWOT': '#7FB3D5',  # Sky Blue
        'CGM': '#76D7C4',  # Light Cyan
        'COUNTS': '#F8C471',  # Light Yellow
        'SPARCE': '#AF7AC5',  # Lavender
        'CEM': '#2ECC71',  # Emerald Green
        'TS-Tweaking': '#E74C3C',  # Alizarin Red
        'Abstract': '#BDC3C7',  # Silver
        'Soft-DTW-CFE': '#117A65'  # Dark Teal
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
cf_results_all = {
    'Native Guide': (cf_ng, prediction_ng),
    'COMTE': (cf_comte, prediction_comte),
    'SETS': (cf_sets, prediction_sets),
    'Wachter Genetic': (cf_w, prediction_w),
    'GLACIER': (cf_glacier, prediction_glacier),
    'Sub-SpaCE': (cf_subspace, prediction_subspace),
    'TSEvo': (cf_tsevo, prediction_tsevo),
    'MG-CF': (cf_mg_cf, prediction_mg_cf),
    'Latent-CF': (cf_latent_cf, prediction_latent_cf),
    'M-CELS': (cf_cels, prediction_cels),
    'TERCE': (cf_terce, prediction_terce),
    'CEM': (cf_cem, prediction_cem),
    'Soft-DTW-CFE': (cf_soft_dtw_cfe, prediction_soft_dtw_cfe),
}

def _achieved_target(pred, target):
    """Whether a method's resulting prediction actually reached target_class."""
    if pred is None:
        return False
    return int(np.argmax(np.asarray(pred).reshape(-1))) == target

# Only keep methods that actually produced a valid counterfactual (errored
# out or ran but never reached target_class isn't a counterfactual to show)
# rather than plotting every attempt regardless of outcome.
cf_results = {name: (cf, pred) for name, (cf, pred) in cf_results_all.items()
              if _achieved_target(pred, target_class)}
excluded_names = [name for name in cf_results_all if name not in cf_results]
if excluded_names:
    print(f"Excluded {len(excluded_names)} method(s) that didn't produce a valid "
          f"counterfactual (no result, or wrong predicted class): {excluded_names}")

# Create enhanced visualization
create_enhanced_visualization(sample, label, original_pred_np, original_class, cf_results, is_multivariate)

# Save the plot but don't show it
# plt.show()  # Disabled to prevent plot display
print("Plot saved. Exiting without displaying.")
