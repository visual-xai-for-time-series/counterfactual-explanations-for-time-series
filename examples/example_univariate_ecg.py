"""
ECG200 Counterfactual Explanations Example

This example demonstrates counterfactual explanation generation for the ECG200 dataset
using two representative methods from each category (Optimization-Based: Wachter, COMTE;
Evolutionary: TSEvo, Sub-SpaCE; Instance-Based: Native Guide, CELS; Latent Space: GLACIER,
Latent-CF; Segment-Based: SETS, TS-CEM; Hybrid: TeRCE, MG-CF) with enhanced visualization.

Features:
- Univariate time series support
- Professional visualization with enhanced styling
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
_log_file = os.path.join(_log_dir, 'example_univariate_ecg.log')
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
from sklearn.metrics import f1_score

# Set up enhanced plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

import cfts.cf_native_guide.native_guide as ng
import cfts.cf_wachter.wachter as w
import cfts.cf_comte.comte as comte
import cfts.cf_sets.sets as sets
import cfts.cf_glacier.glacier as glacier
import cfts.cf_tsevo.tsevo as tsevo
import cfts.cf_subspace.subspace as subspace
from cfts.cf_mg_cf import mg_cf_generate_stumpy
import cfts.cf_latent_cf.latent_cf as latent_cf
import cfts.cf_cels.cels as cels
import cfts.cf_terce.terce as terce
import cfts.cf_cem.cem as cem



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading dataset')
dataloader_train, dataset_train = bd.get_UCR_UEA_dataloader(dataset_name='ECG200', split='train')
dataloader_test, dataset_test = bd.get_UCR_UEA_dataloader(dataset_name='ECG200', split='test')

output_classes = dataset_train.y_shape[1]
input_length = dataset_train.X_shape[2]  # Get the time series length

model = bm.SimpleCNN(output_channels=output_classes, input_length=input_length).to(device)

# --- model persistence: load if exists, otherwise train and save ---
models_dir = os.path.abspath(os.path.join(script_path, '..', 'models'))
os.makedirs(models_dir, exist_ok=True)
model_file = os.path.join(models_dir, f'simple_cnn_ecg200_{output_classes}.pth')

model_loaded = False
if os.path.exists(model_file):
    print(f'Loading saved model from {model_file}')
    state = torch.load(model_file, map_location=device)
    model.load_state_dict(state)
    model_loaded = True
else:
    print(f'No saved model at {model_file}; training will run and the model will be saved.')

print('Preparing training components')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


def trainer(model_, dataloader, criterion_):
    running_loss = 0

    model_.train()

    for _, (inputs, labels) in enumerate(dataloader):
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
    all_preds = []
    all_labels = []

    model_.eval()

    for _, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.reshape(inputs.shape[0], 1, -1)
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)

        preds = model_(inputs)
        loss_val = criterion_(preds, labels.argmax(dim=-1))

        running_loss += loss_val.item()
        all_preds.extend(preds.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.argmax(dim=-1).cpu().numpy())

    val_loss = running_loss / len(dataloader)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')

    return val_loss, val_f1


# only train if we didn't load a saved model
if not model_loaded:
    print('Training model')
    epochs = 300
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_model_state = None

    for epoch in range(epochs):
        train_loss = trainer(model, dataloader_train, criterion)
        val_loss, val_f1 = validator(model, dataloader_test, criterion)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            if epoch % 10 == 0:
                print(f'Epoch {epoch:4d} - Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}, F1: {val_f1:.4f} *** New best ***')
        else:
            if epoch % 10 == 0:
                print(f'Epoch {epoch:4d} - Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}, F1: {val_f1:.4f}')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\nLoaded best model - Val Loss: {best_val_loss:.3f}, F1: {best_val_f1:.4f}')
    
    # save trained model state_dict
    torch.save(model.state_dict(), model_file)
    print(f'Model saved to {model_file}')
else:
    print('Using loaded model; skipping training.')

# Evaluate the model (whether trained or loaded) and show F1 score
print('\nEvaluating model performance...')
model.eval()
val_loss, val_f1 = validator(model, dataloader_test, criterion)
print(f'Model Performance - Val Loss: {val_loss:.3f}, F1 Score: {val_f1:.4f}\n')

print('Generating counterfactual')
# Select a random sample from the test dataset that is correctly classified
model.eval()
sample, label = None, None
original_pred_np, original_class = None, None
original_pred = None
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
            if sample_tensor.shape[0] > sample_tensor.shape[1]:
                sample_tensor = sample_tensor.T
            sample_tensor = sample_tensor.unsqueeze(0)
        
        pred_output = model(sample_tensor)
        pred_np = torch.softmax(pred_output, dim=-1).squeeze().cpu().numpy()
        pred_class = torch.argmax(pred_output, dim=-1).item()
        
        # Check if prediction matches true label
        true_class = np.argmax(candidate_label) if hasattr(candidate_label, 'shape') and len(candidate_label.shape) > 0 else candidate_label
        if pred_class == true_class:
            sample, label = candidate_sample, candidate_label
            original_pred_np, original_class = pred_np, pred_class
            original_pred = pred_output.squeeze().cpu().detach().numpy()
            print(f'Found correctly classified sample {random_idx} after {attempts + 1} attempts')
            break
    
    attempts += 1

if sample is None:
    print(f'Could not find a correctly classified sample after {max_attempts} attempts')
    print('Exiting without generating counterfactuals')
    exit(1)

print('Selected correctly classified sample from test dataset')

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

print(f'Original class: {original_class}')
print(f'Target class for all counterfactuals: {target_class}')
print()

# Dictionary to store timing results
timing_results = {}

# List of all methods to execute
methods = [
    'Native Guide', 'COMTE', 'SETS', 'Wachter Genetic', 'GLACIER',
    'Sub-SpaCE', 'TSEvo', 'MG-CF', 'Latent-CF', 'CELS', 'TERCE', 'CEM-PN',
]

# Initialize progress bar
progress = tqdm(total=len(methods), desc='Generating Counterfactuals', unit='method')

print('Start with native guide')
start_time = time.time()
try:
    cf_ng, prediction_ng = ng.native_guide_uni_cf(sample, model, dataset=dataset_test)
    timing_results['Native Guide'] = time.time() - start_time
    print(f'Native Guide completed in {timing_results["Native Guide"]:.3f} seconds')
except Exception as e:
    cf_ng, prediction_ng = None, None
    timing_results['Native Guide'] = time.time() - start_time
    print(f'Native Guide failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with COMTE')
start_time = time.time()
try:
    cf_comte, prediction_comte = comte.comte_cf_gradient(sample, model, target_class=target_class, dataset=dataset_test)
    timing_results['COMTE'] = time.time() - start_time
    print(f'COMTE completed in {timing_results["COMTE"]:.3f} seconds')
except Exception as e:
    cf_comte, prediction_comte = None, None
    timing_results['COMTE'] = time.time() - start_time
    print(f'COMTE failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with SETS')
start_time = time.time()
try:
    cf_sets, prediction_sets = sets.sets_cf(sample, model, target_class=target_class, dataset=dataset_test)
    timing_results['SETS'] = time.time() - start_time
    print(f'SETS completed in {timing_results["SETS"]:.3f} seconds')
except Exception as e:
    cf_sets, prediction_sets = None, None
    timing_results['SETS'] = time.time() - start_time
    print(f'SETS failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with Genetic Wachter et al.')
start_time = time.time()
try:
    cf_w, prediction_w = w.wachter_genetic_cf(sample, model, target_class=target_class, step_size=np.mean(dataset_test.std) + 0.2, max_steps=100)
    timing_results['Wachter Genetic'] = time.time() - start_time
    print(f'Wachter Genetic completed in {timing_results["Wachter Genetic"]:.3f} seconds')
except Exception as e:
    cf_w, prediction_w = None, None
    timing_results['Wachter Genetic'] = time.time() - start_time
    print(f'Wachter Genetic failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with GLACIER')
start_time = time.time()
try:
    cf_glacier, prediction_glacier = glacier.glacier_cf(sample, model, target_class=target_class, dataset=dataset_test)
    timing_results['GLACIER'] = time.time() - start_time
    print(f'GLACIER completed in {timing_results["GLACIER"]:.3f} seconds')
except Exception as e:
    cf_glacier, prediction_glacier = None, None
    timing_results['GLACIER'] = time.time() - start_time
    print(f'GLACIER failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with Sub-SpaCE')
start_time = time.time()
try:
    cf_subspace, prediction_subspace = subspace.subspace_cf(
        sample, model,
        target_class=target_class,
        dataset=dataset_test,
        population_size=100,
        max_iter=200,  # Increased iterations
        alpha=0.8,  # Even higher weight for classification (validity)
        beta=0.15,  # Lower weight for sparsity
        eta=0.05,   # Lower weight for outlier
        invalid_penalization=20,  # Much lower penalty
        init_pct=0.4,  # Higher initial activation
        reinit=True,
        verbose=False
    )
    timing_results['Sub-SpaCE'] = time.time() - start_time
    print(f'Sub-SpaCE completed in {timing_results["Sub-SpaCE"]:.3f} seconds')
except Exception as e:
    cf_subspace, prediction_subspace = None, None
    timing_results['Sub-SpaCE'] = time.time() - start_time
    print(f'Sub-SpaCE failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with TSEvo')
start_time = time.time()
try:
    cf_tsevo, prediction_tsevo = tsevo.tsevo_cf(sample, model,
                                                target_class=target_class,
                                                dataset=dataset_test,
                                                population_size=30,
                                                generations=50)
    timing_results['TSEvo'] = time.time() - start_time
    print(f'TSEvo completed in {timing_results["TSEvo"]:.3f} seconds')
except Exception as e:
    cf_tsevo, prediction_tsevo = None, None
    timing_results['TSEvo'] = time.time() - start_time
    print(f'TSEvo failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with MG-CF (STUMPY optimized)')
start_time = time.time()
try:
    # MG-CF with STUMPY optimization for faster motif mining
    # Using a smaller subset for faster execution
    subset_size = min(100, len(dataset_test))
    from torch.utils.data import Subset
    dataset_subset = Subset(dataset_test, range(subset_size))
    cf_mg_cf, prediction_mg_cf = mg_cf_generate_stumpy(sample, model, target_class=target_class, dataset=dataset_subset,
                                                        top_k=5,  # Reduce top_k for faster execution
                                                        verbose=False)
    timing_results['MG-CF'] = time.time() - start_time
    print(f'MG-CF completed in {timing_results["MG-CF"]:.3f} seconds')
except Exception as e:
    cf_mg_cf, prediction_mg_cf = None, None
    timing_results['MG-CF'] = time.time() - start_time
    print(f'MG-CF failed: {type(e).__name__}: {str(e)[:100]}')
finally:
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
    print(f'Latent-CF failed: {str(e)}')
finally:
    progress.update(1)

print('Start with CELS')
start_time = time.time()
try:
    # CELS requires training data for nearest unlike neighbor
    X_train = np.array([dataset_test[i][0] for i in range(min(100, len(dataset_test)))])
    y_train = np.array([dataset_test[i][1] for i in range(min(100, len(dataset_test)))])
    cf_cels, prediction_cels = cels.cels_generate(sample, model, X_train, y_train,
                                                 target_class=target_class,
                                                 max_iter=100,
                                                 verbose=False)
    timing_results['CELS'] = time.time() - start_time
    print(f'CELS completed in {timing_results["CELS"]:.3f} seconds')
except Exception as e:
    cf_cels, prediction_cels = None, None
    timing_results['CELS'] = time.time() - start_time
    print(f'CELS failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with TERCE')
start_time = time.time()
try:
    # TERCE requires training data for nearest unlike neighbor and rule mining
    X_train = np.array([dataset_test[i][0] for i in range(min(100, len(dataset_test)))])
    y_train = np.array([np.argmax(dataset_test[i][1]) if hasattr(dataset_test[i][1], 'shape') and len(dataset_test[i][1].shape) > 0 else dataset_test[i][1] for i in range(min(100, len(dataset_test)))])
    cf_terce, pred_class_terce = terce.terce_generate(sample, model, X_train, y_train,
                                                    target_class=target_class,
                                                    n_regions=10,
                                                    window_size_ratio=0.1,
                                                    verbose=False)
    # TERCE returns an integer class, convert to probability array for consistency
    if pred_class_terce is not None:
        prediction_terce = np.zeros(output_classes)
        prediction_terce[pred_class_terce] = 1.0
    else:
        prediction_terce = None
    timing_results['TERCE'] = time.time() - start_time
    print(f'TERCE completed in {timing_results["TERCE"]:.3f} seconds')
except Exception as e:
    cf_terce, prediction_terce = None, None
    timing_results['TERCE'] = time.time() - start_time
    print(f'TERCE failed: {type(e).__name__}: {str(e)[:100]}')
finally:
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
finally:
    progress.update(1)

# Close the progress bar
progress.close()

print()
print('='*80)
print('Combined Results Summary:')
print('='*80)
# Get original prediction info
true_class_idx = np.argmax(label) if hasattr(label, 'shape') and len(label.shape) > 0 else label
print(f'Target Class: {target_class}')
print('-'*80)
print(f'{"Method":<20} {"Status":<10} {"Pred Class":<12} {"Confidence":<12} {"Time (s)":>10}')
print('-'*80)

def format_combined_result(name, prediction, elapsed_time):
    if prediction is None:
        return f'{name:<20} {"Failed":<10} {"-":<12} {"-":<12} {elapsed_time:>10.3f}'
    pred_np = np.asarray(prediction).reshape(-1)
    
    # Check if prediction looks like logits (negative values or not summing to 1) or probabilities
    # Apply softmax if needed
    if np.any(pred_np < 0) or not np.isclose(np.sum(pred_np), 1.0, atol=0.1):
        # Looks like logits, apply softmax
        pred_np = np.exp(pred_np) / np.sum(np.exp(pred_np))
    
    pred_class = int(np.argmax(pred_np))
    confidence = float(np.max(pred_np))
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
print(format_combined_result('CELS', prediction_cels, timing_results['CELS']))
print(format_combined_result('TERCE', prediction_terce, timing_results['TERCE']))
print(format_combined_result('CEM', prediction_cem, timing_results['CEM']))
print('='*80)
print()

# Normalize series to channel-first arrays (C, L) for plotting
def _to_channel_first(a):
    arr = np.asarray(a)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        r, c = arr.shape
        return arr if r <= c else arr.T
    if arr.ndim == 3:
        # take first batch element if present (B, C, L) -> (C, L)
        return arr[0]
    raise ValueError("Unsupported array shape for plotting: %s" % (arr.shape,))

sample_pl = _to_channel_first(sample)
cf_ng_pl = None if cf_ng is None else _to_channel_first(cf_ng)
cf_comte_pl = None if cf_comte is None else _to_channel_first(cf_comte)
cf_sets_pl = None if cf_sets is None else _to_channel_first(cf_sets)
cf_w_pl = None if cf_w is None else _to_channel_first(cf_w)
cf_glacier_pl = None if cf_glacier is None else _to_channel_first(cf_glacier)
cf_tsevo_pl = None if cf_tsevo is None else _to_channel_first(cf_tsevo)
cf_subspace_pl = None if cf_subspace is None else _to_channel_first(cf_subspace)
cf_mg_cf_pl = None if cf_mg_cf is None else _to_channel_first(cf_mg_cf)
cf_latent_cf_pl = None if cf_latent_cf is None else _to_channel_first(cf_latent_cf)
cf_cels_pl = None if cf_cels is None else _to_channel_first(cf_cels)
cf_terce_pl = None if cf_terce is None else _to_channel_first(cf_terce)
cf_cem_pl = None if cf_cem is None else _to_channel_first(cf_cem)

def _fmt_pred(pred):
    """Format a model prediction array into 'label (conf)' or 'None'."""
    if pred is None:
        return "None"
    arr = np.asarray(pred).reshape(-1)
    
    # Check if prediction looks like logits or probabilities and normalize
    if np.any(arr < 0) or not np.isclose(np.sum(arr), 1.0, atol=0.1):
        # Looks like logits, apply softmax
        arr = np.exp(arr) / np.sum(np.exp(arr))
    
    lab = int(np.argmax(arr))
    conf = float(np.max(arr))
    return f"{lab} ({conf:.2f})"

pred_ng_str = _fmt_pred(prediction_ng)
pred_comte_str = _fmt_pred(prediction_comte)
pred_sets_str = _fmt_pred(prediction_sets)
pred_w_str = _fmt_pred(prediction_w)
pred_glacier_str = _fmt_pred(prediction_glacier)
pred_tsevo_str = _fmt_pred(prediction_tsevo)
pred_subspace_str = _fmt_pred(prediction_subspace)
pred_mg_cf_str = _fmt_pred(prediction_mg_cf)
pred_latent_cf_str = _fmt_pred(prediction_latent_cf)
pred_cels_str = _fmt_pred(prediction_cels)
pred_terce_str = _fmt_pred(prediction_terce)
pred_cem_str = _fmt_pred(prediction_cem)
pred_original_str = _fmt_pred(original_pred_np)

def _check_success(pred, target):
    """Check if counterfactual achieved the target class."""
    if pred is None:
        return False
    arr = np.asarray(pred).reshape(-1)
    pred_class = int(np.argmax(arr))
    return pred_class == target

# Check success for all methods
success_ng = _check_success(prediction_ng, target_class)
success_comte = _check_success(prediction_comte, target_class)
success_sets = _check_success(prediction_sets, target_class)
success_w = _check_success(prediction_w, target_class)
success_glacier = _check_success(prediction_glacier, target_class)
success_subspace = _check_success(prediction_subspace, target_class)
success_tsevo = _check_success(prediction_tsevo, target_class)
success_mg_cf = _check_success(prediction_mg_cf, target_class)
success_latent_cf = _check_success(prediction_latent_cf, target_class)
success_cels = _check_success(prediction_cels, target_class)
success_terce = _check_success(prediction_terce, target_class)
success_cem = _check_success(prediction_cem, target_class)

def plot_channels(ax, arr, title=None, styles=None, alpha=1.0):
    """Plot each channel on ax. arr is (C, L). styles can be list of kwargs per channel."""
    if title:
        ax.set_title(title)
    C, L = arr.shape
    x = np.arange(L)
    for ch in range(C):
        style = styles[ch] if styles and ch < len(styles) else {}
        ax.plot(x, arr[ch], **style, alpha=alpha)
    if C > 1:
        ax.legend([f'channel:{i}' for i in range(C)], loc='upper right', fontsize='small')

n_rows = 25  # 1 original + 12 individual CFs + 12 overlays
fig, axs = plt.subplots(n_rows, figsize=(10, 1.75 * n_rows))
fig.suptitle('Counterfactual Explanations - ECG200', y=0.998, fontsize=14)

i = 0
# show true label from dataset and model prediction
true_class_idx = np.argmax(label) if hasattr(label, 'shape') and len(label.shape) > 0 else label
true_label_str = f"Class {true_class_idx}"
plot_channels(axs[i], sample_pl, f'Original sample — true: {true_label_str}, pred: {pred_original_str}', styles=[{'color': 'blue'}])
i += 1

# Individual counterfactual plots
if cf_ng_pl is not None:
    status = '✓' if success_ng else '✗'
    plot_channels(axs[i], cf_ng_pl, f'Native Guide [{status}] — pred: {pred_ng_str}')
else:
    axs[i].set_title('Native Guide [✗ FAILED]')
i += 1

if cf_comte_pl is not None:
    status = '✓' if success_comte else '✗'
    plot_channels(axs[i], cf_comte_pl, f'COMTE [{status}] — pred: {pred_comte_str}')
else:
    axs[i].set_title('COMTE [✗ FAILED]')
i += 1

if cf_sets_pl is not None:
    status = '✓' if success_sets else '✗'
    plot_channels(axs[i], cf_sets_pl, f'SETS [{status}] — pred: {pred_sets_str}')
else:
    axs[i].set_title('SETS [✗ FAILED]')
i += 1

if cf_w_pl is not None:
    status = '✓' if success_w else '✗'
    plot_channels(axs[i], cf_w_pl, f'Wachter Genetic [{status}] — pred: {pred_w_str}')
else:
    axs[i].set_title('Wachter Genetic [✗ FAILED]')
i += 1

if cf_glacier_pl is not None:
    status = '✓' if success_glacier else '✗'
    plot_channels(axs[i], cf_glacier_pl, f'GLACIER [{status}] — pred: {pred_glacier_str}')
else:
    axs[i].set_title('GLACIER [✗ FAILED]')
i += 1

if cf_subspace_pl is not None:
    status = '✓' if success_subspace else '✗'
    plot_channels(axs[i], cf_subspace_pl, f'Sub-SpaCE [{status}] — pred: {pred_subspace_str}')
else:
    axs[i].set_title('Sub-SpaCE [✗ FAILED]')
i += 1

if cf_tsevo_pl is not None:
    status = '✓' if success_tsevo else '✗'
    plot_channels(axs[i], cf_tsevo_pl, f'TSEvo [{status}] — pred: {pred_tsevo_str}')
else:
    axs[i].set_title('TSEvo [✗ FAILED]')
i += 1

if cf_mg_cf_pl is not None:
    status = '✓' if success_mg_cf else '✗'
    plot_channels(axs[i], cf_mg_cf_pl, f'MG-CF [{status}] — pred: {pred_mg_cf_str}')
else:
    axs[i].set_title('MG-CF [✗ FAILED]')
i += 1

if cf_latent_cf_pl is not None:
    status = '✓' if success_latent_cf else '✗'
    plot_channels(axs[i], cf_latent_cf_pl, f'Latent-CF [{status}] — pred: {pred_latent_cf_str}')
else:
    axs[i].set_title('Latent-CF [✗ FAILED]')
i += 1

if cf_cels_pl is not None:
    status = '✓' if success_cels else '✗'
    plot_channels(axs[i], cf_cels_pl, f'CELS [{status}] — pred: {pred_cels_str}')
else:
    axs[i].set_title('CELS [✗ FAILED]')
i += 1

if cf_terce_pl is not None:
    status = '✓' if success_terce else '✗'
    plot_channels(axs[i], cf_terce_pl, f'TERCE [{status}] — pred: {pred_terce_str}')
else:
    axs[i].set_title('TERCE [✗ FAILED]')
i += 1

if cf_cem_pl is not None:
    status = '✓' if success_cem else '✗'
    plot_channels(axs[i], cf_cem_pl, f'CEM [{status}] — pred: {pred_cem_str}')
else:
    axs[i].set_title('CEM [✗ FAILED]')
i += 1

# overlay plots: counterfactual vs original
def overlay(ax, base, other, title, pred_str=None, is_success=False):
    if other is None:
        ax.set_title(f'{title} [✗ FAILED]')
        return
    # include prediction in overlay title if provided
    status = '✓' if is_success else '✗'
    t = f"{title} [{status}] — pred: {pred_str}" if pred_str else f"{title} [{status}]"
    ax.set_title(t)
    plot_channels(ax, base, title=None, styles=[{'linestyle': '--', 'color': 'blue'} for _ in range(base.shape[0])], alpha=0.6)
    plot_channels(ax, other, title=None, styles=[{'linewidth': 1.2} for _ in range(other.shape[0])], alpha=0.9)

overlay(axs[i], sample_pl, cf_ng_pl, 'Native Guide vs Original', pred_ng_str, success_ng)
i += 1
overlay(axs[i], sample_pl, cf_comte_pl, 'COMTE vs Original', pred_comte_str, success_comte)
i += 1
overlay(axs[i], sample_pl, cf_sets_pl, 'SETS vs Original', pred_sets_str, success_sets)
i += 1
overlay(axs[i], sample_pl, cf_w_pl, 'Wachter Genetic vs Original', pred_w_str, success_w)
i += 1
overlay(axs[i], sample_pl, cf_glacier_pl, 'GLACIER vs Original', pred_glacier_str, success_glacier)
i += 1
overlay(axs[i], sample_pl, cf_subspace_pl, 'Sub-SpaCE vs Original', pred_subspace_str, success_subspace)
i += 1
overlay(axs[i], sample_pl, cf_tsevo_pl, 'TSEvo vs Original', pred_tsevo_str, success_tsevo)
i += 1
overlay(axs[i], sample_pl, cf_mg_cf_pl, 'MG-CF vs Original', pred_mg_cf_str, success_mg_cf)
i += 1
overlay(axs[i], sample_pl, cf_latent_cf_pl, 'Latent-CF vs Original', pred_latent_cf_str, success_latent_cf)
i += 1
overlay(axs[i], sample_pl, cf_cels_pl, 'CELS vs Original', pred_cels_str, success_cels)
i += 1
overlay(axs[i], sample_pl, cf_terce_pl, 'TERCE vs Original', pred_terce_str, success_terce)
i += 1
overlay(axs[i], sample_pl, cf_cem_pl, 'CEM vs Original', pred_cem_str, success_cem)
plt.tight_layout(rect=[0, 0.01, 1, 0.999])
plt.savefig('counterfactuals_ecg200.png')
print("\nPlot saved to 'counterfactuals_ecg200.png'. Exiting without displaying.")
# plt.show()  # Disabled to prevent plot display
