import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}/../')

# Import vibration generation functions
from base.vibration_gen import make_segmented_dataset, VibSegments, Tiny1DCNN

import base.model as bm
import base.data as bd

import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
import time
import pickle
from torch.utils.data import DataLoader

import cfts.cf_native_guide.native_guide as ng
import cfts.cf_wachter.wachter as w
import cfts.cf_comte.comte as comte
import cfts.cf_sets.sets as sets
import cfts.cf_dandl.dandl as dandl
import cfts.cf_glacier.glacier as glacier
import cfts.cf_multispace.multispace as ms



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data persistence: load if exists, otherwise generate and save ---
data_dir = os.path.abspath(os.path.join(script_path, 'data'))
os.makedirs(data_dir, exist_ok=True)

# Define generation parameters
generation_params = {
    'fs': 12000,
    'run_duration_s': 10,
    'segment_len': 2048,
    'hop': 512,
    'classes': ["normal","imbalance","misalignment","inner_race","outer_race","ball_fault"],
    'f_res': 3000.0,
    'Q': 50.0,
    'snr_db': 15.0,
    'seed': 42
}

data_file = os.path.join(data_dir, 'vibration_dataset.pkl')

if os.path.exists(data_file):
    print(f'Loading saved vibration dataset from {data_file}')
    with open(data_file, 'rb') as f:
        saved_data = pickle.load(f)
    
    X = saved_data['X']
    y = saved_data['y']
    class_names = saved_data['class_names']
    fs = saved_data['params']['fs']
    
    print(f"Loaded data: {X.shape}, {y.shape}")
    print(f"Classes: {class_names}")
else:
    print('Generating vibration dataset')
    # Generate synthetic vibration data - more challenging detection parameters
    X, y, class_names = make_segmented_dataset(
        fs=generation_params['fs'],
        run_duration_s=generation_params['run_duration_s'],
        segment_len=generation_params['segment_len'],
        hop=generation_params['hop'],
        classes=generation_params['classes'],
        f_res=generation_params['f_res'],
        Q=generation_params['Q'],
        snr_db=generation_params['snr_db'],
        seed=generation_params['seed']
    )
    
    print(f"Generated data: {X.shape}, {y.shape}")
    print(f"Classes: {class_names}")
    
    # Save the generated data
    saved_data = {
        'X': X,
        'y': y,
        'class_names': class_names,
        'params': generation_params
    }
    
    with open(data_file, 'wb') as f:
        pickle.dump(saved_data, f)
    print(f'Dataset saved to {data_file}')

fs = generation_params['fs']

# Split data (80/20 train/test)
n = len(X)
split = int(0.8 * n)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Create dataset objects that mimic the UCR structure
class VibrationDatasetWrapper:
    def __init__(self, X, y, class_names):
        self.X = X
        self.y = y
        self.class_names = class_names
        self.X_shape = X.shape
        self.y_shape = (len(y), len(class_names))  # mimic one-hot shape
        self.std = np.std(X, axis=-1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Return data in the expected format (sample, one-hot label)
        x = self.X[idx]
        y_onehot = np.zeros(len(self.class_names))
        y_onehot[self.y[idx]] = 1
        return x, y_onehot

# Create datasets
dataset_train = VibrationDatasetWrapper(X_train, y_train, class_names)
dataset_test = VibrationDatasetWrapper(X_test, y_test, class_names)

# Create dataloaders
dataloader_train = DataLoader(VibSegments(X_train, y_train, zscore=True), 
                             batch_size=128, shuffle=True, num_workers=0)
dataloader_test = DataLoader(VibSegments(X_test, y_test, zscore=True), 
                            batch_size=256, shuffle=False, num_workers=0)

output_classes = len(class_names)

# Use the Tiny1DCNN from vibration_gen instead of SimpleCNN
model = Tiny1DCNN(n_classes=output_classes).to(device)

# --- model persistence: load if exists, otherwise train and save ---
models_dir = os.path.abspath(os.path.join(script_path, '..', 'models'))
os.makedirs(models_dir, exist_ok=True)
model_file = os.path.join(models_dir, f'tiny1dcnn_vibration_{output_classes}.pth')

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

    for inputs, labels in dataloader:
        inputs = inputs.to(device)  # VibSegments already returns (batch, 1, length) format
        labels = labels.to(device)  # VibSegments returns class indices directly

        optimizer.zero_grad()
        preds = model_(inputs)
        loss_val = criterion_(preds, labels)
        loss_val.backward()
        optimizer.step()

        running_loss += loss_val.item()

    train_loss = running_loss / len(dataloader)

    return train_loss


def validator(model_, dataloader, criterion_):
    running_loss = 0

    model_.eval()
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model_(inputs)
            loss_val = criterion_(preds, labels)

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
            print(f'Val loss: {val_loss:.3f}')
        print(f'Train epoch {epoch:4d} loss: {train_loss:.3f}')

    # save trained model state_dict
    torch.save(model.state_dict(), model_file)
    print(f'Model saved to {model_file}')
else:
    print('Using loaded model; skipping training.')

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
        # sample is already (1, length) from the vibration dataset, reshape to (1, 1, length)
        sample_tensor = torch.tensor(candidate_sample, dtype=torch.float32, device=device)
        if len(sample_tensor.shape) == 2:
            sample_tensor = sample_tensor.unsqueeze(0)  # (1, 1, length)
        elif len(sample_tensor.shape) == 1:
            sample_tensor = sample_tensor.reshape(1, 1, -1)
        
        pred_output = model(sample_tensor)
        pred_np = torch.softmax(pred_output, dim=-1).squeeze().cpu().numpy()
        pred_class = torch.argmax(pred_output, dim=-1).item()
        
        # Check if prediction matches true label
        true_class = np.argmax(candidate_label)
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

print(f'True class: {class_names[np.argmax(label)]} (index: {np.argmax(label)})')
print(f'Predicted class: {class_names[original_class]} (index: {original_class})')

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
    target_class = (original_class + 1) % len(class_names)

print(f'Target class for all counterfactuals: {class_names[target_class]} (index: {target_class})')
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
start_time = time.time()
cf_w, prediction_w = w.wachter_genetic_cf(sample, model, target=target_class, step_size=np.mean(dataset_test.std) + 0.2, max_steps=100)
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

print()
print('='*80)
print('Combined Results Summary:')
print('='*80)
target_class_name = class_names[target_class]
print(f'Target Class: {target_class} ({target_class_name})')
print('-'*80)
print(f'{"Method":<20} {"Status":<10} {"Pred Class":<25} {"Confidence":<12} {"Time (s)":>10}')
print('-'*80)

def format_combined_result(name, prediction, elapsed_time):
    if prediction is None:
        return f'{name:<20} {"Failed":<10} {"-":<25} {"-":<12} {elapsed_time:>10.3f}'
    pred_np = np.asarray(prediction).reshape(-1)
    pred_class = int(np.argmax(pred_np))
    confidence = float(np.max(pred_np))
    class_name = class_names[pred_class]
    return f'{name:<20} {"Success":<10} {pred_class} ({class_name}){" "*(25-len(class_name)-4)} {confidence:<12.4f} {elapsed_time:>10.3f}'

original_class_name = class_names[original_class]
print(f'{"Original":<20} {"-":<10} {original_class} ({original_class_name}){" "*(25-len(original_class_name)-4)} {original_pred_np[original_class]:<12.4f} {"-":>10}')
print(format_combined_result('Native Guide', prediction_ng, timing_results['Native Guide']))
print(format_combined_result('COMTE', prediction_comte, timing_results['COMTE']))
print(format_combined_result('SETS', prediction_sets, timing_results['SETS']))
print(format_combined_result('MOC (Dandl)', prediction_moc, timing_results['MOC (Dandl)']))
print(format_combined_result('Wachter Gradient', prediction_wg, timing_results['Wachter Gradient']))
print(format_combined_result('Wachter Genetic', prediction_w, timing_results['Wachter Genetic']))
print(format_combined_result('GLACIER', prediction_glacier, timing_results['GLACIER']))
print(format_combined_result('Multi-SpaCE', prediction_multispace, timing_results['Multi-SpaCE']))
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
cf_moc_pl = None if cf_moc is None else _to_channel_first(cf_moc)
cf_wg_pl = None if cf_wg is None else _to_channel_first(cf_wg)
cf_w_pl = None if cf_w is None else _to_channel_first(cf_w)
cf_glacier_pl = None if cf_glacier is None else _to_channel_first(cf_glacier)
cf_multispace_pl = None if cf_multispace is None else _to_channel_first(cf_multispace)

def _fmt_pred(pred):
    """Format a model prediction array into 'label (conf)' or 'None'."""
    if pred is None:
        return "None"
    arr = np.asarray(pred).reshape(-1)
    lab = int(np.argmax(arr))
    conf = float(np.max(arr))
    return f"{lab} ({conf:.2f})"

pred_ng_str = _fmt_pred(prediction_ng)
pred_comte_str = _fmt_pred(prediction_comte)
pred_sets_str = _fmt_pred(prediction_sets)
pred_moc_str = _fmt_pred(prediction_moc)
pred_wg_str = _fmt_pred(prediction_wg)
pred_w_str = _fmt_pred(prediction_w)
pred_glacier_str = _fmt_pred(prediction_glacier)
pred_multispace_str = _fmt_pred(prediction_multispace)
pred_original_str = _fmt_pred(original_pred_np)

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

n_rows = 17  # Individual plots + overlay plots (added Multi-SpaCE)
fig, axs = plt.subplots(n_rows, figsize=(10, 2.2 * n_rows))
fig.suptitle('Counterfactual Explanations')

i = 0
# show true label from dataset and model prediction
true_class_idx = np.argmax(label) if label is not None else -1
true_class_name = class_names[true_class_idx] if true_class_idx >= 0 else "Unknown"
true_label_str = f"{true_class_name} ({true_class_idx})"
plot_channels(axs[i], sample_pl, f'Original sample — true: {true_label_str}, pred: {pred_original_str}')
i += 1

# Individual counterfactual plots
if cf_ng_pl is not None:
    plot_channels(axs[i], cf_ng_pl, f'Native Guide Counterfactual — pred: {pred_ng_str}')
else:
    axs[i].set_title('Native Guide Counterfactual (none)')
i += 1

if cf_comte_pl is not None:
    plot_channels(axs[i], cf_comte_pl, f'COMTE Counterfactual — pred: {pred_comte_str}')
else:
    axs[i].set_title('COMTE Counterfactual (none)')
i += 1

if cf_sets_pl is not None:
    plot_channels(axs[i], cf_sets_pl, f'SETS Counterfactual — pred: {pred_sets_str}')
else:
    axs[i].set_title('SETS Counterfactual (none)')
i += 1

if cf_moc_pl is not None:
    plot_channels(axs[i], cf_moc_pl, f'MOC Counterfactual — pred: {pred_moc_str}')
else:
    axs[i].set_title('MOC Counterfactual (none)')
i += 1

if cf_wg_pl is not None:
    plot_channels(axs[i], cf_wg_pl, f'Wachter et al Gradient Counterfactual — pred: {pred_wg_str}')
else:
    axs[i].set_title('Wachter et al Gradient Counterfactual (none)')
i += 1

if cf_w_pl is not None:
    plot_channels(axs[i], cf_w_pl, f'Wachter et al Genetic Counterfactual — pred: {pred_w_str}')
else:
    axs[i].set_title('Wachter et al Genetic Counterfactual (none)')
i += 1

if cf_glacier_pl is not None:
    plot_channels(axs[i], cf_glacier_pl, f'GLACIER Counterfactual — pred: {pred_glacier_str}')
else:
    axs[i].set_title('GLACIER Counterfactual (none)')
i += 1

if cf_multispace_pl is not None:
    plot_channels(axs[i], cf_multispace_pl, f'Multi-SpaCE Counterfactual — pred: {pred_multispace_str}')
else:
    axs[i].set_title('Multi-SpaCE Counterfactual (none)')
i += 1

# overlay plots: counterfactual vs original
def overlay(ax, base, other, title, pred_str=None):
    if other is None:
        ax.set_title(f'{title} (none)')
        return
    # include prediction in overlay title if provided
    t = f"{title} — pred: {pred_str}" if pred_str else title
    ax.set_title(t)
    plot_channels(ax, other, title=None, styles=[{'linewidth': 1.2} for _ in range(other.shape[0])], alpha=0.9)
    plot_channels(ax, base, title=None, styles=[{'linestyle': '--'} for _ in range(base.shape[0])], alpha=0.6)

overlay(axs[i], sample_pl, cf_ng_pl, 'Native Guide vs Original', pred_ng_str)
i += 1
overlay(axs[i], sample_pl, cf_comte_pl, 'COMTE vs Original', pred_comte_str)
i += 1
overlay(axs[i], sample_pl, cf_sets_pl, 'SETS vs Original', pred_sets_str)
i += 1
overlay(axs[i], sample_pl, cf_moc_pl, 'MOC vs Original', pred_moc_str)
i += 1
overlay(axs[i], sample_pl, cf_wg_pl, 'Wachter Gradient vs Original', pred_wg_str)
i += 1
overlay(axs[i], sample_pl, cf_w_pl, 'Wachter Genetic vs Original', pred_w_str)
i += 1
overlay(axs[i], sample_pl, cf_glacier_pl, 'GLACIER vs Original', pred_glacier_str)
i += 1
overlay(axs[i], sample_pl, cf_multispace_pl, 'Multi-SpaCE vs Original', pred_multispace_str)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('counterfactuals_vibration.png')
print("\nPlot saved to 'counterfactuals_vibration.png'. Exiting without displaying.")
# plt.show()  # Disabled to prevent plot display
