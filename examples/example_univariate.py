"""
FordA Counterfactual Explanations Example

This example demonstrates counterfactual explanation generation for the FordA dataset
using various methods (Native Guide, COMTE, SETS, MOC, Wachter, GLACIER, Multi-SpaCE, 
TSEvo, LASTS, and TSCF) with enhanced visualization.

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
import cfts.cf_dandl.dandl as dandl
import cfts.cf_glacier.glacier as glacier
import cfts.cf_multispace.multispace as ms
import cfts.cf_tsevo.tsevo as tsevo
import cfts.cf_lasts.lasts as lasts
import cfts.cf_tscf.tscf as tscf
import cfts.cf_subspace.subspace as subspace
import cfts.cf_fastpace.fastpace as fastpace
import cfts.cf_time_cf.time_cf as time_cf
import cfts.cf_sg_cf.sg_cf as sg_cf
from cfts.cf_mg_cf import mg_cf_generate_stumpy
import cfts.cf_latent_cf.latent_cf as latent_cf
import cfts.cf_cgm.cgm as cgm
import cfts.cf_discox.discox as discox
import cfts.cf_cels.cels as cels
from cfts.cf_fft_cf.fft_cf import fft_nn_cf
import cfts.cf_terce.terce as terce
import cfts.cf_ab_cf.ab_cf as ab_cf
from cfts.cf_sparce.sparce import sparce_gan_cf
from cfts.cf_counts.counts import counts_cf_with_pretrained_model
import cfts.cf_cfwot.cfwot as cfwot



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading dataset')
dataloader_train, dataset_train = bd.get_UCR_UEA_dataloader(split='train')
dataloader_test, dataset_test = bd.get_UCR_UEA_dataloader(split='test')

output_classes = dataset_train.y_shape[1]

model = bm.SimpleCNN(output_channels=output_classes).to(device)

# --- model persistence: load if exists, otherwise train and save ---
models_dir = os.path.abspath(os.path.join(script_path, '..', 'models'))
os.makedirs(models_dir, exist_ok=True)
model_file = os.path.join(models_dir, f'simple_cnn_{output_classes}.pth')

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
        inputs = inputs.reshape(inputs.shape[0], 1, -1)
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)

        preds = model_(inputs)
        loss_val = criterion_(preds, labels.argmax(dim=-1))

        running_loss += loss_val.item()

    val_loss = running_loss / len(dataloader)  # fixed to use provided dataloader length

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
    'Native Guide', 'COMTE', 'COMTE-TS', 'SETS', 'MOC (Dandl)', 
    'Wachter Gradient', 'Wachter Genetic', 'GLACIER', 'Multi-SpaCE', 
    'Sub-SpaCE', 'TSEvo', 'LASTS', 'TSCF', 'FASTPACE', 'TIME-CF', 
    'SG-CF', 'MG-CF', 'Latent-CF', 'CGM', 'DiSCoX', 'CELS', 
    'FFT-CF', 'TERCE', 'AB-CF', 'SPARCE', 'CounTS', 'CFWoT'
]

# Initialize progress bar
progress = tqdm(total=len(methods), desc='Generating Counterfactuals', unit='method')

print('Start with native guide')
start_time = time.time()
try:
    # Native Guide doesn't support explicit target class, it finds the nearest different class
    cf_ng, prediction_ng = ng.native_guide_uni_cf(sample, dataset_test, model)
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
    cf_comte, prediction_comte = comte.comte_cf(sample, dataset_test, model, target_class=target_class)
    timing_results['COMTE'] = time.time() - start_time
    print(f'COMTE completed in {timing_results["COMTE"]:.3f} seconds')
except Exception as e:
    cf_comte, prediction_comte = None, None
    timing_results['COMTE'] = time.time() - start_time
    print(f'COMTE failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with COMTE-TS')
start_time = time.time()
try:
    cf_comte_ts, prediction_comte_ts = comte.comte_ts_cf(sample, dataset_test, model, target_class=target_class)
    timing_results['COMTE-TS'] = time.time() - start_time
    print(f'COMTE-TS completed in {timing_results["COMTE-TS"]:.3f} seconds')
except Exception as e:
    cf_comte_ts, prediction_comte_ts = None, None
    timing_results['COMTE-TS'] = time.time() - start_time
    print(f'COMTE-TS failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with SETS')
start_time = time.time()
try:
    cf_sets, prediction_sets = sets.sets_cf(sample, dataset_test, model, target_class=target_class)
    timing_results['SETS'] = time.time() - start_time
    print(f'SETS completed in {timing_results["SETS"]:.3f} seconds')
except Exception as e:
    cf_sets, prediction_sets = None, None
    timing_results['SETS'] = time.time() - start_time
    print(f'SETS failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with Dandl et al.')
start_time = time.time()
try:
    cf_moc, prediction_moc = dandl.moc_cf(sample, dataset_test, model, target_class=target_class)
    timing_results['MOC (Dandl)'] = time.time() - start_time
    print(f'MOC completed in {timing_results["MOC (Dandl)"]:.3f} seconds')
except Exception as e:
    cf_moc, prediction_moc = None, None
    timing_results['MOC (Dandl)'] = time.time() - start_time
    print(f'MOC failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with Gradient Wachter et al.')
start_time = time.time()
try:
    cf_wg, prediction_wg = w.wachter_gradient_cf(sample, dataset_test, model, target=target_class)
    timing_results['Wachter Gradient'] = time.time() - start_time
    print(f'Wachter Gradient completed in {timing_results["Wachter Gradient"]:.3f} seconds')
except Exception as e:
    cf_wg, prediction_wg = None, None
    timing_results['Wachter Gradient'] = time.time() - start_time
    print(f'Wachter Gradient failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with Genetic Wachter et al.')
start_time = time.time()
try:
    cf_w, prediction_w = w.wachter_genetic_cf(sample, model, target=target_class, step_size=np.mean(dataset_test.std) + 0.2, max_steps=100)
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
    cf_glacier, prediction_glacier = glacier.glacier_cf(sample, dataset_test, model, target_class=target_class)
    timing_results['GLACIER'] = time.time() - start_time
    print(f'GLACIER completed in {timing_results["GLACIER"]:.3f} seconds')
except Exception as e:
    cf_glacier, prediction_glacier = None, None
    timing_results['GLACIER'] = time.time() - start_time
    print(f'GLACIER failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with Multi-SpaCE')
start_time = time.time()
try:
    # Multi-SpaCE doesn't support explicit target class, it finds the nearest different class
    cf_multispace, prediction_multispace = ms.multi_space_cf(sample, dataset_test, model, 
                                                              population_size=30, 
                                                              max_iterations=50,
                                                              sparsity_weight=0.3,
                                                              validity_weight=0.7)
    timing_results['Multi-SpaCE'] = time.time() - start_time
    print(f'Multi-SpaCE completed in {timing_results["Multi-SpaCE"]:.3f} seconds')
except Exception as e:
    cf_multispace, prediction_multispace = None, None
    timing_results['Multi-SpaCE'] = time.time() - start_time
    print(f'Multi-SpaCE failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with Sub-SpaCE')
start_time = time.time()
try:
    cf_subspace, prediction_subspace = subspace.subspace_cf(
        sample, dataset_test, model,
        desired_class=target_class,
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
    cf_tsevo, prediction_tsevo = tsevo.tsevo_cf(sample, dataset_test, model, 
                                                target_class=target_class,
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

print('Start with LASTS')
start_time = time.time()
try:
    cf_lasts, prediction_lasts = lasts.lasts_cf(sample, dataset_test, model, 
                                                target_class=target_class,
                                                latent_dim=32,
                                                n_iterations=100,
                                                train_ae_epochs=50,
                                                verbose=False)
    timing_results['LASTS'] = time.time() - start_time
    print(f'LASTS completed in {timing_results["LASTS"]:.3f} seconds')
except Exception as e:
    cf_lasts, prediction_lasts = None, None
    timing_results['LASTS'] = time.time() - start_time
    print(f'LASTS failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with TSCF')
start_time = time.time()
try:
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
except Exception as e:
    cf_tscf, prediction_tscf = None, None
    timing_results['TSCF'] = time.time() - start_time
    print(f'TSCF failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with FASTPACE')
start_time = time.time()
try:
    cf_fastpace, prediction_fastpace = fastpace.fastpace_cf(sample, dataset_test, model, 
                                                            target=target_class,
                                                            n_planning_steps=10,
                                                            intervention_step_size=0.3,
                                                            lambda_proximity=1.0,
                                                            lambda_plausibility=0.5,
                                                            max_refinement_iterations=500,
                                                            verbose=False)
    timing_results['FASTPACE'] = time.time() - start_time
    print(f'FASTPACE completed in {timing_results["FASTPACE"]:.3f} seconds')
except Exception as e:
    cf_fastpace, prediction_fastpace = None, None
    timing_results['FASTPACE'] = time.time() - start_time
    print(f'FASTPACE failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with TIME-CF')
start_time = time.time()
try:
    # TIME-CF uses dataset for shapelet extraction and TimeGAN training
    from torch.utils.data import Subset
    subset_size = min(100, len(dataset_test))
    dataset_subset = Subset(dataset_test, range(subset_size))
    cf_time_cf, prediction_time_cf = time_cf.time_cf_generate(sample, dataset_subset, model, 
                                                             target_class=target_class,
                                                             n_shapelets=10,
                                                             M=32,
                                                             timegan_epochs=50,
                                                             verbose=False)
    timing_results['TIME-CF'] = time.time() - start_time
    print(f'TIME-CF completed in {timing_results["TIME-CF"]:.3f} seconds')
except Exception as e:
    cf_time_cf, prediction_time_cf = None, None
    timing_results['TIME-CF'] = time.time() - start_time
    print(f'TIME-CF failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with SG-CF')
start_time = time.time()
try:
    # SG-CF is too slow for this dataset - skipping to allow other methods to run
    cf_sg_cf, prediction_sg_cf = None, None
    timing_results['SG-CF'] = time.time() - start_time
    print(f'SG-CF skipped (too slow for this example)')
except Exception as e:
    cf_sg_cf, prediction_sg_cf = None, None
    timing_results['SG-CF'] = time.time() - start_time
    print(f'SG-CF failed: {str(e)}')
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
    cf_mg_cf, prediction_mg_cf = mg_cf_generate_stumpy(sample, dataset_subset, model, 
                                                        target=target_class,
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
    cf_latent_cf, prediction_latent_cf = latent_cf.latent_cf_generate(sample, dataset_test, model, 
                                                                      target=target_class,
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

print('Start with CGM (Conditional Generative Models)')
start_time = time.time()
try:
    cf_cgm, prediction_cgm = cgm.cgm_generate(sample, dataset_test, model, 
                                             target=target_class,
                                             latent_dim=16,
                                             max_iter=100,
                                             lambda_validity=1.0,
                                             lambda_proximity=0.5,
                                             lambda_sparsity=0.01,
                                             train_vae=True,
                                             verbose=False)
    timing_results['CGM'] = time.time() - start_time
    print(f'CGM completed in {timing_results["CGM"]:.3f} seconds')
except Exception as e:
    cf_cgm, prediction_cgm = None, None
    timing_results['CGM'] = time.time() - start_time
    print(f'CGM failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with DiSCoX')
start_time = time.time()
try:
    cf_discox, prediction_discox = discox.discox_cf(sample, dataset_test, model, 
                                                   target_class=target_class,
                                                   window_size=20,
                                                   max_iterations=100,
                                                   verbose=False)
    timing_results['DiSCoX'] = time.time() - start_time
    print(f'DiSCoX completed in {timing_results["DiSCoX"]:.3f} seconds')
except Exception as e:
    cf_discox, prediction_discox = None, None
    timing_results['DiSCoX'] = time.time() - start_time
    print(f'DiSCoX failed: {str(e)}')
finally:
    progress.update(1)

print('Start with CELS')
start_time = time.time()
try:
    # CELS requires training data for nearest unlike neighbor
    X_train = np.array([dataset_test[i][0] for i in range(min(100, len(dataset_test)))])
    y_train = np.array([dataset_test[i][1] for i in range(min(100, len(dataset_test)))])
    cf_cels, prediction_cels = cels.cels_generate(sample, model, X_train, y_train,
                                                 target=target_class,
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

print('Start with FFT-CF')
start_time = time.time()
try:
    # Using nearest neighbor FFT blending approach
    cf_fft_cf, prediction_fft_cf = fft_nn_cf(sample, dataset_test, model, 
                                             target_class=target_class,
                                             k=5,
                                             blend_ratio=0.5,
                                             frequency_bands="all",
                                             verbose=False)
    timing_results['FFT-CF'] = time.time() - start_time
    print(f'FFT-CF completed in {timing_results["FFT-CF"]:.3f} seconds')
except Exception as e:
    cf_fft_cf, prediction_fft_cf = None, None
    timing_results['FFT-CF'] = time.time() - start_time
    print(f'FFT-CF failed: {type(e).__name__}: {str(e)[:100]}')
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

print('Start with AB-CF')
start_time = time.time()
try:
    # AB-CF requires training data for nearest unlike neighbor retrieval
    X_train = np.array([dataset_test[i][0] for i in range(min(100, len(dataset_test)))])
    y_train = np.array([np.argmax(dataset_test[i][1]) if hasattr(dataset_test[i][1], 'shape') and len(dataset_test[i][1].shape) > 0 else dataset_test[i][1] for i in range(min(100, len(dataset_test)))])
    cf_ab_cf, pred_class_ab_cf = ab_cf.ab_cf_generate(sample, model, X_train, y_train,
                                                     target_class=target_class,
                                                     n_segments=10,
                                                     window_size_ratio=0.1,
                                                     verbose=False)
    # AB-CF returns an integer class, convert to probability array for consistency
    if pred_class_ab_cf is not None:
        prediction_ab_cf = np.zeros(output_classes)
        prediction_ab_cf[pred_class_ab_cf] = 1.0
    else:
        prediction_ab_cf = None
    timing_results['AB-CF'] = time.time() - start_time
    print(f'AB-CF completed in {timing_results["AB-CF"]:.3f} seconds')
except Exception as e:
    cf_ab_cf, prediction_ab_cf = None, None
    timing_results['AB-CF'] = time.time() - start_time
    print(f'AB-CF failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with SPARCE-GAN')
start_time = time.time()
try:
    # SPARCE uses GAN-based architecture to generate sparse counterfactuals
    cf_sparce, pred_class_sparce = sparce_gan_cf(sample, dataset_test, model,
                                                  target=target_class,
                                                  num_epochs=50,
                                                  lambda_sparse=1.0,
                                                  verbose=False)
    # SPARCE returns prediction probabilities array directly
    prediction_sparce = pred_class_sparce
    timing_results['SPARCE'] = time.time() - start_time
    print(f'SPARCE completed in {timing_results["SPARCE"]:.3f} seconds')
except Exception as e:
    cf_sparce, prediction_sparce = None, None
    timing_results['SPARCE'] = time.time() - start_time
    print(f'SPARCE failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with CounTS')
start_time = time.time()
try:
    # CounTS is a self-interpretable model that performs counterfactual reasoning
    # It requires training a VAE model on the dataset
    cf_counts, prediction_counts = counts_cf_with_pretrained_model(sample, dataset_test, model,
                                                                   target=target_class,
                                                                   latent_dim=16,
                                                                   hidden_dim=64,
                                                                   train_epochs=30,
                                                                   max_iter=500,
                                                                   verbose=False)
    timing_results['CounTS'] = time.time() - start_time
    print(f'CounTS completed in {timing_results["CounTS"]:.3f} seconds')
except Exception as e:
    cf_counts, prediction_counts = None, None
    timing_results['CounTS'] = time.time() - start_time
    print(f'CounTS failed: {type(e).__name__}: {str(e)[:100]}')
finally:
    progress.update(1)

print('Start with CFWoT')
start_time = time.time()
try:
    # CFWoT is a reinforcement learning approach that doesn't require training dataset
    cf_cfwot, pred_class_cfwot = cfwot(sample, model,
                                       target=target_class,
                                       M_E=200,
                                       M_T=100,
                                       verbose=False,
                                       device=device)
    # CFWoT returns an integer class, convert to probability array for consistency
    if pred_class_cfwot is not None:
        prediction_cfwot = np.zeros(output_classes)
        prediction_cfwot[pred_class_cfwot] = 1.0
    else:
        prediction_cfwot = None
    timing_results['CFWoT'] = time.time() - start_time
    print(f'CFWoT completed in {timing_results["CFWoT"]:.3f} seconds')
except Exception as e:
    cf_cfwot, prediction_cfwot = None, None
    timing_results['CFWoT'] = time.time() - start_time
    print(f'CFWoT failed: {type(e).__name__}: {str(e)[:100]}')
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
    pred_class = int(np.argmax(pred_np))
    confidence = float(np.max(pred_np))
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
print(format_combined_result('Sub-SpaCE', prediction_subspace, timing_results['Sub-SpaCE']))
print(format_combined_result('TSEvo', prediction_tsevo, timing_results['TSEvo']))
print(format_combined_result('LASTS', prediction_lasts, timing_results['LASTS']))
print(format_combined_result('TSCF', prediction_tscf, timing_results['TSCF']))
print(format_combined_result('FASTPACE', prediction_fastpace, timing_results['FASTPACE']))
print(format_combined_result('TIME-CF', prediction_time_cf, timing_results['TIME-CF']))
print(format_combined_result('SG-CF', prediction_sg_cf, timing_results['SG-CF']))
print(format_combined_result('MG-CF', prediction_mg_cf, timing_results['MG-CF']))
print(format_combined_result('Latent-CF', prediction_latent_cf, timing_results['Latent-CF']))
print(format_combined_result('DiSCoX', prediction_discox, timing_results['DiSCoX']))
print(format_combined_result('CELS', prediction_cels, timing_results['CELS']))
print(format_combined_result('FFT-CF', prediction_fft_cf, timing_results['FFT-CF']))
print(format_combined_result('TERCE', prediction_terce, timing_results['TERCE']))
print(format_combined_result('AB-CF', prediction_ab_cf, timing_results['AB-CF']))
print(format_combined_result('SPARCE', prediction_sparce, timing_results['SPARCE']))
print(format_combined_result('CounTS', prediction_counts, timing_results['CounTS']))
print(format_combined_result('CFWoT', prediction_cfwot, timing_results['CFWoT']))
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
cf_comte_ts_pl = None if cf_comte_ts is None else _to_channel_first(cf_comte_ts)
cf_sets_pl = None if cf_sets is None else _to_channel_first(cf_sets)
cf_moc_pl = None if cf_moc is None else _to_channel_first(cf_moc)
cf_wg_pl = None if cf_wg is None else _to_channel_first(cf_wg)
cf_w_pl = None if cf_w is None else _to_channel_first(cf_w)
cf_glacier_pl = None if cf_glacier is None else _to_channel_first(cf_glacier)
cf_multispace_pl = None if cf_multispace is None else _to_channel_first(cf_multispace)
cf_tsevo_pl = None if cf_tsevo is None else _to_channel_first(cf_tsevo)
cf_lasts_pl = None if cf_lasts is None else _to_channel_first(cf_lasts)
cf_tscf_pl = None if cf_tscf is None else _to_channel_first(cf_tscf)
cf_subspace_pl = None if cf_subspace is None else _to_channel_first(cf_subspace)
cf_fastpace_pl = None if cf_fastpace is None else _to_channel_first(cf_fastpace)
cf_time_cf_pl = None if cf_time_cf is None else _to_channel_first(cf_time_cf)
cf_sg_cf_pl = None if cf_sg_cf is None else _to_channel_first(cf_sg_cf)
cf_mg_cf_pl = None if cf_mg_cf is None else _to_channel_first(cf_mg_cf)
cf_latent_cf_pl = None if cf_latent_cf is None else _to_channel_first(cf_latent_cf)
cf_cgm_pl = None if cf_cgm is None else _to_channel_first(cf_cgm)
cf_discox_pl = None if cf_discox is None else _to_channel_first(cf_discox)
cf_cels_pl = None if cf_cels is None else _to_channel_first(cf_cels)
cf_fft_cf_pl = None if cf_fft_cf is None else _to_channel_first(cf_fft_cf)
cf_terce_pl = None if cf_terce is None else _to_channel_first(cf_terce)
cf_ab_cf_pl = None if cf_ab_cf is None else _to_channel_first(cf_ab_cf)
cf_sparce_pl = None if cf_sparce is None else _to_channel_first(cf_sparce)
cf_counts_pl = None if cf_counts is None else _to_channel_first(cf_counts)
cf_cfwot_pl = None if cf_cfwot is None else _to_channel_first(cf_cfwot)

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
pred_comte_ts_str = _fmt_pred(prediction_comte_ts)
pred_sets_str = _fmt_pred(prediction_sets)
pred_moc_str = _fmt_pred(prediction_moc)
pred_wg_str = _fmt_pred(prediction_wg)
pred_w_str = _fmt_pred(prediction_w)
pred_glacier_str = _fmt_pred(prediction_glacier)
pred_multispace_str = _fmt_pred(prediction_multispace)
pred_tsevo_str = _fmt_pred(prediction_tsevo)
pred_lasts_str = _fmt_pred(prediction_lasts)
pred_tscf_str = _fmt_pred(prediction_tscf)
pred_subspace_str = _fmt_pred(prediction_subspace)
pred_fastpace_str = _fmt_pred(prediction_fastpace)
pred_time_cf_str = _fmt_pred(prediction_time_cf)
pred_sg_cf_str = _fmt_pred(prediction_sg_cf)
pred_mg_cf_str = _fmt_pred(prediction_mg_cf)
pred_latent_cf_str = _fmt_pred(prediction_latent_cf)
pred_cgm_str = _fmt_pred(prediction_cgm)
pred_discox_str = _fmt_pred(prediction_discox)
pred_cels_str = _fmt_pred(prediction_cels)
pred_fft_cf_str = _fmt_pred(prediction_fft_cf)
pred_terce_str = _fmt_pred(prediction_terce)
pred_ab_cf_str = _fmt_pred(prediction_ab_cf)
pred_sparce_str = _fmt_pred(prediction_sparce)
pred_counts_str = _fmt_pred(prediction_counts)
pred_cfwot_str = _fmt_pred(prediction_cfwot)
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
success_comte_ts = _check_success(prediction_comte_ts, target_class)
success_sets = _check_success(prediction_sets, target_class)
success_moc = _check_success(prediction_moc, target_class)
success_wg = _check_success(prediction_wg, target_class)
success_w = _check_success(prediction_w, target_class)
success_glacier = _check_success(prediction_glacier, target_class)
success_multispace = _check_success(prediction_multispace, target_class)
success_subspace = _check_success(prediction_subspace, target_class)
success_tsevo = _check_success(prediction_tsevo, target_class)
success_lasts = _check_success(prediction_lasts, target_class)
success_tscf = _check_success(prediction_tscf, target_class)
success_fastpace = _check_success(prediction_fastpace, target_class)
success_time_cf = _check_success(prediction_time_cf, target_class)
success_sg_cf = _check_success(prediction_sg_cf, target_class)
success_mg_cf = _check_success(prediction_mg_cf, target_class)
success_latent_cf = _check_success(prediction_latent_cf, target_class)
success_cgm = _check_success(prediction_cgm, target_class)
success_discox = _check_success(prediction_discox, target_class)
success_cels = _check_success(prediction_cels, target_class)
success_fft_cf = _check_success(prediction_fft_cf, target_class)
success_terce = _check_success(prediction_terce, target_class)
success_ab_cf = _check_success(prediction_ab_cf, target_class)
success_sparce = _check_success(prediction_sparce, target_class)
success_counts = _check_success(prediction_counts, target_class)
success_cfwot = _check_success(prediction_cfwot, target_class)

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

n_rows = 55  # 1 original + 27 individual CFs + 27 overlays (includes CGM, SPARCE, CounTS, CFWoT)
fig, axs = plt.subplots(n_rows, figsize=(10, 1.75 * n_rows))
fig.suptitle('Counterfactual Explanations - FordA', y=0.998, fontsize=14)

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

if cf_comte_ts_pl is not None:
    status = '✓' if success_comte_ts else '✗'
    plot_channels(axs[i], cf_comte_ts_pl, f'COMTE-TS [{status}] — pred: {pred_comte_ts_str}')
else:
    axs[i].set_title('COMTE-TS [✗ FAILED]')
i += 1

if cf_sets_pl is not None:
    status = '✓' if success_sets else '✗'
    plot_channels(axs[i], cf_sets_pl, f'SETS [{status}] — pred: {pred_sets_str}')
else:
    axs[i].set_title('SETS [✗ FAILED]')
i += 1

if cf_moc_pl is not None:
    status = '✓' if success_moc else '✗'
    plot_channels(axs[i], cf_moc_pl, f'MOC [{status}] — pred: {pred_moc_str}')
else:
    axs[i].set_title('MOC [✗ FAILED]')
i += 1

if cf_wg_pl is not None:
    status = '✓' if success_wg else '✗'
    plot_channels(axs[i], cf_wg_pl, f'Wachter Gradient [{status}] — pred: {pred_wg_str}')
else:
    axs[i].set_title('Wachter Gradient [✗ FAILED]')
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

if cf_multispace_pl is not None:
    status = '✓' if success_multispace else '✗'
    plot_channels(axs[i], cf_multispace_pl, f'Multi-SpaCE [{status}] — pred: {pred_multispace_str}')
else:
    axs[i].set_title('Multi-SpaCE [✗ FAILED]')
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

if cf_lasts_pl is not None:
    status = '✓' if success_lasts else '✗'
    plot_channels(axs[i], cf_lasts_pl, f'LASTS [{status}] — pred: {pred_lasts_str}')
else:
    axs[i].set_title('LASTS [✗ FAILED]')
i += 1

if cf_tscf_pl is not None:
    status = '✓' if success_tscf else '✗'
    plot_channels(axs[i], cf_tscf_pl, f'TSCF [{status}] — pred: {pred_tscf_str}')
else:
    axs[i].set_title('TSCF [✗ FAILED]')
i += 1

if cf_fastpace_pl is not None:
    status = '✓' if success_fastpace else '✗'
    plot_channels(axs[i], cf_fastpace_pl, f'FASTPACE [{status}] — pred: {pred_fastpace_str}')
else:
    axs[i].set_title('FASTPACE [✗ FAILED]')
i += 1

if cf_time_cf_pl is not None:
    status = '✓' if success_time_cf else '✗'
    plot_channels(axs[i], cf_time_cf_pl, f'TIME-CF [{status}] — pred: {pred_time_cf_str}')
else:
    axs[i].set_title('TIME-CF [✗ FAILED]')
i += 1

if cf_sg_cf_pl is not None:
    status = '✓' if success_sg_cf else '✗'
    plot_channels(axs[i], cf_sg_cf_pl, f'SG-CF [{status}] — pred: {pred_sg_cf_str}')
else:
    axs[i].set_title('SG-CF [✗ FAILED]')
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

if cf_cgm_pl is not None:
    status = '✓' if success_cgm else '✗'
    plot_channels(axs[i], cf_cgm_pl, f'CGM [{status}] — pred: {pred_cgm_str}')
else:
    axs[i].set_title('CGM [✗ FAILED]')
i += 1

if cf_discox_pl is not None:
    status = '✓' if success_discox else '✗'
    plot_channels(axs[i], cf_discox_pl, f'DiSCoX [{status}] — pred: {pred_discox_str}')
else:
    axs[i].set_title('DiSCoX [✗ FAILED]')
i += 1

if cf_cels_pl is not None:
    status = '✓' if success_cels else '✗'
    plot_channels(axs[i], cf_cels_pl, f'CELS [{status}] — pred: {pred_cels_str}')
else:
    axs[i].set_title('CELS [✗ FAILED]')
i += 1

if cf_fft_cf_pl is not None:
    status = '✓' if success_fft_cf else '✗'
    plot_channels(axs[i], cf_fft_cf_pl, f'FFT-CF [{status}] — pred: {pred_fft_cf_str}')
else:
    axs[i].set_title('FFT-CF [✗ FAILED]')
i += 1

if cf_terce_pl is not None:
    status = '✓' if success_terce else '✗'
    plot_channels(axs[i], cf_terce_pl, f'TERCE [{status}] — pred: {pred_terce_str}')
else:
    axs[i].set_title('TERCE [✗ FAILED]')
i += 1

if cf_ab_cf_pl is not None:
    status = '✓' if success_ab_cf else '✗'
    plot_channels(axs[i], cf_ab_cf_pl, f'AB-CF [{status}] — pred: {pred_ab_cf_str}')
else:
    axs[i].set_title('AB-CF [✗ FAILED]')
i += 1

if cf_sparce_pl is not None:
    status = '✓' if success_sparce else '✗'
    plot_channels(axs[i], cf_sparce_pl, f'SPARCE [{status}] — pred: {pred_sparce_str}')
else:
    axs[i].set_title('SPARCE [✗ FAILED]')
i += 1

if cf_counts_pl is not None:
    status = '✓' if success_counts else '✗'
    plot_channels(axs[i], cf_counts_pl, f'CounTS [{status}] — pred: {pred_counts_str}')
else:
    axs[i].set_title('CounTS [✗ FAILED]')
i += 1

if cf_cfwot_pl is not None:
    status = '✓' if success_cfwot else '✗'
    plot_channels(axs[i], cf_cfwot_pl, f'CFWoT [{status}] — pred: {pred_cfwot_str}')
else:
    axs[i].set_title('CFWoT [✗ FAILED]')
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
overlay(axs[i], sample_pl, cf_comte_ts_pl, 'COMTE-TS vs Original', pred_comte_ts_str, success_comte_ts)
i += 1
overlay(axs[i], sample_pl, cf_sets_pl, 'SETS vs Original', pred_sets_str, success_sets)
i += 1
overlay(axs[i], sample_pl, cf_moc_pl, 'MOC vs Original', pred_moc_str, success_moc)
i += 1
overlay(axs[i], sample_pl, cf_wg_pl, 'Wachter Gradient vs Original', pred_wg_str, success_wg)
i += 1
overlay(axs[i], sample_pl, cf_w_pl, 'Wachter Genetic vs Original', pred_w_str, success_w)
i += 1
overlay(axs[i], sample_pl, cf_glacier_pl, 'GLACIER vs Original', pred_glacier_str, success_glacier)
i += 1
overlay(axs[i], sample_pl, cf_multispace_pl, 'Multi-SpaCE vs Original', pred_multispace_str, success_multispace)
i += 1
overlay(axs[i], sample_pl, cf_subspace_pl, 'Sub-SpaCE vs Original', pred_subspace_str, success_subspace)
i += 1
overlay(axs[i], sample_pl, cf_tsevo_pl, 'TSEvo vs Original', pred_tsevo_str, success_tsevo)
i += 1
overlay(axs[i], sample_pl, cf_lasts_pl, 'LASTS vs Original', pred_lasts_str, success_lasts)
i += 1
overlay(axs[i], sample_pl, cf_tscf_pl, 'TSCF vs Original', pred_tscf_str, success_tscf)
i += 1
overlay(axs[i], sample_pl, cf_fastpace_pl, 'FASTPACE vs Original', pred_fastpace_str, success_fastpace)
i += 1
overlay(axs[i], sample_pl, cf_time_cf_pl, 'TIME-CF vs Original', pred_time_cf_str, success_time_cf)
i += 1
overlay(axs[i], sample_pl, cf_sg_cf_pl, 'SG-CF vs Original', pred_sg_cf_str, success_sg_cf)
i += 1
overlay(axs[i], sample_pl, cf_mg_cf_pl, 'MG-CF vs Original', pred_mg_cf_str, success_mg_cf)
i += 1
overlay(axs[i], sample_pl, cf_latent_cf_pl, 'Latent-CF vs Original', pred_latent_cf_str, success_latent_cf)
i += 1
overlay(axs[i], sample_pl, cf_cgm_pl, 'CGM vs Original', pred_cgm_str, success_cgm)
i += 1
overlay(axs[i], sample_pl, cf_discox_pl, 'DiSCoX vs Original', pred_discox_str, success_discox)
i += 1
overlay(axs[i], sample_pl, cf_cels_pl, 'CELS vs Original', pred_cels_str, success_cels)
i += 1
overlay(axs[i], sample_pl, cf_fft_cf_pl, 'FFT-CF vs Original', pred_fft_cf_str, success_fft_cf)
i += 1
overlay(axs[i], sample_pl, cf_terce_pl, 'TERCE vs Original', pred_terce_str, success_terce)
i += 1
overlay(axs[i], sample_pl, cf_ab_cf_pl, 'AB-CF vs Original', pred_ab_cf_str, success_ab_cf)
i += 1
overlay(axs[i], sample_pl, cf_sparce_pl, 'SPARCE vs Original', pred_sparce_str, success_sparce)
i += 1
overlay(axs[i], sample_pl, cf_counts_pl, 'CounTS vs Original', pred_counts_str, success_counts)
i += 1
overlay(axs[i], sample_pl, cf_cfwot_pl, 'CFWoT vs Original', pred_cfwot_str, success_cfwot)

plt.tight_layout(rect=[0, 0.01, 1, 0.999])
plt.savefig('counterfactuals_forda.png')
print("\nPlot saved to 'counterfactuals_forda.png'. Exiting without displaying.")
# plt.show()  # Disabled to prevent plot display
