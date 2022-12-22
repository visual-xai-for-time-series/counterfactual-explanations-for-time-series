import torch

import numpy as np


def detach_to_numpy(data):
    # move pytorch data to cpu and detach it to numpy data
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    # convert numpy array to pytorch and move it to the device
    return torch.from_numpy(data).float().to(device)


def wachter_genetic_uni_cf(sample, model, target=None, max_steps=10000, step_size=0.1):

    device = next(model.parameters()).device
    def model_predict(data):
        return detach_to_numpy(model(numpy_to_torch(data, device)))
    
    # prepare inputs and configuration
    shape = sample.reshape(1, -1).shape

    sample_cf = np.copy(sample)
    y_cf = model_predict(sample_cf.reshape(-1, *shape))[0]
    label_cf = np.argmax(y_cf)
    if not target:
        target = 0
        if label_cf == 0:
            target += 1
    
    print(y_cf)
    # Iterate until the counterfactual prediction is different from the original prediction or the maximum number of steps is reached
    for step in range(max_steps):
        # Compute the model prediction for each possible change to the input features
        y_preds = []
        for i in range(shape[1]):
            z = np.zeros(sample_cf.shape)
            z[i] += step_size
            y_pred = model_predict((sample_cf + z).reshape(-1, *shape))[0]
            y_preds.append(y_pred)
        
        # Find the index of the input feature that results in the greatest change in the model prediction
        i_best = np.argmax(np.abs(np.array(y_preds) - y_cf), axis=0)
        i_best = i_best[target]
        
        # Update the counterfactual input values and prediction
        sample_cf[i_best] += step_size
        y_cf = model_predict(sample_cf.reshape(-1, *shape))[0]
        print(y_cf)
        
        # If the counterfactual prediction is different from the original prediction, return the counterfactual input values and prediction
        if np.argmax(y_cf) == target:
            return sample_cf, y_cf
        
    # If the maximum number of steps is reached without finding a counterfactual, return None
    return None, None

