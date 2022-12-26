import numpy as np

import torch
import torch.nn as nn

from torch.optim import Adam


def detach_to_numpy(data):
    # move pytorch data to cpu and detach it to numpy data
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    # convert numpy array to pytorch and move it to the device
    return torch.from_numpy(data).float().to(device)


####
# Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR
#
# https://arxiv.org/abs/1711.00399
#
# There has been much discussion of the right to explanation in the EU General 
# Data Protection Regulation, and its existence, merits, and disadvantages. 
# Implementing a right to explanation that opens the black box of algorithmic 
# decision-making faces major legal and technical barriers. Explaining the 
# functionality of complex algorithmic decision-making systems and their 
# rationale in specific cases is a technically challenging problem. Some 
# explanations may offer little meaningful information to data subjects, 
# raising questions around their value. Explanations of automated decisions 
# need not hinge on the general public understanding how algorithmic systems 
# function. Even though such interpretability is of great importance and should 
# be pursued, explanations can, in principle, be offered without opening the 
# black box. Looking at explanations as a means to help a data subject act 
# rather than merely understand, one could gauge the scope and content of 
# explanations according to the specific goal or action they are intended to 
# support. From the perspective of individuals affected by automated 
# decision-making, we propose three aims for explanations: (1) to inform and 
# help the individual understand why a particular decision was reached, (2) to 
# provide grounds to contest the decision if the outcome is undesired, and (3) 
# to understand what would need to change in order to receive a desired result 
# in the future, based on the current decision-making model. We assess how each 
# of these goals finds support in the GDPR. We suggest data controllers should 
# offer a particular type of explanation, unconditional counterfactual 
# explanations, to support these three aims. These counterfactual explanations
# describe the smallest change to the world that can be made to obtain a 
# desirable outcome, or to arrive at the closest possible world, without 
# needing to explain the internal logic of the system.
#
# This is a genetic model-agnostic variant using the sensitivity of the model.
#
####
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
    
    # Iterate until the counterfactual prediction is different from the original prediction or the maximum number of steps is reached
    for _ in range(max_steps):
        # Compute the model prediction for each possible change to the input features
        y_preds = []
        for i in range(shape[1]):
            # plus variant
            z_plus = np.zeros(sample_cf.shape)
            z_plus[i] += step_size
            y_pred = model_predict((sample_cf + z_plus).reshape(-1, *shape))[0]
            y_preds.append(y_pred)
            
            # minus variant
            z_minus = np.zeros(sample_cf.shape)
            z_minus[i] -= step_size
            y_pred = model_predict((sample_cf + z_minus).reshape(-1, *shape))[0]
            y_preds.append(y_pred)
        
        # Find the index of the input feature that results in the greatest change in the model prediction
        i_best = np.argmax(np.abs(np.array(y_preds) - y_cf), axis=0)
        i_best = i_best[target]
        i_best_idx = int(i_best / 2)
        # Update the counterfactual input values and prediction
        if i_best_idx % 2 == 0:
            sample_cf[i_best_idx] += step_size
        else:
            sample_cf[i_best_idx] -= step_size
        
        y_cf = model_predict(sample_cf.reshape(-1, *shape))[0]
        
        # If the counterfactual prediction is different from the original prediction, return the counterfactual input values and prediction
        if np.argmax(y_cf) == target:
            return sample_cf, y_cf
        
    # If the maximum number of steps is reached without finding a counterfactual, return None
    return None, None


####
# Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR
#
# https://arxiv.org/abs/1711.00399
#
# There has been much discussion of the right to explanation in the EU General 
# Data Protection Regulation, and its existence, merits, and disadvantages. 
# Implementing a right to explanation that opens the black box of algorithmic 
# decision-making faces major legal and technical barriers. Explaining the 
# functionality of complex algorithmic decision-making systems and their 
# rationale in specific cases is a technically challenging problem. Some 
# explanations may offer little meaningful information to data subjects, 
# raising questions around their value. Explanations of automated decisions 
# need not hinge on the general public understanding how algorithmic systems 
# function. Even though such interpretability is of great importance and should 
# be pursued, explanations can, in principle, be offered without opening the 
# black box. Looking at explanations as a means to help a data subject act 
# rather than merely understand, one could gauge the scope and content of 
# explanations according to the specific goal or action they are intended to 
# support. From the perspective of individuals affected by automated 
# decision-making, we propose three aims for explanations: (1) to inform and 
# help the individual understand why a particular decision was reached, (2) to 
# provide grounds to contest the decision if the outcome is undesired, and (3) 
# to understand what would need to change in order to receive a desired result 
# in the future, based on the current decision-making model. We assess how each 
# of these goals finds support in the GDPR. We suggest data controllers should 
# offer a particular type of explanation, unconditional counterfactual 
# explanations, to support these three aims. These counterfactual explanations
# describe the smallest change to the world that can be made to obtain a 
# desirable outcome, or to arrive at the closest possible world, without 
# needing to explain the internal logic of the system.
#
# This is a optimization variant using the gradients of the model.
#
####
def wachter_gradient_uni_cf(sample, dataset, model, target=None, lb=None, lb_step=None, max_cfs=1000, full_random=False, distance='euclidean'):

    device = next(model.parameters()).device
    
    # prepare inputs and configuration
    shape = sample.reshape(1, -1).shape
    sample_t = numpy_to_torch(sample, device)

    y_cf = detach_to_numpy(model(sample_t.reshape(-1, *shape)))[0]
    label_cf = np.argmax(y_cf)
    if not target:
        target = 0
        if label_cf == 0:
            target += 1
    target_t = numpy_to_torch(np.array(target), device).long().reshape(1)
    
    def manhattan_dist(x, y):
        return torch.sum(torch.abs(x - y))
    
    def euclidean_dist(x, y):
        return torch.sqrt(torch.sum(torch.abs(x - y) ** 2))
    
    def dist(x, y):
        if distance == 'euclidean':
            return euclidean_dist(x, y)
        else:
            return manhattan_dist(x, y)
    
    def dist_labels(x, y):
        dist_loss = nn.CrossEntropyLoss()
        return dist_loss(x, y)

    def loss_fn(pred, cf):
        return lb * dist_labels(pred, target_t) ** 2 + dist(sample_t, cf)

    sample_cf = torch.rand(1, *shape).float().to(device)
    sample_cf = (sample_cf - torch.mean(sample_cf)) / torch.std(sample_cf)
    
    if not full_random:
        dataset_len = len(dataset)
        dataset_ridx = torch.randint(dataset_len, (1,))
        sample_cf = torch.from_numpy(dataset[dataset_ridx][0].reshape(1, *shape)).float().to(device)
    
    if not lb:
        lb = int(dist(sample_t, sample_cf) / 2)
    if not lb_step:
        lb_step = int(dist(sample_t, sample_cf) / 2)
    
    sample_cf.requires_grad_(True)
    
    optimizer = Adam([sample_cf], lr=0.001)
    
    cfs = []
    for _ in range(max_cfs):
        
        sample_cf.retain_grad()
    
        optimizer.zero_grad()
        pred = model(sample_cf)
        loss = loss_fn(pred, sample_cf)
        loss.backward(retain_graph=True)
        optimizer.step()
        
        lb += lb_step
        
        y_cf = detach_to_numpy(model(sample_cf))[0]
        sample_cf_np = detach_to_numpy(sample_cf.reshape(-1))
        cfs.append([sample_cf_np, loss.item(), y_cf])
        
        if np.abs(np.argmax(y_cf) - target) < 0.5:
            break
    
    cfs = sorted(cfs, key=lambda x: x[1])

    return cfs[0][0], cfs[0][2]
