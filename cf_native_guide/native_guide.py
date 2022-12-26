import torch

import numpy as np

from captum.attr import GradientShap

from sklearn.neighbors import NearestNeighbors


def detach_to_numpy(data):
    # move pytorch data to cpu and detach it to numpy data
    return data.cpu().detach().numpy()


def numpy_to_torch(data, device):
    # convert numpy array to pytorch and move it to the device
    return torch.from_numpy(data).float().to(device)


####
# Instance-based Counterfactual Explanations for Time Series Classification
#
# https://arxiv.org/abs/2009.13211
#
# In recent years, there has been a rapidly expanding focus on explaining the 
# predictions made by black-box AI systems that handle image and tabular data. 
# However, considerably less attention has been paid to explaining the predictions 
# of opaque AI systems handling time series data. In this paper, we advance a 
# novel model-agnostic, case-based technique -- Native Guide -- that generates 
# counterfactual explanations for time series classifiers. Given a query time 
# series, Tq, for which a black-box classification system predicts class, c, a 
# counterfactual time series explanation shows how Tq could change, such that 
# the system predicts an alternative class, c′. The proposed instance-based 
# technique adapts existing counterfactual instances in the case-base by 
# highlighting and modifying discriminative areas of the time series that 
# underlie the classification. Quantitative and qualitative results from two 
# comparative experiments indicate that Native Guide generates plausible, 
# proximal, sparse and diverse explanations that are better than those produced 
# by key benchmark counterfactual methods.
#
####
def native_guide_uni_cf(sample, dataset, model, weight_function=GradientShap, iterate=None, sub_len=1):
    
    device = next(model.parameters()).device
    def model_predict(data):
        return detach_to_numpy(model(numpy_to_torch(data, device)))
    
    # prepare inputs and configuration
    shape = sample.reshape(1, -1).shape
    time_series_data = np.array([x[0] for x in dataset])
    
    # set iterate to length of time series -1
    if not iterate:
        iterate = shape[1]
    
    # get predictions for the sample and the dataset
    predictions_for_data = model_predict(time_series_data.reshape(-1, *shape))
    predictions_for_sample = model_predict(sample.reshape(1, *shape))
    label_data = np.argmax(predictions_for_data, axis=1)
    label_sample = np.argmax(predictions_for_sample)
    
    # get indices for the not predicted class as train instances
    train_indices = label_data != label_sample
    time_series_candidates = time_series_data[train_indices,...]
    label_data_candidates = label_data[train_indices,...]
    
    # get nearest neighbor with a different class and then set it as the native guide
    k_for_candidates = min(int(shape[1] * 0.25), len(time_series_candidates) - 1)

    neigh = NearestNeighbors(n_neighbors=1, metric='euclidean')
    neigh.fit(time_series_candidates)
    candidate_neighbors = neigh.kneighbors(sample.reshape(1, -1), k_for_candidates)
    reference_label = label_sample

    native_guide = None
    cf_label = None
    for x in candidate_neighbors[1][0][1:]:
        if not np.all(label_data_candidates[x] == reference_label):
            native_guide = time_series_candidates[x]
            cf_label = label_data_candidates[x]
            break

    # get weights to change time series accordingly with subsequence around
    weights = weight_function(model)
    baselines = numpy_to_torch(time_series_data.reshape(-1, *shape), device)
    attributions = weights.attribute(numpy_to_torch(native_guide.reshape(1, *shape), device), baselines, target=int(cf_label))

    # find most influential subarray in attributions for a certain length
    def find_most_influential_array(length):
        attributions_flatted = detach_to_numpy(attributions.reshape(-1))
        max_len = len(attributions_flatted)
        candidates = []
        for i in range(max_len):
            subarray_sum = np.sum(attributions_flatted[i:i+length])
            candidates.append((i, subarray_sum))
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

        return candidates[0][0]
    
    # take starting point as most influential point and exchange sample to native guide
    for i in range(iterate):
        cf = sample.copy()
        starting_idc = find_most_influential_array(i+sub_len)
        cf[starting_idc:starting_idc+(i+sub_len)] = native_guide[starting_idc:starting_idc+(i+sub_len)]

        y_cf = model_predict(cf.reshape(1, *shape)).reshape(-1)
        if cf_label == np.argmax(y_cf):
            break

    return cf, y_cf

