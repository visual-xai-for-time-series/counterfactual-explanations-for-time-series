import torch
import numpy as np
import pandas as pd
import os
import urllib.request
import zipfile
from pathlib import Path

from aeon.datasets import load_classification
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from torch.utils.data import DataLoader
import scipy.sparse as sp


def collate_sparse(batch):
    xs, ys = zip(*batch)

    def to_tensor(a):
        if sp.isspmatrix(a):
            arr = a.toarray()
        else:
            arr = np.asarray(a)
        # ensure float32 tensor and remove extraneous dims
        return torch.from_numpy(arr.squeeze().astype(np.float32))

    xs_t = [to_tensor(x) for x in xs]
    ys_t = [to_tensor(y) for y in ys]

    return torch.stack(xs_t), torch.stack(ys_t)


class TimeSeriesDataset(Dataset):

    def __init__(self, X, y, name=None, mapping=None):
        # Save input data and metadata as attributes of the TimeSeriesDataset instance
        self.X = X
        self.y = y
        self.name = name
        self.mapping = mapping

        self.min = np.amin(X, axis=-1)
        self.max = np.amax(X, axis=-1)

        self.std = np.std(X, axis=-1)

        self.X_shape = X.shape
        self.y_shape = y.shape

    def __repr__(self):
        # Return a string representation of the UCRDataset instance, including its name and shape
        return f'<UCRDataset {self.name} {self.X.shape} {self.y.shape}>'

    def __len__(self):
        # Return the length of the UCRDataset, which is the number of time series in the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # Return the input time series and label at the given index
        inputs = self.X[idx]
        label = self.y[idx]

        return inputs, label


def get_UCR_UEA_dataset(dataset_name='FordA', split='train'):

    # Load and process the specified UCR/UEA dataset
    X, y = load_classification(name=dataset_name, split=split)

    # One-hot encode the labels
    encoder = OneHotEncoder(categories='auto', sparse_output=False)
    y = encoder.fit_transform(np.expand_dims(y, axis=-1))

    # Create an instance of the TimeSeriesDataset class
    dataset = TimeSeriesDataset(X=X, y=y, name=dataset_name, mapping=encoder.categories_)

    return dataset


def get_UCR_UEA_dataloader(dataset_name='FordA', split='train', batch_size=256, shuffle=True):

    # Load the specified UCR/UEA dataset
    dataset = get_UCR_UEA_dataset(dataset_name, split)

    # Create a dataloader for the dataset with the specified batch size and shuffle behavior
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_sparse)
    return dataloader, dataset


# Move example usage into a main guard
def main():
    # Original UCR/UEA datasets
    print("=== UCR/UEA Datasets ===")
    data_train_loader, dataset_train = get_UCR_UEA_dataloader()
    print(f"FordA: {dataset_train}")
    print(f"Sample shape: {dataset_train[0][0].shape}")

    data_train_loader, dataset_train = get_UCR_UEA_dataloader('SpokenArabicDigits')
    print(f"SpokenArabicDigits: {dataset_train}")
    print(f"Sample shape: {dataset_train[0][0].shape}")


if __name__ == "__main__":
    main()
