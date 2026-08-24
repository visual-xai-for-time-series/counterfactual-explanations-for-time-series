import os
import torch
import numpy as np
import scipy.sparse as sp

from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import tsc_zenodo
try:
    # Public as of aeon >= ~1.5; older pinned releases (examples/requirements.txt
    # currently pins aeon==1.3.0) only expose it from the private loader module.
    from aeon.datasets import download_dataset
except ImportError:
    from aeon.datasets._data_loaders import download_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader

# Local cache for datasets that aeon can't fetch through its usual zenodo-backed
# registry (see _load_classification_with_fallback below). Kept outside the aeon
# install so it works regardless of whether site-packages is writable, and
# outside the repo's tracked files (matches the existing `data/` gitignore entry).
_DATASET_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'data')


def _load_classification_with_fallback(dataset_name, split):
    """Load a UCR/UEA dataset by name, backing off to a direct download when
    aeon's registry doesn't know it.

    aeon's `load_classification` only downloads datasets listed in its own
    tsc_zenodo/tsr_zenodo dictionaries; archive datasets that have dropped out
    of that list between aeon releases (e.g. FaultDetectionA, present on
    timeseriesclassification.com but absent from tsc_zenodo as of aeon 1.5)
    raise a ValueError instead of downloading. In that case, fetch the dataset
    directly from timeseriesclassification.com/aeon-toolkit and load it from
    the local copy instead.

    `load_equal_length`/`load_no_missing` ask for the archive's pre-padded,
    imputed variant when one exists, so unequal-length collections (e.g.
    SpokenArabicDigits, where each sample has a different number of time
    steps) come back as a plain (N, C, T) array instead of a ragged list.
    """
    try:
        return load_classification(
            name=dataset_name, split=split,
            load_equal_length=True, load_no_missing=True,
        )
    except ValueError:
        if dataset_name in tsc_zenodo:
            raise  # a real failure (bad split, network error, ...), not a missing registry entry

        # download_dataset() pulls the .ts files straight from
        # timeseriesclassification.com rather than going through tsc_zenodo,
        # extracting to <save_path>/<dataset_name>/. Re-running load_classification
        # with extract_path pointed at that parent directory picks it up locally.
        dataset_dir = download_dataset(dataset_name, save_path=_DATASET_CACHE_DIR)
        extract_path = os.path.dirname(dataset_dir)
        return load_classification(
            name=dataset_name, split=split, extract_path=extract_path,
            load_equal_length=True, load_no_missing=True,
        )


def _stack_ragged(X):
    """Zero-pad a collection of per-sample arrays to a common length.

    aeon represents unequal-length collections as a plain Python list of
    (channels, length) arrays rather than a single ndarray. Passing that
    straight to np.amin/amax/std (as TimeSeriesDataset does) raises "setting
    an array element with a sequence" instead of just working. This is a
    fallback for datasets where even the equal-length ("_eq") archive variant
    requested above isn't available; well-formed (N, C, T) input is returned
    unchanged.
    """
    if isinstance(X, np.ndarray) and X.dtype != object:
        return X

    max_len = max(sample.shape[-1] for sample in X)
    padded = np.zeros((len(X), *X[0].shape[:-1], max_len), dtype=np.result_type(*(s.dtype for s in X)))
    for i, sample in enumerate(X):
        padded[i, ..., :sample.shape[-1]] = sample
    return padded


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
        # Pad any remaining ragged (unequal-length) samples to a common length
        # before treating X as a regular (N, C, T) array below.
        X = _stack_ragged(X)

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

    # Load and process the specified UCR/UEA dataset (falls back to a direct
    # download for datasets aeon's registry doesn't know about, and requests
    # equal-length variants so unequal-length datasets don't come back ragged)
    X, y = _load_classification_with_fallback(dataset_name, split)

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


if __name__ == '__main__':
    print('=== UCR/UEA Datasets ===')
    _, dataset_train = get_UCR_UEA_dataloader()
    print(f'FordA: {dataset_train}')
    print(f'Sample shape: {dataset_train[0][0].shape}')

    _, dataset_train = get_UCR_UEA_dataloader('SpokenArabicDigits')
    print(f'SpokenArabicDigits: {dataset_train}')
    print(f'Sample shape: {dataset_train[0][0].shape}')
