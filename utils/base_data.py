import numpy as np

from sktime.datasets import load_UCR_UEA_dataset

from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import OneHotEncoder


class UCRDataset(Dataset):

    def __init__(self, X, y, name=None, mapping=None):
        # Save input data and metadata as attributes of the UCRDataset instance
        self.X = X
        self.y = y
        self.name = name
        self.mapping = mapping
        
    def __repr__(self):
        # Return a string representation of the UCRDataset instance, including its name and shape
        return f'<UCRDataset {self.name} {self.X.shape}>'
    
    def __len__(self):
        # Return the length of the UCRDataset, which is the number of time series in the dataset
        return len(self.X)
    
    def __getitem__(self, idx):
        # Return the input time series and label at the given index
        inputs = self.X[idx]
        label = self.y[idx]
        
        return inputs, label


def get_UCR_dataset(dataset_name='FordA', split='train'):
    
    # Load and process the specified UCR/UEA dataset
    X, y = load_UCR_UEA_dataset(name=dataset_name, split=split)
    
    # One-hot encode the labels
    encoder = OneHotEncoder(categories='auto', sparse=False)
    y = encoder.fit_transform(np.expand_dims(y, axis=-1))
    
    # Reshape the input time series data into a single list of time series
    X = np.array([x.to_numpy() for z in X.to_numpy() for x in z])

    # Create an instance of the UCRDataset class
    dataset = UCRDataset(X=X, y=y, name=dataset_name, mapping=encoder.categories_)

    return dataset


def get_UCR_dataloader(dataset_name='FordA', split='train', batch_size=256, shuffle=True):
    
    # Load the specified UCR/UEA dataset
    dataset = get_UCR_dataset(dataset_name, split)

    # Create a dataloader for the dataset with the specified batch size and shuffle behavior
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset


data_train = get_UCR_dataloader()
print(data_train)
