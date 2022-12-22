import torch
import torch.nn as nn


class AbstractModel:
    
    def __init__(self, func, *args, **kwargs):
        self.func, self.args, self.kwargs = func, args, kwargs
    def __call__(self, *more_args, **more_kwargs):
        all_kwargs = {**self.kwargs, **more_kwargs}
        return self.func(*self.args, *more_args, **all_kwargs)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Define the first convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 9, kernel_size=3, stride=2),  # 1 input channel, 9 output channels, kernel size 3, stride 2
            nn.ReLU(inplace=True)  # Apply ReLU activation function
        )
        
        # Define the second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv1d(9, 18, kernel_size=3, stride=2),  # 9 input channels, 18 output channels, kernel size 3, stride 2
            nn.ReLU(inplace=True)  # Apply ReLU activation function
        )
        
        # Define the third convolutional layer
        self.conv3 = nn.Sequential(
            nn.Conv1d(18, 36, kernel_size=3, stride=2),  # 18 input channels, 36 output channels, kernel size 3, stride 2
            nn.ReLU(inplace=True)  # Apply ReLU activation function
        )
        
        # Define the first fully-connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(36 * 61, 100),  # 36 * 61 input features, 100 output features
            nn.Dropout(0.5),  # Apply dropout with rate 0.5
            nn.ReLU(inplace=True)  # Apply ReLU activation function
        )
        
        # Define the second fully-connected layer
        self.fc2 = nn.Sequential(
            nn.Linear(100, 2),  # 100 input features, 2 output features
            nn.Softmax()  # Apply softmax activation function
        )
        
    def forward(self, x):
        # Pass the input through the convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten the output of the convolutional layers for input to the fully-connected layers
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        # Pass the output through the fully-connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
