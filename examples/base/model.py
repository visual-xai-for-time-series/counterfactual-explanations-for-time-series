import torch
import torch.nn as nn


class AbstractModel:

    def __init__(self, func, *args, **kwargs):
        self.func, self.args, self.kwargs = func, args, kwargs

    def __call__(self, *more_args, **more_kwargs):
        all_kwargs = {**self.kwargs, **more_kwargs}
        return self.func(*self.args, *more_args, **all_kwargs)


class SimpleCNN(nn.Module):
    def __init__(self, output_channels=2, input_length=500):
        super(SimpleCNN, self).__init__()

        # Define the first convolutional layer with batch normalization
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # Define the second convolutional layer with batch normalization
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # Define the third convolutional layer with batch normalization
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Define the fourth convolutional layer
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Calculate the output size after convolutions
        conv_output_size = self._calculate_conv_output_size(input_length)
        
        # Define the first fully-connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        
        # Define the second fully-connected layer
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Define the output layer with softmax
        self.fc3 = nn.Sequential(
            nn.Linear(128, output_channels),
            nn.Softmax(dim=1)
        )

    def _calculate_conv_output_size(self, input_length):
        """Calculate the flattened output size after all convolutional layers."""
        # Simulate forward pass through conv layers
        x = torch.randn(1, 1, input_length)
        x = self.conv1[0](x)  # Just the conv layer
        x = self.conv2[0](x)
        x = self.conv3[0](x)
        x = self.conv4[0](x)
        return x.shape[1] * x.shape[2]  # channels * length

    def forward(self, x):
        # Pass the input through the convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Flatten the output of the convolutional layers for input to the fully-connected layers
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        # Pass the output through the fully-connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class SimpleCNNMulti(nn.Module):
    """
    CNN model designed for multivariate time series classification.
    Supports configurable input channels for handling multiple time series variables.
    """
    def __init__(self, input_channels=1, output_channels=2, sequence_length=500):
        super(SimpleCNNMulti, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Define the first convolutional layer
        self.conv1 = nn.Sequential(
            # input_channels input channels, 16 output channels, kernel size 5, stride 2
            nn.Conv1d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # Define the second convolutional layer
        self.conv2 = nn.Sequential(
            # 16 input channels, 32 output channels, kernel size 5, stride 2
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # Define the third convolutional layer
        self.conv3 = nn.Sequential(
            # 32 input channels, 64 output channels, kernel size 5, stride 2
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Define the fourth convolutional layer
        self.conv4 = nn.Sequential(
            # 64 input channels, 128 output channels, kernel size 3, stride 2
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Calculate the size of flattened features after convolutions
        # This is a rough calculation - actual size may vary based on input length
        self.feature_size = self._calculate_conv_output_size(sequence_length)
        
        # Global Average Pooling to make the model more robust to sequence length variations
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Define fully-connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(128, 256),  # Use 128 due to global avg pooling
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Output layer
        self.fc3 = nn.Linear(128, output_channels)

    def _calculate_conv_output_size(self, input_length):
        """Calculate the output size after all convolutional layers."""
        # Simulate the forward pass through conv layers to get output size
        x = torch.randn(1, self.input_channels, input_length)
        x = self.conv1[0](x)  # Just the conv layer, not the full sequential
        x = self.conv2[0](x)
        x = self.conv3[0](x)
        x = self.conv4[0](x)
        return x.shape[1] * x.shape[2]  # channels * length

    def forward(self, x):
        # Pass the input through the convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Apply global average pooling
        x = self.global_avg_pool(x)  # (batch, 128, 1)
        
        # Flatten for fully connected layers
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # (batch, 128)

        # Pass through fully-connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def create_cnn_model(input_channels=1, sequence_length=500, num_classes=2):
    """
    Create a CNN model suitable for time series classification.
    
    Parameters:
    - input_channels: Number of input channels (1 for univariate, >1 for multivariate)
    - sequence_length: Length of input time series
    - num_classes: Number of output classes
    
    Returns:
    - PyTorch model
    """
    if input_channels == 1:
        # Use SimpleCNN for univariate data
        model = SimpleCNN(output_channels=num_classes)
    else:
        # Use SimpleCNNMulti for multivariate data
        model = SimpleCNNMulti(
            input_channels=input_channels,
            output_channels=num_classes,
            sequence_length=sequence_length
        )
    
    return model
