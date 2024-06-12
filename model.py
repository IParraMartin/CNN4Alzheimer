import torch
import torch.nn as nn
from torchsummary import summary

"""
We build our convolutional network with the following parameters:
    - in_channels = 1 (since we are using grayscale)
    - classes = 4 (those are our possible outcomes)
    - n_filters = 8 (kind of like inspectors that go through the image)
    - dropout_rate = 0.5 (chance of dropping connections to prevent overfitting)
"""

class DementiaModel(nn.Module):
    def __init__(self, in_channels=1, n_filters=8, classes=4, dropout_rate=0.5):
        super().__init__()

        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_channels, 
                      n_filters, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, 
                         stride=2),
            nn.Dropout(dropout_rate),
            nn.Conv2d(n_filters, 
                      n_filters * 2, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, 
                         stride=2),
            nn.Dropout(dropout_rate),
            nn.Conv2d(n_filters * 2, 
                      n_filters * 4, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, 
                         stride=2),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(n_filters * 4 * 16 * 16, 
                      classes)
        )
    
    """
    Note that the last layer has a multiplication to calculate the input features.
    The image is downsampled by the MaxPool2d layers three times, each reducing
    the dimensionality by a factor of 2. Therefore, starting with 128x128, we get:
    128x128 -> 64x64 -> 32x32 -> 16x16. The final feature map size is 
    n_filters * 4 * 16 * 16 (n_filters multiplied by 4 due to the three convolutional layers).
    """

    def forward(self, x):
        return self.cnn_block(x)
    
if __name__ == "__main__":
    # initilaize the model
    model = DementiaModel()
    # make some random tensors to check the model:
    # B, D, W, H -> Batch size, color channels, width, and height
    x = torch.randn(64, 1, 128, 128)
    # pass them to see te output shape
    print(model(x).shape)
    # print the summary of the model
    summary(model, (1, 128, 128))

