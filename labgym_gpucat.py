import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class GPUCat(nn.Module):
    def __init__(self, num_behaviors) -> None:
        """Define model architecture."""
        super().__init__()
        # Animal boxes from detectron can have different shapes
        # Two options: Resize before hand or run CNN and the use adaptive pooling to fixed size for linear layer

        # Stealing Henry's logic???
        # kernel = -1
        # if framesize < 500:
        #     kernel = 3
        # elif framesize < 1000:
        #     kernel = 5
        # elif framesize < 1500:
        #     kernel = 7
        # else:
        #     kernel = 9

        self.num_behaviors = num_behaviors

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5), stride=(2,2), padding=2)
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=8, kernel_size=(5,5), stride=(2,2), padding=2)
        # 8 out channels at (H, W) output size, added to ensure that despite variable image input sizes from cropping
        # always the same size for the final layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc_1 = nn.Linear(in_features=8*6*6, out_features=self.num_behaviors)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        # torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            torch.nn.init.normal_(conv.weight.data, 0.0, sqrt((1.0/(5*5*conv.in_channels))))
            torch.nn.init.constant_(conv.bias, 0.0)
        # Initialize parameters for final fully connected layer
        torch.nn.init.normal_(self.fc_1.weight, 0.0, sqrt((1.0/(self.fc_1.in_features))))
        torch.nn.init.constant_(self.fc_1.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        N, C, H, W = x.shape

        # Pass x through conv, relu, then maxpool w/ no maxpool for last conv layer
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # Adaptive pool to ensure same size output for final linea layer
        x = self.adaptive_pool(x)

        # You want to flatten out each N from (C, H, W) to C*H*W
        # Start at dim=1 to skip N
        x = torch.flatten(x, start_dim=1)
        # Note no activation for fully connected layer 1
        x = self.fc_1(x)
        # Returns predictions as [batch size, label space=2]
        return x