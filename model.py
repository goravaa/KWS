# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """
    A building block for efficient models.
    Performs a depthwise convolution followed by a pointwise convolution.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = F.relu(self.bn1(x))
        x = self.pointwise(x)
        x = F.relu(self.bn2(x))
        return x

class KWSLiteCNN(nn.Module):
    """
    A lightweight, fast CNN for keyword spotting, optimized for CPU inference.
    """
    def __init__(self, num_classes):
        super(KWSLiteCNN, self).__init__()
        # Initial standard convolution
        self.init_conv = nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.init_bn = nn.BatchNorm2d(16)

        # Sequence of efficient separable convolutions
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=(2, 1)),
            DepthwiseSeparableConv(32, 32),
            DepthwiseSeparableConv(32, 64, stride=(2, 1)),
            DepthwiseSeparableConv(64, 64),
        )
        
        # Adaptive pooling and final classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes)
                )

    def forward(self, x):
        # Input shape: (batch, channels=1, n_mfcc, time_steps)
        x = F.relu(self.init_bn(self.init_conv(x)))
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x

