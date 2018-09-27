import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
def Alexnet():
    model = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride = 4)
        nn.LocalResponseNorm(size = 96)
        nn.MaxPool2d(kernel_size = 3, stride = 2)
        nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, stride = 1)
        nn.LocalResponseNorm(256)
        nn.MaxPool2d(kernel_size = 3, stride = 2)
        nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 1)
        nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, stride = 1)
        nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, stride = 1)
        nn.MaxPool2d(kernel_size = 3, stride = 2)
        nn.Linear()
    )
'''

class SiamFC(nn.Module):
    def __init__(self):
        super(SiamFC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True)
        )

        self.branch = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )

    def forward(self, z, x):
        z = self.branch(z)
        x = self.branch(x)

        out = self.xcorr(z, x)

        return out

    def xcorr(self, z, x):
        return F.conv2d(z, x)

    