#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

class Simplify_Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Simplify_Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1,stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1,stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1,stride=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU(inplace=True)

        self.upconv1 = nn.ConvTranspose2d(in_channels=16,out_channels=16,kernel_size=4,padding=1,stride=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU(inplace=True)

        self.upconv2 = nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=4,padding=1,stride=2)
        self.bn5 = nn.BatchNorm2d(16)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv_last = nn.Conv2d(in_channels=32,out_channels=num_classes,kernel_size=1,stride=1)


    def forward(self, x):

        x1 = self.relu1(self.bn1(self.conv1(x)))
        x2 = self.relu2(self.bn2(self.conv2(x1)))
        x3 = self.relu3(self.bn3(self.conv3(x2)))

        up1 = torch.cat([x2,self.relu4(self.bn4(self.upconv1(x3)))],dim=1)
        up2 = torch.cat([x1,self.relu5(self.bn5(self.upconv2(up1)))],dim=1)

        up3 = self.conv_last(up2)

        out = interpolate(up3, scale_factor=2, mode='bilinear', align_corners=False)

        return out
