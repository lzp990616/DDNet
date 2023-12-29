# coding=utf-8
import pdb

import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder

from PIL import Image
import cv2
import random
import os
# import neuron
# import surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from model_net import *
from conv_dnm import *
# from spikingjelly.clock_driven import neuron, surrogate, functional
# from spikingjelly.activation_based import neuron, layer
# rom spikingjelly.clock_driven import surrogate, functional

# from spikingjelly.activation_based import neuron, surrogate,  functional
# from torchinfo import summary
bn_momentum = 0.1


class ConvBlock0(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, dilation_rate=(2, 2), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvBlock1(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.ReLU(nplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvBlock2(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dilation_rate=2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=dilation_rate,
                      dilation=(dilation_rate, dilation_rate),
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.ReLU(nplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)



class DDNet(nn.Module):
    def __init__(self, device=DEVICE, m=10, flag=0):
        super(DDNet, self).__init__()

        # encoder
        self.enco1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )

        print("Build enceder done..")

        # between encoder and decoder
        self.midco = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )

        self.out7 = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=32),
            nn.Conv2d(512, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )
        
        self.out7_dnm = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=32),
            conv_dnm(512, 1, 10),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )
        
        
        # decoder

        self.deco1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )
        self.out6 = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=16),
            nn.Conv2d(512, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )

        self.deco2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.out5 = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=8),
            nn.Conv2d(256, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )

        self.out5_dnm = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=32),
            conv_dnm(256, 1, 10),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )


        self.deco3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )
        self.out4 = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=4),
            nn.Conv2d(128, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )
        
        self.out4_dnm = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=32),
            conv_dnm(128, 1, 10),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )

        self.deco4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
        )
        self.out3 = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )
        
        self.out3_dnm = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=32),
            conv_dnm(64, 1, 5),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )
        

        self.deco5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            # nn.Conv2d(64, 1, kernel_size=1, stride=1),
            # nn.BatchNorm2d(1, momentum=bn_momentum),
            # nn.ReLU(),
        )
        self.conv2d = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
        self.Dnm_conv2d = conv_dnm(64, 1, m)
        # ICModel
        # self.ICModel_conv = ICModel(64, 1)
        self.flag = flag


    def forward(self, x):
        id = []
        # encoder
        
        x = self.enco1(x)  # 1-2
        conv_2 = x
        x, id1 = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1, return_indices=True)  # 保留最大值的位置
        id.append(id1)
        x = self.enco2(x)  # 2-4
        conv_4 = x
        x, id2 = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1, return_indices=True)
        id.append(id2)
        x = self.enco3(x)  # 5-7
        conv_7 = x

        x, id3 = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1, return_indices=True)
        id.append(id3)
        x = self.enco4(x)  # 8-10
        conv_10 = x
        x, id4 = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1, return_indices=True)
        id.append(id4)
        x = self.enco5(x)  # 11-13
        conv_13 = x
        x, id5 = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1, return_indices=True)
        id.append(id5)

        # between encoder and decoder
        x = self.midco(x)  # 14-16
        Out7 = self.out7(x)

        # decoder
        x = F.max_unpool2d(x, id[4], kernel_size=2)
        concat_1 = torch.cat((x, conv_13), axis=1)  # 拼接16池化后的结果和13

        x = self.deco1(concat_1)    # 17-19
        Out6 = self.out6(x)
        x = F.max_unpool2d(x, id[3], kernel_size=(2, 2))
        concat_2 = torch.cat((x, conv_10), axis=1)  # 拼接19池化后的结果和10

        x = self.deco2(concat_2)    # 20-22
        Out5 = self.out5(x)
        x = F.max_unpool2d(x, id[2], kernel_size=(2, 2))
        concat_3 = torch.cat((x, conv_7), axis=1)  # 拼接22池化后的结果和7

        x = self.deco3(concat_3)    # 23-25
        Out4 = self.out4(x)
        x = F.max_unpool2d(x, id[1], kernel_size=(2, 2))
        concat_4 = torch.cat((x, conv_4), axis=1)  # 拼接25池化后的结果和4

        x = self.deco4(concat_4)    # 26-27
        Out3 = self.out3(x)
        # Out3 = self.out3_dnm(x)
        x = F.max_unpool2d(x, id[0], kernel_size=(2, 2))
        concat_5 = torch.cat((x, conv_2), axis=1)  # 拼接27池化后的结果和2

        x = self.deco5(concat_5)    # 28-29
        # pdb.set_trace()
        if self.flag == 1:
            x = self.Dnm_conv2d(x)  # DNM
        else:
            x = self.conv2d(x)      # DSegNet
        Out1 = x
        
        Out2 = F.sigmoid(x)    
        # x = F.sigmoid(x)
        
        # sigmod, 
        return Out1, Out3, Out4, Out5, Out6, Out7
        # out2 + out_m - out_f == out1 (2+9-8=1)  ,out_f,out_m




# model = DDNet().cuda()
# # print(model)
# #
# summary(model, (3, 384, 384))
# print(model(torch.rand(4, 3, 384, 384).cuda()).shape)
