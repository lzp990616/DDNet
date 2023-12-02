import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
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

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Soma(nn.Module):
    def __init__(self, k, qs):
        super(Soma, self).__init__()
        self.params = nn.ParameterDict({'k': nn.Parameter(k)})
        self.params.update({'qs': nn.Parameter(qs)})

    def forward(self, x):
        y = 1 / (1 + torch.exp(-self.params['k'] * (x - self.params['qs'])))
        return y


class Membrane(nn.Module):
    def __init__(self):
        super(Membrane, self).__init__()

    def forward(self, x):
        x = torch.sum(x, 1) 
        return x


class Dendritic(nn.Module):
    def __init__(self):
        super(Dendritic, self).__init__()

    def forward(self, x):
        x = torch.prod(x, 2)  # prod or sum 
        return x


class Synapse(nn.Module):

    def __init__(self, w, q, k):
        super(Synapse, self).__init__()
        self.params = nn.ParameterDict({'w': nn.Parameter(w)})
        self.params.update({'q': nn.Parameter(q)})
        self.params.update({'k': nn.Parameter(k)})

    def forward(self, x):
        num, _ = self.params['w'].shape
        x = torch.unsqueeze(x, 1)
        x = x.repeat((1, num, 1))  # copy m
        y = 1 / (1 + torch.exp(
            torch.mul(-self.params['k'], (torch.mul(x, self.params['w']) - self.params['q']))))  # k*(w*x-q)

        return y


class BASE_DNM(nn.Module):
    def __init__(self, dim, M, kv=5, qv=0.3):  # , device=torch.device('cuda:0')):

        w = torch.rand([M, dim])  # .to(device)
        q = torch.rand([M, dim])  # .to(device)
        # k = torch.tensor(kv)
        # qs = torch.tensor(qv)
        k = torch.rand(1)
        qs = torch.rand(1)

        super(BASE_DNM, self).__init__()
        self.model = nn.Sequential(
            Synapse(w, q, k),
            Dendritic(),
            Membrane(),
            Soma(k, qs)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class DNM_Linear(nn.Module):
    def __init__(self, input_size, out_size, M, k=0.5, qs=0.1, activation=F.relu):
        super(DNM_Linear, self).__init__()

        DNM_W = torch.rand([out_size, M, input_size])  # .cuda() # [size_out, M, size_in]
        dendritic_W = torch.rand([input_size])  # .cuda() # size_out, M, size_in]
        membrane_W = torch.rand([M])  # .cuda() # size_out, M, size_in]
        q = torch.rand([out_size, M, input_size])  # .cuda()
        torch.nn.init.constant_(q, qs)  # 设置q的初始值
        k = torch.tensor(k).to(DEVICE)
        qs = torch.tensor(qs).to(DEVICE)

        self.params = nn.ParameterDict({'DNM_W': nn.Parameter(DNM_W)})
        self.params.update({'q': nn.Parameter(q)})
        self.params.update({'dendritic_W': nn.Parameter(dendritic_W)})
        self.params.update({'membrane_W': nn.Parameter(membrane_W)})
        self.k = k
        self.qs = qs
        self.activation = activation

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['DNM_W'].shape
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 2)
        x = x.repeat(1, out_size, M, 1)
        
        x = torch.relu(torch.mul(self.k, (torch.mul(x, self.params['DNM_W']) - self.params['q'])))

        # x = torch.mul(x, self.params['DNM_W'])
        # x = F.relu(self.k * (x - self.params['q']))

        # Dendritic
        # x = torch.mul(x, self.params['dendritic_W'])
        # x = x * self.params['dendritic_W']
        x = torch.sum(x, 3)
        # x = torch.sigmoid(x)
        # x = F.relu(x)
        pdb.set_trace()
        # Membrane
        # x = torch.mul(x, self.params['membrane_W'])
        # x = x * self.params['membrane_W']
        
        x = torch.sum(x, 2)

        # Soma
        if self.activation != None:  # sigmode or relu
            x = self.activation(self.k * (x - self.qs))

        return x


class DNM_Conv(nn.Module):
    def __init__(self, input_size, out_size, M, activation=F.relu):
        super(DNM_Conv, self).__init__()
        DNM_W = torch.rand([out_size, M, input_size])  # .cuda() # [size_out, M, size_in]  [num_class, M, 512 * 3 * 3]
        DNM_q = torch.rand([out_size, M, input_size])
        dendritic_W = torch.rand([input_size])  # .cuda() # size_out, M, size_in]
        membrane_W = torch.rand([M])  # .cuda() # size_out, M, size_in]
        # q = torch.rand([out_size, M, input_size])  # .cuda()
        # torch.nn.init.constant_(q, qs)  # 设置q的初始值
        # torch.nn.init.normal_(dendritic_W, mean=0,std=1)
        # torch.nn.init.normal_(dendritic_W, mean=0,std=1)
        # torch.nn.init.normal_(dendritic_W, mean=0,std=1)
        # k = torch.tensor(k).to(DEVICE)
        qs = torch.rand(1)
        qs = torch.tensor(qs).to(DEVICE)
        self.params = nn.ParameterDict({'DNM_W': nn.Parameter(DNM_W)})
        self.params.update({'q': nn.Parameter(DNM_q)})
        self.params.update({'dendritic_W': nn.Parameter(dendritic_W)})
        self.params.update({'membrane_W': nn.Parameter(membrane_W)})
        # self.k = k
        self.qs = qs
        self.activation = activation
        
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['DNM_W'].shape
        # pdb.set_trace()
        # x.shape (batch*64*256*256)    (0,1,2,3) ==> (0,2,3,1) => (0,1,2,3) (permute(0,3,1,2))
        x = torch.permute(x, (0, 2, 3, 1))  # torch.Size([8, 256, 256, 64])
        x = torch.unsqueeze(x, 3)
        x = torch.unsqueeze(x, 4)
        
        x = self.norm1(x) # norm
        
        x = x.repeat(1, 1, 1, out_size, M, 1)
        # x = F.relu(torch.mul(self.k, (torch.mul(x, self.params['DNM_W']) - self.params['q'])))
        x = F.relu(torch.mul(x, self.params['DNM_W']) - self.params['q'])
        # x = torch.mul(x, self.params['DNM_W'])
        # x = F.relu(self.k * (x - self.params['q']))
        
        
        
        
        # Dendritic
        x = self.norm2(x) # norm、
        
        # x = torch.mul(x, self.params['dendritic_W'])
        # x = x * self.params['dendritic_W']
        x = torch.sum(x, 5)
        # x = torch.sigmoid(x)
        #x = F.relu(x)
        # pdb.set_trace()
        # Membrane
        # x = torch.mul(x, self.params['membrane_W'])
        # x = x * self.params['membrane_W']
        x = torch.sum(x, 4)
        x = torch.permute(x, (0, 3, 1, 2))
          
        # Soma
        if self.activation != None:
            #x = self.activation(self.k * (x - self.qs))
            x = self.activation(x - self.qs)
        return x


class ConvDNMBase(nn.Module):
    def __init__(self, in_channal=64, out_channal=1, m=10):
        super(ConvDNMBase, self).__init__()
        self.inchannel = 64
        
        # self.k = DNM_Conv(in_channal, out_channal, 10, activation=None)
        self.DNM_Conv1 = DNM_Conv(in_channal, out_channal, m, activation=None)
        # self.DNM_Linear2 = DNM_Linear(1024, out_channal, 10, activation=None)
    def forward(self, x):
        out = self.DNM_Conv1(x)
        return out


class conv_dnm(nn.Module):  
    def __init__(self,  input_size, out_size, M):
        super(conv_dnm, self ).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1)
        self.Dnm_conv2d = ConvDNMBase(input_size, out_size, M)
        # self.conv2d = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        # input_size = x.size(0)  # batch_size
        # x = self.conv1(x)
        x = self.Dnm_conv2d(x)
        # x = self.conv2d(x)
        # out0 = F.sigmoid(x)

        return x


# model = conv_dnm()
# # # print(model)
# # #
# summary(model, (3, 384, 384))
