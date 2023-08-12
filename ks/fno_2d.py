# Adapted from official FNO implementation: https://github.com/neuraloperator/neuraloperator

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce

import scipy.io

import sys
from utilities import *

torch.manual_seed(0)
np.random.seed(0)


def compl_mul2d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixy,ioxy->boxy", a, b)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        size1 = x.shape[-2]
        size2 = x.shape[-1]

        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3], norm="ortho")

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, size1, size2//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)


        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(size1, size2), dim=[2,3], norm="ortho")
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, in_dim, out_dim, modes1, modes2, width, pad_amount=None, pad_dim='1'):
        """
            `pad_dim` can be '1', '2', or 'both', and this decides which of the two space dimensions to pad
            `pad_amount` is a tuple that determines how much to pad each dimension by, if `pad_dim`
            specifies that dimension should be padded.
        """
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2

        self.pad_amount = pad_amount
        self.pad_dim = pad_dim

        self.width_list = [width*2//4, width*3//4, width*4//4, width*4//4, width*5//4]
        self.grid_dim = 2

        self.fc0 = nn.Linear(in_dim+self.grid_dim, self.width_list[0])

        self.conv0 = SpectralConv2d(self.width_list[0], self.width_list[1], self.modes1*4//4, self.modes2*4//4)
        self.conv1 = SpectralConv2d(self.width_list[1], self.width_list[2], self.modes1*3//4, self.modes2*3//4)
        self.conv2 = SpectralConv2d(self.width_list[2], self.width_list[3], self.modes1*2//4, self.modes2*2//4)
        self.conv3 = SpectralConv2d(self.width_list[3], self.width_list[4], self.modes1*2//4, self.modes2*2//4)
        self.w0 = nn.Conv1d(self.width_list[0], self.width_list[1], 1)
        self.w1 = nn.Conv1d(self.width_list[1], self.width_list[2], 1)
        self.w2 = nn.Conv1d(self.width_list[2], self.width_list[3], 1)
        self.w3 = nn.Conv1d(self.width_list[3], self.width_list[4], 1)

        self.fc1 = nn.Linear(self.width_list[4], self.width_list[4]*2)
        self.fc2 = nn.Linear(self.width_list[4]*2, self.width_list[4]*2)
        self.fc3 = nn.Linear(self.width_list[4]*2, out_dim)

    def forward(self, x):

        batchsize = x.shape[0]
        size_x, size_y= x.shape[1], x.shape[2]
        grid = self.get_grid(size_x, size_y, batchsize, x.device)

        x = torch.cat((x, grid.permute(0, 2, 3, 1)), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        if self.pad_amount: # pad the domain if input is non-periodic
            if self.pad_dim == '1':
                x = x.permute(0, 1, 3, 2) # new shape: (batch, dim, dom_size2, dom_size1)
                x = F.pad(x, [0,self.pad_amount[0]])
                x = x.permute(0, 1, 3, 2)
                size_x += self.pad_amount[0]
            elif self.pad_dim == '2':
                x = F.pad(x, [0,self.pad_amount[1]])
                size_y += self.pad_amount[1]
            elif self.pad_dim == 'both':
                x = x.permute(0, 1, 3, 2) # new shape: (batch, timesteps, dim, dom_size2, dom_size1)
                x = F.pad(x, [0,self.pad_amount[0]])
                x = x.permute(0, 1, 3, 2)
                x = F.pad(x, [0,self.pad_amount[1]])
                size_x += self.pad_amount[0]
                size_y += self.pad_amount[1]

        x1 = self.conv0(x)
        x2 = self.w0(x.reshape((batchsize, self.width_list[0], size_x * size_y))).view(batchsize, self.width_list[1], size_x, size_y)
        x = x1 + x2
        x = F.selu(x) 

        x1 = self.conv1(x)
        x2 = self.w1(x.reshape((batchsize, self.width_list[1], size_x * size_y))).view(batchsize, self.width_list[2], size_x, size_y)
        x = x1 + x2
        x = F.selu(x) 

        x1 = self.conv2(x)
        x2 = self.w2(x.reshape((batchsize, self.width_list[2], size_x * size_y))).view(batchsize, self.width_list[3], size_x, size_y)
        x = x1 + x2
        x = F.selu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.reshape((batchsize, self.width_list[3], size_x * size_y))).view(batchsize, self.width_list[4], size_x, size_y)
        x = x1 + x2

        if self.pad_amount: # remove padding
            if self.pad_dim == '1':
                x = x[:, :, :-self.pad_amount[0]]
            elif self.pad_dim == '2':
                x = x[..., :-self.pad_amount[1]]
            elif self.pad_dim == 'both':
                x = x[:, :, :-self.pad_amount[0]]
                x = x[..., :-self.pad_amount[1]]

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.selu(x)
        x = self.fc2(x)
        x = F.selu(x)
        x = self.fc3(x)
        return x

    def get_grid(self, S1, S2, batchsize, device):
        gridx = torch.tensor(np.linspace(0, 1, S1), dtype=torch.float)
        gridx = gridx.reshape(1, 1, S1, 1).repeat([batchsize, 1, 1, S2])
        gridy = torch.tensor(np.linspace(0, 1, S2), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, S2).repeat([batchsize, 1, S1, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

class Net2d(nn.Module):
    def __init__(self, in_dim, out_dim, modes1, modes2, width, pad_amount=None, pad_dim='1'):
        super(Net2d, self).__init__()
        self.conv1 = SimpleBlock2d(in_dim, out_dim, modes1, modes2, width, pad_amount, pad_dim)

    def forward(self, x):
        x = self.conv1(x)
        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c