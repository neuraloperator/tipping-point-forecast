"""
`FourierLayer1d` from official FNO implementation

This file implements a recurrent neural operator (RNO) based on the gated recurrent unit (GRU)
and Fourier neural operator (FNO) architectures.

In particular, the RNO has an identical architecture to the finite-dimensional GRU, 
with the exception that linear matrix-vector multiplications are replaced by linear 
Fourier layers (see Li et al., 2021), and for regression problems, the output nonlinearity
is replaced with a SELU activation.

We call this model the 1D RNO, in the sense that it solves ODE trajectories and 
takes the FFT in time only.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce

import scipy.io

import sys
sys.path.append('../nonstationary_lorenz')

from fourier_1d_ode import SpectralConv1d
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)



class FourierLayer1d(nn.Module):
    def __init__(self, modes, width):
        super(FourierLayer1d, self).__init__()

        self.modes = modes
        self.width = width

        self.conv = SpectralConv1d(self.width, self.width, self.modes)
        self.w = nn.Conv1d(self.width, self.width, 1)

    def forward(self, x):
        # x must be shape (batch, channels, domain_size)
        return self.conv(x) + self.w(x)

class RNO_cell(nn.Module):
    def __init__(self, in_dim, out_dim, modes, width):
        super(RNO_cell, self).__init__()

        self.modes = modes
        self.width = width
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.f1 = FourierLayer1d(self.modes, self.width)
        self.f2 = FourierLayer1d(self.modes, self.width)
        self.f3 = FourierLayer1d(self.modes, self.width)
        self.f4 = FourierLayer1d(self.modes, self.width)
        self.f5 = FourierLayer1d(self.modes, self.width)
        self.f6 = FourierLayer1d(self.modes, self.width)

        self.b1 = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.))) # constant bias terms
        self.b2 = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))
        self.b3 = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))
    
    def forward(self, x, h):
        z = torch.sigmoid(self.f1(x) + self.f2(h) + self.b1)
        r = torch.sigmoid(self.f3(x) + self.f4(h) + self.b2)
        h_hat = F.selu(self.f5(x) + self.f6(r * h) + self.b3) # selu for regression problem

        h_next = (1. - z) * h + z * h_hat

        return h_next

class RNO_layer(nn.Module):
    def __init__(self, in_dim, out_dim, modes, width, return_sequences=False):
        super(RNO_layer, self).__init__()

        self.modes = modes
        self.width = width
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.return_sequences = return_sequences

        self.cell = RNO_cell(in_dim, out_dim, modes, width)
        self.bias_h = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))

    def forward(self, x, h=None):
        batch_size, timesteps, dim, domain_size = x.shape

        if h is None:
            h = torch.zeros((batch_size, self.width, domain_size)).to(x.device)
            h += self.bias_h

        outputs = []
        for i in range(timesteps):
            h = self.cell(x[:, i], h)
            if self.return_sequences:
                outputs.append(h)

        if self.return_sequences:
            return torch.stack(outputs, dim=1)
        else:
            return h


class RNO_1D(nn.Module):
    def __init__(self, in_dim, out_dim, modes, width, padding=None):
        super(RNO_1D, self).__init__()

        self.modes = modes
        self.width = width
        self.padding = padding
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.p = nn.Linear(in_dim + 1, self.width) # input channel_dim is in_dim + 1: u is in-dim, and time label t is 1 dim

        self.layer1 = RNO_layer(in_dim, out_dim, modes, width, return_sequences=True)
        self.layer2 = RNO_layer(in_dim, out_dim, modes, width, return_sequences=True)
        self.layer3 = RNO_layer(in_dim, out_dim, modes, width, return_sequences=False)

        self.q = nn.Linear(self.width, out_dim)
    
    def forward(self, x, init_hidden_states=[None, None, None]): # h must be padded if using padding
        batch_size, timesteps, domain_size, dim = x.shape

        h1, h2, h3 = init_hidden_states

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)

        x = x.permute(0, 1, 3, 2)
        if self.padding:
            x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        hidden_seq1 = self.layer1(x, h1)
        hidden_seq2 = self.layer2(hidden_seq1, h2)
        h = self.layer3(hidden_seq2, h3) # output shape is (batch, width, domain_size)

        # Save final hidden states
        final_hidden_states = [hidden_seq1[:,-1], hidden_seq2[:,-1], h]

        h_unpad = h[..., :-self.padding] # pad the domain if input is non-periodic

        h_unpad = h_unpad.permute(0, 2, 1)
        pred = self.q(h_unpad)

        return pred, final_hidden_states

    def predict(self, x, num_steps, forcing=None): # num_steps is the number of steps ahead to predict
        # forcing is an array of length `num_steps - 1` that is appended to the end of the 
            # input dimensions of `pred` at subsequent steps ahead
            # `forcing` should have shape (batch, num_steps - 1, domain_size, forcing_dim)
        output = []
        states = [None, None, None]
        
        for i in range(num_steps):
            pred, states = self.forward(x, states)
            output.append(pred)
            x = pred.reshape((pred.shape[0], 1, pred.shape[1], pred.shape[2]))
            if forcing is not None and i < num_steps - 1:
                forcing_term = torch.unsqueeze(forcing[:, i], 1)
                x = torch.cat((x, forcing_term), dim=-1)

        return torch.stack(output, dim=1)

    def predict_teacher_forcing(self, x, num_steps, forcing=None):
        """
            Same parameters as in `predict`, but in this case, teacher forcing
            is used.
                `x` is the same as in `predict`
                `forcing` entirely replaces the autoregressive input, as opposed to
                being concatenated like in `predict`
        """
        output = []
        states = [None, None, None]
        
        for i in range(num_steps):
            pred, states = self.forward(x, states)
            output.append(pred)
            x = pred.reshape((pred.shape[0], 1, pred.shape[1], pred.shape[2]))
            if forcing is not None and i < num_steps - 1:
                x = torch.unsqueeze(forcing[:, i], 1)

        return torch.stack(output, dim=1)

    def get_grid(self, shape, device):
        batchsize, steps, size_x = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, steps, 1, 1])
        return gridx.to(device)

    def count_params(self):
        # Credit: Vadim Smolyakov on PyTorch forum
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return int(sum([np.prod(p.size()) for p in model_parameters]))