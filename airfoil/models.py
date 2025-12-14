import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU
from math import prod
import numpy as np

class GRU_spatial_net(torch.nn.Module):
    """
    GRU model for spatiotemporal data.
    Consolidated to support both Re1000 (standard dropout) and Re5000 (embedding dropout) configurations.
    """
    def __init__(self, in_channels, out_channels, spatial_shape, hidden_units, num_layers, emb_dropout=0.0, dropout=0.0):
        super(GRU_spatial_net, self).__init__()

        self.hidden_units = hidden_units
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_shape = spatial_shape
        self.num_layers = num_layers
        
        self.drop = nn.Dropout(p=emb_dropout) if emb_dropout > 0. else None

        self.p1 = nn.Linear(in_channels * int(prod(spatial_shape)), self.hidden_units // 2)
        self.p2 = nn.Linear(self.hidden_units // 2, self.hidden_units)
        self.p3 = nn.Linear(self.hidden_units, 3 * self.hidden_units // 2)
        
        # Standard dropout within GRU
        self.gru = GRU(input_size=3 * self.hidden_units // 2, 
                       hidden_size=hidden_units, 
                       num_layers=num_layers, 
                       batch_first=True, 
                       dropout=dropout)
        
        self.q1 = nn.Linear(self.hidden_units, self.hidden_units * 2)
        self.q2 = nn.Linear(self.hidden_units * 2, out_channels * int(prod(spatial_shape)))

        self.bias_h = nn.Parameter(torch.normal(torch.tensor(0.), torch.tensor(1.)))

    def forward(self, x, init_hidden_states=None): 
        # init_hidden_states has shape (num_layers, batch_size, hidden_units)
        shape = x.shape # assumes shape is (batch, history, channels, x1, x2, ..., xn)
        
        # Flatten spatial dims
        x = x.reshape((shape[0], shape[1], -1))

        if init_hidden_states is None:
            init_hidden_states = torch.zeros(self.num_layers, shape[0], self.hidden_units).to(x.device)
            init_hidden_states += self.bias_h

        x = F.selu(self.p1(x))
        if self.drop: x = self.drop(x)
        
        x = F.selu(self.p2(x))
        if self.drop: x = self.drop(x)
        
        x = F.selu(self.p3(x))

        out, h = self.gru(x, init_hidden_states)
        pred = out[:,-1]

        pred = F.selu(self.q1(pred))
        pred = self.q2(pred)

        pred = pred.reshape((shape[0], self.out_channels, *self.spatial_shape))

        return pred, h

    def predict(self, x, num_steps, grid_function=None):
        output = []
        states = None
        
        for _ in range(num_steps):
            pred, states = self.forward(x, states)
            output.append(pred)
            x = pred.unsqueeze(1)
            if grid_function:
                grid = grid_function((x.shape[0], x.shape[1], 1, x.shape[-2], x.shape[-1]), x.device)
                x = torch.cat((x, grid), dim=2)

        return torch.stack(output, dim=1)