
import torch
import torch.nn as nn
from neuralop.models import RNO

class RNOWrapper(nn.Module):
    def __init__(self, in_dim, out_dim, modes, width, n_layers=4, padding=None):
        super(RNOWrapper, self).__init__()

        self.rno = RNO(
            n_modes=(modes,),
            hidden_channels=width,
            in_channels=in_dim,
            out_channels=out_dim,
            n_layers=n_layers,
            positional_embedding="grid",
            domain_padding=padding
        )

    def forward(self, x, init_hidden_states=None):
        # Permute input to match RNO expectation
        x_permuted = x.permute(0, 1, 3, 2)
        
        # Forward pass through RNO
        pred_permuted, final_hidden_states = self.rno(
            x_permuted, 
            init_hidden_states=init_hidden_states, 
            return_hidden_states=True
        )
        
        # Permute output back
        pred = pred_permuted.permute(0, 2, 1) # (batch, spatial, out_channels)
                
        return pred, final_hidden_states

    def predict(self, x, num_steps, forcing=None):
        output = []
        states = None
        
        for i in range(num_steps):
            pred, states = self.forward(x, states)
            output.append(pred)
            
            # Prepare next input
            x = pred.unsqueeze(1)
            
            # Handle forcing
            if forcing is not None and i < num_steps - 1:
                forcing_term = forcing[:, i].unsqueeze(1) # (batch, 1, spatial, forcing_dim)
                x = torch.cat((x, forcing_term), dim=-1) # Concatenate along channel dim (last dim)
        
        return torch.stack(output, dim=1)

    def count_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        import numpy as np
        return int(sum([np.prod(p.size()) for p in model_parameters]))
