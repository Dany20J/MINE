import torch
import torch.nn as nn
import torch.nn.utils as utils

class T(nn.Module):
    """
    The neural network T for MINE, with Spectral Normalization (SN) applied
    to enforce a 1-Lipschitz constraint on the weight matrices.
    """
    def __init__(self, input_dim, hidden_dim=200):
        super().__init__()
        
        input_size = input_dim * 2 # Concatenation of X and Z
        
        # Helper function to apply SN
        def SNLinear(in_f, out_f):
            # Apply SN to the nn.Linear layer. 
            # We use n_power_iterations=1 for a fast approximation.
            return utils.spectral_norm(nn.Linear(in_f, out_f), name='weight', n_power_iterations=1)

        # The network architecture now uses the SNLinear function
        self.net = nn.Sequential(
            # Layer 1
            SNLinear(input_size, hidden_dim),
            nn.LeakyReLU(),
            
            # Layer 2
            SNLinear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            
            # Layer 3
            SNLinear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            
            # Layer 4
            SNLinear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            
            # Output Layer (Output dim = 1)
            nn.Linear(hidden_dim, 1)
            # No final activation is typically used for the MINE score T(x,z)
        )

    def forward(self, x, z):
        # Concatenate x and z along the feature dimension (dim=1)
        h = torch.cat((x, z), dim=1)
        return self.net(h)