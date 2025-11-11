import torch
import torch.nn as nn
import torch.nn.utils as utils

class T(nn.Module):
    """
    Tighter MINE (T-MINE) score network (critic) using Spectral Normalization 
    and Layer Normalization for stability.

    Assumes both inputs (x and z) have the dimension specified by input_dim.
    """
    def __init__(self, input_dim, hidden_dim=64):
        # We assume X and Z both have dimension input_dim
        super().__init__()
        
        # Total input dimension is the concatenation of X and Z
        input_size = input_dim * 2 
        # Set a fixed depth for stability when depth is not specified
        num_layers = 1
        
        def SNLinear(in_f, out_f):
            # 1-Lipschitz constraint via Spectral Normalization
            return utils.spectral_norm(nn.Linear(in_f, out_f), name='weight', n_power_iterations=1)

        layers = []
        
        # 1. Input Layer
        layers.append(SNLinear(input_size, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(hidden_dim)) # Added for training stability

        # 2. Intermediate Hidden Layers (num_layers - 2 = 2 layers)
        for _ in range(num_layers - 2):
            layers.append(SNLinear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))

        # 3. Output Layer (Mapping to scalar score)
        layers.append(SNLinear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Concatenates inputs x and z and returns the scalar MI score."""
        h = torch.cat((x, z), dim=1)
        return self.net(h)