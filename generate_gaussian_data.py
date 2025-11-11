import torch


# Z=ρX+(1−ρ)^2 * ϵ
def generate_gaussian_data(batch_size, input_dim, rho):
    x = torch.randn(batch_size, input_dim)
    
    epsilon = torch.randn(batch_size, input_dim)
    
   
    Z = rho * x + torch.sqrt(torch.tensor(1.0 - rho**2)) * epsilon
    
    return x, Z


def generate_fixed_z_gaussian_data(batch_size, input_dim, rho, L):
    """
    For each fixed z in a batch, generate L correlated x values.
    Returns:
        x: [B, L, D]
        z: [B, L, D] (same values repeated along L for each batch)
    """
    z_base = torch.randn(batch_size, input_dim)  # fixed z per group
    epsilon = torch.randn(batch_size, L, input_dim)
    z = z_base.unsqueeze(1).expand(-1, L, -1)    # repeat z across L

    x = rho * z + torch.sqrt(torch.tensor(1.0 - rho**2)) * epsilon
    return x, z

