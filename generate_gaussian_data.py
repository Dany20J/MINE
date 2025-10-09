import torch

def generate_gaussian_data(batch_size, input_dim, rho):
    """
    Generates d-dimensional correlated Gaussian data (X, Z).
    
    This is based on the linear transformation: Z = rho * X + sqrt(1-rho^2) * epsilon
    """
    # Generate X ~ N(0, I)
    x = torch.randn(batch_size, input_dim)
    
    # Generate independent noise epsilon ~ N(0, I)
    epsilon = torch.randn(batch_size, input_dim)
    
    # Create Z correlated with X
    Z = rho * x + torch.sqrt(torch.tensor(1.0 - rho**2)) * epsilon
    
    return x, Z