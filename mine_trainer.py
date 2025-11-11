from model import T
from mine_f_loss import EMA_MINE_F_Loss
from generate_gaussian_data import generate_gaussian_data
import torch
import torch.optim as optim
import numpy as np

class MineTrainer:
    def __init__(self, rho, input_dim, batch_size, num_steps, mine_loss, lr=1e-4, alpha=1):
        self.rho = rho
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.alpha = alpha
        self.num_steps = num_steps
        self.lr = lr
        self.mine_loss_obj = mine_loss
        
    def mi(self, alpha):
        # alpha > 0, alpha != 1
        return 0.5 * self.input_dim * np.log(alpha / (1 - self.rho**2))


    def train(self):
        """
        Trains MINE for a specific correlation rho and returns the final MI estimate.
        """
        
        # Calculate True MI for comparison: I(X;Z) = -d/2 * log(1 - rho^2)
        true_mi = self.mi(self.alpha)

        # Model Setup
        model = T(input_dim=self.input_dim)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Training Loop
        mi_estimates = []
        
        for _ in range(self.num_steps):
            # --- Data Generation for the current batch ---
            x_joint, z_joint = generate_gaussian_data(self.batch_size, self.input_dim, self.rho)
            
            # --- Create Marginal Samples (shuffling trick) ---
            # The joint sample is (x_joint, z_joint)
            # The marginal sample is (x_shuffled, z_joint)
            shuffle_indices = torch.randperm(self.batch_size)
            x_marginal = x_joint[shuffle_indices]
            z_marginal = z_joint
            
            # --- Forward Pass ---
            T_joint = model(x_joint, z_joint)
            T_marginal = model(x_marginal, z_marginal)
            
            
            # --- Optimization Step ---
            loss, mi_estimate = self.mine_loss_obj(T_joint, T_marginal)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            mi_estimates.append(mi_estimate.item())
            
        
        # Return the average of the last 50 estimates for stability
        final_mi_estimate = np.mean(mi_estimates[-50:]) 
        return final_mi_estimate, true_mi