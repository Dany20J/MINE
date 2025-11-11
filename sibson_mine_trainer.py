from model import T
from mine_sibson_loss import EMA_MINESibsonLoss
from generate_gaussian_data import generate_fixed_z_gaussian_data
import torch
import torch.optim as optim
import numpy as np


class SibsonMineTrainer:
    def __init__(self, rho, input_dim, batch_size, num_steps, mine_loss, lr=1e-4, alpha=1.5):
        self.rho = rho
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lr = lr
        self.mine_loss_obj = mine_loss
        self.alpha = alpha
        
    def mi(self):
        # True Sibson MI (analytical, for reference)
        return 0.5 * self.input_dim * np.log(self.alpha / (1 - self.rho**2))

    def train(self):
        true_mi = self.mi()
        model = T(input_dim=self.input_dim)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        mi_estimates = []
        repeat_fixed = 10  # number of X samples per fixed Z

        for _ in range(self.num_steps):
            # --- Generate grouped (x, z) ---
            x_joint, z_joint = generate_fixed_z_gaussian_data(
                self.batch_size, self.input_dim, self.rho, L=repeat_fixed
            )
            # both tensors: [B, L, D]

            # Flatten for model input
            B, L, D = x_joint.shape
            x_flat = x_joint.reshape(-1, D)
            z_flat = z_joint.reshape(-1, D)

            # Group indices to track which Z each X belongs to
            group_idx = torch.arange(B).repeat_interleave(L)

            # --- Joint samples ---
            T_joint = model(x_flat, z_flat)

            # --- Marginal samples ---
            perm = torch.randperm(B * L)
            z_marginal_flat = z_flat[perm]
            T_marginal = model(x_flat, z_marginal_flat)

            # --- Optimization ---
            loss, mi_estimate = self.mine_loss_obj(
                T_joint, T_marginal, group_idx=group_idx
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            mi_estimates.append(mi_estimate.item())

        final_mi_estimate = np.mean(mi_estimates[-50:])
        return final_mi_estimate, true_mi
