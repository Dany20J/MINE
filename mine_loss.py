import torch

class EMA_MINELoss:
    """
    A stateful class implementing the stabilized MINE loss with 
    Exponential Moving Average (EMA) for the log-denominator.
    
    This is independent of the network model architecture.
    """
    def __init__(self, alpha=0.01):
        """
        Initializes the EMA state.
        
        Args:
            alpha (float): The smoothing factor for the EMA (e.g., 0.01 or 0.001).
        """
        self.ma_et = 1.0  # EMA of E[exp(T_marginal)]. Initialized to 1.0.
        self.alpha = alpha

    def __call__(self, T_output_joint, T_output_marginal):
        """
        Calculates the stabilized MINE loss and updates the EMA state.
        
        Args:
            T_output_joint (torch.Tensor): T scores for joint samples.
            T_output_marginal (torch.Tensor): T scores for marginal/shuffled samples.
        
        Returns:
            tuple: (loss, mi_estimate)
        """
        
        # 1. Calculate the raw batch mean of the exponential term
        mean_exp_marginal_batch = torch.mean(torch.exp(T_output_marginal))
        
        # 2. Update the EMA (ma_et)
        # Use .item() and detach the value because the EMA update is outside the 
        # current step's backpropagation path.
        self.ma_et = (1.0 - self.alpha) * self.ma_et + self.alpha * mean_exp_marginal_batch.item()
        
        # 3. Calculate the stabilized loss
        
        # Convert the stored scalar EMA value back to a tensor for calculation
        # and place it on the correct device.
        current_ma_et = torch.tensor(self.ma_et, device=T_output_joint.device)
        
        # E_P_XZ [T(x, z)]
        term1 = torch.mean(T_output_joint)
        
        # log(EMA[E_P_X @ P_Z [e^T(x', z)]])
        # Use + 1e-8 for numerical stability, preventing log(0)
        term2 = torch.log(current_ma_et + 1e-8) 

        term2 = torch.log(torch.mean(torch.exp(T_output_marginal)))
        
        
        mi_estimate = term1 - term2
        
        # We minimize the negative MI estimate
        loss = -mi_estimate
        
        return loss, mi_estimate
