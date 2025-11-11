import torch

class EMA_MINE_F_Loss:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, T_output_joint, T_output_marginal):

    
        term1 = torch.mean(T_output_joint)
        

        term2 = torch.mean(torch.exp(T_output_marginal - 1))        
        
        mi_estimate = term1 - term2
        
        loss = -mi_estimate
        
        return loss, mi_estimate
