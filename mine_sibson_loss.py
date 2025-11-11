import torch


class EMA_MINESibsonLoss:
    def __init__(self, alpha, softmax_temp=10.0):
        """
        alpha: Sibson parameter (α > 1)
        softmax_temp: sharpness of soft-max approximation to max_y
        """
        self.alpha = alpha
        self.beta = alpha / (1 - alpha)
        self.softmax_temp = softmax_temp

    def __call__(self, T_output_joint, T_output_marginal, group_idx):
        """
        Compute the Sibson-MINE loss given model outputs and grouping indices.
        group_idx indicates which x's share the same fixed z.
        """
        # Ensure positivity for g
        g_joint = torch.exp(T_output_joint)
        g_marginal = torch.exp(T_output_marginal)

        # Numerator: E_{Pxy}[g(X,Y)]
        numerator = torch.mean(g_joint)

        # Denominator: softmax over groups for max_y (E_{Px}[g^β(X, y)])^{1/β}
        unique_groups = torch.unique(group_idx)
        group_expectations = []

        for g in unique_groups:
            vals = g_marginal[group_idx == g]
            exp_term = torch.mean(vals ** self.beta)
            group_expectations.append(exp_term)

        group_expectations = torch.stack(group_expectations)

        # Soft approximation of max_y
        weights = torch.softmax(group_expectations * self.softmax_temp, dim=0)
        denominator = torch.sum(weights * group_expectations)

        # Stable log form
        log_ratio = (
            torch.log(numerator + 1e-12)
            - (1.0 / self.beta) * torch.log(denominator + 1e-12)
        )

        mi_estimate = (self.alpha / (self.alpha - 1)) * log_ratio

        # For training, we minimize -MI
        loss = -mi_estimate
        return loss, mi_estimate
