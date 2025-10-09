import matplotlib.pyplot as plt

def plot_mine_results(RHOS, true_results, mine_results, mine_results_f, INPUT_DIM):
    plt.figure(figsize=(9, 6))
    
    # 1. Plot True MI (Analytic Solution)
    plt.plot(RHOS, true_results, 
             label=r'True Analytic $I(X; Z)$', 
             color='black', 
             linestyle='--',
             linewidth=1.0)
    
    # 2. Plot MINE Estimates (First Run)
    plt.plot(RHOS, mine_results, 
             label=f'MINE Estimate (d={INPUT_DIM})', 
             color='red', 
             linestyle='-',
             linewidth=1.0)
    
    # 3. Plot MINE Estimates (Second Run)
    plt.plot(RHOS, mine_results_f, 
             label=f'MINE-f Estimate (d={INPUT_DIM})', 
             color='blue', 
             linestyle='-',
             linewidth=1.0)
    
    plt.xlabel(r'Correlation Coefficient ($\rho$)', fontsize=12)
    plt.ylabel(r'Mutual Information $I(X; Z)$ (nats)', fontsize=12)
    plt.title(f'MINE Estimates vs MINE-f Estimates vs. True MI for Correlated Gaussian', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.show()