import numpy as np
import matplotlib.pyplot as plt

from mine_trainer import MineTrainer
from plot_mine_results import plot_mine_results
from mine_loss import EMA_MINELoss
from mine_f_loss import EMA_MINE_F_Loss

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MINE and MINE-f models for various correlation coefficients.')
    parser.add_argument('--input_dim', type=int, default=20, help='Dimension of the input space (default: 2)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training (default: 256)')
    parser.add_argument('--num_steps', type=int, default=10000, help='Number of steps for training (default: 1000)')
    parser.add_argument('--rhos', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], help='List of correlation coefficients to test (default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])')
    parser.add_argument('--load', type=str, default=None, help='File to load results from (default: None)')
    args = parser.parse_args()
    if args.load is not None:
        mine_results = np.load(args.load)
        plot_mine_results(mine_results['RHOS'], mine_results['true_results'], mine_results['mine_results'], mine_results['mine_results_f'], mine_results['INPUT_DIM'])
        exit(0)
    
    # Hyperparameters
    INPUT_DIM = args.input_dim
    BATCH_SIZE = args.batch_size
    NUM_STEPS = args.num_steps 
    # List of correlation coefficients to test
    RHOS = args.rhos
    NEG_RHOS = [-r for r in RHOS]
    RHOS = NEG_RHOS + [0] + RHOS
    RHOS = sorted(RHOS)
    print(RHOS)

    mine_results = []
    mine_results_f = []
    true_results = []

    print(f"Starting MINE experiments for d={INPUT_DIM} with {NUM_STEPS} steps per rho...")

    for rho in RHOS:
        mine_trainer = MineTrainer(rho, INPUT_DIM, BATCH_SIZE, NUM_STEPS, EMA_MINELoss())
        mine_mi, true_mi = mine_trainer.train()
        mine_trainer_f = MineTrainer(rho, INPUT_DIM, BATCH_SIZE, NUM_STEPS, EMA_MINE_F_Loss())
        mine_mi_f, _ = mine_trainer_f.train()
        
        mine_results.append(mine_mi)
        mine_results_f.append(mine_mi_f)
        true_results.append(true_mi)
        print(f"rho: {rho:.1f} | True MI: {true_mi:.4f} | MINE Est: {mine_mi:.4f}")

    
    
    print("\nExperiment Complete. Plot generated with three distinct lines.")
    
    np.savez('mine_results.npz', mine_results=mine_results, mine_results_f=mine_results_f, true_results=true_results, RHOS=RHOS, INPUT_DIM=INPUT_DIM)
    
    plot_mine_results(RHOS, true_results, mine_results, mine_results_f, INPUT_DIM)