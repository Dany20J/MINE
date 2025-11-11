import numpy as np
import matplotlib.pyplot as plt

from mine_trainer import MineTrainer
from sibson_mine_trainer import SibsonMineTrainer
from plot_mine_results import plot_mine_results
from mine_loss import EMA_MINELoss
from mine_f_loss import EMA_MINE_F_Loss
from mine_sibson_loss import EMA_MINESibsonLoss

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MINE and MINE-f models for various correlation coefficients.')
    parser.add_argument('--input_dim', type=int, default=10, help='Dimension of the input space (default: 2)')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training (default: 256)')
    parser.add_argument('--num_steps', type=int, default=2000, help='Number of steps for training (default: 1000)')
    parser.add_argument('--rhos', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99], help='List of correlation coefficients to test (default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])')
    parser.add_argument('--load', type=str, default=None, help='File to load results from (default: None)')
    parser.add_argument('--save', type=str, default=None, help='File to save results to (default: None)')
    parser.add_argument('--alpha', type=float, default=1, help='Alpha parameter for MINE-sibson loss (default: 1)')
    parser.add_argument('--mine-f', type=bool, default=False, help='Calculate MINE-f (default: False)')
    parser.add_argument('--mine-sibson', type=bool, default=True, help='Calculate MINE-sibson (default: False)')
    parser.add_argument('--mine', type=bool, default=False, help='Calculate MINE (default: True)')
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
    mine_results_sibson = []

    print(f"Starting MINE experiments for d={INPUT_DIM} with {NUM_STEPS} steps per rho...")

    for rho in RHOS:
        if args.mine:
            mine_trainer = MineTrainer(rho, INPUT_DIM, BATCH_SIZE, NUM_STEPS, EMA_MINELoss())
            mine_mi, true_mi = mine_trainer.train()
            mine_results.append(mine_mi)
            true_results.append(true_mi)
            print(f"rho: {rho:.1f} | MINE Est: {mine_mi:.4f} | True MI: {true_mi:.4f}")
        
        if args.mine_sibson:
            mine_trainer_sibson = SibsonMineTrainer(rho, INPUT_DIM, BATCH_SIZE, NUM_STEPS, EMA_MINESibsonLoss(alpha=5), alpha=5)
            mine_mi_sibson, true_mi = mine_trainer_sibson.train()
            
            mine_results_sibson.append(mine_mi_sibson)
            true_results.append(true_mi)
            print(f"rho: {rho:.1f} | MINE-sibson Est: {mine_mi_sibson:.4f} | True MI: {true_mi:.4f}")
        
        if args.mine_f:
            mine_trainer_f = MineTrainer(rho, INPUT_DIM, BATCH_SIZE, NUM_STEPS, EMA_MINE_F_Loss())
            mine_mi_f, _ = mine_trainer_f.train()
            mine_results_f.append(mine_mi_f)
            print(f"rho: {rho:.1f} | MINE-f Est: {mine_mi_f:.4f}")
        
    # for rho in RHOS:
    #     mine_trainer = MineTrainer(rho, INPUT_DIM, BATCH_SIZE, NUM_STEPS, EMA_MINELoss())
    #     mine_mi, true_mi = mine_trainer.train()
    #     mine_trainer_f = MineTrainer(rho, INPUT_DIM, BATCH_SIZE, NUM_STEPS, EMA_MINE_F_Loss())
    #     mine_mi_f, _ = mine_trainer_f.train()
    #     mine_trainer_sibson = MineTrainer(rho, INPUT_DIM, BATCH_SIZE, NUM_STEPS, EMA_MINESibsonLoss())
    #     mine_mi_sibson, _ = mine_trainer_sibson.train()
        
    #     mine_results.append(mine_mi)
    #     mine_results_f.append(mine_mi_f)
    #     mine_results_sibson.append(mine_mi_sibson)
    #     true_results.append(true_mi)
    #     print(f"rho: {rho:.1f} | True MI: {true_mi:.4f} | MINE Est: {mine_mi:.4f} | MINE-f Est: {mine_mi_f:.4f} | MINE-sibson Est: {mine_mi_sibson:.4f}")

    
    
    print("\nExperiment Complete. Plot generated with three distinct lines.")
    
    if args.save is None:
        args.save = f"mine_results_{INPUT_DIM}.npz"
    if args.mine:
        np.savez(args.save, mine_results=mine_results, true_results=true_results, RHOS=RHOS, INPUT_DIM=INPUT_DIM)
    
    if args.mine_f:
        np.savez(args.save, mine_results_f=mine_results_f, true_results=true_results, RHOS=RHOS, INPUT_DIM=INPUT_DIM)
    
    if args.mine_sibson:
        np.savez(args.save, mine_results_sibson=mine_results_sibson, true_results=true_results, RHOS=RHOS, INPUT_DIM=INPUT_DIM)
    
    if args.mine:
        plot_mine_results(RHOS, INPUT_DIM, true_results, mine_results=mine_results, mine_results_f=None, mine_results_sibson=None)
    
    if args.mine_f:
        plot_mine_results(RHOS, INPUT_DIM, true_results, mine_results_f=mine_results_f, mine_results_sibson=None)
    
    if args.mine_sibson:
        plot_mine_results(RHOS, INPUT_DIM, true_results, mine_results_sibson=mine_results_sibson, mine_results_f=None)
