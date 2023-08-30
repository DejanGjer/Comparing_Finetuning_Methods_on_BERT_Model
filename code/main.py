import argparse
import wandb
from datetime import datetime as dt
import os

from wandb_config import sweep_config
import config as base_config
from train import Train

def train(params=None):
    run = None
    if params is not None:
        run = wandb.init(project=sweep_config['project'])
    else:
        run = wandb.init()
    parameters = wandb.config if params is None else params
    print("Training parameters")
    print(parameters)
    trainer = Train(parameters, base_config, run_name=run.name)
    trainer.run()
    wandb.finish()

def prepare_params(args):
    params = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    }
    return params


if __name__ == "__main__":
    print("Dependencies loaded")
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Emotion Classification")

    # Add arguments
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate for the training (default: 3e-5)')
    
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for regularization (default: 0.01)')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')

    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs (default: 2)')
    
    parser.add_argument('--new_sweep', action='store_true', default=False,
                        help='Create a new sweep (default: False)')
    
    parser.add_argument('--sweep_id', type=str, default=None,
                        help='Sweep ID to continue runs of already existent sweep (default: None)')
    
    parser.add_argument('--run_count', type=int, default=0, 
                        help='Number of runs for the sweep (default: 1)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # login to wandb
    api_key = os.environ['WANDB_API_KEY']
    wandb.login(key=api_key)

    sweep_id = None
    if args.new_sweep:
        # create sweep
        sweep_id = wandb.sweep(sweep_config, project=sweep_config['project'])
    elif args.sweep_id is not None:
        sweep_id = args.sweep_id

    if sweep_id is not None:
        # set up the sweep agent
        if args.run_count != 0:
            wandb.agent(sweep_id, function=train, count=args.run_count, project=sweep_config['project'])
        else: 
            wandb.agent(sweep_id, function=train, project=sweep_config['project'])
    else:
        # run the training
        train(params=prepare_params(args))
    

    

