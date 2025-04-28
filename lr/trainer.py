from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.utils import shuffle
import torch
import numpy as np
from pprint import pprint
import sys
from tqdm import tqdm
import datetime
import json
import os
import glob
import time
import logging

from util import log
from model_svgd import SVGD
from load_data import load_uci_dataset


#import pdb

class Trainer:
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

        # Create timestamp for unique log directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = f'{config.method}_{timestamp}'
        self.train_dir = os.path.join('train_dir', self.filepath)
        self.log_dir = os.path.join('logs', self.filepath)

        # Create directories
        for directory in [self.train_dir, self.log_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Save config to log directory
        config_dict = vars(config)
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)

        # Create log file
        self.log_file = os.path.join(self.log_dir, 'training.log')
        self.setup_logging()

        if self.config.clean:
            files = glob.glob(self.train_dir + '/*')
            for f in files:
                os.remove(f)

        # Create model
        self.model = SVGD(config)
        
        # Create optimizer
        if config.method == 'svgd':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=config.learning_rate)
        elif config.method == 'svgd_kfac':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=config.learning_rate)
        elif config.method == 'mixture_kfac':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=config.learning_rate)
        elif config.method in ['SGLD', 'pSGLD']:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1.0)

        # Initialize step counter
        self.step = 0

    def setup_logging(self):
        """Setup logging to both file and console"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_metrics(self, metrics, step):
        """Log metrics to both file and console"""
        train_ll, train_acc, valid_ll, valid_acc, test_ll, test_acc = metrics
        metrics_str = (
            f"Step {step:4d} | "
            f"Train LL: {train_ll:.4f} | Train Acc: {train_acc:.4f} | "
            f"Valid LL: {valid_ll:.4f} | Valid Acc: {valid_acc:.4f} | "
            f"Test LL: {test_ll:.4f} | Test Acc: {test_acc:.4f}"
        )
        self.logger.info(metrics_str)

    def train_step(self, x_batch, y_batch):
        self.optimizer.zero_grad()
        
        # Convert inputs to tensors if they aren't already
        if not isinstance(x_batch, torch.Tensor):
            x_batch = torch.tensor(x_batch, dtype=torch.float32)
        if not isinstance(y_batch, torch.Tensor):
            y_batch = torch.tensor(y_batch, dtype=torch.float32)
        
        ll, acc = self.model((x_batch, y_batch), training=True)
            
        if self.config.method == 'svgd':
            grads = self.model.svgd_grads
        elif self.config.method == 'svgd_kfac':
            grads = self.model.kfac_grads
        elif self.config.method == 'mixture_kfac':
            grads = self.model.mixture_grads
        elif self.config.method in ['SGLD', 'pSGLD']:
            grads = self.model.psgld_grads

        # Apply gradients manually since we're using custom gradients
        for param, grad in zip(self.model.parameters(), grads):
            param.grad = grad
            
        self.optimizer.step()
        
        # Update step counters
        self.step += 1
        self.model.step += 1.0
        
        return ll.item(), acc.item()

    def evaluate(self):
        def get_lik_and_acc(X, y):
            n = len(X)
            ll, acc = [], []
            batch_size = 2000
            
            # Convert to tensors
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            
            for i in range(n // batch_size + 1):
                start = i * batch_size
                end = min((i+1)*batch_size, n)
                x_batch = X[start:end]
                y_batch = y[start:end]
                
                with torch.no_grad():
                    ll_i, acc_i = self.model((x_batch, y_batch))
                ll.append(ll_i)
                acc.append(acc_i)
            return torch.mean(torch.tensor(ll)), torch.mean(torch.tensor(acc))

        train_ll, train_acc = get_lik_and_acc(self.dataset.x_train, self.dataset.y_train)
        valid_ll, valid_acc = get_lik_and_acc(self.dataset.x_valid, self.dataset.y_valid)
        test_ll, test_acc = get_lik_and_acc(self.dataset.x_test, self.dataset.y_test)

        return train_ll, train_acc, valid_ll, valid_acc, test_ll, test_acc

    def train(self):
        self.logger.info("Training Starts!")
        n_updates = 1
        total_steps = 2000  # Run for exactly 2000 steps

        # Print header
        self.logger.info("-" * 120)
        self.logger.info(f"{'Step':>6} | {'Train LL':>10} | {'Train Acc':>10} | {'Valid LL':>10} | {'Valid Acc':>10} | {'Test LL':>10} | {'Test Acc':>10}")
        self.logger.info("-" * 120)

        # Create progress bar
        pbar = tqdm(total=total_steps, desc="Training", unit="step")
        last_eval_step = 0

        while n_updates <= total_steps:
            # Shuffle data at the start of each epoch
            x_train, y_train = shuffle(self.dataset.x_train, self.dataset.y_train)
            
            # Calculate batches for this epoch
            max_batches = self.config.n_train // self.config.batch_size 

            for bi in range(max_batches):
                if n_updates > total_steps:
                    break

                start = bi * self.config.batch_size
                end = min((bi+1) * self.config.batch_size, self.config.n_train)

                x_batch = x_train[start:end]
                y_batch = y_train[start:end]

                ll, acc = self.train_step(x_batch, y_batch)

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'Train LL': f'{ll:.4f}',
                    'Train Acc': f'{acc:.4f}'
                })

                # Evaluate every 200 steps
                if n_updates - last_eval_step >= 200:
                    metrics = self.evaluate()
                    self.log_metrics(metrics, n_updates)
                    last_eval_step = n_updates

                n_updates += 1

        pbar.close()
        self.logger.info("Training Completed!")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoches', type=int, default=2, required=False)
    parser.add_argument('--method', type=str, default='svgd', choices=['SGLD', 'pSGLD', 'svgd', 'svgd_kfac', 'mixture_kfac'], required=True)
    parser.add_argument('--n_particles', type=int, default=20, required=False)
    parser.add_argument('--batch_size', type=int, default=256, required=False)
    parser.add_argument('--dataset', type=str, default='ionosphere', required=False, 
                       choices=['ionosphere', 'breastcancer', 'heart-disease', 'austrailia-credit', 
                               'sonar', 'banknote', 'mammographic-masses', 'parkinsons', 'tic-tac-toe'])
    parser.add_argument('--trial', type=int, default=1, required=False)
    parser.add_argument('--learning_rate', type=float, default=5e-3, required=False)
    parser.add_argument('--kernel', type=str, default='rbf', required=False)
    parser.add_argument('--clean', action='store_true', default=False)
    parser.add_argument('--savepath', type=str, default='results/', required=False)
    parser.add_argument('--checkpoint', type=str, default=None, required=False)
    parser.add_argument('--save', action='store_true', default=False)
    config = parser.parse_args()
    
    if not config.save:
        log.warning("nothing will be saved.")

    # Load dataset
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_uci_dataset(config.dataset, random_state=config.trial)
    
    # Set dataset dimensions
    config.n_train, config.dim = x_train.shape
    
    # Create dataset object
    from collections import namedtuple
    dataset = namedtuple("dataset", "x_train, x_valid, x_test, y_train, y_valid, y_test")(
        x_train=x_train, x_valid=x_valid, x_test=x_test,
        y_train=y_train, y_valid=y_valid, y_test=y_test
    )
    
    # Create trainer and start training
    trainer = Trainer(config, dataset)
    trainer.train()


if __name__ == '__main__':
    main()

