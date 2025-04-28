#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
from tabulate import tabulate

# Define datasets and methods
datasets = ["ionosphere", "breastcancer", "heart-disease", "austrailia-credit", "sonar", "banknote", "mammographic-masses", "parkinsons", "tic-tac-toe"]
methods = ["mixture_kfac", "SGLD", "pSGLD", "svgd", "svgd_kfac"]
learning_rate = 0.01
n_trials = 5

# Initialize results dictionary
results = {}

# Process each dataset
for dataset in datasets:
    results[dataset] = {}
    
    # Process each method
    for method in methods:
        ll_values = []
        acc_values = []
        
        # Process each trial
        for trial in range(1, n_trials + 1):
            log_file = f"logs/{dataset}/{method}_lr{learning_rate}_trial{trial}.log"
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_content = f.read()
                
                # Extract final log likelihood and accuracy
                ll_match = re.search(r'Final log likelihood: ([\d\.-]+)', log_content)
                acc_match = re.search(r'Final accuracy: ([\d\.-]+)', log_content)
                
                if ll_match and acc_match:
                    ll_values.append(float(ll_match.group(1)))
                    acc_values.append(float(acc_match.group(1)))
        
        # Calculate statistics
        if ll_values and acc_values:
            results[dataset][method] = {
                'll_mean': np.mean(ll_values),
                'll_std': np.std(ll_values),
                'acc_mean': np.mean(acc_values),
                'acc_std': np.std(acc_values)
            }
        else:
            results[dataset][method] = {
                'll_mean': np.nan,
                'll_std': np.nan,
                'acc_mean': np.nan,
                'acc_std': np.nan
            }

# Create tables
ll_table_data = []
acc_table_data = []

for dataset in datasets:
    ll_row = [dataset]
    acc_row = [dataset]
    
    for method in methods:
        if method in results[dataset]:
            ll_row.append(f"{results[dataset][method]['ll_mean']:.4f} ± {results[dataset][method]['ll_std']:.4f}")
            acc_row.append(f"{results[dataset][method]['acc_mean']:.4f} ± {results[dataset][method]['acc_std']:.4f}")
        else:
            ll_row.append("N/A")
            acc_row.append("N/A")
    
    ll_table_data.append(ll_row)
    acc_table_data.append(acc_row)

# Print tables
print("\nLog Likelihood Results:")
print(tabulate(ll_table_data, headers=["Dataset"] + methods, tablefmt="grid"))

print("\nAccuracy Results:")
print(tabulate(acc_table_data, headers=["Dataset"] + methods, tablefmt="grid"))

# Save tables to files
with open("logs/log_likelihood_results.txt", "w") as f:
    f.write(tabulate(ll_table_data, headers=["Dataset"] + methods, tablefmt="grid"))

with open("logs/accuracy_results.txt", "w") as f:
    f.write(tabulate(acc_table_data, headers=["Dataset"] + methods, tablefmt="grid"))

print("\nResults saved to logs/log_likelihood_results.txt and logs/accuracy_results.txt") 