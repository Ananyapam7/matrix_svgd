#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create logs directory if it doesn't exist
mkdir -p logs

# Define datasets
datasets=("ionosphere" "breastcancer" "heart-disease" "austrailia-credit" "sonar" "banknote" "mammographic-masses" "parkinsons" "tic-tac-toe")

# Define methods
methods=("mixture_kfac" "SGLD" "pSGLD" "svgd" "svgd_kfac")

# Set learning rate and number of trials
learning_rate=0.01
n_trials=5

# Run experiments for each dataset
for dataset in "${datasets[@]}"; do
  echo "Running experiments for dataset: $dataset"
  
  # Create dataset-specific log directory
  mkdir -p "logs/${dataset}"
  
  # Run each method with the specified learning rate for multiple trials
  for method in "${methods[@]}"; do
    echo "  Method: $method"
    
    for trial in $(seq 1 $n_trials); do
      echo "    Trial: $trial"
      
      # Run the trainer with the specified parameters
      python trainer.py \
        --method $method \
        --learning_rate $learning_rate \
        --trial $trial \
        --dataset $dataset \
        --n_epoches 2 \
        --batch_size 256 \
        > "logs/${dataset}/${method}_lr${learning_rate}_trial${trial}.log" 2>&1 &
      
      # Wait for the process to complete
      wait
    done
  done
done

# Generate results table using a separate Python script
echo "Generating results table..."
python generate_results_table.py

echo "All experiments completed and results table generated."
