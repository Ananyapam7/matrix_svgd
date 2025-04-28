import math
import time
import numpy as np
import torch
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from ucimlrepo import fetch_ucirepo

import sys
from math import pi

import scipy.io
from sklearn.datasets import load_svmlight_file

DATASET_ID_MAP = {
    'ionosphere': 52,      # Ionosphere
    'breastcancer': 17,    # Breast Cancer Wisconsin (Diagnostic)
    'heart-disease': 45,   # Heart Disease
    'austrailia-credit': 143,  # Australia Credit Approval
    'sonar': 151,          # Connectionist Bench (Sonar, Mines vs. Rocks)
    'banknote': 267,       # Banknote Authentication
    'mammographic-masses': 161,  # Mammographic Masses
    'parkinsons': 174,     # Parkinson's Disease
    'tic-tac-toe': 101     # Tic-Tac-Toe
}

def load_uci_dataset(dataset='ionosphere', random_state=42):
    """
    Load and preprocess a UCI dataset for logistic regression.
    
    Args:
        dataset (str): Name of the dataset to load (must be in DATASET_ID_MAP)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_valid, X_test, y_train, y_valid, y_test) as PyTorch tensors
        
    Raises:
        ValueError: If dataset_name is not found in the mapping
    """
    # Get dataset ID
    dataset_id = DATASET_ID_MAP.get(dataset)
    if dataset_id is None:
        raise ValueError(f"Dataset '{dataset}' not found. Available datasets: {list(DATASET_ID_MAP.keys())}")
    
    # Fetch dataset from UCI ML Repository
    dataset = fetch_ucirepo(id=dataset_id)
    
    # Print dataset information
    print(f"Dataset: {dataset.metadata.name}")
    print(f"Instances: {dataset.metadata.num_instances}, Features: {dataset.metadata.num_features}")
    
    # Get features and targets
    X = dataset.data.features
    y = dataset.data.targets
    
    # If y is a DataFrame, convert it to a Series (assuming single target column)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]  # Take the first column as the target
    
    # Convert target values to numeric if they're categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        # Get unique classes and map them to integers
        unique_classes = y.unique()
        class_to_int = {cls: i for i, cls in enumerate(unique_classes)}
        y = y.map(class_to_int)
    
    # Ensure y is float type
    y = y.astype(float)
    
    # Handle NaN values in features
    if X.isna().any().any():
        print(f"Found NaN values in features. Replacing with column means.")
        X = X.fillna(X.mean())
    
    # Handle NaN values in target
    if y.isna().any():
        print(f"Found NaN values in target. Dropping those rows.")
        # Create a mask for rows without NaN in y
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
    
    # Handle duplicate column names
    if len(X.columns) != len(set(X.columns)):
        print(f"Found duplicate column names. Renaming to make them unique.")
        # Create a mapping of old column names to new unique names
        new_columns = []
        seen = {}
        for col in X.columns:
            if col in seen:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_columns.append(col)
        X.columns = new_columns
    
    # Handle categorical features
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    print(f"Categorical features: {len(categorical_cols)}, Numerical features: {len(numerical_cols)}")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ])
    
    # First split into train+valid and test
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Then split train+valid into train and valid
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=0.2, random_state=random_state, stratify=y_train_valid
    )
    
    # Apply preprocessing
    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    X_test = preprocessor.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_valid = torch.tensor(y_valid.values, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    # Add bias term
    X_train = torch.cat([X_train, torch.ones(X_train.shape[0], 1)], dim=1)
    X_valid = torch.cat([X_valid, torch.ones(X_valid.shape[0], 1)], dim=1)
    X_test = torch.cat([X_test, torch.ones(X_test.shape[0], 1)], dim=1)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_valid.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

