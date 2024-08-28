"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np
from typing import Tuple

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X, drop_first=True)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return y.dtype in ['float64', 'int64'] and len(y.unique()) / len(y) > 0.05

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    _, counts = np.unique(Y, return_counts=True)
    probabilities = counts / len(Y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    _, counts = np.unique(Y, return_counts=True)
    probabilities = counts / len(Y)
    return 1 - np.sum(probabilities ** 2)

def mse(Y: pd.Series) -> float:
    """
    Function to calculate the mean squared error
    """
    return np.mean((Y - Y.mean()) ** 2)

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    if criterion == "information_gain" or criterion == "entropy":
        score_function = entropy
    elif criterion == "gini_index":
        score_function = gini_index
    elif criterion == "mse":
        score_function = mse
    else:
        raise ValueError(f"Invalid criterion: {criterion}")

    base_score = score_function(Y)
    weighted_score = 0

    for value in attr.unique():
        subset = Y[attr == value]
        weight = len(subset) / len(Y)
        weighted_score += weight * score_function(subset)

    return base_score - weighted_score

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series) -> str:
    """
    Function to find the optimal attribute to split about.
    """
    best_gain = -float('inf')
    best_attribute = None

    for attribute in features:
        gain = information_gain(y, X[attribute], criterion)
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute

    return best_attribute

def find_best_split(X: pd.Series, y: pd.Series, criterion: str) -> Tuple[float, float]:
    """
    Function to find the best split point for a real-valued attribute
    """
    sorted_indices = np.argsort(X)
    sorted_X, sorted_y = X.iloc[sorted_indices], y.iloc[sorted_indices]

    best_gain = -np.inf
    best_split = None

    for i in range(1, len(sorted_X)):
        if sorted_X.iloc[i] != sorted_X.iloc[i-1]:
            split = (sorted_X.iloc[i] + sorted_X.iloc[i-1]) / 2
            left_y, right_y = sorted_y[:i], sorted_y[i:]
            
            if criterion == "mse":
                gain = mse(y) - (len(left_y) / len(y) * mse(left_y) + len(right_y) / len(y) * mse(right_y))
            else:
                gain = information_gain(y, X <= split, criterion)

            if gain > best_gain:
                best_gain = gain
                best_split = split

    return best_gain, best_split

def split_data(X: pd.DataFrame, y: pd.Series, attribute: str, value: float) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    """
    Function to split the data according to an attribute.
    """
    if check_ifreal(X[attribute]):
        left_mask = X[attribute] <= value
        right_mask = ~left_mask
    else:
        left_mask = X[attribute] == value
        right_mask = ~left_mask

    return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])