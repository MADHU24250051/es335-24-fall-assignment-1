from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    assert y_hat.size == y.size, "Size mismatch between predictions and true values"
    return (y_hat == y).mean()

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    true_positives = ((y_hat == cls) & (y == cls)).sum()
    predicted_positives = (y_hat == cls).sum()
    
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    true_positives = ((y_hat == cls) & (y == cls)).sum()
    actual_positives = (y == cls).sum()
    
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size, "Size mismatch between predictions and true values"
    return np.sqrt(((y_hat - y) ** 2).mean())

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size, "Size mismatch between predictions and true values"
    return (y_hat - y).abs().mean()