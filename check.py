from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

np.random.seed(42)


@dataclass
class DecisionTree:

    criterion: Literal["information_gain", "gini_index"]  # criterion for classification
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        self.tree = self.construct_classification_tree(X, y, self.max_depth)

    def construct_classification_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> dict:
        """
        Function to construct the classification decision tree
        """
        if depth == 0 or len(y.unique()) == 1:
            return y.mode().iloc[0]

        best_attribute = opt_split_attribute(X, y, self.criterion, X.columns)
        if best_attribute is None:
            return y.mode().iloc[0]

        tree = {best_attribute: {}}

        if check_ifreal(X[best_attribute]):
            unique_values = X[best_attribute].unique()
            unique_values.sort()
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                (left_X, left_y), (right_X, right_y) = split_data(X, y, best_attribute, threshold)
                if not left_X.empty and not right_X.empty:
                    tree[best_attribute][f"less_than_{threshold}"] = self.construct_classification_tree(left_X, left_y, depth - 1)
                    tree[best_attribute][f"greater_than_or_equal_to_{threshold}"] = self.construct_classification_tree(right_X, right_y, depth - 1)
        else:
            for value in X[best_attribute].unique():
                (left_X, left_y), (right_X, right_y) = split_data(X, y, best_attribute, value)
                tree[best_attribute][value] = self.construct_classification_tree(left_X, left_y, depth - 1)

        return tree

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to run the decision tree on test inputs
        """
        predictions = []
        for index, row in X.iterrows():
            prediction = self.traverse_tree(self.tree, row)
            predictions.append(prediction)
        return pd.Series(predictions)

    def traverse_tree(self, tree, row):
        """
        Function to traverse the tree and make a prediction
        """
        if isinstance(tree, dict):
            attribute = list(tree.keys())[0]
            value = row[attribute]
            for condition, subtree in tree[attribute].items():
                if "less_than_" in condition:
                    threshold = float(condition.split("_")[2])
                    if value < threshold:
                        return self.traverse_tree(subtree, row)
                elif "greater_than_or_equal_to_" in condition:
                    threshold = float(condition.split("_")[4])
                    if value >= threshold:
                        return self.traverse_tree(subtree, row)
                elif condition == value:
                    return self.traverse_tree(subtree, row)
        else:
            return tree

    def plot(self) -> None:
        """
        Function to plot the tree
        """
        def plot_tree(tree, indent=0):
            if isinstance(tree, dict):
                attribute = list(tree.keys())[0]
                print("  " * indent + f"?({attribute})")
                for value, subtree in tree[attribute].items():
                    print("  " * (indent + 1) + f"Y: {value}")
                    plot_tree(subtree, indent + 2)
            else:
                print("  " * indent + f"Class: {tree}")

        plot_tree(self.tree)

import pandas as pd
from math import log2


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if y.dtype == 'float64':
        return True
    if y.dtype == 'int64':
        return True
    else:
        return False


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    entropy = 0
    for i in Y.unique():
        p = (Y == i).sum() / len(Y)
        entropy += -p * log2(p)
    return entropy


def conditional_entropy(Y: pd.Series, attr: pd.Series) -> float:
    c_entropy = 0
    for i in attr.unique():
        p = (attr == i).sum() / len(attr)
        c_entropy += p * entropy(Y[attr == i])
    return c_entropy


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    gini = 1
    unique_values = Y.unique()
    for value in unique_values:
        p = (Y == value).sum() / len(Y)
        gini -= p ** 2
    return gini


def conditional_gini(Y: pd.Series, attr: pd.Series) -> float:
    conditional_gini = 0
    for value in attr.unique():
        p = (attr == value).sum() / len(attr)
        conditional_gini += p * gini_index(Y[attr == value])
    return conditional_gini


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    if criterion == 'information_gain':
        return entropy(Y) - conditional_entropy(Y, attr)
    elif criterion == 'gini':
        return gini_index(Y) - conditional_gini(Y, attr)


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    max_information_gain = 0
    best_attr = None
    for attr in X.columns:
        info_gain_temp = information_gain(y, X[attr], criterion)
        if info_gain_temp > max_information_gain:
            max_information_gain = info_gain_temp
            best_attr = attr
    return best_attr


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    if check_ifreal(X[attribute]):
        left_X = X[X[attribute] < value]
        right_X = X[X[attribute] >= value]
        left_y = y[X[attribute] < value]
        right_y = y[X[attribute] >= value]
    else:
        left_mask = X[attribute] == value
        left_X = X[left_mask]
        left_y = y[left_mask]
        right_X = X[~left_mask]
        right_y = y[~left_mask]
    return (left_X, left_y), (right_X, right_y)


# Sample data for classification with real input and discrete output
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
