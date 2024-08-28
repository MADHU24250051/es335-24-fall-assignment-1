"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from .utils import *

np.random.seed(42)


@dataclass
class DecisionTree:

    criterion: Literal["information_gain", "gini_index", "mse"]  # criterion for classification and regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # Convert discrete input features to one-hot vectors
        X = one_hot_encoding(X)


        if check_ifreal(y):
            # Regression case: real output
            #print("Regression")
            self.tree = self.construct_regression_tree(X, y, self.max_depth)
        else:
            # Classification case: discrete output
            #print("Classification")
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
            split_value = X[best_attribute].mean()  # You might want to use a more sophisticated method to find the split value
            (left_X, left_y), (right_X, right_y) = split_data(X, y, best_attribute, split_value)
            tree[best_attribute][f"<={split_value}"] = self.construct_classification_tree(left_X, left_y, depth - 1)
            tree[best_attribute][f">{split_value}"] = self.construct_classification_tree(right_X, right_y, depth - 1)
        else:
            for value in X[best_attribute].unique():
                (subset_X, subset_y), _ = split_data(X, y, best_attribute, value)
                tree[best_attribute][value] = self.construct_classification_tree(subset_X, subset_y, depth - 1)

        return tree

    def construct_regression_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> dict:
        """
        Function to construct the regression decision tree
        """
        if depth == 0 or len(y) <= 1:
            return y.mean()

        best_attribute = opt_split_attribute(X, y, "mse", X.columns)
        
        if best_attribute is None:
            return y.mean()

        tree = {best_attribute: {}}

        if check_ifreal(X[best_attribute]):
            split_value = X[best_attribute].mean()  # You might want to use a more sophisticated method to find the split value
            (left_X, left_y), (right_X, right_y) = split_data(X, y, best_attribute, split_value)
            tree[best_attribute][f"<={split_value}"] = self.construct_regression_tree(left_X, left_y, depth - 1)
            tree[best_attribute][f">{split_value}"] = self.construct_regression_tree(right_X, right_y, depth - 1)
        else:
            for value in X[best_attribute].unique():
                (subset_X, subset_y), _ = split_data(X, y, best_attribute, value)
                tree[best_attribute][value] = self.construct_regression_tree(subset_X, subset_y, depth - 1)

        return tree

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to run the decision tree on test inputs
        """

        # Convert discrete input features to one-hot vectors
        X = one_hot_encoding(X)

        predictions = []
        for index, row in X.iterrows():
            prediction = self.traverse_tree(self.tree, row)
            predictions.append(prediction)

        return pd.Series(predictions)

    def traverse_tree(self, tree, row):
        """
        Function to traverse the tree and make a prediction
        """

        if not isinstance(tree, dict):
                return tree

        attribute = list(tree.keys())[0]
        value = row[attribute]

        if isinstance(value, (int, float)):
            for condition, subtree in tree[attribute].items():
                if "<=" in condition:
                    threshold = float(condition.split("<=")[1])
                    if value <= threshold:
                        return self.traverse_tree(subtree, row)
                elif ">" in condition:
                    threshold = float(condition.split(">")[1])
                    if value > threshold:
                        return self.traverse_tree(subtree, row)
        else:
            if value in tree[attribute]:
                return self.traverse_tree(tree[attribute][value], row)
        
        # If no matching condition is found, return the most common value or mean
        return max(tree[attribute].values(), key=lambda x: x if isinstance(x, (int, float)) else len(x))

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """

        def plot_tree(tree, indent=0):
            if not isinstance(tree, dict):
                print("  " * indent + f"Prediction: {tree}")
                return

            attribute = list(tree.keys())[0]
            print("  " * indent + f"?({attribute})")
            for value, subtree in tree[attribute].items():
                print("  " * (indent + 1) + f"{value}:")
                plot_tree(subtree, indent + 2)

        plot_tree(self.tree)


# Example usage
if __name__ == "__main__":
    # Sample data for classification

    
    data_classification = {
    'Color': ["Red", "Yellow", "Green", "Red", "Yellow"],
    'Shape': ["Round", "Round", "Round", "Oval", "Oval"],
    'Fruit': ["Apple", "Lemon", "Apple", "Grape", "Lemon"],
}
    
    X_classification = pd.DataFrame(data_classification)
    X_classification = X_classification.iloc[:, :-1]
    y_classification = pd.Series(data_classification['Fruit'])

    # Sample data for regression
    data_regression = {
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
    "Temp": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "High", "High", "Normal", "High"],
    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
    "Minutes Played": [20, 24, 40, 50, 60, 10, 4, 10, 60, 40, 45, 40, 35, 20]
}

    X_regression = pd.DataFrame(data_regression)
    X_regression = X_regression.iloc[:, :-1]
    y_regression = pd.Series(data_regression['Minutes Played'])

    # Classification tree
    tree_classification = DecisionTree("information_gain", max_depth=5)
    tree_classification.fit(X_classification, y_classification)
    print("Classification Tree:")
    tree_classification.plot()

    # Regression tree
    tree_regression = DecisionTree("mse", max_depth=5)
    tree_regression.fit(X_regression, y_regression)
    print("Regression Tree:")
    tree_regression.plot()







    '''data_classification_real = {
    'Temperature': [70.0, 75.0, 80.0, 85.0, 90.0],
    'Humidity': [30.0, 40.0, 50.0, 60.0, 70.0],
    'Weather': ["Sunny", "Cloudy", "Rainy", "Sunny", "Cloudy"]
}'''

    data_classification_real = { "Temperature" : [40,48,60,72,80,90], "Weather" : ["No","No","Yes","Yes","Yes","No"] }
    X_classification_real = pd.DataFrame(data_classification_real)
    X_classification_real = X_classification_real.iloc[:, :-1]
    y_classification_real = pd.Series(data_classification_real['Weather'])

    # Create and fit the decision tree for classification
    tree_classification_real = DecisionTree("information_gain", max_depth=5)
    tree_classification_real.fit(X_classification_real, y_classification_real)

    # Print the tree
    print("Classification Tree with Real Input:")
    tree_classification_real.plot()

    # Predict on new data
''' new_data_classification_real = pd.DataFrame({
        'Temperature': [72.0, 78.0, 82.0],
        'Humidity': [35.0, 45.0, 55.0]
    })
    predictions_classification_real = tree_classification_real.predict(new_data_classification_real)
    print("Classification Predictions with Real Input:", predictions_classification_real)'''




'''


    data_regression_real = {
    'Temperature': [70.0, 75.0, 80.0, 85.0, 90.0],
    'Humidity': [30.0, 40.0, 50.0, 60.0, 70.0],
    'Price': [5.0, 10.0, 15.0, 20.0, 25.0]
}
    X_regression_real = pd.DataFrame(data_regression_real)
    y_regression_real = pd.Series(data_regression_real['Price'])

    # Create and fit the decision tree for regression
    tree_regression_real = DecisionTree("mse", max_depth=5)
    tree_regression_real.fit(X_regression_real, y_regression_real)

    # Print the tree
    print("Regression Tree with Real Input:")
    tree_regression_real.plot()

    # Predict on new data
    new_data_regression_real = pd.DataFrame({
        'Temperature': [72.0, 78.0, 82.0],
        'Humidity': [35.0, 45.0, 55.0]
    })
    predictions_regression_real = tree_regression_real.predict(new_data_regression_real)
    print("Regression Predictions with Real Input:", predictions_regression_real)'''