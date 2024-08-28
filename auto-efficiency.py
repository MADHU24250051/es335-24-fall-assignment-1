import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values

# Remove rows with missing values
data = data.replace('?', np.nan)
data = data.dropna()

data.drop('car name', axis=1, inplace=True)

y = data['mpg'] 
X = data.drop('mpg', axis=1)


from tree.base import DecisionTree

# Split the data into training and test set

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tree.utils import mse

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTree(criterion="mse",max_depth=5)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate the model
print('MSE on custom decision tree:', mean_squared_error(y_test, y_pred))


#Compare the performance of your model with the decision tree module from scikit learn
from sklearn.tree import DecisionTreeRegressor


model = DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('MSE om sklearn decision tree:', mean_squared_error(y_test, y_pred))


