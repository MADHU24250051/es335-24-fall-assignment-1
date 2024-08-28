import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import accuracy, precision, recall

def split_data(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(X)
    indices = np.random.permutation(n)
    test_size = int(test_size * n)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]


def nested_cross_validation(X, y, max_depths, n_splits=5):
    best_depths = []
    outer_scores = []
    
    for outer_fold in range(n_splits):
        X_train_outer, X_test_outer, y_train_outer, y_test_outer = split_data(X, y, test_size=1/n_splits, random_state=42+outer_fold)
        
        best_depth = None
        best_score = -np.inf
        
        for depth in max_depths:
            fold_scores = []
            
            for inner_fold in range(n_splits):
                X_train_inner, X_val_inner, y_train_inner, y_val_inner = split_data(X_train_outer, y_train_outer, test_size=1/n_splits, random_state=42+inner_fold)
                
                dt = DecisionTree(criterion='information_gain', max_depth=depth)
                dt.fit(X_train_inner, y_train_inner)
                
                y_pred_inner = dt.predict(X_val_inner)

                #print(inner_fold,depth,outer_fold)

                #print(y_pred_inner.reset_index(drop=True))
                #print(y_val_inner.copy().reset_index(drop=True))

                #print(y_pred_inner)
                #print(y_val_inner)
                score = np.mean(y_pred_inner.copy().reset_index(drop=True) == y_val_inner.copy().reset_index(drop=True))
                fold_scores.append(score)

            #print(fold_scores)            
            mean_score = np.mean(fold_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_depth = depth
        
        best_depths.append(best_depth)
        
        # Train the model with the best depth on the outer fold
        best_dt = DecisionTree(criterion='information_gain', max_depth=best_depth)
        best_dt.fit(X_train_outer, y_train_outer)
        y_pred_outer = best_dt.predict(X_test_outer)
        outer_scores.append(np.mean(y_pred_outer.copy().reset_index(drop=True) == y_test_outer.copy().reset_index(drop=True)))
    
    return best_depths, outer_scores

# Generate the dataset
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# Convert to DataFrames
X_df = pd.DataFrame(X, columns=['feature1', 'feature2'])
y_series = pd.Series(y, name='target')

# Split the data
X_train, X_test, y_train, y_test = split_data(X_df, y_series, test_size=0.3, random_state=42)

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Generated Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class')
#plt.show()


dt = DecisionTree(criterion='information_gain', max_depth=5)
dt.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt.predict(X_test)

y_test = y_test.reset_index(drop=True)
y_pred = y_pred.reset_index(drop=True)

# Calculate metrics
accuracy = accuracy(y_test, y_pred)
precision = precision(y_test, y_pred, cls=1)
recall = recall(y_test, y_pred, cls=1)

print(f"Accuracy: {accuracy:.4f}")
print(f"Per-class Precision: {precision}")
print(f"Per-class Recall: {recall}")

# Plot the decision tree
dt.plot()

# Define the range of depths to search
max_depths = range(1, 11)

# Perform nested cross-validation
best_depths, outer_scores = nested_cross_validation(X_df, y_series, max_depths)

print(f"Best depths for each fold: {best_depths}")
print(f"Mean outer fold score: {np.mean(outer_scores):.4f}")
print(f"Optimum depth (mode): {max(set(best_depths), key=best_depths.count)}")