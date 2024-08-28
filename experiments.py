import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values

# Function to create fake data
def create_fake_data(N, M):
    X = np.random.randint(2, size=(N, M))
    y = np.random.randint(2, size=N)

    X = pd.DataFrame(X)
    y = pd.Series(y)
    return X, y

# Function to calculate average time (and std) taken by fit() and predict() for different N and M
def calculate_average_time(N_values, M_values):
    fit_times = np.zeros((len(N_values), len(M_values)))
    predict_times = np.zeros((len(N_values), len(M_values)))
    
    for i, N in enumerate(N_values):
        for j, M in enumerate(M_values):
            X, y = create_fake_data(N, M)
            fit_time_list = []
            predict_time_list = []

            #print(i,j)

            try:
            
                for _ in range(num_average_time):
                    start_time = time.time()
                    dt = DecisionTree(criterion='information_gain')
                    dt.fit(X, y)
                    fit_time_list.append(time.time() - start_time)
                    
                    X_test, _ = create_fake_data(N, M)
                    start_time = time.time()
                    dt.predict(X_test)
                    predict_time_list.append(time.time() - start_time)
                
                fit_times[i, j] = np.mean(fit_time_list)
                predict_times[i, j] = np.mean(predict_time_list)
            except:
                pass
    
    return fit_times, predict_times

# Function to plot the results
def plot_results(N_values, M_values, fit_times, predict_times):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot fit times
    for j, M in enumerate(M_values):
        axs[0].plot(N_values, fit_times[:, j], label=f'M={M}')
    axs[0].set_xlabel('Number of Samples (N)')
    axs[0].set_ylabel('Average Time (s)')
    axs[0].set_title('Time taken to fit the Decision Tree')
    axs[0].legend()
    
    # Plot predict times
    for j, M in enumerate(M_values):
        axs[1].plot(N_values, predict_times[:, j], label=f'M={M}')
    axs[1].set_xlabel('Number of Samples (N)')
    axs[1].set_ylabel('Average Time (s)')
    axs[1].set_title('Time taken to predict using the Decision Tree')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

# Run the functions
N_values = [100, 200, 500, 1000, 2000]
M_values = [10, 20, 50, 100]
fit_times, predict_times = calculate_average_time(N_values, M_values)
plot_results(N_values, M_values, fit_times, predict_times)