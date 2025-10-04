import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from timeit import timeit

# 01 - Exercicio kaggle 
"""
PROBLEMS ENCOUNTERED & SOLUTIONS LEARNED:

PROBLEM: Dataset contained NaN values at index 213 (3530, NaN)
SOLUTION: Implemented data cleaning with pd.dropna() before processing .
 - The row data_train[x = 213][y = 213] had the pair (3530, NaN).

PROBLEM: Needed to verify manual implementation correctness
SOLUTION: Cross-validated with scikit-learn LinearRegression

PROBLEM: Matrix operations had different performance characteristics
SOLUTION: Benchmarked np.dot(), mixed methods, and @ operator using timeit
"""


def benchmark_matrix_methods(X, y):
    """Benchmark different matrix multiplication methods."""
    print("\n=== BENCHMARKING MATRIX METHODS ===")
    
    # Each timeit executes lambda 10,000 times and returns total time
    time_full_npdot = timeit(
        lambda: np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y)), 
        number=10000
    )
    print(f"full npdot = {time_full_npdot}")
    
    time_partial_npdot = timeit(
        lambda: np.linalg.inv(np.dot(X.T, X)) @ (np.dot(X.T, y)), 
        number=10000
    )
    print(f"partial npdot = {time_partial_npdot}")
    
    time_at = timeit(
        lambda: np.linalg.inv(X.T @ X) @ (X.T @ y), 
        number=10000
    )
    print(f"full @ = {time_at}")
    print("\n================================")


def least_squares(X, output_targets):
    """
    Compute linear regression parameters using normal equation.
    
    Parameters:
    -----------
    X : numpy.ndarray, shape (n_samples, n_features)
        Feature matrix where each row is a sample, each column a feature
    y : numpy.ndarray, shape (n_samples,)
        Target values to predict
        
    Returns:
    --------
    theta : numpy.ndarray, shape (n_features,)
        Learned parameters (coefficients) for each feature
    """
    # theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    model_weights = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, output_targets))
    return model_weights


# =============================================================================
# DATA LOADING AND CLEANING
# =============================================================================

# The row data_train[x = 213][y = 213] had the pair (3530, NaN). 
# It was needed to clean the data before continue the training

path = kagglehub.dataset_download("andonians/random-linear-regression")
print("Path to dataset files:", path)

# Load and clean training data 
data_train =  pd.read_csv(f"{path}/train.csv") 
train_valids = data_train.dropna(subset=['x', 'y']) # Remove any row with NaN in either column pd.dropna
x_train = train_valids['x'].to_numpy(dtype=float).reshape(-1, 1)
y_train = train_valids['y'].to_numpy(dtype=float).reshape(-1, 1)

# Load and clean test data
data_test = pd.read_csv(f"{path}/test.csv")
test_valids = data_test.dropna(subset=['x', 'y'])
x_test = test_valids['x'].to_numpy(dtype=float).reshape(-1, 1)
y_test = test_valids['y'].to_numpy(dtype=float).reshape(-1, 1)


# benchmark
benchmark_matrix_methods(x_train, y_train)



# TRAINING MODELS AND COMPARISON 
# =============================================================================
train_params = least_squares(x_train, y_train)
print(f"TP = {train_params}")

training_pred = np.dot(x_train, train_params) 

# Scikit-learn model (no intercept since manual formulation)
TRAIN_sklearn_model = LinearRegression(fit_intercept=False) # Set to False since not including the linear coef
TRAIN_sklearn_model.fit(x_train, y_train)
print(f"Scikit-learn (LR) = {TRAIN_sklearn_model.coef_}")

parameter_diff = abs(train_params - TRAIN_sklearn_model.coef_)

print(f" Validação com LinearRegression: {parameter_diff}")

# =============================================================================



# TESTING AND EVALUATION
model_ytest = np.dot(y_test, train_params)
TEST_sklear_model = LinearRegression(fit_intercept=False)
TEST_sklear_model.fit(x_test, y_test)

print(f"training_pred: {train_params}")
print(f"sklearn: {TEST_sklear_model.coef_}")

parameter_diff = abs(train_params - TRAIN_sklearn_model.coef_)
print(f"Validação com LinearRegression: {parameter_diff}")



# Training data visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Regression Model Analysis', fontsize=16)

axes[0].scatter(x_train, y_train, alpha=0.4, label='Data')
axes[0].plot(x_train, training_pred, alpha=0.9, color='red', label='Model')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].legend()
axes[0].grid(True)

axes[1].scatter(y_train, training_pred, label='Model')
axes[1].set_xlabel('y_actual')
axes[1].set_ylabel('y_predicted')
axes[1].set_title('Actual vs Predicted')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

pass