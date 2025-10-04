import numpy as np
import matplotlib.pyplot as plt

# Aula 02/10
"""
GRADIENT DESCENT IMPLEMENTATION - LINEAR REGRESSION

KEY CONCEPTS:
• Gradient Descent: Iterative optimization algorithm
• Stochastic Gradient Descent: Updating parameters per sample
• Learning Rate: Step size control for convergence stability
• Epochs: Complete passes through the dataset

MATHEMATICAL FOUNDATIONS:
• Residual Calculation: error = ŷ - y_actual
• Gradient: ∇J(θ) = 2 * error * X[i].T
• Coefficient Update: θ_new = θ_old - α * ∇J(θ)
• Cost Function: Sum of Squared Errors

IMPLEMENTATION DETAILS:
• Feature Matrix Extension: Adding bias term [x, 1]
• Parameter Initialization: Starting point affects convergence
• Error Tracking: Monitoring training progress
• Array Indexing: Understanding numpy slicing syntax

OBSERVATIONS:
• Low learning rate prevents overshooting but requires more epochs
• Per-sample updates (stochastic) vs batch updates
• Error should decrease monotonically with proper learning rate
• Convergence to optimal linear fit parameters
"""


def predict_outputs(X, theta):
    """Make predictions using linear model: ŷ = Xθ"""
    return np.dot(X, theta)

def calculate_residual(X_sample, theta, y_actual):
    """Calculate residual (error) for a single data point: y_hat - y_actual"""
    return np.dot(X_sample, theta) - y_actual

def gradient_descent(X, y, initial_theta, learning_rate, n_epochs):
    """
    Perform gradient descent to optimize linear regression parameters.
    
    Parameters:
    -----------
    X : numpy.ndarray, shape (n_samples, n_columns)     | Feature matrix with lineaar coef term 
    y : numpy.ndarray, shape (n_samples, 1)             | Target values
    initial_theta : numpy.ndarray, shape (n_columns, 1) | Initial parameter guess
    learning_rate : float                               | Step size for parameter updates
    n_epochs : int                                      | Number of training iterations
        
    Returns:
    --------
    theta_optimized : numpy.ndarray                     | Optimized parameters
    errors : numpy.ndarray                              | Sum of squared errors per epoch
    """
    theta_current = initial_theta.copy()
    errors = np.zeros((n_epochs, 1))
    
    for epoch in range(n_epochs):
        total_squared_error = 0.0

        for i in range(len(X)):
            # Calculate error for current data point
            residual = calculate_residual(X[[i], :], theta_current, y[i])
            
            # Update parameters: θ = θ - α * 2 * e * Xᵀ
            gradient = 2 * residual * X[[i], :].T
            theta_current = theta_current - (learning_rate * gradient)
            
            total_squared_error += residual**2

        errors[epoch] = total_squared_error
    
    print(theta_current)    
    return theta_current, errors


def main():
    x = np.array([[0],
                [0.08],
                [0.16],
                [0.41]])


    y = np.array([[0],
                [0.49],
                [0.98],
                [2.45]])


    # Extended matrix: [x, 1] for linear model y = a*x + b 
    # y_pred | y_hat | ŷ = a * x + b
    X_extended = np.column_stack((x, np.ones((4,1))))


    # This method needs initial guess for the coefficients
    theta_0 = np.array([[5.5],
                        [0.1]])

    # Learning
    alpha = 0.2   # Learning rate, the idea is keeping the learn rate low value and grow the number of epochs
    n_epochs = 100

    # Train model using gradient descent
    optimized_theta, epoch_errors = gradient_descent(X_extended, y, theta_0, alpha, n_epochs)
    
    print("Optimized parameters:")
    print(f"Slope (a): {optimized_theta[0, 0]:.4f}")
    print(f"Intercept (b): {optimized_theta[1, 0]:.4f}")

    # Plot error convergence
    plt.figure(figsize=(10, 4))
    plt.plot(epoch_errors, '.-b')
    plt.xlabel('Epoch')
    plt.ylabel('Sum of Squared Errors')
    plt.title('Gradient Descent Error Convergence')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
# # learning how to use select array

# X_extend[:,0] # all rows, column 0
# X_extend[:,1] # all rows, column 1
# X_extend[0,:] # row 1, all columns
# X_extend[1,:] # row 2, all columns
# X_extend[2,:] # row 3, all columns
# X_extend[3,:] # row 4, all columns



