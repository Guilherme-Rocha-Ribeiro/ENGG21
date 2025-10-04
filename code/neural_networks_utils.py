import numpy as np

def predict_outputs(X, theta):
    """Make predictions using linear model: ŷ = Xθ"""
    return np.dot(X, theta)

def calculate_residual(X_sample, theta, y_actual):
    """Calculate residual (error) for a single data point: y_hat - y_actual"""
    return np.dot(X_sample, theta) - y_actual

def least_squares(X, y):
    """Normal equation: θ = (XᵀX)⁻¹Xᵀy"""
    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

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

