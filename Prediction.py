import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    
    # Add bias unit to the input layer (X)
    X = np.hstack([np.ones((m, 1)), X])  # Add a column of ones for the bias term
    
    # Layer 2 (Hidden Layer)
    z2 = np.dot(X, Theta1.T)
    a2 = sigmoid(z2)  # Apply sigmoid activation
    a2 = np.hstack([np.ones((m, 1)), a2])  # Add bias unit to the hidden layer
    
    # Layer 3 (Output Layer)
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)  # Apply sigmoid activation
    
    # Prediction
    p = np.argmax(a3, axis=1)  # Get the index of the maximum output value (prediction)
    
    return p
#rd