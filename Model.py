import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Sigmoid gradient function (used for backpropagation)
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    """
    Computes the cost and gradients of the neural network.
    
    Args:
        nn_params (np.ndarray): Unrolled parameters (Theta1 and Theta2).
        input_layer_size (int): Size of the input layer (number of features).
        hidden_layer_size (int): Number of units in the hidden layer.
        num_labels (int): Number of possible output labels (0-9 for digits).
        X (np.ndarray): Input feature matrix.
        y (np.ndarray): Output labels.
        lamb (float): Regularization parameter.
    
    Returns:
        J (float): Computed cost.
        grad (np.ndarray): Unrolled gradients.
    """
    
    # Reshape nn_params back into Theta1 and Theta2 (the weight matrices for 2 layers)
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1))

    # Setup useful variables
    m = X.shape[0]

    # Add bias unit to input layer (a1)
    a1 = np.hstack([np.ones((m, 1)), X])  # X is (m x input_layer_size)
    
    # Forward propagation
    z2 = np.dot(a1, Theta1.T)  # (m x hidden_layer_size)
    a2 = sigmoid(z2)
    a2 = np.hstack([np.ones((m, 1)), a2])  # Add bias unit to hidden layer
    
    z3 = np.dot(a2, Theta2.T)  # (m x num_labels)
    a3 = sigmoid(z3)  # This is the final output (m x num_labels)
    
    # Vectorize the labels
    y = y.astype(int)  # Ensure y is of integer type
    y_vect = np.eye(num_labels)[y]  # Convert y to one-hot encoded (m x num_labels)
  # Convert y to one-hot encoded (m x num_labels)
    
    # Cost function with regularization
    epsilon = 1e-5  # Small value to prevent log(0)
    J = (1 / m) * np.sum(-y_vect * np.log(a3 + epsilon) - (1 - y_vect) * np.log(1 - a3 + epsilon))
    J += (lamb / (2 * m)) * (np.sum(np.square(Theta1[:, 1:])) + np.sum(np.square(Theta2[:, 1:])))  # Regularization
    
    # Backpropagation
    Delta3 = a3 - y_vect  # Output layer error (m x num_labels)
    Delta2 = np.dot(Delta3, Theta2)[:, 1:] * sigmoid_gradient(z2)  # Hidden layer error (m x hidden_layer_size)
    
    # Gradients
    Theta1_grad = (1 / m) * np.dot(Delta2.T, a1)
    Theta2_grad = (1 / m) * np.dot(Delta3.T, a2)
    
    # Regularization for gradients (skip for bias term)
    Theta1_grad[:, 1:] += (lamb / m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] += (lamb / m) * Theta2[:, 1:]
    
    # Unroll gradients into a single vector
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
    
    return J, grad
#rd