import numpy as np

def initialise(output_size, input_size, epsilon=0.15):
    """
    Randomly initializes the weights (Theta) for a layer with `output_size` units 
    and `input_size` features, including the bias term.
    
    Args:
        output_size (int): Number of units in the layer (rows of Theta).
        input_size (int): Number of features (columns of Theta).
        epsilon (float): Range for random initialization, default is 0.15.
    
    Returns:
        np.ndarray: Matrix of random weights of shape (output_size, input_size + 1).
                    The extra column is for the bias term.
    """
    # Randomly initialize the weights to small values in the range [-epsilon, +epsilon]
    theta = np.random.rand(output_size, input_size + 1) * 2 * epsilon - epsilon
    return theta
#rd