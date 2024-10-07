from scipy.io import loadmat
import numpy as np
from Model import neural_network
from Randinitialise import initialise
from Prediction import predict
from scipy.optimize import minimize

# Load MNIST data
data = loadmat('mnist-original.mat')
X = data['data'].T  # Transpose for proper shape
y = data['label'][0]

# Normalize the features
X = X / 255.0

# Split into training and test sets
X_train, y_train = X[:60000], y[:60000]
X_test, y_test = X[60000:], y[60000:]

input_layer_size = 784
hidden_layer_size = 100
num_labels = 10

# Initialize thetas
initial_Theta1 = initialise(hidden_layer_size, input_layer_size)
initial_Theta2 = initialise(num_labels, hidden_layer_size)

initial_nn_params = np.concatenate([initial_Theta1.flatten(), initial_Theta2.flatten()])
lambda_reg = 0.1
myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg)

# Train the neural network using minimize
results = minimize(neural_network, x0=initial_nn_params, args=myargs, 
                   options={'disp': True, 'maxiter': 100}, method="L-BFGS-B", jac=True)

nn_params = results["x"]
Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)

# Test set accuracy
pred = predict(Theta1, Theta2, X_test)
print('Test Set Accuracy: {:.2f}%'.format(np.mean(pred == y_test) * 100))

# Save Thetas
np.savetxt('Theta1.txt', Theta1)
np.savetxt('Theta2.txt', Theta2)
#rd