download the minst-original.mat file using the link provided in minst-original.txt

GUI.py
The code provides a GUI application for handwritten digit recognition. 
It allows users to draw a digit on a canvas and then predicts the digit using a machine learning model. 
The application uses Tkinter for the GUI, PIL for image processing, and NumPy for numerical computations.


main.py
The code implements a neural network for handwritten digit recognition using the MNIST dataset. 
It loads and preprocesses the data, initializes the neural network, trains it using the training data, and tests its accuracy on the test data. 
The trained model is then saved to files Theta1.txt and Theta2.txt.


Model.py
The code implements a neural network for handwritten digit recognition using the MNIST dataset. 
It loads and preprocesses the data, initializes and trains the network, tests its accuracy, and saves the trained model.


Prediction.py
The code implements a neural network for handwritten digit recognition using the MNIST dataset. 
It initializes and trains the network, tests its accuracy, and saves the trained model. 
The network has an input layer, hidden layer, and output layer, and uses gradient descent for optimization.


Randinitialise.py
The code implements a neural network for handwritten digit recognition using the MNIST dataset. 
It has an input layer, hidden layer, and output layer, and uses gradient descent for optimization. 
The network is trained using the minimize function from scipy.optimize and the trained model is saved to files Theta1.txt and Theta2.txt.