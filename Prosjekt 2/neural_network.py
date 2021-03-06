import numpy as np
import matplotlib.pyplot as plt
from math import floor
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import datasets
# Local files
from franke import FrankeFunction, make_data



def accuracy_func(y, y_pred):
    """
    Functions calculates accuracy of classification.

    Params:
    --------
        y: vector
            True labels, 1D array.
        y_pred: vector
            True labels which is one hot-encoded,
            and is a matrix of probabilities.
    Returns:
    --------
        float which is the fraction of accuratly predicted classes.
    """
    corr = 0
    wrong = 0
    N = y.shape[0]
    for i in range(N):
        if np.argmax(y[i]) == np.argmax(y_pred[i]):
            corr += 1
        else:
            wrong += 1
    return (corr)/N


class FFNN:
    """
    Feed Foward Neural network. Takes a list of nodes in a given layer
    and a list of activation function for each hidden layer.
    Can do both classification and regression depending on the
    choice of activation in the last hidden layer.

    This function takes a flexible no. of nodes and layers.

    Params:
    --------
        layers: list
            list of no of nodes for each layer, including
            input and output layer.

        activation_functions: list of strings
            list of strings specifying each activation function
            for each hidden layer.

    Returns
    --------
        y: vector
            Vector of predicted outputs. If back_prop function is called
            before prediction, then we train the network.
    """
    def __init__(self, layers, activation_functions):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.activation_functions = []
        self.activation_derivatives = []

        if len(activation_functions)+1 != len(layers):
            print("Warning: No. of activation functions does not match number of hidden layers")


        for i in range(len(layers) - 1):
            # He initialization and normal distribution of weights
            self.weights.append(
                np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i])
            )

            self.biases.append(np.zeros((1, layers[i + 1])))

            if activation_functions[i] == "sigmoid":
                self.activation_functions.append(lambda x : 1 / (1 + np.exp(-x)))
                self.activation_derivatives.append(lambda a : a * (1 - a))

            elif activation_functions[i] == "relu":
                self.activation_functions.append(lambda x : np.maximum(x, 0))
                self.activation_derivatives.append(lambda a : np.where(a > 0, 1, 0))

            elif activation_functions[i] == "relu6":
                self.activation_functions.append(lambda x : np.minimum(np.maximum(x, 0), 6))
                self.activation_derivatives.append(lambda a : np.ones(a.shape) * np.logical_and(a > 0, a < 6))

            elif activation_functions[i] == "leaky_relu":
                self.activation_functions.append(lambda x : np.where(x > 0, x, x * 0.01))
                self.activation_derivatives.append(lambda a : np.where(a > 0, 1, 0.01))

            elif activation_functions[i] == "tanh":
                self.activation_functions.append(lambda x : np.tanh(x))
                self.activation_derivatives.append(lambda a : 1 - a**2)

            elif activation_functions[i] == "softmax":
                self.activation_functions.append(lambda x : np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1))
                self.activation_derivatives.append(lambda a : a*(1-a))

            elif activation_functions[i] == "binary":
                activation_functions[i].append(lambda x : np.where(x >= 0.5, 1, 0))
                activation_derivatives[i].append(lambda a : 0)

            elif activation_functions[i] == "identity":
                self.activation_functions.append(lambda x: x)
                self.activation_derivatives.append(lambda a: np.ones(a.shape))

            else:
                print("Warning: Activation function", activation_functions[i], "for layer", i, "is not implemented.")


    def forward_pass(self, x):
        """
        Predicts a output based on initial input and
        pass it through network and activates with activation Functions
        for each hidden layer.

        Params:
            x: Array

        Returns:
            Array of activated output from input and hidden layers
        """
        for i in range(len(self.layers) - 1):
            x = self.activation_functions[i]((x @ self.weights[i]) + self.biases[i])
        return x

    def back_prop(self, x, y, learning_rate, epochs, batch_size):
        """
        This function trains the neural network
        for classification or regression.

        Params:
        --------
            x: array
                array of shape (no. of observations, no. of features).
            y: array
                vector of output data.
            learning_rate: float
                the learning rate of the stochastic gradient descent.
            epochs: int
                specifies the no. of times the whole dataset has
                been passed through and back the network.
            batch_size: int
                size of mini-batches to part the dataset in.

        Returns:
        --------
            None
        """
        N = x.shape[0]
        for e in range(epochs):
            for batch in range(floor(N / batch_size)):
                random_idx = np.random.randint(0, N, size=batch_size)
                x_batch = x[random_idx]
                y_batch = y[random_idx]
                z = []
                a = [x_batch]
                L = len(self.layers)

                # Forward pass, store a list and z list
                for i in range(L - 1):
                    z.append((a[i] @ self.weights[i]) + self.biases[i])
                    a.append(self.activation_functions[i](z[i]))

                # Initialize for last layer, [-1]
                gradient_cost = a[-1] - y_batch  # True for MSE and Cross Entropy (given softmax activation for last layer)
                delta = (
                    self.activation_derivatives[-1](a[-1]) * gradient_cost
                )  # Local gradient, delta_L (last layer)
                self.dB = [np.mean(delta, axis=0)]
                self.dW = [a[-2].T @ delta]

                # Back propagate, fill these lists: dW and dB
                for i in range(2, L):
                    """
                    Loop going between all hidden layers,
                    and goes from last layer to first,
                    must start from 2 because we use -i.
                    """
                    delta = (
                        delta @ self.weights[-i + 1].T
                    ) * self.activation_derivatives[-i](a[-i])
                    self.dB.append(np.mean(delta, axis=0))
                    self.dW.append(a[-i - 1].T @ delta)

                # Loop over layers and update all weights and biases with lists dW and dB
                for i in range(L - 1):
                    self.weights[i] -= learning_rate * self.dW[-i - 1] / N
                    self.biases[i] -= learning_rate * self.dB[-i - 1]


if __name__ == "__main__":

    # Define layers and no. of nodes in each layer
    net = FFNN(
        layers=[2, 80, 40, 25, 10, 1],
        activation_functions=["tanh", "relu", "tanh", "leaky_relu", "identity"],
    )
    # MSE= 0.054 for ["leaky_relu", "tanh", "relu", "identity"], [2, 40, 25, 10, 1], epochs= 1000, eta=0.001
    # MSE= 0.054 for ["leaky_relu", "leaky_relu", "relu", "identity"], [2, 40, 25, 10, 1], epochs =1000, eta=0.001
    # MSE = 0.058 for ["leaky_relu", "leaky_relu", "sigmoid", "identity"], [2, 40, 25, 10, 1], epochs=10000, eta=0.001

    x, y, z = make_data(1000)  # 100 no. of points

    # Define input data
    X = np.concatenate((x, y), axis=1)

    # Split in train and test
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    # Train on training set
    net.back_prop(X_train, z_train, learning_rate=0.01, epochs=1000, batch_size=60)

    # Predict on test set
    z_pred = net.forward_pass(X_test)

    # Print MSE
    print(mean_squared_error(z_test, z_pred))

    # FFNN from sklearn
    sklearn_FFNN = MLPRegressor(
        hidden_layer_sizes=(100, 70, 40, 10),
        alpha=0,
        activation="relu",
        learning_rate_init=0.01,
        max_iter=100,
        batch_size=30,
    ).fit(X_train, np.array(z_train).ravel())

    sk_z_pred = sklearn_FFNN.predict(X_test)
    print(mean_squared_error(z_test, sk_z_pred))

    # Make 3D plot of predicted surface of franke function with trained FFNN
    l = np.linspace(0, 1, 101)
    xm, ym = np.meshgrid(l, l)
    X = np.concatenate(
        (xm.flatten().reshape(-1, 1), ym.flatten().reshape(-1, 1)), axis=1
    )
    zflat = net.forward_pass(X)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(xm, ym, zflat.reshape(xm.shape))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

    # Test NN for classification:
    # Import MNIST dataset
    digits = datasets.load_digits()
    X = digits.images # Images
    X /= np.max(X) # Scaling
    a = digits.target # Labels

    # Make output data one-hot encoded
    y = np.zeros((X.shape[0],10))
    for i in range(X.shape[0]):
        y[i, int (a[i])] = 1

    # Flatten images
    n_inputs = X.shape[0]
    X = X.reshape(n_inputs, -1) # shape = (1797, 64), n_inputs, n_features
    net = FFNN(
        layers=[X.shape[1], 300, 200, 150, 100, 70, 10],
        #activation_functions=["relu6", "relu6", "relu6", "relu6", "relu6", "softmax"],
        #activation_functions=["tanh", "tanh", "tanh", "tanh", "tanh", "softmax"],
        activation_functions=["tanh" ,"tanh", "tanh", "tanh", "tanh", "softmax"]
    )
    # Accuracy = 0.96, for layers=[X.shape[1], 300, 200, 150, 100, 70, 10]
    # and activation_functions=["tanh", "tanh", "tanh", "tanh", "tanh", "softmax"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train on training set
    net.back_prop(X_train, y_train, learning_rate=0.1, epochs=1000, batch_size=15)

    # Predict on test set return probabilities of being in a given class
    y_pred = net.forward_pass(X_test)

    accuracy = accuracy_func(y_test, y_pred)
    print(f"Accuracy of NN = {accuracy:2.2f}")

    if y_test.shape != y_pred.shape:
        print("Warning: predicted output and test output does not have same shape.")

    # Revrese one-hot encoding
    y_test_new = []
    y_pred_new = []
    for i in range(y_pred.shape[0]):
        y_test_new.append(np.argmax(y_test[i]))
        y_pred_new.append(np.argmax(y_pred[i]))

    plt.title("Accuracy scores of MNIST dataset")
    sns.heatmap(
        confusion_matrix(y_test_new, y_pred_new),
        cmap="Blues",
        annot=True,
        fmt="d",
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()
