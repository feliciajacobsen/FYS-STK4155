import numpy as np
import matplotlib.pyplot as plt
from math import floor
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets
#Local files
from franke import make_data
from neural_network import accuracy_func


class logistic_regression:
    """
    Estimate logistic regression coefficients
    using stochastic gradient descent with mini-batches.

    Params:
        X: array
            design matrix of shape shape (no. of observations, no. of features)
        y: vector
            vector of output data
    """
    def __init__(self, X, y, initialization=None):
        self.X = X
        self.y = y
        if initialization==None:
            self.weights = (np.random.random((y.shape[1], X.shape[1])) - 0.5)

    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=1).reshape(-1,1)


    def SGD(self, eta, epochs, lam, batch_size):
        """
        Linear regression algorithm using
        stochastic gradient descent with mini-batches.

        Params:
            X: Array
                array of observations(rows) and features(columns).
            y: vector
                vector of output values.
            eta: float
                learning rate.
            epochs: int
                the number of epochs to perform SGD.
            lam: float
                value of L2 regularization. If zero, then we use OLS.
            batch_size: int
                the no. of mini-batches.

        Returns:
            weights: vector
                Array of estimated coefficients for regression model.
            MSE: list
                List of computed MSE for predicted output and true output.
        """

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)

        acc = []
        y_pred = []
        N = y_train.shape[0] # 1437

        for i in range(epochs):
            for j in range(floor(N / batch_size)): # 0, ..., 94
                random_idx = np.random.randint(0, N, size=batch_size)

                y_pred = self.softmax(X_train[random_idx, :] @ self.weights.T)

                cost_gradient = (y_pred - y_train[random_idx,:]).T @ X_train[random_idx,:]

                self.weights -= (eta * cost_gradient / batch_size) + ((eta * lam / N) * self.weights)

            acc.append(accuracy_func(y_test,self.softmax(X_test @ self.weights.T)))

        y_pred_new = self.softmax(X_test @ self.weights.T)

        return y_test, y_pred_new, np.array(acc)

if __name__ == "__main__":
    # Import MNIST dataset
    digits = datasets.load_digits()
    X = digits.images # Images
    X /= np.max(X)
    a = digits.target # Labels

    # Make output data one-hot encoded
    y = np.zeros((X.shape[0],10))
    for i in range(X.shape[0]):
        y[i, int (a[i])] = 1

    # Flatten images
    n_inputs = X.shape[0]
    X = X.reshape(n_inputs, -1) # shape = (1797, 64), n_inputs, n_features

    reg = logistic_regression(X,y,initialization=None)
    epochs = 1000
    lam = 0.0
    y_test, y_pred, acc = reg.SGD(eta=0.001, epochs=epochs, lam=0.1, batch_size=15)

    print(f"Total accuracy og test data after training is {accuracy_func(y_test,y_pred):1.2f}")
    plt.title(f"Accuracy (%) on the MNIST test data")
    epochs_arr = np.linspace(0, epochs - 1, epochs)
    plt.plot(epochs_arr, acc)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

    lam = [1,0.1,0.01,0.001]
    for l in lam:
        y_test, y_pred, acc = reg.SGD(eta=0.001, epochs=epochs, lam=i, batch_size=15)
        plt.plot(epochs_arr, acc, label=r"log($\lambda$)="+f"{log10(i):1.1f}, accuracy = {accuracy_func(y_test,y_pred):1.2f}")
        plt.legend()

    plt.title("Accuracy (%) on the MNIST test data with L2 regularization")
    plt.plot(epochs_arr, acc)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
