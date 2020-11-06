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
    def __init__(self, X, y, betas=None):
        self.X = X
        self.y = y
        if betas==None:
            self.betas = (np.random.random(X.shape[1]) - 0.5).reshape(-1, 1)


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
            betas: vector
                Array of estimated coefficients for regression model.
            MSE: list
                List of computed MSE for predicted output and true output.
        """

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)

        MSE = []
        N = y_train.shape[0]
        for i in range(epochs):
            for j in range(floor(N / batch_size)):
                random_idx = np.random.randint(0, N, size=batch_size)

                if lam == 0:
                    cost_gradient = np.mean(
                        -2 * (y_train[random_idx] - (X_train[random_idx, :] @ self.betas)), axis=0
                    )
                if lam > 0:
                    cost_gradient = np.mean(-2 * X_train[random_idx, :].T*(y_train[random_idx] - (X_train[random_idx, :] @ self.betas)).T+ 2 * lam * self.betas)

                self.betas -= eta * cost_gradient

                y_pred = 1. / (1. + np.exp(-(X_test @ self.betas)))

            MSE.append(mean_squared_error(y_test, y_pred))

        return y_pred, MSE

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

    reg = logistic_regression(X,y)
    epochs=100
    y_pred, MSE = reg.SGD(eta=0.001, epochs=epochs, lam=0.001, batch_size=15)

    epochs_arr = np.linspace(0, epochs - 1, epochs)
    plt.plot(epochs_arr, MSE)
    plt.show()
