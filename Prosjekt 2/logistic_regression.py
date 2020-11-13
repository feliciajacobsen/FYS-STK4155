import numpy as np
import matplotlib.pyplot as plt
from math import floor, log10
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
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
        --------
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
        --------
            y_test: vector
                1D Array of true output
            y_pred_new: vector
                Array of estimated output based on test input values
            acc: Array
                array of accuracies of predicted test data for every epoch
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
    epochs_arr = np.linspace(0, epochs-1, epochs)
    y_test, y_pred, acc = reg.SGD(eta=0.1, epochs=epochs, lam=lam, batch_size=15)

    print(f"Total accuracy og test data after training is {accuracy_func(y_test,y_pred):1.2f}")
    #Total accuracy og test data after training is 0.98

    etas = [0.1, 0.01, 0.001, 0.0001]
    for i in etas:
        reg = logistic_regression(X,y,initialization=None)
        y_test, y_pred, acc = reg.SGD(eta=i, epochs=epochs, lam=lam, batch_size=15)
        plt.plot(epochs_arr, acc, label=r"$\eta$ =" + f"{i:1.1e}, accuracy = {accuracy_func(y_test,y_pred):1.2f}")
        plt.legend()
    plt.title("Accuracy (%) on the MNIST test data")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()


    lams = [1000,10,0.1,0.0001,0.000001]
    for l in lams:
        reg = logistic_regression(X,y,initialization=None)
        y_test, y_pred, acc = reg.SGD(eta=0.01, epochs=epochs, lam=l, batch_size=15)
        plt.plot(epochs_arr, acc, label=r"log10($\lambda$)="+f"{log10(l):1.1f}, accuracy = {accuracy_func(y_test,y_pred):1.2f}")
        plt.legend()

    plt.title("Accuracy (%) on the MNIST test data with L2 regularization")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    sklearn_FFNN = MLPClassifier(
        hidden_layer_sizes=(X.shape[0], 10),
        alpha=0.001,
        learning_rate="constant",
        learning_rate_init=0.01,
        max_iter=1000,
    ).fit(X_train, y_train)

    sk_y_pred = sklearn_FFNN.predict(X_test)
    print(f"Total accuracy og test data predicted by sklearn={accuracy_func(y_test,sk_y_pred):1.2f}")
