import numpy as np
from matplotlib import pyplot as plt
from math import floor
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
# Local files
from franke import make_data



def SGD(X, y, initialization, eta, epochs, lam, batch_size):
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
        betas: vector
            Array of estimated coefficients for regression model.
        MSE: list
            List of computed MSE for predicted output and true output.
    """
    if initialization==True:
        betas = initialization

    else:
        betas = (np.random.random(X.shape[1]) - 0.5).reshape(-1, 1) * 0.1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    MSE = []
    N = y_train.shape[0]
    for i in range(epochs):
        for j in range(floor(N / batch_size)):
            random_idx = np.random.randint(0, N, size=batch_size)
            cost_gradient = -2 * X_train[random_idx, :].T @ (y_train[random_idx] - (X_train[random_idx, :] @ betas)) + 2 * lam * betas
            betas -= eta * cost_gradient / batch_size
        MSE.append(mean_squared_error(y_test, (X_test @ betas)))
    return betas, MSE


if __name__ == "__main__":
    # Get input and output data
    x, y, z = make_data(n_points=1000)

    # Gather output data in common martix
    X = PolynomialFeatures(5).fit_transform(np.column_stack((x, y)))

    epochs = 1000
    epochs_arr = np.linspace(0, epochs - 1, epochs)
    lam = 0.0

    betas_result, MSE_result = SGD(X=X, y=z, initialization=None, eta=0.01, epochs=epochs, lam=lam, batch_size=15)
    print(np.mean(MSE_result))

    batch_size = np.array([5,10,20,30,40])
    for i in batch_size:
        # Perform SGD
        betas_result, MSE_result = SGD(X=X, y=z, initialization=None, eta=0.001, epochs=epochs, lam=lam, batch_size=i)
        if lam == 0:
            plt.title("OLS regression with SGD using batches")
        else:
            plt.title(f"SGD with mini-batches with Ridge regression and penalty={lam:1.2f}")
        plt.plot(epochs_arr, MSE_result, label="No. of mini-batches=%s" %i)
        plt.legend()

    plt.ylabel("MSE")
    plt.xlabel("Epochs")
    plt.show()

    etas = np.array([0.01, 0.001, 0.0001, 0.00001])
    for i in etas:
        # Perform SGD
        betas_result, MSE_result = SGD(X=X, y=z,initialization=None, eta=i, epochs=epochs, lam=lam, batch_size=15)
        if lam == 0:
            plt.title("OLS regression with SGD and batches")
        else:
            plt.title(f"Ridge regression with SGD and batches, penalty={lam:1.2f}")
        plt.plot(epochs_arr, MSE_result, label=r"$\eta=$"+f"{i:1.1e}")
        plt.legend()

    plt.ylabel("MSE")
    plt.xlabel("Epochs")
    plt.show()

    log_lambdas = np.linspace(-6, 1, 8)
    lambdas = 10 ** log_lambdas
    MSE = np.zeros((len(etas), len(lambdas)))
    betas_result = np.zeros((len(etas), len(lambdas)))
    for i, eta in enumerate(etas):
        for j, lam in enumerate(lambdas):
             b, m = SGD(X=X, y=z, initialization=None, eta=eta, epochs=epochs, lam=lam, batch_size=15)
             MSE[i,j] = np.mean(np.array(m))
    plt.title(f"MSE of SGD with 15 batches with L2 penalty and {epochs:1.0f} epochs")
    sns.heatmap(
        MSE.T,
        cmap="RdYlGn_r",
        xticklabels=[str(eta) for eta in etas],
        yticklabels=[str(lam) for lam in log_lambdas],
        annot=True,
        fmt="1.2f",
    )
    plt.xlabel("Learning rate")
    plt.ylabel("log(L2 penalty)")
    plt.show()
