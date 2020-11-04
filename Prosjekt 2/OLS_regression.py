import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import floor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from franke import make_data

def OLS_CV(K, degree, x, y, z):
    """
    Function takes no. of folds(K) as argument and
    model complexity(polynomial degrees) and cross validates
    K times. Plots MSE of test set vs. polynomial degree.

    Params:
    K: int
        integer of no. of folds in CV.
    degree: list
        list of degrees to include in model.
    z: array
        array of response data.
    """

    z_OLS_pred_test = []
    z_OLS_pred_train = []
    MSE = []  # MSE for every fold
    MSE_train = []
    MSE_for_every_degree = []
    MSE_for_every_degree_train = []

    for i in degree:
        X = PolynomialFeatures(i).fit_transform(np.column_stack((x, y)))

        # Split design matrix into train- and test-set
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

        # Scale data
        scaler1 = StandardScaler(with_std=False).fit(X_train)
        X_train, X_test = scaler1.transform(X_train), scaler1.transform(X_test)
        scaler2 = StandardScaler(with_std=False).fit(z_train)
        z_train, z_test = scaler2.transform(z_train), scaler2.transform(z_test)

        # Fold length
        length = floor(z_train.shape[0] / K)

        for j in range(K):
            begin = j * length
            end = (j + 1) * length

            # split training set into temp and validation set
            X_vali = X_train[begin:end]
            z_vali = z_train[begin:end]

            # k-1 folds used to train the model
            X_temp = np.concatenate((X_train[:begin], X_train[end:]), axis=0)
            z_temp = np.concatenate((z_train[:begin], z_train[end:]), axis=0)

            # fit
            reg = LinearRegression()
            reg.fit(X_temp, z_temp)

            # predict
            z_OLS_pred_test.append(reg.predict(X_vali))  # X @ coef.T
            z_OLS_pred_train.append(reg.predict(X_temp))

            # Evaluate model based on validation data
            MSE.append(np.mean((z_vali - z_OLS_pred_test[-1]) ** 2))
            MSE_train.append(np.mean((z_temp - z_OLS_pred_train[-1]) ** 2))

        MSE_for_every_degree.append(np.mean(np.array(MSE)))
        MSE_for_every_degree_train.append(np.mean(np.array(MSE_train)))

    best_index = np.argmin(MSE_for_every_degree)
    print(
        f"Best degree={degree[best_index]:1.1f}, MSE={MSE_for_every_degree[best_index]:1.4f}"
    )

    plt.plot(degree, MSE_for_every_degree, label="Test-set")
    plt.plot(degree, MSE_for_every_degree_train, label="Training-set")
    plt.legend(loc="best")
    plt.title("OLS with 5-fold CV")
    plt.xlabel("Polynomial degree / Model complexity")
    plt.ylabel("MSE")
    plt.show()

if __name__ == "__main__":
    x, y, z = make_data(1000)
    degree = [i for i in range(1,20+1)]
    K = 5
    OLS_CV(K, degree, x, y, z)
