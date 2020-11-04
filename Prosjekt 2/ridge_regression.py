import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import floor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from franke import make_data

def Ridge_CV(K, degrees, x, y, z):
    """
    Function tunes polynomial degree in model
    and penalty lambda in Ridge by CV

    Params:
    K: int
        Number of folds
    degrees: list
        List of integers with degrees to use
    z: array
        array of response data

    Returns:
    None
    """

    log_lambdas = np.linspace(-5, 1, 7)
    lambdas = 10 ** log_lambdas


    # Matrix with MSE values corresponding to a given value of polynomial degreee
    # and a given lambda parameter
    MSE_degree_lambda = np.zeros((len(degrees), len(lambdas)))
    z_ridge_pred_test = []

    # Loop over model complexity
    for i, deg in enumerate(degrees):
        X = PolynomialFeatures(deg).fit_transform(np.column_stack((x, y)))
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

        scaler1 = StandardScaler(with_std=False).fit(X_train)
        scaler2 = StandardScaler(with_std=False).fit(z_train)
        X_train, X_test = scaler1.transform(X_train), scaler1.transform(X_test)
        z_train, z_test = scaler2.transform(z_train), scaler2.transform(z_test)

        length = floor(z_train.shape[0] / K)

        # Loop over penalty parameter
        for j, lam in enumerate(lambdas):
            # Loop over folds
            MSE = []
            for k in range(K):
                begin = k * length
                end = (k + 1) * length

                X_vali = X_train[begin:end]
                z_vali = z_train[begin:end]

                X_temp = np.concatenate((X_train[:begin], X_train[end:]), axis=0)
                z_temp = np.concatenate((z_train[:begin], z_train[end:]), axis=0)

                reg = Ridge(alpha=lam)
                reg.fit(X_temp, z_temp)

                z_ridge_pred_test.append(reg.predict(X_vali))

                MSE.append(mean_squared_error(z_vali, z_ridge_pred_test[-1]))
            MSE_degree_lambda[i, j] = np.mean(np.array(MSE))

    # Plot, heatmap
    plt.title(f"MSE of Ridge with {K:1.1f}-fold CV")
    sns.heatmap(
        MSE_degree_lambda.T,
        cmap="RdYlGn_r",
        xticklabels=[str(deg) for deg in degrees],
        yticklabels=[str(lam) for lam in log_lambdas],
        annot=True,
        fmt="1.2f",
    )
    plt.xlabel("Poly. degree")
    plt.ylabel(r"$log_{10} \lambda$")
    plt.show()

    # Obtain tuned degree and lambda
    best_index = np.argwhere(MSE_degree_lambda == np.min(MSE_degree_lambda))
    print(
        f"Ridge CV: Best degree={degrees[best_index[0,0]]:1.1f}, with best lambda={lambdas[best_index[0,1]]:1.1e}"
    )
    print(f"With minimum MSE={np.min(MSE_degree_lambda):1.4f}")

if __name__ == "__main__":
    x, y, z = make_data(1000)
    Ridge_CV(5, [i for i in range(4,10+1)], x, y, z)
