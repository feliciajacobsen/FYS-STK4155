import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from imageio import imread

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

from math import floor

np.random.seed(22)


def FrankeFunction(x, y):
    """
    Defining Franke function that take two
    independent variables and returns
    """
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


# Generate data
n_points = 1000
x = np.random.rand(n_points).reshape(-1, 1)
y = np.random.rand(n_points).reshape(-1, 1)


def design_matrix(x, y, degree):
    # return PolynomialFeatures(degree).fit_transform(np.column_stack((x, y)))
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)
    N = len(x)
    # calculating no. of features from the choice of degree
    l = int((degree + 1) * (degree + 2) / 2)
    X = np.ones((N, l))

    for i in range(1, degree + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x ** (i - k)) * (y ** k)

    return X


def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def MSE_R2_vs_poly(x, y, degrees, z):
    z_OLS_pred_test, z_OLS_pred_train = [], []
    MSE_test, MSE_train = [], []
    R2_test, R2_train = [], []

    for degree in degrees:
        X = design_matrix(x, y, degree)

        # Split design matrix into train- and test-set
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

        # Scale data
        scaler = StandardScaler(with_std=False).fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
        scaler2 = StandardScaler(with_std=False).fit(z_train)
        z_train, z_test = scaler2.transform(z_train), scaler2.transform(z_test)

        # Model fitted on training set
        OLS = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train

        # Predicted response based on trained OLS model
        z_OLS_pred_test.append((X_test @ OLS))
        z_OLS_pred_train.append((X_train @ OLS))

        # Computing MSE for test set based on OLS estimates
        MSE_test.append(np.mean((z_test - z_OLS_pred_test[-1]) ** 2))
        MSE_train.append(np.mean((z_train - z_OLS_pred_train[-1]) ** 2))

        # Computing R2
        R2_test.append(R2(z_test, z_OLS_pred_test[-1]))
        R2_train.append(R2(z_train, z_OLS_pred_train[-1]))

    return OLS, MSE_test, MSE_train, R2_test, R2_train


def plot_MSE_R2_vs_degree(x, y, degrees, z):
    OLS, MSE_test, MSE_train, R2_test, R2_train = MSE_R2_vs_poly(x, y, degrees, z)

    best_index = np.argmin(MSE_test)
    print(
        f"Best degree= {degrees[best_index]:1.1f} with MSE={MSE_test[best_index]:1.2f}, and R2={R2_test[best_index]:1.2f}."
    )

    plt.subplot(211)
    plt.title("OLS")
    plt.title("Training and test prediction scores for OLS")
    plt.plot(degrees, MSE_test, label="Test-set")
    plt.plot(degrees, MSE_train, label="Training-set")
    plt.ylabel("MSE")
    plt.legend(loc="best")

    plt.subplot(212)
    plt.plot(degrees, R2_test, label="Test-set")
    plt.plot(degrees, R2_train, label="Training-set")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("R2 score")
    plt.legend(loc="best")
    plt.show()

    # Kjoreeksempel:
    # Best degree= 5.0 with MSE=0.04, and R2=0.69.
    return None


def betas_variance_plot(degree):
    """
    Computing variance for every estimator
    in OLS with a feature matrix up 5th degree
    """
    X = design_matrix(x, y, degree)
    betas, MSE_test, MSE_train, R2_test, R2_train = MSE_R2_vs_poly(x, y, [5], z)
    variance = []
    for i in range(len(betas)):
        variance.append(0.2 ** 2 * np.linalg.inv(X.T @ X)[i, i])
    x_arranged = np.linspace(0, X.shape[1] - 1, X.shape[1])
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.errorbar(x_arranged, betas, yerr=variance, fmt="b.", capsize=3.5)
    plt.ylabel(r"$\hat{\beta}^{OLS}$")
    plt.title(f"OLS estimators with corresponding variance, degree={degree:1.1f}")
    ax.set_xticks(x_arranged.tolist())
    ax.set_xticklabels(
        (
            r"$1$",
            r"$x$",
            r"$y$",
            r"$x^2$",
            r"$xy$",
            r"$y^2$",
            r"$x^3$",
            r"$x^2 y$",
            r"$x y^2$",
            r"$y^3$",
            r"$x^4$",
            r"$x^3 y$",
            r"$x^2 y^2$",
            r"$x y^3$",
            r"$y^4$",
            r"$x^5$",
            r"$x^4 y$",
            r"$x^3 y^2$",
            r"$x^2 y^3$",
            r"$x y^4$",
            r"$y^5$",
        ),
        fontdict={"fontsize": 6},
    )
    plt.show()

    return None


def bootstrap_indices(N):
    boot_index = np.random.randint(0, N, N)

    # Out of bag samples, indices not in bootstrap sample
    OOB_index = [x for x in range(N) if x not in boot_index]

    return boot_index, OOB_index


def bootstrap_bias_variance_plot(degrees, n_bootstraps):
    z_OLS_pred_test, z_OLS_pred_train = [], []
    MSE_test_b, MSE_train_b = [], []
    MSE_test, MSE_train = [], []
    R2_test, R2_train = [], []
    # variance, bias_squared = [], []
    var, bias = np.zeros(len(degrees)), np.zeros(len(degrees))
    mean_pred = np.zeros(len(degrees))
    z = FrankeFunction(x, y)

    lins = np.linspace(0, 1, 101)
    xm, ym = np.meshgrid(lins, lins)

    for j, degree in tqdm(enumerate(degrees)):
        X = design_matrix(x, y, degree)

        # Split design matrix into train- and test-set
        X_train, X_test, f_train, f_test = train_test_split(X, z, test_size=0.2)
        z_train = f_train + np.random.normal(0, 0.2, f_train.shape)
        z_test = f_test + np.random.normal(0, 0.2, f_test.shape)
        f_test = f_test.reshape(-1)  # NB! important reshape!

        # Scale data
        scaler = StandardScaler(with_std=False).fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Store results in matrix
        z_pred = np.zeros((n_bootstraps, X_test.shape[0]))

        # 3D plot of meshgrid (for visualization)
        # reg = LinearRegression()
        # reg.fit(X_train, z_train)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        # X_mesh = design_matrix(xm.flatten(),  ym.flatten(), degree)
        # ax.plot_surface(xm, ym, reg.predict(X_mesh).reshape(xm.shape))
        # plt.show()

        for i in range(n_bootstraps):
            # Split design matrix into train- and test-set
            train_index, test_index = bootstrap_indices(X_train.shape[0])

            # Split data with bootstrap sampling
            X_temp, X_vali = X_train[train_index], X_train[test_index]
            z_temp, z_vali = z_train[train_index], z_train[test_index]

            # Create model
            reg = LinearRegression()
            reg.fit(X_temp, z_temp)

            MSE_test_b.append((z_test - reg.predict(X_test)) ** 2)
            MSE_train_b.append((z_temp - reg.predict(X_temp)) ** 2)
            z_pred[i, :] = reg.predict(X_test).reshape(
                -1
            )  # must reshape to fit into matrix

        # Compute the variance
        var[j] = np.mean(np.var(z_pred, axis=0))
        # Compute the bias
        bias[j] = np.mean(
            (f_test - np.mean(z_pred, axis=0)) ** 2
        )  # average over data points, then average over bootstraps
        # print(f_test.shape, (np.mean(z_pred, axis=0)).shape)

        # Taking mean of MSE evaluated by bootstrap samples of every degree
        MSE_test.append(np.mean(np.array(MSE_test_b)))
        MSE_train.append(np.mean(np.array(MSE_train_b)))
    best_index = np.argmin(MSE_test)
    print(
        f"Best degree= {degrees[best_index]:1.1f} with MSE={MSE_test[best_index]:1.2f}"
    )

    plt.figure(1)
    plt.title(
        f"MSE w bootstrap, no. datapoints={n_points:1.1f} ,resampling={n_bootstraps:1.1f} times"
    )
    plt.plot(degrees, MSE_test, label="Test-set")
    plt.plot(degrees, MSE_train, label="Training-set")
    plt.ylabel("MSE")
    plt.xlabel("Polynomial degree")
    plt.legend(loc="best")

    plt.figure(2)
    plt.title("Bias variance of OLS with bootstrap")
    plt.plot(degrees, var, "--", label="Variance")
    plt.plot(degrees, bias, "--", label=r"$Bias^2$")
    plt.plot(degrees, MSE_test, label="MSE test")
    plt.ylabel("Error")
    plt.xlabel("Polynomial degree")
    plt.legend(loc="best")
    plt.show()


def OLS_CV(K, degree, z):
    """
    Function takes no. of folds(K) as argument and
    model complexity(polynomial degrees) and cross validates
    K times. Plots MSE of test set vs. polynomial degree.
    """

    z_OLS_pred_test = []
    z_OLS_pred_train = []
    MSE = []  # MSE for every fold
    MSE_train = []
    MSE_for_every_degree = []
    MSE_for_every_degree_train = []

    for i in degree:
        X = design_matrix(x, y, i)

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
        f"Best degree={degree[best_index]:1.1f}, MSE={MSE_for_every_degree[best_index]:1.2f}"
    )

    plt.plot(degree, MSE_for_every_degree, label="Test-set")
    plt.plot(degree, MSE_for_every_degree_train, label="Training-set")
    plt.legend(loc="best")
    plt.title("OLS with 5-fold CV")
    plt.xlabel("Polynomial degree / Model complexity")
    plt.ylabel("MSE")
    plt.show()


def Ridge_CV(K, degrees, z):
    """
    Function tunes polynomial degree in model
    and penalty lambda in Ridge by CV
    """

    log_lambdas = np.linspace(-5, 1, 7)
    lambdas = 10 ** log_lambdas

    # Matrix with MSE values corresponding to a given value of polynomial degreee
    # and a given lambda parameter
    MSE_degree_lambda = np.zeros((len(degrees), len(lambdas)))
    z_ridge_pred_test = []

    # Loop over model complexity
    for i, deg in enumerate(degrees):
        X = design_matrix(x, y, deg)
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

                Ridge = (
                    np.linalg.inv(
                        X_temp.T @ X_temp + lam * np.identity(X_temp.shape[1])
                    )
                    @ X_temp.T
                    @ z_temp
                )

                z_ridge_pred_test.append((X_vali @ Ridge))

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

    # Kjoreeksempel : Ridge CV: Best degree=6.0, with best lambda=1.0e-05


def Ridge_bootstrap(degrees, n_bootstraps, z):
    z_ridge_pred_test = []
    log_lambdas = np.linspace(-5, 1, 7)
    lambdas = 10 ** log_lambdas
    MSE_degree_lambda = np.zeros((len(degrees), len(lambdas)))

    for i, deg in enumerate(degrees):
        X = design_matrix(x, y, deg)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

        scaler1 = StandardScaler(with_std=False).fit(X_train)
        scaler2 = StandardScaler(with_std=False).fit(z_train)
        X_train, X_test = scaler1.transform(X_train), scaler1.transform(X_test)
        z_train, z_test = scaler2.transform(z_train), scaler2.transform(z_test)

        for j, lam in enumerate(lambdas):
            MSE = []

            for b in range(n_bootstraps):
                train_index, test_index = bootstrap_indices((X_train.shape[0]))
                X_temp, X_vali = X_train[train_index], X_train[test_index]
                z_temp, z_vali = z_train[train_index], z_train[test_index]

                scaler = StandardScaler().fit(X_temp)
                X_temp = scaler.transform(X_temp)
                X_vali = scaler.transform(X_vali)

                reg = Ridge(alpha=lam)
                reg.fit(X_temp, z_temp)

                z_ridge_pred_test.append(reg.predict(X_vali))

                MSE.append(mean_squared_error(z_vali, z_ridge_pred_test[-1]))

            MSE_degree_lambda[i, j] = np.mean(np.array(MSE))

    # Obtain tuned degree and lambda
    best_index = np.argwhere(MSE_degree_lambda == np.min(MSE_degree_lambda))
    print(
        f"Ridge bootstrap: Best degree={degrees[best_index[0,0]]:1.1f}, with best lambda={lambdas[best_index[0,1]]:1.1e}"
    )

    # Heatmap of MSE
    plt.title(f"MSE of Ridge with {n_bootstraps:1.1f} bootstraps")
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

    # Kjoreeksempel: Ridge bootstrap: Best degree=9.0, with best lambda=1.0e-02


def Lasso_CV(K, degrees, log_lambdas, z):
    lambdas = 10 ** log_lambdas
    MSE_degree_lambda = np.zeros((len(degrees), len(lambdas)))
    z_lasso_pred_test = []

    for i, deg in enumerate(degrees):
        X = design_matrix(x, y, deg)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

        scaler1 = StandardScaler(with_std=False).fit(X_train)
        scaler2 = StandardScaler(with_std=False).fit(z_train)
        X_train, X_test = scaler1.transform(X_train), scaler1.transform(X_test)
        z_train, z_test = scaler2.transform(z_train), scaler2.transform(z_test)

        length = floor(z_train.shape[0] / K)

        for j, lam in enumerate(lambdas):
            MSE = []
            for k in range(K):
                begin = k * length
                end = (k + 1) * length

                X_vali = X_train[begin:end]
                z_vali = z_train[begin:end]

                X_temp = np.concatenate((X_train[:begin], X_train[end:]), axis=0)
                z_temp = np.concatenate((z_train[:begin], z_train[end:]), axis=0)

                reg = Lasso(alpha=lam, max_iter=10000, tol=1e-4)
                reg.fit(X_temp, z_temp)
                z_lasso_pred_test.append(reg.predict(X_vali))

                MSE.append(mean_squared_error(z_vali, z_lasso_pred_test[-1]))
            MSE_degree_lambda[i, j] = np.mean(np.array(MSE))

    # Plot, heatmap
    plt.title(f"MSE of Lasso with {K:1.1f}-fold CV")
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
        f"Lasso CV : Best degree={degrees[best_index[0,0]]:1.0f}, with best lambda={lambdas[best_index[0,1]]:1.1e}"
    )

    # Kjoreeksempel: Lasso CV : Best degree=5, with best lambda=1.0e-05


def Lasso_bootstrap(degrees, log_lambdas, n_bootstraps, z):
    z_lasso_pred_test = []
    lambdas = 10 ** log_lambdas
    MSE_degree_lambda = np.zeros((len(degrees), len(lambdas)))

    for i, deg in enumerate(degrees):
        X = design_matrix(x, y, deg)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

        scaler1 = StandardScaler(with_std=False).fit(X_train)
        scaler2 = StandardScaler(with_std=False).fit(z_train)
        X_train, X_test = scaler1.transform(X_train), scaler1.transform(X_test)
        z_train, z_test = scaler2.transform(z_train), scaler2.transform(z_test)

        for j, lam in enumerate(lambdas):
            MSE = []
            var_b, bias_b = [], []
            for b in range(n_bootstraps):
                train_index, test_index = bootstrap_indices((X_train.shape[0]))
                X_temp, X_vali = X_train[train_index], X_train[test_index]
                z_temp, z_vali = z_train[train_index], z_train[test_index]

                reg = Lasso(alpha=lam, max_iter=10000, tol=1e-4)
                reg.fit(X_temp, z_temp)

                z_lasso_pred_test.append(reg.predict(X_vali))

                MSE.append(mean_squared_error(z_vali, z_lasso_pred_test[-1]))
            MSE_degree_lambda[i, j] = np.mean(np.array(MSE))

    best_index = np.argwhere(MSE_degree_lambda == np.min(MSE_degree_lambda))
    print(
        f"Lasso bootstrap: Best degree={degrees[best_index[0,0]]:1.1f}, with best lambda={lambdas[best_index[0,1]]:1.1e}"
    )

    # Heatmap of MSE
    plt.title(f"MSE of Lasso with {n_bootstraps:1.1f} bootstraps")
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

    # Kjoreeksempel: Lasso bootstrap: Best degree=7.0, with best lambda=1.0e-05


def terrain(K, degrees, log_lambdas, method):
    # Load the terrain
    z_mesh = imread("lyngdal.tif")  # 3601 x 1801 data
    x = np.linspace(0, z_mesh.shape[0] - 1, z_mesh.shape[0]).reshape(-1, 1)
    y = np.linspace(0, z_mesh.shape[1] - 1, z_mesh.shape[1]).reshape(-1, 1)

    # Make 3D plot of MSE of test for lambas and degree
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    x_mesh, y_mesh = np.meshgrid(x, y)
    ax.plot_surface(x_mesh, y_mesh, z_mesh.T, cmap="winter")
    plt.show()

    # cut down data to 100000 points
    x, y, z = (
        x_mesh.flatten().reshape(-1, 1),
        y_mesh.flatten().reshape(-1, 1),
        z_mesh.T.flatten().reshape(-1, 1),
    )

    N = np.array(range(x.shape[0]))
    kept_data = np.random.permutation(N)[:10000]
    x_reduced, y_reduced, z_reduced = x[kept_data], y[kept_data], z[kept_data]

    lambdas = 10 ** log_lambdas
    MSE_degree_lambda = np.zeros((len(degrees), len(lambdas)))
    z_pred_test = []

    if method == "OLS":
        MSE_for_every_fold, MSE_test = [], []
        for i, deg in enumerate(degrees):
            X = design_matrix(x_reduced, y_reduced, deg)
            X_train, X_test, z_train, z_test = train_test_split(
                X, z_reduced, test_size=0.2
            )

            scaler1 = StandardScaler(with_std=False).fit(X_train)
            scaler2 = StandardScaler(with_std=False).fit(z_train)
            X_train, X_test = scaler1.transform(X_train), scaler1.transform(X_test)
            z_train, z_test = scaler2.transform(z_train), scaler2.transform(z_test)

            a = np.random.permutation(X_train.shape[0])

            length = floor(z_train.shape[0] / K)
            for k in range(K):
                begin = k * length
                end = (k + 1) * length
                X_vali = X_train[a[begin:end]]
                z_vali = z_train[a[begin:end]]
                X_temp = np.concatenate((X_train[a[:begin]], X_train[a[end:]]), axis=0)
                z_temp = np.concatenate((z_train[a[:begin]], z_train[a[end:]]), axis=0)
                reg = LinearRegression()
                reg.fit(X_temp, z_temp)
                z_pred_test.append(reg.predict(X_vali))
                MSE_for_every_fold.append(np.mean((z_vali - z_pred_test[-1]) ** 2))
            MSE_test.append(np.mean(np.array(MSE_for_every_fold)))
        best_index = np.argmin(MSE_test)
        print(
            f"Best degree with OLS is degree={degrees[best_index]:1.1f}, with MSE={MSE_test[best_index]:1.3f}"
        )
        plt.title("5-fold CV on reduced real terrain data, 10000 points")
        plt.plot(degrees, MSE_test, label="MSE-test")
        plt.plot(degrees[best_index], MSE_test[best_index], ".", label="best MSE")
        plt.legend(loc="best")
        plt.show()

        X_design = design_matrix(x, y, degrees[best_index])
        X_train, X_test, z_train, z_test = train_test_split(X_design, z, test_size=0.2)
        reg = LinearRegression()
        reg.fit(X_train, z_train)
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.plot_surface(x_mesh, y_mesh, z_mesh.T, alpha=0.8)
        ax.plot_surface(
            x_mesh, y_mesh, reg.predict(X_design).T.reshape(x_mesh.shape), alpha=0.8
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    else:
        for i, deg in enumerate(degrees):
            X = design_matrix(x_reduced, y_reduced, deg)
            X_train, X_test, z_train, z_test = train_test_split(
                X, z_reduced, test_size=0.2
            )

            scaler1 = StandardScaler(with_std=False).fit(X_train)
            scaler2 = StandardScaler(with_std=False).fit(z_train)
            X_train, X_test = scaler1.transform(X_train), scaler1.transform(X_test)
            z_train, z_test = scaler2.transform(z_train), scaler2.transform(z_test)

            length = floor(z_train.shape[0] / K)
            a = np.random.permutation(X_train.shape[0])
            for j, lam in enumerate(lambdas):
                MSE = []
                for k in range(K):
                    begin = k * length
                    end = (k + 1) * length
                    X_vali = X_train[a[begin:end]]
                    z_vali = z_train[a[begin:end]]
                    X_temp = np.concatenate(
                        (X_train[a[:begin]], X_train[a[end:]]), axis=0
                    )
                    z_temp = np.concatenate(
                        (z_train[a[:begin]], z_train[a[end:]]), axis=0
                    )

                    if method == "Lasso":
                        reg = Lasso(alpha=lam, max_iter=10000, tol=1e-4)

                    if method == "Ridge":
                        reg = Ridge(alpha=lam)

                    reg.fit(X_temp, z_temp)
                    z_pred_test.append(reg.predict(X_vali))
                    MSE.append(mean_squared_error(z_vali, z_pred_test[-1]))
                MSE_degree_lambda[i, j] = np.mean(np.array(MSE))
        # Obtain tuned degree and lambda
        best_index = np.argwhere(MSE_degree_lambda == np.min(MSE_degree_lambda))
        print(
            method
            + f" CV : Best degree={degrees[best_index[0,0]]:1.0f}, with best lambda={lambdas[best_index[0,1]]:1.1e}"
        )

        # Plot, heatmap
        plt.title(f"MSE of " + method + f" with {K:1.1f}-fold CV")
        sns.heatmap(
            MSE_degree_lambda.T,
            cmap="RdYlGn_r",
            xticklabels=[str(deg) for deg in degrees],
            yticklabels=[str(lam) for lam in log_lambdas],
            annot=True,
            fmt="1.1e",
        )
        plt.xlabel("Poly. degree")
        plt.ylabel(r"$log_{10} \lambda$")
        plt.show()

        """
        Kjoreeksempel: Ridge CV : Best degree=14, with best lambda=1.0e-05
        """


if __name__ == "__main__":
    np.random.seed(22)
    p = [i for i in range(1, 10)]
    z = FrankeFunction(x, y) + np.random.normal(0, 0.2, x.shape)
    #plot_MSE_R2_vs_degree(x, y, p, z)
    #bootstrap_bias_variance_plot(p, 100)
    #betas_variance_plot(5)
    #OLS_CV(5, p, z)
    #Ridge_CV(5, [i for i in range(3,10+1)], z)
    #Ridge_bootstrap([i for i in range(3,10+1)],100, z)
    #Lasso_CV(5, [i for i in range(3, 10 + 1)], np.linspace(-5, 1, 7), z)
    #Lasso_bootstrap([i for i in range(3, 10 + 1)], np.linspace(-5, 1, 7), 100, z)
    #terrain(5, [i for i in range(3, 15 + 1)], np.linspace(-5, 1, 7), "OLS")
    #terrain(5, [i for i in range(5, 15 + 1)], np.linspace(-5, 1, 7), "Ridge")
    #terrain(5, [i for i in range(5, 15 + 1)], np.linspace(-5, 1, 7), "Lasso")
