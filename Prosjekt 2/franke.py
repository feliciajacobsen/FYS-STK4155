import numpy as np

def FrankeFunction(x, y):
    """
    Defining Franke function that take two
    independent variables and returns franke Function
    fit on these two variables.

    Params:
    --------
    x: Array
        independent input variable
    y: Array
        independent input variable

    Returns:
    --------
        array with same dimension as x and y.
    """
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)

    return term1 + term2 + term3 + term4

def make_data(n_points):
    """
    Generate random x and ydata which is randomly distributed
    between 0 and 1.

    Params:
    --------
    n_points: int
        no. of observations for our dataset

    Returns:
    --------
        x : Array
            1D array of random values between 0 and 1
        y : Array
            1D array of random values between 0 and 1
        z : Array
            1D array of Franke function output with added noise
    """
    np.random.seed(22) # Ensures we get same dataset everytime make_data is called
    x = np.random.rand(n_points).reshape(-1, 1)
    y = np.random.rand(n_points).reshape(-1, 1)
    z = FrankeFunction(x,y) + np.random.normal(0,0.2,x.shape)
    
    return x, y, z
