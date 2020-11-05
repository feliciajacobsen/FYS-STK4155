import numpy as np
import matplotlib.pyplot as plt


class logistic_regression:
    """
    Estimate logistic regression coefficients
    using stochastic gradient descent with mini-batches.

    Params:
    
    """
    def __init__(self, X, y, betas, eta, epochs, lam, batch_size):
