import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import AdaBoostClassifier
# Local files:
from neural_network import FFNN


def corr_matrix(X, y):
    """
    Function calculates and plots correlation matrix of input
    data X and target data y.

    Params:
    -------
        X: array
            array of input data
        y: 1d array
            vector of target data

    Returns:
    -------
        None

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    labels = load_breast_cancer().feature_names
    df = pd.DataFrame(data=X_train)
    sns.set(font_scale=0.5)
    sns.heatmap(
        df.corr().abs(),
        annot=True,
        fmt="1.1f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
    )
    plt.tight_layout()
    plt.show()


def feature_treshold(X, thresh):
    """
    Function calculates Pearson correlation matrix and selects upper triangular
    matrix in order to select out one in a pair of features to select out
    according to predetermined threshold value.

    Params:
    -------
        X: Array
            array of input training data
        thresh: float between 0 and 1
            threhold used for filtering features.

    Returns:
    -------
        List of indices of kept features with respect to input data X.
    """
    labels = load_breast_cancer().feature_names
    df = pd.DataFrame(data=X)

    # Absolute value of correlation matrix
    corr = df.corr().abs()

    # Select upper triangular matrix and remove diag elements
    upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

    # Indexes of dropped features which is over treshold value of 0.95
    dropped_features = [
        column for column in upper_tri.columns if any(upper_tri[column] > thresh)
    ]

    kept_features = [x for x in range(X_train.shape[1]) if x not in dropped_features]

    # Printing labels of features which has been dropped
    #print(labels[dropped_features])

    return kept_features



def recursive_feat_elimination(X, y, method, no_of_features):
    """
    Function reduces features down to a predetermined no. of features.
    The function trains with AdaBoostClassifier for every iteration until
    desiered number of features is reached.

    Params:
    --------
        X: array
            input training data
        y: 1d array
            target training data
        method: string
            performs feature selection only if mehtod is "boost"
        no_of_features: int
            Number of desiered features to keep in data set.

    Returns:
    --------
        Indices of kept features with respect to input data X.

    """
    labels = load_breast_cancer().feature_names
    featues_idx = np.arange(X.shape[1])

    if method == "boost":
        model = AdaBoostClassifier(n_estimators=300, algorithm="SAMME.R", learning_rate=1)
        rfe = RFE(estimator=model, n_features_to_select=no_of_features)
        rfe.fit(X, y)
        features_idx = rfe.support_

    elif method == "network":
        print("Warning: Gini index of Neural Network can not be extracted.")

    else:
        print("Warning: Classification model", model, "is not implemented.")

    # Print names of dropped features
    dropped_feat = [x for x in range(X_train.shape[1]) if x not in features_idx]

    return features_idx


def ada_boost(X_train, y_train, X_test, y_test):
    """
    Function takes training input and target data which is split into
    training and test set beforehand. The function plots a confusion matrix
    as well as three different accuracy scores.


    Params:
    -------
        X_train: Array
            Array of training input data
        y_train: 1D array
            1D array or vector which contains training output/target data.
        X_test: Array
            array of test input data used for prediction.
        y_test: 1D array
            1d array or vector which contains test target data used for scoring
            the prediction of the boosting model.

    Returns:
    -------
        None


    """
    abc1 = AdaBoostClassifier(n_estimators=300, algorithm="SAMME.R")
    abc2 = AdaBoostClassifier(n_estimators=300, algorithm="SAMME.R")
    abc3 = AdaBoostClassifier(n_estimators=300, algorithm="SAMME.R")

    abc1.fit(X_train, y_train)
    print("Accuracy obtained from AdaBoost with all features:")
    print(accuracy_score(y_test, abc1.predict(X_test)))

    # Selected features off RFE
    N = 25 # No. of features to select by RFE
    new_idx = recursive_feat_elimination(X_train, y_train, "boost", N)  # Index of kept features
    abc2.fit(X_train[:, new_idx], y_train)  # Train on training set with reduced features
    y_pred = abc2.predict(
        X_test[:, new_idx]
    )  # Predict on test set with reduced features
    print("Accuracy obtained from AdaBoost and RFE of", N ,"features:")
    print(accuracy_score(y_test, y_pred))  # Print accuracy score

    # Plot confusion matrix
    plt.title("Accuracy scores of Wisconsin breast cancer dataset")
    sns.heatmap(confusion_matrix(y_test, y_pred), cmap="Blues", annot=True, fmt="d")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

    # Selected features from correlation filter
    thresh = 0.95
    new_idx = feature_treshold(X_train, thresh)
    abc3.fit(X_train[:, new_idx], y_train)
    y_pred = abc3.predict(X_test[:, new_idx])

    print("Accuracy obtained from AdaBoost and feature treshold of", thresh, ":")
    print(accuracy_score(y_test, y_pred))



def from_prob_to_class(y_pred):
    """
    Function takes array of probabilities and make the array binary to match
    true target array.

    Params:
    --------
        y_pred: vector
            1D array of probabilities of having in benign cancer.

    Returns:
    --------
        1D array of predicted classes consisting of 0=Malignant or 1=Benign
    """

    N = y_pred.shape[0]
    y_pred_new = np.zeros(N)
    for i in range(N):
        if y_pred[i] >= 0.5:
            y_pred_new[i] = 1
        else:
            y_pred_new[i] = 0
    return y_pred_new


def find_stumps(X_train, y_train, X_test, y_test):
    """
    Function plots the accuracy for different numbers of estimatros
    in the AdaBoosted forest.

    Params:
    --------
        X_train: Array
            Array of training input data
        y_train: 1D array
            1D array or vector which contains training output/target data.
        X_test: Array
            array of test input data used for prediction.
        y_test: 1D array
            1d array or vector which contains test target data used for scoring
            the prediction of the boosting model.

    Returns:
    --------
        None
    """
    N = 200
    y_pred = []
    acc = []
    model = []
    for i in range(1, N+1):
        reg = AdaBoostClassifier(n_estimators=i, algorithm="SAMME.R")
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        acc.append(accuracy_score(y_train, np.array(y_pred)))
        del reg
    plt.plot(np.linspace(0,N-1,N), acc)
    plt.xlabel("Number of estimators")
    plt.ylabel("Accuracy score")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    data = load_breast_cancer(return_X_y=True) # return data as tuple
    X = np.array(data[0]) # transform tuple to array
    y = np.array(data[1])  # 1 = Benign, 0 = Malignant

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #find_stumps(X_train, y_train, X_test, y_test)


    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    corr_matrix(X, y)
    feature_treshold(X_train)
    recursive_feat_elimination(X_train, y_train, "boost", 20)
    ada_boost(X_train, y_train.ravel(), X_test, y_test.ravel())

    new_idx = feature_treshold(X_train, 0.95)
    net = FFNN(layers=[len(new_idx), 300, 200, 150, 100, 70, 25, 1],
        activation_functions=["tanh", "tanh", "tanh", "tanh", "tanh", "tanh", "sigmoid"])

    # Train on training set
    net.back_prop(X_train[:, new_idx], y_train, learning_rate=0.01, epochs=1000, mini_batches=30)
    y_pred = net.forward_pass(X_test[:, new_idx])
    y_pred_new = from_prob_to_class(y_pred)
    print(accuracy_score(y_test, y_pred_new))
