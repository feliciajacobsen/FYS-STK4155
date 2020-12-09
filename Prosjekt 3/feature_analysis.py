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
    plt.show()


def feature_treshold(X, y):
    labels = load_breast_cancer().feature_names
    df = pd.DataFrame(data=X)

    # Absolute value of correlation matrix
    corr = df.corr().abs()

    # Select upper triangular matrix and remove diag elements
    upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

    # Indexes of dropped features which is over treshold value of 0.95
    dropped_features = [
        column for column in upper_tri.columns if any(upper_tri[column] > 0.95)
    ]

    kept_features = [x for x in range(X_train.shape[1]) if x not in dropped_features]

    # Printing labels of features which has been dropped
    #print(labels[dropped_features])

    return kept_features



def recursive_feat_elimination(X, y, method, no_of_features):
    labels = load_breast_cancer().feature_names
    featues_idx = np.arange(X.shape[1])

    if method == "boost":
        model = AdaBoostClassifier(n_estimators=200, algorithm="SAMME.R", learning_rate=0.5)
        rfe = RFE(estimator=model, n_features_to_select=no_of_features)
        rfe.fit(X, y)
        features_idx = rfe.support_

    elif method == "network":
        print("Warning: Gini index of Neural Network can not be extracted.")

    else:
        print("Warning: Classification model", model, "is not implemented.")

    # Print names of kept features
    #print(labels[features_idx])

    return features_idx


def ada_boost(X_train, y_train, X_test, y_test):
    abc1 = AdaBoostClassifier(n_estimators=300, algorithm="SAMME.R", learning_rate=0.5)
    abc2 = AdaBoostClassifier(n_estimators=300, algorithm="SAMME.R", learning_rate=0.5)
    abc3 = AdaBoostClassifier(n_estimators=300, algorithm="SAMME.R", learning_rate=0.5)

    abc1.fit(X_train, y_train)
    print("Accuracy obtained from AdaBoost with all features:")
    print(accuracy_score(y_test, abc1.predict(X_test)))

    # Selected features off RFE
    N = 15 # No. of features to select by RFE
    new_idx = recursive_feat_elimination(X_train, y_train, "boost", N)  # Index of kept features
    abc2.fit(X_train[:, new_idx], y_train)  # Train on training set with reduced features
    y_pred = abc2.predict(
        X_test[:, new_idx]
    )  # Predict on test set with reduced features
    print("Accuracy obtained from AdaBoost and RFE of", N ,"features:")
    print(accuracy_score(y_test, y_pred))  # Print accuracy score
    # 0.9824561403508771

    # Plot confusion matrix
    plt.title("Accuracy scores of Wisconsin breast cancer dataset")
    sns.heatmap(confusion_matrix(y_test, y_pred), cmap="Blues", annot=True, fmt="d")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

    # Selected features from treshold of 0.95 correlation index
    new_idx = feature_treshold(X_train, y_train)
    abc3.fit(X_train[:, new_idx], y_train)
    y_pred = abc3.predict(X_test[:, new_idx])

    print("Accuracy obtained from AdaBoost and feature treshold of 0.95 :")
    print(accuracy_score(y_test, y_pred))
    # 1.0


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



if __name__ == "__main__":
    data = load_breast_cancer(return_X_y=True)
    X = np.array(data[0])
    y = np.array(data[1])  # 1 = Benign, 0 = Malignant

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    #corr_matrix(X, y)
    #feature_treshold(X_train, y_train)
    #recursive_feat_elimination(X_train, y_train, "boost", 20)

    ada_boost(X_train, y_train, X_test, y_test)

    # Train on training set

    new_idx = feature_treshold(X_train, y_train) # np.arange(X_train.shape[1])
    net = FFNN(
        layers=[len(new_idx), 100, 80, 40, 20, 1],
        activation_functions=["tanh", "tanh", "tanh", "tanh", "sigmoid"],
    )
    # accuracy = 0.92 with all features
    # accuracy = 0.9649 with 23
    net.back_prop(X_train[:, new_idx], y_train, learning_rate=0.01, epochs=3000, mini_batches=30)
    y_pred = net.forward_pass(X_test[:, new_idx])
    y_pred_new = from_prob_to_class(y_pred)
    print(accuracy_score(y_test, y_pred_new))
