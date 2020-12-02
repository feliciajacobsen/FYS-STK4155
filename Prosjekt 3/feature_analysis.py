from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    # Dropping columns with high corr
    new_df = df.drop(df.columns[dropped_features], axis=1)

    # Printing labels of features which has been dropped
    #print(labels[dropped_feat])

    return kept_features


def recursive_feat_elimination(X, y):
    labels = load_breast_cancer().feature_names

    abc = AdaBoostClassifier(
        n_estimators=200,
        algorithm="SAMME.R",
        learning_rate=0.5,
    )
    rfe = RFE(estimator=abc, n_features_to_select=20)
    rfe.fit(X_train, y_train)

    # Print names of kept features
    #print(labels[rfe.support_])

    return rfe.support_


def ada_boost(X_train, y_train, X_test, y_test):
    abc = AdaBoostClassifier(
        n_estimators=200,
        algorithm="SAMME.R",
        learning_rate=0.5,
    )

    # Selected features off RFE
    new_idx = recursive_feat_elimination(X_train, y_train) # Index of kept features
    abc.fit(X_train[:,new_idx], y_train) # Train on training set with reduced features
    y_pred = abc.predict(X_test[:,new_idx]) # Predict on test set with reduced features

    print(accuracy_score(y_test, y_pred)) # Print accuracy score
    # 0.9824561403508771

    # Plot confusion matrix
    plt.title("Accuracy scores of Wisconsin breast cancer dataset")
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        cmap="Blues",
        annot=True,
        fmt="d",
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

    # Selected features from treshold of 0.95 correlation index
    new_idx = feature_treshold(X_train, y_train)
    abc.fit(X_train[:,new_idx], y_train)
    y_pred = abc.predict(X_test[:,new_idx])

    print(accuracy_score(y_test, y_pred))
    # 1.0 on one occasion




if __name__ == "__main__":
    data = load_breast_cancer(return_X_y=True)
    X = np.array(data[0])
    y = np.array(data[1]) # 1 = Benign, 0 = Malignant

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #corr_matrix(X, y)
    #feature_treshold(X_train, y_train)
    #recursive_feat_elimination(X_train, y_train)

    ada_boost(X_train, y_train, X_test, y_test)

    net = FFNN(
        layers=[2, 80, 20, 1],
        activation_functions=["tanh", "leaky_relu", "sigmoid"],
    )
    # Train on training set
    new_idx = feature_treshold(X_train, y_train)
    net.back_prop(X_train[:,new_idx], y_train, learning_rate=0.01, epochs=1000, batch_size=60)
    y_pred = net.forward_pass(X_test[:,new_idx])
    #print(accuracy_score(y_test, y_pred))
