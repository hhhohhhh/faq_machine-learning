#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   desicion_tree_examples.py
@Version:  
@Desc:
@Time: 2022/6/3 20:25
@Contact :   wang_feicheng@163.com
@License :   Create by wangfc27441 on 2022/6/3, Copyright 2022 wangfc27441. All rights reserved.

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/3 20:25   wangfeicheng      1.0         None
'''

from __future__ import division, print_function
import sys
import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt


# Import helper functions
from utils.ml_utils.data_manipulation import train_test_split
from utils.ml_utils.loss_functions import  accuracy_score,mean_squared_error,calculate_variance
from utils.ml_utils.data_normalization import standardize
from models.decision_tree.decision_tree_model import ClassificationTree,RegressionTree
from models.decision_tree.gbdt_model import GBDTClassifier,GBDTRegressor


def test_classification_tree(test_decision_tree=True,test_gbdt=True):
    print ("-- Classification Tree --")
    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,seed=1234)
    if test_decision_tree:
        clf = ClassificationTree(basetree_type='CART')
        clf.fit(X_train, y_train)

        # clf.print_tree()
        y_train_pred = clf.predict(X_train)
        train_accuracy = accuracy_score(y_train,y_train_pred)
        y_pred = clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print ("{} with tree_depth of {}: train_accuracy={}, test_accuracy= {},"
               .format('Decision_tree',clf._get_tree_depth(), train_accuracy, test_accuracy,))

    if test_gbdt:
        clf = GBDTClassifier(n_estimators=100,max_depth=4)
        clf.fit(X_train, y_train)
        # clf.print_tree()
        y_train_pred = clf.predict(X_train)
        train_accuracy = accuracy_score(y_train,y_train_pred)
        y_pred = clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print ("{} with max_depth of {}: train_accuracy={}, test_accuracy= {}"
               .format('GBDT',clf.max_depth, train_accuracy, test_accuracy))



    # Plot().plot_in_2d(X_test, y_pred,
    #     title="Decision Tree",
    #     accuracy=accuracy,
    #     legend_labels=data.target_names)

def test_regression_tree(test_decision_tree=True,test_gbdt=True):

    print ("-- Regression Tree --")

    # Load temperature data
    data_path = os.path.join('corpus','ml_data','TempLinkoping2016.txt')
    data = pd.read_csv(data_path, sep="\t")

    time = np.atleast_2d(data["time"]).T
    temp = np.atleast_2d(data["temp"]).T

    # X = np.around(time*366,decimals=0)
    X = standardize(time)        # Time. Fraction of the year [0, 1]
    y = temp[:, 0]  # Temperature. Reduce to one-dim

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    if test_decision_tree:
        model = RegressionTree()
        model.fit(X_train, y_train)
        # model.print_tree()
        y_pred = model.predict(X_test)
        y_pred_line = model.predict(X)
        mse = mean_squared_error(y_test, y_pred)
        print ("{} with tree_depth of {} :Mean Squared Error:{}".format("RegressionTree", model._get_tree_depth(),mse))
    if test_gbdt:
        model = GBDTRegressor(n_estimators=100,max_depth=4)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_line = model.predict(X)
        mse = mean_squared_error(y_test, y_pred)
        print ("{} with tree_depth of {} :Mean Squared Error:{}".format("GBDTRegressor", model.max_depth,mse))



    # Color map
    cmap = plt.get_cmap('viridis')

    # Plot the results
    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    m3 = plt.scatter(366 * X_test, y_pred, color='black', s=10)
    plt.suptitle("Regression Tree")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"), loc='lower right')
    plt.show()




def plot_tree_regression():
    """
    https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
    """

    # Import the necessary modules and libraries
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    import matplotlib.pyplot as plt

    # Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(16))

    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_1.fit(X, y)
    regr_2.fit(X, y)

    # Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)

    # Plot the results
    plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
    plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # test_classification_tree()
    test_regression_tree()