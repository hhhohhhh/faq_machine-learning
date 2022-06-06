#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   loss_functions.py
@Version:  
@Desc:
@Time: 2022/6/3 20:31
@Contact :   wang_feicheng@163.com
@License :   Create by wangfc27441 on 2022/6/3, Copyright 2022 wangfc27441. All rights reserved.

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/3 20:31   wangfeicheng      1.0         None
'''

from __future__ import division
import numpy as np
import math
import sys











def mean_squared_error(y_true, y_pred):
    """ Returns the mean squared error between y_true and y_pred """
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


def calculate_variance(X):
    """ Return the variance of the features in dataset X
    """
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))

    return variance


def calculate_std_dev(X):
    """ Calculate the standard deviations of the features in dataset X """
    std_dev = np.sqrt(calculate_variance(X))
    return std_dev


def euclidean_distance(x1, x2):
    """ Calculates the l2 distance between two vectors """
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def calculate_covariance_matrix(X, Y=None):
    """ Calculate the covariance matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)


def calculate_correlation_matrix(X, Y=None):
    """ Calculate the correlation matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance = (1 / n_samples) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(Y), 1)
    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_y.T))

    return np.array(correlation_matrix, dtype=float)



def calculate_entropy(y):
    """ Calculate the entropy of label array y """
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy


def calculate_gini_index(y):
    """
    @time: 2022/6/4 21:27
    @author: wangfeicheng
    @version: 1.0
    @description:  计算一个集合的 gini 指数

    @params:
    @return:
    """
    unique_labels = np.unique(y)
    gini_index = 0
    for label in unique_labels:
        count =  len(y[y==label])
        p = count/len(y)
        gini_index += p*(1-p)
    return gini_index


class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0


class MeanSquaredLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        # return 0.5 * np.power((y - y_pred), 2)
        return 0.5 * np.sum(np.power((y - y_pred), 2))

    def gradient(self, y, y_pred):
        """
        将 y_pred 看过参数， 对 loss 求导后得到 MeanSquaredLoss  - gradient == residual
        """
        return -(y - y_pred)


class AbsoluteLoss(Loss):
    """This is AbsoluteLoss(Loss)"""

    def __init__(self):
        pass


class HuberLoss(Loss):
    """This is HuberLoss"""

    def __init__(self):
        pass




class SotfMaxLoss(Loss):
    def gradient(self, y, p):
        return y - p


class LeastSquaresLoss():
    """Least squares loss"""

    def gradient(self, actual, predicted):
        return actual - predicted

    def hess(self, actual, predicted):
        return np.ones_like(actual)



class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    # def acc(self, y, p):
    #     return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)



if __name__ == '__main__':
    log2 = lambda x: math.log(x) / math.log(2)
    print(log2(2**9))
    # print(log2)
