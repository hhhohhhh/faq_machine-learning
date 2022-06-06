#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   gbdt_model.py
@Version:  
@Desc:
@Time: 2022/6/5 16:00
@Contact :   wang_feicheng@163.com
@License :   Create by wangfc27441 on 2022/6/5, Copyright 2022 wangfc27441. All rights reserved.

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/5 16:00   wangfeicheng      1.0         None
'''
import numpy as np
import progressbar

from utils.misc import bar_widgets
from utils.ml_utils.loss_functions import MeanSquaredLoss,SotfMaxLoss
from utils.ml_utils.data_manipulation import to_categorical
from models.decision_tree.decision_tree_model import RegressionTree,ClassificationTree





class GBDT(object):
    """Super class of GradientBoostingClassifier and GradientBoostinRegressor.
    Uses a collection of regression trees that trains on predicting the gradient
    of the loss function.
    Parameters:
    -----------
    n_estimators: int
        树的数量
        The number of classification trees that are used.
    learning_rate: float
        梯度下降的学习率
        The step length that will be taken when following the negative gradient during
        training.
    min_samples_split: int
        每棵子树的节点的最小数目（小于后不继续切割）
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        每颗子树的最小纯度（小于后不继续切割）
        The minimum impurity required to split the tree further.
    max_depth: int
        每颗子树的最大层数（大于后不继续切割）
        The maximum depth of a tree.
    regression: boolean
        是否为回归问题
        True or false depending on if we're doing regression or classification.
    """

    def __init__(self, n_estimators, learning_rate=0.5, min_samples_split=2,
                 min_impurity=1e-7, max_depth=float('inf'), regression=True):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression

        # 进度条 processbar
        self.bar = progressbar.ProgressBar(widgets=bar_widgets)

        self.loss = MeanSquaredLoss()
        if not self.regression:
            self.loss = SotfMaxLoss()

        # 分类问题也使用回归树，利用残差去学习概率
        self.trees = []
        for i in range(self.n_estimators):
            # 使用  RegressionTree 初始化 n_estimators
            self.trees.append(RegressionTree(min_samples_split=self.min_samples_split,
                                             min_impurity=self.min_impurity,
                                             max_depth=self.max_depth))

    def fit(self, X, y):
        """
        @time: 2022/6/5 19:23
        @author: wangfeicheng
        @version: 1.0
        @description:
        To improve the model such that:
            F(x) + h(x) = y
        Then
            h(x) = y - F(x)
        Just fit a regression tree h to data
            (x1,y1-F(x1)), ...

        Loss function L(y; F(x)) = (y - F(x))^2/2
        We want to minimize J = sum(L(y_i; F(x_i))),
        Notice that F(x1); F(x2); ... F(xn) are just some numbers. We can treat F(xi ) as parameters and take derivatives
            dJ/dF(x_i) = -(y_i - F(x_i))
        So we can interpret residuals as negative gradients.
            y_i - F(x_i) = - dJ/dF(x_i)

        For regression with square loss:
            residual                    ~= negative gradient
            fit h to residual           ~= fit h to negative gradient
            update F based on residual  ~=  update F based on negative gradient
            (residual = y - F(x) = h(x) ~=  tree(x))
        @params:
        @return:
        """
        if np.ndim(y) ==1:
            y = np.expand_dims(y,axis=1)
        # 让第一棵树去拟合模型
        self.trees[0].fit(X, y)
        y_pred = self.trees[0].predict(X)
        for i in self.bar(range(1, self.n_estimators)):
            # 计算 gradient
            gradient = self.loss.gradient(y, y_pred)
            # fit h to negative gradient
            self.trees[i].fit(X,  - gradient)
            # y = F(x) + h(x)
            # h(x) ~= tree(x)
            # Todo: 当 X 较大的时候,self.tree.predict() 方法使用 for-loop 循环方式预测会很慢
            y_pred += np.multiply(self.learning_rate, self.trees[i].predict(X))


    def predict(self,X):
        y_pred = self.trees[0].predict(X)
        for tree in self.trees[1:]:
            # 使用训练的时候同样的 learning_rate 来进行预测
            y_pred += np.multiply(self.learning_rate, tree.predict(X))


        if not self.regression:
            # y 被转换为one-hot形式进行拟合， y_pred  为 (n_samples,n_class)
            # Turn into probability distribution
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            # Set label to the value that maximizes probability
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred


class GBDTRegressor(GBDT):
    pass


class GBDTClassifier(GBDT):
    def __init__(self, n_estimators=200, learning_rate=.5, min_samples_split=2,
                 min_info_gain=1e-7, max_depth=2, debug=False):
        super(GBDTClassifier, self).__init__(n_estimators=n_estimators,
                                             learning_rate=learning_rate,
                                             min_samples_split=min_samples_split,
                                             min_impurity=min_info_gain,
                                             max_depth=max_depth,
                                             regression=False)


    def fit(self, X, y):
        if np.ndim(y) ==1:
            # 转换为 one-hot 形式后 使用 regression tree fit on y(one-hot)
            y = to_categorical(y)
        super(GBDTClassifier,self).fit(X,y)

