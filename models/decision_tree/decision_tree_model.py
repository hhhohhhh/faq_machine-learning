#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   decision_tree_model.py
@Version:  
@Desc:
@Time: 2022/6/3 19:28
@Contact :   wang_feicheng@163.com
@License :   Create by wangfc27441 on 2022/6/3, Copyright 2022 wangfc27441. All rights reserved.

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/3 19:28   wangfeicheng      1.0         None
'''
from  collections import deque
import numpy as np
from utils.ml_utils.loss_functions import calculate_entropy, calculate_variance,calculate_gini_index
from utils.ml_utils.data_manipulation import divide_on_feature


class DecisionNode(object):
    """这是DecisionNode
    Class that represents a decision node or leaf in the decision tree
    Parameters:
    -----------
    feature_i: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature_i against to determine the prediction.
    current_depth : int
    leaf_value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
        """

    def __init__(self, feature_i=None, threshold=None, leaf_value=None, current_depth=None,
                 true_branch=None, false_branch=None):
        self.feature_i = feature_i
        self.threshold = threshold
        self.leaf_value = leaf_value
        self.current_depth = current_depth
        self.true_branch = true_branch
        self.false_branch = false_branch


class DecisionTree(object):
    """这是DecisionTree
    Super class of RegressionTree and ClassificationTree.
    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    loss: function
        Loss function that is used for Gradient Boosting models to calculate impurity.

    basetree_type:
        在ID3算法中我们使用了信息增益来选择特征，信息增益大的优先选择。
        在C4.5算法中，采用了信息增益比来选择特征，以减少信息增益容易选择特征值多的特征的问题。
        CART分类树算法使用基尼系数来代替信息增益比，基尼系数代表了模型的不纯度，基尼系数越小，则不纯度越低，特征越好
    """

    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float("inf"), loss=None,
                 basetree_type='CART'):
        self.root: DecisionNode = None  # Root node in dec. tree
        # Minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = max_depth
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        # 切割树的方法，gini，方差等
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf
        # 树节点取值的方法，分类树：选取出现最多次数的值，回归树：取所有值的平均值
        self._leaf_value_calculation = None
        # If y is one-hot encoded (multi-dim) or not (one-dim)
        self.one_dim = None
        # If Gradient Boost
        self.loss = loss
        # 新增参数 basetree_type 来选择不同的数 : ID3, C4.5, CART
        self.basetree_type = basetree_type

        # if self.basetree_type in ['ID3','c4.5']:
        #     self._init_largest_impurity = 0
        # elif self.basetree_type == 'CART':
        self._init_largest_impurity = -float("inf")


    def fit(self, X, y, loss=None):
        """ Build decision tree """
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)
        self.loss = loss

    def _build_tree(self, X, y, current_depth=0) -> DecisionNode:
        """ Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data
            Decision Tree 的数据结构是怎么样设计的 ?
        leaf_value = most frequent class or mean
        while current_depth > max_depth or num_samples < min_samples_split:
                stop
            if impurity  < min_impurity:
                stop
            else:
                feature_i,threshold = maximum(impurity)
                left tree Xs: feature_i > threshold
                right tree Xs: feature_i < threshold
                current_depth +=1
                true_branch = self._build_tree(Xs,Ys,tree_depth)
                false_branch = self._build_tree(Xs,Ys,tree_depth)
                root = DecisionNode(feature_i, threshold, leaf_value, current_depth, true_branch, false_branch)
        true_branch = None
        false_branch =None
        root = DecisionNode(feature_i=None, threshold=None, leaf_value, current_depth, true_branch, false_branch)
        return root
        """
        if np.ndim(y) == 1:
            # 分类问题 and 回归问题
            y = np.expand_dims(y, axis=1)
        n_samples, n_features = np.shape(X)
        leaf_value = self._leaf_value_calculation(y)
        current_node = DecisionNode(feature_i=None, threshold=None, leaf_value=leaf_value, current_depth=current_depth)

        if current_depth < self.max_depth and n_samples > self.min_samples_split:
            largest_impurity, best_criteria, best_sets = self._maximum_impurity(X, y)
            if largest_impurity > self.min_impurity:
                feature_i = best_criteria["feature_i"]
                threshold = best_criteria["threshold"]
                leftX = best_sets["leftX"]
                leftY = best_sets["leftY"]
                rightX = best_sets["rightX"]
                rightY = best_sets["rightY"]
                true_branch = self._build_tree(leftX, leftY, current_depth + 1)
                false_branch = self._build_tree(rightX, rightY, current_depth + 1)
                current_node = DecisionNode(feature_i, threshold, leaf_value, current_depth, true_branch, false_branch)
        return current_node

    def _maximum_impurity(self, X, y):
        """
        @time: 2022/6/3 20:49
        @author: wangfeicheng
        @version: 1.0
        @description:  循环遍历每个 feature 及 每个 unique_feature_value，获取 impurity 提升最多的分割 split = (feature, value)

        @params:
        @return:
        """
        largest_impurity = self._init_largest_impurity
        best_criteria = None  # Feature index and threshold
        best_sets = None  # Subsets of the data
        n_samples, n_features = np.shape(X)
        Xy = np.concatenate((X, y), axis=1)
        #
        for feature_i in range(n_features):
            feature_values = X[:, feature_i]
            unique_feature_values = np.unique(feature_values)
            # 对每个值遍历？
            for threshold in unique_feature_values:
                # 按照 threshold 分割为左右两个部分
                left_xys, right_xys = divide_on_feature(Xy, feature_i, threshold)
                if len(left_xys) > 0 and len(right_xys) > 0:
                    left_ys = left_xys[:, n_features:]
                    right_ys = right_xys[:, n_features:]
                    impurity = self._impurity_calculation(y, left_ys, right_ys)
                    if impurity > largest_impurity:
                        largest_impurity = impurity
                        best_criteria = {"feature_i": feature_i, "threshold": threshold}
                        best_sets = {
                            "leftX": left_xys[:, :n_features],  # X of left subtree
                            "leftY": left_xys[:, n_features:],  # y of left subtree
                            "rightX": right_xys[:, :n_features],  # X of right subtree
                            "rightY": right_xys[:, n_features:]  # y of right subtree
                        }
        return largest_impurity, best_criteria, best_sets

    def predict(self, X):
        """
        @time: 5/30/2022 11:38 AM
        @author: wangfc
        @version: 1.0
        @description: Classify samples one by one and return the set of labels
        找到分割的 feature,判断该feature的值是否属于 left tree or right tree 进行 child node
        feature_i > split > tree > feature_i ...
        @params:
        @return:
        """
        y_pred = []
        for x in X:
            y_pred.append(self.predict_value(x))
        # 转换 ndarray
        return np.array(y_pred)

    def predict_value(self, x):
        """
        @time: 5/30/2022 11:39 AM
        @author: wangfc
        @version: 1.0
        @description:

        @params:
        @return:
        """
        if np.ndim(x) == 1:
            x = np.expand_dims(x, axis=0)
        node = self.root
        while node.feature_i is not None:
            feature_i = node.feature_i
            threshold = node.threshold
            left, right = divide_on_feature(x, feature_i, threshold)
            if left.shape[0] > 0:
                node = node.true_branch
            else:
                node = node.false_branch
        return node.leaf_value

    def _get_tree_depth(self, node=None):
        """

        """
        if node is None:
            node = self.root
        current_depth, left_depth, right_depth = [node.current_depth] * 3

        if node.true_branch:
            left_depth = self._get_tree_depth(node.true_branch)
        if node.false_branch:
            right_depth = self._get_tree_depth(node.false_branch)

        tree_depth = max(current_depth, left_depth, right_depth)
        return tree_depth

        return tree_depth

    def print_tree(self, tree: DecisionNode = None, leaf_width=None):
        """ Recursively print the decision tree """

        if not tree:
            tree = self.root
            tree_depth = self._get_tree_depth()

        # 使用 bfs 进行搜索 或者 递归的方式
        # node_queue = deque

        if leaf_width is None:
            leaf_width = tree_depth

        node_with = len("feature_i:t")
        # # If we're at leaf => print the label
        if tree.leaf_value is not None:
            s = "value={0}".format(tree.leaf_value)
            print("{0:^{1}}".format(s,node_with*leaf_width))

        # dfs : Go deeper down the tree
        # if tree.true_branch or tree.false_branch:
        #     # Print test
        #     s = "feature_{0}:{1}".format(tree.feature_i,tree.threshold)
        #     print("{0:^{1}}".format(s, node_with*leaf_width))
        #     # Print the true scenario
        #     self.print_tree(tree.true_branch, leaf_width=leaf_width-1)
        #     # Print the false scenario
        #     self.print_tree(tree.false_branch, leaf_width=leaf_width+1)

        # bfs:
        if tree.true_branch or tree.false_branch:
            # Print test
            feature = "feature_{0}:{1}".format(tree.feature_i,tree.threshold)
            print("{0:^{1}}".format(feature, node_with * leaf_width))

        if tree.true_branch:
            # Print the true scenario
            self.print_tree(tree.true_branch, leaf_width=leaf_width-1)
        if tree.false_branch:
            # Print the false scenario
            self.print_tree(tree.false_branch, leaf_width=leaf_width+1)

    def _impurity_calculation(self):
        raise NotImplementedError

    def _leaf_value_calculation(self):
        raise NotImplementedError


class ClassificationTree(DecisionTree):
    """这是ClassificationTree
    """

    def _calculate_information_gain(self, y, y1, y2):
        # Calculate information gain
        p = len(y1) / len(y)
        # 因为每次分割的y值相同，因此其实不必每次计算该值
        entropy = calculate_entropy(y)
        info_gain = entropy - p * \
                    calculate_entropy(y1) - (1 - p) * \
                    calculate_entropy(y2)

        return info_gain

    def _calculate_gini_index_reduction(self,y,y1,y2):
        """
        计算分割之后的 gini_index： gini_index 越小表示纯度越高，
        因为我们的 父类是 使用 self._maximum_impurity()作为分割点的选择
        """
        gini_index_original = calculate_gini_index(y)
        frac_1  = len(y1)/len(y)
        gini_index_split = frac_1*calculate_gini_index(y1) + (1-frac_1)* calculate_gini_index(y2)
        gini_index_reduction  = gini_index_original - gini_index_split
        return  gini_index_reduction

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # Count number of occurences of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
                # print("most_common :",most_common)
        return most_common

    def fit(self, X, y):
        if self.basetree_type in ['ID3','c4.5']:
            self._impurity_calculation = self._calculate_information_gain
        elif self.basetree_type == 'CART':
            self._impurity_calculation = self._calculate_gini_index_reduction
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)


class RegressionTree(DecisionTree):
    """这是RegressionTree"""

    def _calculate_reduction_variance(self, y, y1, y2):
        """
        y.shape = (n_samples,1)

        y.shape = (n_smaples, n_class),when y  is one-hot encoding
        """
        variance_total = calculate_variance(y)
        variance_1 = calculate_variance(y1)
        variance_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        variance_reduction = variance_total - frac_1 * variance_1 - (1 - frac_1) * variance_2
        return sum(variance_reduction)

    def _mean_of_y(self, y):
        """
        分类问题 or 回归问题
        y.shape = (n_samples,1)

        one-hot 分类问题
        y.shape = (n_samples, n_class)
        """
        value = np.mean(y,axis=0)
        return value

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_reduction_variance
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)


class XGBootTree(DecisionTree):
    pass
