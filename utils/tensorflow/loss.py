#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/6/25 9:34 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/25 9:34   wangfc      1.0         None


一般来说，监督学习的目标函数由损失函数和正则化项组成。（Objective = Loss + Regularization）

对于keras模型，目标函数中的正则化项一般在各层中指定:
   例如使用Dense的 kernel_regularizer 和 bias_regularizer等参数指定权重使用l1或者l2正则化项，
   此外还可以用kernel_constraint 和 bias_constraint等参数约束权重的取值范围，这也是一种正则化手段。

损失函数在模型编译时候指定。

内置的损失函数一般有类的实现和函数的实现两种形式。

    如：CategoricalCrossentropy 和 categorical_crossentropy 都是类别交叉熵损失函数，前者是类的实现形式，后者是函数的实现形式。

常用的一些内置损失函数说明如下。
    mean_squared_error（均方误差损失，用于回归，简写为 mse, 类与函数实现形式分别为 MeanSquaredError 和 MSE）

    mean_absolute_error (平均绝对值误差损失，用于回归，简写为 mae, 类与函数实现形式分别为 MeanAbsoluteError 和 MAE)

    mean_absolute_percentage_error (平均百分比误差损失，用于回归，简写为 mape, 类与函数实现形式分别为 MeanAbsolutePercentageError 和 MAPE)

    Huber(Huber损失，只有类实现形式，用于回归，介于mse和mae之间，对异常值比较鲁棒，相对mse有一定的优势)

    binary_crossentropy (二元交叉熵，用于二分类，类实现形式为 BinaryCrossentropy)

    categorical_crossentropy(类别交叉熵，用于多分类，要求label为onehot编码，类实现形式为 CategoricalCrossentropy)

    sparse_categorical_crossentropy(稀疏类别交叉熵，用于多分类，要求label为序号编码形式，类实现形式为 SparseCategoricalCrossentropy)

    hinge(合页损失函数，用于二分类，最著名的应用是作为支持向量机SVM的损失函数，类实现形式为 Hinge)

    kld(相对熵损失，也叫KL散度，常用于最大期望算法EM的损失函数，两个概率分布差异的一种信息度量。类与函数实现形式分别为 KLDivergence 或 KLD)

    cosine_similarity(余弦相似度，可用于多分类，类实现形式为 CosineSimilarity)



如果有需要，也可以自定义损失函数，自定义损失函数需要接收两个张量y_true,y_pred作为输入参数，并输出一个标量作为损失函数值。

你可以传递一个现有的损失函数名，或者一个 TensorFlow/Theano 符号函数。 该符号函数为每个数据点返回一个标量，有以下两个参数:
y_true: 真实标签。TensorFlow/Theano 张量。
y_pred: 预测值。TensorFlow/Theano 张量，其 shape 与 y_true 相同。
实际的优化目标是所有数据点的输出数组的平均值。

tf.keras.losses.BinaryCrossentropy() vs  tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)


"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.keras.losses import Loss



def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        '''
        设计 loss fun 作为 一个 wrapper，从而可以调整参数设置
        Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        https://gombru.github.io/2019/04/03/ranking_loss/
        y_true: 0 表示 negative pairs, 1 表示 positive pairs
        y_pred: distance(pair_a,pair_b)
        '''
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    return contrastive_loss




def huber_loss(y_true,y_pred):
    """
    @time:  2021/5/26 14:44
    @author:wangfc
    @version:
    @description: 自定义 huber_loss function， 用于 实验 自定义 loss

    @params:
    @return:
    """
    threshold =1
    error  =  y_true - y_pred
    # 判断是否小于阈值
    is_small_error = math_ops.abs(error) < threshold
    small_error_loss = 0.5*math_ops.square(error)
    big_error_loss = threshold * (math_ops.abs(error) - 0.5 * threshold)
    loss = tf.where(is_small_error,small_error_loss,big_error_loss)
    return loss


class HuberLoss(Loss):
    """
    设计 loss 为一个 class
    """
    def __init__(self,threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def call(self,y_true, y_pred):
        """
        @time:  2021/5/26 14:44
        @author:wangfc
        @version:
        @description: 自定义 huber_loss， 用于 实验 自定义 loss

        @params:
        @return:
        """
        error = y_true - y_pred
        # 判断是否小于阈值
        is_small_error = math_ops.abs(error) < self.threshold
        small_error_loss = 0.5 * math_ops.square(error)
        big_error_loss = self.threshold * (math_ops.abs(error) - 0.5 * self.threshold)
        loss = tf.where(is_small_error, small_error_loss, big_error_loss)
        return loss



class CustomMSE(tf.keras.losses.Loss):
    """
    https://keras.io/guides/training_with_built_in_methods/

    If you need a loss function that takes in parameters beside y_true and y_pred,
    you can subclass the tf.keras.losses.Loss class and implement the following two methods:

        __init__(self): accept parameters to pass during the call of your loss function
        call(self, y_true, y_pred): use the targets (y_true) and the model predictions (y_pred) to compute the model's loss
    """
    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        return mse + reg * self.regularization_factor


def test_bce():
    y_true = [[0., 1.], [0., 0.]]
    y_pred = [[0.6, 0.4], [0.4, 0.6]]
    # Using 'auto'/'sum_over_batch_size' reduction type.
    bce = tf.keras.losses.BinaryCrossentropy()
    bce(y_true, y_pred).numpy()

    bce = tf.keras.losses.BinaryCrossentropy(
          reduction=tf.keras.losses.Reduction.SUM)
    bce(y_true, y_pred).numpy()

    bce = tf.keras.losses.BinaryCrossentropy(
        reduction = tf.keras.losses.Reduction.NONE)
    bce(y_true, y_pred).numpy()



if __name__ == '__main__':
    test_bce()