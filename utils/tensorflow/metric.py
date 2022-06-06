#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/6/25 9:28 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/25 9:28   wangfc      1.0         None

metrics，它用来在训练过程中监测一些性能指标，而这个性能指标是什么可以由我们来指定。指定的方法有两种：
1.直接使用字符串 :
   'ce', 'acc'
2. 使用tf.keras.metrics下的类创建的实例化对象或者函数:
   tf.keras.metrics.binary_accuracy,
   f.keras.metrics.MeanSquaredError()

    tf.keras.metrics.Accuracy(准确率，用于分类，可以用字符串"Accuracy"表示，Accuracy=(TP+TN)/(TP+TN+FP+FN)，要求y_true和y_pred都为类别序号编码)
    tf.keras.metrics.AUC(ROC曲线(TPR vs FPR)下的面积，用于二分类，直观解释为随机抽取一个正样本和一个负样本，正样本的预测值大于负样本的概率)
    tf.keras.metrics.Precision(精确率，用于二分类，Precision = TP/(TP+FP))
    tf.keras.metrics.Recall(召回率，用于二分类，Recall = TP/(TP+FN))
    tf.keras.metrics.TopKCategoricalAccuracy(多分类TopK准确率，要求y_true(label)为onehot编码形式)
    tf.keras.metrics.CategoricalAccuracy(分类准确率，与Accuracy含义相同，要求y_true(label)为onehot编码形式)
    tf.keras.metrics.SparseCategoricalAccuracy(稀疏分类准确率，与Accuracy含义相同，要求y_true(label)为序号编码形式)


1. 编写函数形式的评估指标:
自定义评估指标需要接收两个张量y_true,y_pred作为输入参数，并输出一个标量作为评估值。

参数
y_true: 真实标签，Theano/Tensorflow 张量。
y_pred: 预测值。和 y_true 相同尺寸的 Theano/TensorFlow 张量。

返回值
返回一个表示全部数据点平均值的张量。

如果编写函数形式的评估指标，则只能取epoch中各个batch计算的评估指标结果的平均值作为整个epoch上的评估指标结果，这个结果通常会偏离整个epoch数据一次计算的结果。


2. tf.keras.metrics.Metric进行子类化，实现评估指标的计算逻辑，从而得到评估指标的类的实现形式。
由于训练的过程通常是分批次训练的，而评估指标要跑完一个epoch才能够得到整体的指标结果。
__init__(self): 创建与计算指标结果相关的一些中间变量 create state variables for your metric.
update_state(self, y_true, y_pred, sample_weight=None):
     在每个batch后更新相关中间变量的状态。将每一次更新的数据作为一组数据，这样在真正计算的时候会计算出每一组的结果，然后求多组结果的平均值。但并不会直接计算，计算还是在调用 result()时完成
result(self):which uses the state variables to compute the final results.
reset_states(self), which reinitializes the state of the metric.


"""
from typing import List, Text

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.metrics import MeanMetricWrapper
from tensorflow.python.ops import math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras import backend as K


def test_accuracy():
    # 评估函数使用原理
    m = tf.keras.metrics.Accuracy()

    # 使用 y_true 和 y_pred 进行更新,要求y_true和y_pred都为类别序号编码
    # y_true: 真实值，如果真实值是 one-hot 形式呢
    # y_pred: 如何获取
    y_true = [1, 2, 3, 4]
    y_pred = [0, 2, 3, 4]
    m.update_state(y_true=y_true, y_pred=y_pred)

    print('Final result: ', m.result().numpy())
    # Final result: 0.75

    # reset_states(), 清除之前的计算结果，相当于复位重新开始计算
    # m.reset_states()

    y_true = [1, 2, 3, 4]
    y_pred = [0, 2, 3, 1]
    m.update_state(y_true=y_true, y_pred=y_pred)
    print('Final result: ', m.result().numpy())


def test_categorical_accuracy():
    """
    accuracy: y_true 和 y_pred都为具体标签的情况,具体的label index（如y_true=[1, 2, 1], y_pred=[0, 1, 1]）

    categorical_accuracy: y_true为 onehot 标签，y_pred为向量的情况。

    sparse_categorical_accuracy : y_true为非onehot的形式。和 categorical_accuracy功能一样，其

    top_k_categorical_accuracy: 在categorical_accuracy的基础上加上top_k。
    categorical_accuracy要求样本在真值类别上的预测分数是在所有类别上预测分数的最大值，才算预测对，
    而top_k_categorical_accuracy只要求样本在真值类别上的预测分数排在其在所有类别上的预测分数的前k名就行。

    """
    categorical_accuracy = tf.keras.metrics.CategoricalAccuracy()
    y_true = [2, 2, 1, 0]
    y_true_onehot = tf.one_hot(y_true, depth=3)

    y_true_onehot_multilabel = np.array(
        [[1., 1., 0.],
         [0., 0., 1.],
         [0., 1., 0.],
         [1., 0., 0.]])

    y_pred_probability = tf.nn.sigmoid(tf.random.uniform((4, 3)))
    y_pred = tf.argmax(y_pred_probability, axis=-1)

    categorical_accuracy.reset_states()
    categorical_accuracy(y_true=y_true_onehot_multilabel, y_pred=y_pred_probability)
    categorical_accuracy.result()

    # 验证 precision
    precision = tf.keras.metrics.Precision()
    precision(y_true=y_true_onehot_multilabel, y_pred=y_pred_probability)
    precision.result()


def get_keras_metric(metric_name: Text, threshold=0.5) -> tf.keras.metrics.Metric:
    """
    当你的标签和预测值都是具体的label index: Accuracy
    当你的标签是onehot形式，而prediction是向量形式: categorical_accuracy
    当你的标签是具体的label index，而prediction是向量形式 :SparseCategoricalAccuracy

    binary_accuracy和accuracy最大的不同就是，它适用于2分类的情况


    # top_k_categorical_accuracy
    # sparse_top_k_categorical_accuracy

    """
    if metric_name.lower() in ['accuracy',"acc"]:
        metric = tf.keras.metrics.Accuracy
    elif metric_name.lower() == "categorical_accuracy":
        metric = tf.keras.metrics.CategoricalAccuracy
    elif metric_name.lower() == "categorical_accuracy":
        metric = tf.keras.metrics.SparseCategoricalAccuracy()
    elif metric_name.lower() == 'binary_accuracy':
        metric = tf.keras.metrics.BinaryAccuracy(threshold=threshold)
    elif metric_name.lower() == 'recall':
        metric = tf.keras.metrics.Recall
    elif metric_name.lower() == 'precision':
        metric = tf.keras.metrics.Precision
    return metric


class SparseCategoricalAccuracy_(tf.keras.metrics.Metric):
    """
    @time:  2021/6/25 10:16
    @author:wangfc
    @version: https://www.pythonf.cn/read/143274
    @description:

    自定义评估指标需要继承tf.keras.metrics.Metric类，并重写__init__、update_state和result 三个方法。
    __init__() :所有状态变量都应通过以下方法在此方法中创建self.add_weight()
    update_state():对状态变量进行所有更新
    result():根据状态变量计算并返回指标值。


    @params:
    @return:
    """

    def __init__(self, name='SparseCategoricalAccuracy', **kwargs):
        super(SparseCategoricalAccuracy_, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total.assign(0)
        self.count.assign(0)


class CatgoricalTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name='categorical_true_positives', **kwargs):
        super(CatgoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        values = tf.equal(tf.cast(y_true, 'int32'), tf.cast(y_pred, 'int32'))
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)


# 类形式的自定义评估指标
class KS(tf.keras.metrics.Metric):
    """
    @time:  2021/6/25 11:41
    @author:wangfc
    @version: https://www.bookstack.cn/read/eat_tensorflow2_in_30_days-zh/5-6,%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87metrics.md
    @description:

    @params:
    @return:
    """

    def __init__(self, name="ks", **kwargs):
        super(KS, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(
            name="tp", shape=(101,), initializer="zeros")
        self.false_positives = self.add_weight(
            name="fp", shape=(101,), initializer="zeros")

    @tf.function
    def update_state(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.bool)
        y_pred = tf.cast(100 * tf.reshape(y_pred, (-1,)), tf.int32)
        for i in tf.range(0, tf.shape(y_true)[0]):
            if y_true[i]:
                self.true_positives[y_pred[i]].assign(
                    self.true_positives[y_pred[i]] + 1.0)
            else:
                self.false_positives[y_pred[i]].assign(
                    self.false_positives[y_pred[i]] + 1.0)
        return (self.true_positives, self.false_positives)

    @tf.function
    def result(self):
        cum_positive_ratio = tf.truediv(
            tf.cumsum(self.true_positives), tf.reduce_sum(self.true_positives))
        cum_negative_ratio = tf.truediv(
            tf.cumsum(self.false_positives), tf.reduce_sum(self.false_positives))
        ks_value = tf.reduce_max(tf.abs(cum_positive_ratio - cum_negative_ratio))
        return ks_value


@keras_export('keras.metrics.mulilabel_categorical_accuracy')
@dispatch.add_dispatch_support
def multilabel_categorical_accuracy(y_true, y_pred, threshold=0.5):
    """Calculates how often predictions matches multi-hot labels.
    计算多标签的准确率：当 probabiliies >threshold 与 y_true 完全相同的时候，我们认为 1 ，否认为 0

    而不是像原来的 categorical_accuracy 使用 argmax of logits and probabilities 判断 label_index 是否相同来获得 1 or 0.

    Args:
      y_true: One-hot ground truth values.
      y_pred: The prediction values.

    Returns:
      Categorical accuracy values.
    """
    y_pred_label = tf.cast(tf.less(threshold, y=y_pred), tf.int32)
    label_size = y_pred_label.shape[-1]
    is_equal = math_ops.cast(math_ops.equal(tf.cast(y_true, tf.int32), y_pred_label), tf.int32)
    is_complete_equal = math_ops.equal(math_ops.reduce_sum(is_equal, axis=-1), label_size)
    return math_ops.cast(is_complete_equal, K.floatx())


@keras_export('keras.metrics.MultilabelCategoricalAccuracy')
class MultilabelCategoricalAccuracy(MeanMetricWrapper):
    """Calculates how often predictions matches one-hot labels.
      https://blog.csdn.net/menghaocheng/article/details/83479754

      @tf_export 是一个修饰符。修饰符的本质是一个函数. @tf_export修饰器为所修饰的函数取了个名字！
      @keras_export


      tf_export的实现在tensorflow/python/util/tf_export.py中：
      tf_export = functools.partial(api_export, api_name=TENSORFLOW_API_NAME)
      keras_export = functools.partial(api_export, api_name=KERAS_API_NAME)


      等号的右边的理解分两步：
      1.functools.partial
      2.api_export
      functools.partial是偏函数,它的本质简而言之是为函数固定某些参数。
      如：functools.partial(FuncA, p1)的作用是把函数FuncA的第一个参数固定为p1；
      又如functools.partial(FuncB, key1="Hello")的作用是把FuncB中的参数key1固定为“Hello"。

      functools.partial(api_export, api_name=TENSORFLOW_API_NAME)的意思是把api_export的api_name这个参数固定为TENSORFLOW_API。
      其中TENSORFLOW_API_NAME = 'tensorflow'。
      api_export是实现了__call__()函数的类

      tf_export= functools.partial(api_export, api_name=TENSORFLOW_API_NAME)的写法等效于：
      funcC = api_export(api_name=TENSORFLOW_API_NAME)
      tf_export = funcC

      对于funcC = api_export(api_name=TENSORFLOW_API_NAME)，会导致__init__(api_name=TENSORFLOW_API_NAME)被调用
      然后调用像函数一样调用funcC()实际上就会调用__call__()

      因此@tf_export("app.run")最终的结果是用上面这个__call__()来作为修饰器。这是一个带参数的修饰器


      _, undecorated_func = tf_decorator.unwrap(func)
      将对象展开到tfdecorator列表和最终目标列表中。undecorated_func获得的返回对象就是我们有@tf_export修饰的函数。
      标注3：self.set_attr(undecorated_func, api_names_attr, self._names) 设置属性。

    """

    def __init__(self, name='multilabel_categorical_accuracy', dtype=None):
        super(MultilabelCategoricalAccuracy, self).__init__(
            multilabel_categorical_accuracy, name, dtype=dtype)


if __name__ == '__main__':
    test_accuracy()

    test_categorical_accuracy()

    y_true = tf.constant([2, 1])
    y_pred = tf.constant([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])

    sparse_accuracy = SparseCategoricalAccuracy_()
    sparse_accuracy(y_true=y_true, y_pred=y_pred)
    sparse_accuracy.result()
