#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.***
@time: 2021/1/11 9:00 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/11 9:00   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：*** 2019
 * 注意：本内容仅限于***内部传阅，禁止外泄以及用于其他的商业目的

"""
import logging

logger = logging.getLogger(__name__)

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix
from tensorflow_estimator.python.estimator.canned.metric_keys import MetricKeys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix

"""Multiclass
https://github.com/guillaumegenthial/tf_metrics
__author__ = "Guillaume Genthial"
"""


# 定义 eval 时候的评测函数 （dev数据为单标签的数据，故定义的也是 分类的评估函数，而不是真正的多标签的评估函数）
def metric_fnV1(per_example_loss, label_ids, logits, is_real_example):
    """
    tf.metrics.precision 是计算二分类问题的
    :param per_example_loss:
    :param label_ids:
    :param logits:
    :param is_real_example:
    :return:
    """
    # 获取预测的logits的最大值
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    # 转化 label_ids 为 label_index_ls （单标签的）
    label_index_ls = tf.argmax(label_ids, axis=-1)

    # 计算准确率
    accuracy = tf.metrics.accuracy(labels=label_index_ls, predictions=predictions, weights=is_real_example)
    precision = tf.metrics.precision(labels=label_index_ls, predictions=predictions, weights=is_real_example)
    recall = tf.metrics.recall(labels=label_index_ls, predictions=predictions, weights=is_real_example)
    # 计算 f1 = f1_score, f1_update_op
    f1_value = (2 * precision[0] * recall[0]) / (precision[0] + recall[0])
    f1_update_op = (2 * precision[1] * recall[1]) / (precision[1] + recall[1])
    f1_score = (f1_value, f1_update_op)
    # 计算loss
    # loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        'f1-score': f1_score
        # "loss": loss,
    }


def metric_fnV2(per_example_loss, label_ids, logits, is_real_example, num_classes, average="macro", beta=1):
    # 获取预测的logits的最大值
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    # 转化 label_ids 为 label_index_ls （单标签的）
    label_index_ls = tf.argmax(label_ids, axis=-1)
    # 计算准确率
    accuracy = tf.metrics.accuracy(labels=label_index_ls, predictions=predictions, weights=is_real_example)

    # pr_and_op, re_and_op, fb_and_op = precision_recall_fbeta(label_index_ls, predictions, num_classes)
    return {
        "accuracy": accuracy,
        # "precision": pr_and_op,
        # "recall": re_and_op,
        # 'f1-score': fb_and_op
        # "loss": loss,
    }


def precision_recall_fbeta(labels, predictions, num_classes, pos_indices=None,
                           weights=None, average='macro', beta=1):
    """Multi-class precision metric for Tensorflow
    使用
    """
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    pr, re, fb = metrics_from_confusion_matrix(
        cm, pos_indices, average=average)
    pr_op, re_op, fb_op = metrics_from_confusion_matrix(
        op, pos_indices, average=average, beta=beta)

    return (pr, pr_op), (re, re_op), (fb, fb_op)


def precision(labels, predictions, num_classes, pos_indices=None,
              weights=None, average='micro'):
    """Multi-class precision metric for Tensorflow
    Parameters
    ----------
    labels : Tensor of tf.int32 or tf.int64
        The true labels
    predictions : Tensor of tf.int32 or tf.int64
        The predictions, same shape as labels
    num_classes : int
        The number of classes
    pos_indices : list of int, optional
        The indices of the positive classes, default is all
    weights : Tensor of tf.int32, optional
        Mask, must be of compatible shape with labels
    average : str, optional
        'micro': counts the total number of true positives, false
            positives, and false negatives for the classes in
            `pos_indices` and infer the metric from it.
        'macro': will compute the metric separately for each class in
            `pos_indices` and average. Will not account for class
            imbalance.
        'weighted': will compute the metric separately for each class in
            `pos_indices` and perform a weighted average by the total
            number of true labels for each class.
    Returns
    -------
    tuple of (scalar float Tensor, update_op)
    """
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    pr, _, _ = metrics_from_confusion_matrix(
        cm, pos_indices, average=average)
    op, _, _ = metrics_from_confusion_matrix(
        op, pos_indices, average=average)
    return (pr, op)


def recall(labels, predictions, num_classes, pos_indices=None, weights=None,
           average='micro'):
    """Multi-class recall metric for Tensorflow
    Parameters
    ----------
    labels : Tensor of tf.int32 or tf.int64
        The true labels
    predictions : Tensor of tf.int32 or tf.int64
        The predictions, same shape as labels
    num_classes : int
        The number of classes
    pos_indices : list of int, optional
        The indices of the positive classes, default is all
    weights : Tensor of tf.int32, optional
        Mask, must be of compatible shape with labels
    average : str, optional
        'micro': counts the total number of true positives, false
            positives, and false negatives for the classes in
            `pos_indices` and infer the metric from it.
        'macro': will compute the metric separately for each class in
            `pos_indices` and average. Will not account for class
            imbalance.
        'weighted': will compute the metric separately for each class in
            `pos_indices` and perform a wei average by the total
            number of true labels for each class.
    Returns
    -------
    tuple of (scalar float Tensor, update_op)
    """
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    _, re, _ = metrics_from_confusion_matrix(
        cm, pos_indices, average=average)
    _, op, _ = metrics_from_confusion_matrix(
        op, pos_indices, average=average)
    return (re, op)


def f1(labels, predictions, num_classes, pos_indices=None, weights=None,
       average='micro'):
    return fbeta(labels, predictions, num_classes, pos_indices, weights,
                 average)


def fbeta(labels, predictions, num_classes, pos_indices=None, weights=None,
          average='micro', beta=1):
    """Multi-class fbeta metric for Tensorflow
    Parameters
    ----------
    labels : Tensor of tf.int32 or tf.int64
        The true labels
    predictions : Tensor of tf.int32 or tf.int64
        The predictions, same shape as labels
    num_classes : int
        The number of classes
    pos_indices : list of int, optional
        The indices of the positive classes, default is all
    weights : Tensor of tf.int32, optional
        Mask, must be of compatible shape with labels
    average : str, optional
        'micro': counts the total number of true positives, false
            positives, and false negatives for the classes in
            `pos_indices` and infer the metric from it.
        'macro': will compute the metric separately for each class in
            `pos_indices` and average. Will not account for class
            imbalance.
        'weighted': will compute the metric separately for each class in
            `pos_indices` and perform a weighted average by the total
            number of true labels for each class.
    beta : int, optional
        Weight of precision in harmonic mean
    Returns
    -------
    tuple of (scalar float Tensor, update_op)
    """
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    _, _, fbeta = metrics_from_confusion_matrix(
        cm, pos_indices, average=average, beta=beta)
    _, _, op = metrics_from_confusion_matrix(
        op, pos_indices, average=average, beta=beta)
    return (fbeta, op)


def safe_div(numerator, denominator):
    """Safe division, return 0 if denominator is 0"""
    numerator, denominator = tf.to_float(numerator), tf.to_float(denominator)
    zeros = tf.zeros_like(numerator, dtype=numerator.dtype)
    denominator_is_zero = tf.equal(denominator, zeros)
    return tf.where(denominator_is_zero, zeros, numerator / denominator)


def pr_re_fbeta(cm, pos_indices, beta=1):
    """Uses a confusion matrix to compute precision, recall and fbeta"""
    num_classes = cm.shape[0]
    neg_indices = [i for i in range(num_classes) if i not in pos_indices]
    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[neg_indices, neg_indices] = 0
    diag_sum = tf.reduce_sum(tf.diag_part(cm * cm_mask))

    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[:, neg_indices] = 0
    tot_pred = tf.reduce_sum(cm * cm_mask)

    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[neg_indices, :] = 0
    tot_gold = tf.reduce_sum(cm * cm_mask)

    pr = safe_div(diag_sum, tot_pred)
    re = safe_div(diag_sum, tot_gold)
    fbeta = safe_div((1. + beta ** 2) * pr * re, beta ** 2 * pr + re)

    return pr, re, fbeta


def metrics_from_confusion_matrix(cm, pos_indices=None, average='micro',
                                  beta=1):
    """Precision, Recall and F1 from the confusion matrix
    Parameters
    ----------
    cm : tf.Tensor of type tf.int32, of shape (num_classes, num_classes)
        The streaming confusion matrix.
    pos_indices : list of int, optional
        The indices of the positive classes
    beta : int, optional
        Weight of precision in harmonic mean
    average : str, optional
        'micro', 'macro' or 'weighted'
    """
    num_classes = cm.shape[0]
    if pos_indices is None:
        pos_indices = [i for i in range(num_classes)]

    if average == 'micro':
        return pr_re_fbeta(cm, pos_indices, beta)
    elif average in {'macro', 'weighted'}:
        precisions, recalls, fbetas, n_golds = [], [], [], []
        # 对每个 正样本的 index 统计
        for idx in pos_indices:
            pr, re, fbeta = pr_re_fbeta(cm, [idx], beta)
            precisions.append(pr)
            recalls.append(re)
            fbetas.append(fbeta)
            cm_mask = np.zeros([num_classes, num_classes])
            # 所有的行
            cm_mask[idx, :] = 1
            n_golds.append(tf.to_float(tf.reduce_sum(cm * cm_mask)))

        if average == 'macro':
            pr = tf.reduce_mean(precisions)
            re = tf.reduce_mean(recalls)
            fbeta = tf.reduce_mean(fbetas)
            return pr, re, fbeta
        if average == 'weighted':
            n_gold = tf.reduce_sum(n_golds)
            pr_sum = sum(p * n for p, n in zip(precisions, n_golds))
            pr = safe_div(pr_sum, n_gold)
            re_sum = sum(r * n for r, n in zip(recalls, n_golds))
            re = safe_div(re_sum, n_gold)
            fbeta_sum = sum(f * n for f, n in zip(fbetas, n_golds))
            fbeta = safe_div(fbeta_sum, n_gold)
            return pr, re, fbeta
    else:
        raise NotImplementedError()


def get_exporter_compare_fn(compare_key):
    def exporter_compare_fn(best_eval_result, current_eval_result):
        """Compares two evaluation results and returns true if the 2nd one is smaller.

        Both evaluation results should have the values for MetricKeys.LOSS, which are
        used for comparison.

        Args:
          best_eval_result: best eval metrics.
          current_eval_result: current eval metrics.

        Returns:
          True if the loss of current_eval_result is smaller; otherwise, False.

        Raises:
          ValueError: If input eval result is None or no loss is available.
        """
        logger.info(f"使用 compare_key={compare_key}进行比较:"
                    f"best_eval_result[{compare_key}]vs current_eval_result[{compare_key}] ={best_eval_result[compare_key]} vs {current_eval_result[compare_key]}")

        if not best_eval_result or compare_key not in best_eval_result:
            raise ValueError(
                f'best_eval_result cannot be empty or no loss is found in it\n'
                f'compare_key={compare_key}\nbest_eval_result={best_eval_result} ,type ={type(best_eval_result)}\n'
                f'current_eval_result={current_eval_result},type ={type(current_eval_result)}')

        if not current_eval_result or compare_key not in current_eval_result:
            raise ValueError(
                'current_eval_result cannot be empty or no loss is found in it.')

        if compare_key == MetricKeys.LOSS:
            is_true = best_eval_result[compare_key] > current_eval_result[compare_key]
            if is_true:
                logger.info("我们保存当前的模型")
            return best_eval_result[compare_key] > current_eval_result[compare_key]
        if compare_key in [MetricKeys.ACCURACY, MetricKeys.PRECISION, MetricKeys.RECALL, "f1-score"]:
            is_true = best_eval_result[compare_key] < current_eval_result[compare_key]
            if is_true:
                logger.info("我们保存当前的模型")
            return best_eval_result[compare_key] < current_eval_result[compare_key]

    return exporter_compare_fn


# # 计算micro average的precision、recall和f1
# def safe_div(numerator, denominator):
#     """安全除，分母为0时返回0"""
#     numerator, denominator = tf.cast(numerator,dtype=tf.float64), tf.cast(denominator,dtype=tf.float64)
#     zeros = tf.zeros_like(numerator, dtype=numerator.dtype) # 创建全0Tensor
#     denominator_is_zero = tf.equal(denominator, zeros) # 判断denominator是否为零
#     return tf.where(denominator_is_zero, zeros, numerator / denominator) # 如果分母为0，则返回零
#
# def pr_re_f1(cm,pos_indices):
#     num_classes = cm.shape[0]
#
#     # 计算 负类型的 index
#     neg_indices = [i for i in range(num_classes) if i not in pos_indices]
#     cm_mask = np.ones([num_classes, num_classes])
#     cm_mask[neg_indices, neg_indices] = 0  # 将负样本预测正确的位置清零零
#     diag_sum = tf.reduce_sum(tf.diag_part(cm * cm_mask))  # 正样本预测正确的数量
#
#     cm_mask = np.ones([num_classes, num_classes])
#     cm_mask[:, neg_indices] = 0  # 将负样本对应的列清零
#     tot_pred = tf.reduce_sum(cm * cm_mask)  # 所有被预测为正的样本数量
#
#     cm_mask = np.ones([num_classes, num_classes])
#     cm_mask[neg_indices, :] = 0  # 将负样本对应的行清零
#     tot_gold = tf.reduce_sum(cm * cm_mask)  # 所有正样本的数量
#
#     pr = safe_div(diag_sum, tot_pred)
#     re = safe_div(diag_sum, tot_gold)
#     f1 = safe_div(2. * pr * re, pr + re)
#     return pr, re, f1
#
# def compute_micro_statistics(labels, predictions, num_classes, pos_indices):
#     """
#     :param cm:
#     :param pos_indices:  正类的 index
#     :return:
#     """
#     # cm是上一个batch的混淆矩阵，op是更新完当前batch的
#     cm, op = _streaming_confusion_matrix(labels, predictions, num_classes)
#     pr, re, f1 = pr_re_f1(op,pos_indices)
#     return pr, re, f1
#
# def compute_marco_statitics(labels, predictions, num_classes):
#     # cm是上一个batch的混淆矩阵，op是更新完当前batch的
#     cm, op = _streaming_confusion_matrix(labels, predictions, num_classes)
#     precisions, recalls, f1s, n_golds = [], [], [], []
#     pos_indices = [0, 1]  # 正例
#     # 计算每个正例的 precison, recall 和 f1
#     for idx in pos_indices:
#         pr, re, f1 = pr_re_f1(cm, [idx])
#         precisions.append(pr)
#         recalls.append(re)
#         f1s.append(f1)
#         cm_mask = np.zeros([num_classes, num_classes])
#         cm_mask[idx, :] = 1
#         n_golds.append(tf.to_float(tf.reduce_sum(cm * cm_mask)))
#     pr = tf.reduce_mean(precisions)
#     re = tf.reduce_mean(recalls)
#     f1 = tf.reduce_mean(f1s)
#     return pr, re,f1


if __name__ == '__main__':

    batch_num = 2
    labels = np.array([0, 0, 1, 2, 1, 2, 2, 2], dtype=np.int64).reshape(batch_num, -1)
    predictions = np.array([0, 1, 2, 2, 0, 0, 1, 2], dtype=np.int64).reshape(batch_num, -1)
    print("使用 sklearn 进行评估")
    for i in range(labels.shape[0]):
        accuracy_value = accuracy_score(y_true=labels[i], y_pred=predictions[i])
        precision_value = precision_score(y_true=labels[i], y_pred=predictions[i], average='macro')
        recall_value = recall_score(y_true=labels[i], y_pred=predictions[i], average='macro')
        f1_value = f1_score(y_true=labels[i], y_pred=predictions[i], average='macro')
        cm = confusion_matrix(y_true=labels[i], y_pred=predictions[i])
        clas_report = classification_report(y_true=labels[i], y_pred=predictions[i])
        print(
            f'y_true={labels[i]},y_pred={predictions[i]}\naccuracy={accuracy_value},precision={precision_value},recall={recall_value},f1={f1_value},'
            f'\nconfusion_matrix=\n{cm}\nclas_report=\n{clas_report}')

    labels_re = labels.reshape(-1)
    predictions_re = predictions.reshape(-1)
    precision_value = precision_score(y_true=labels_re, y_pred=predictions_re, average='macro')
    recall_value = recall_score(y_true=labels_re, y_pred=predictions_re, average='macro')
    f1_value = f1_score(y_true=labels_re, y_pred=predictions_re, average='macro')
    cm = confusion_matrix(y_true=labels_re, y_pred=predictions_re)
    clas_report = classification_report(y_true=labels_re, y_pred=predictions_re)
    print(
        f'y_true={labels_re},y_pred={predictions_re}\naccuracy={accuracy_value},precision={precision_value},recall={recall_value},f1={f1_value},'
        f'\nconfusion_matrix=\n{cm}\nclas_report=\n{clas_report}')

    labels_tensor = tf.convert_to_tensor(labels)
    predictions_tensor = tf.convert_to_tensor(predictions)
    num_classes = 3  # 3个类别0,1,2

    # pr, re, f1 = compute_micro_statistics(labels=labels_tensor,predictions=predictions_tensor,num_classes=num_classes,pos_indices= [0, 1])
    # pr, re, f1 = computer_macro_statistics(labels=labels,predictions=predictions,num_classes=num_classes,pos_indices= [0, 1])

    labels_placehold = tf.placeholder(tf.int64, [None])
    predictions_placehold = tf.placeholder(tf.int64, [None])
    tf_precision = precision(labels=labels_placehold, predictions=predictions_placehold, num_classes=3, average='macro')

    # Isolate the variables stored behind the scenes by the metric operation
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # Get the local variables true_positive and false_positive
        stream_vars = [i for i in tf.local_variables()]

        for i in range(labels_tensor.shape[0]):
            batch_labels = labels[i]
            batch_predictions = predictions[i]
            print(
                f"batch_i={i} before run update_op,stream_vars=\n{sess.run(stream_vars)},\nprecision_value={precision_value}")
            sess.run(tf_precision[1],
                     feed_dict={labels_placehold: batch_labels, predictions_placehold: batch_predictions})
            precision_value = sess.run(tf_precision[0])
            print(
                f"batch_i={i} after run update_op,stream_vars=\n{sess.run(stream_vars)},\nprecision_value={precision_value}")
