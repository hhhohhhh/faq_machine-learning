#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/11 17:07 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/11 17:07   wangfc      1.0         None
"""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

from functools import partial
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib import data as contrib_data
from tensorflow.contrib import metrics as contrib_metrics
from tensorflow.contrib import tpu as contrib_tpu

from albert import modeling, optimization
from albert.fine_tuning_utils import _create_model_from_hub, _create_model_from_scratch
from utils.tensorflow_v1.metrics import metric_fnV2


def create_model(albert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, task_name,
                 hub_module, multilabel_classifier=False, use_einsum=True):
    """Creates a classification model.
    新版 albet 增加参数： use_einsum
    """
    if hub_module:
        tf.logging.info("creating model from hub_module: %s", hub_module)
        output_layer = _create_model_from_hub(hub_module, is_training, input_ids,
                                              input_mask, segment_ids)
    else:
        tf.logging.info("creating model from albert_config")
        output_layer = _create_model_from_scratch(albert_config, is_training,
                                                  input_ids, input_mask, segment_ids,
                                                  use_one_hot_embeddings, use_einsum=use_einsum)

    if isinstance(output_layer, tuple):
        output_layer = output_layer[0]

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        if task_name != "sts-b" and multilabel_classifier == False:
            # logits = (batch_size, output_dim) ---> probabilities=  (batch_size, output_dim)
            probabilities = tf.nn.softmax(logits, axis=-1)
            # predictions =  (batch_size, 1)
            predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
            # log_probs = (batch_size, output_dim)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            # one_hot_labels =  (batch_size, ouput_dim)
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            # (batch_size, output_dim) ---> (batch_size)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        elif task_name == "sts-b":
            probabilities = logits
            logits = tf.squeeze(logits, [-1])
            predictions = logits
            per_example_loss = tf.square(logits - labels)
        elif multilabel_classifier == True:
            tf.logging.info("创建多标签模型 in create_model().")
            # 在多标签分类中，使用 sigmoid() 来获取概率
            #  logits = (batch_size, output_dim) ---> probabilities = (batch_size, output_dim) : 不对 logits 进行 softmax 而是进行 sigmoid
            probabilities = tf.nn.sigmoid(logits)
            # predictions =  (batch_size, 1)
            predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
            labels = tf.cast(labels, tf.float32)
            # 先对 logits 进行 sigmoid， 然后 对 label进行 one-hot encoding，然后计算 loss
            # logits = (batch_size, output_dim)，labels = (batch_size, output_dim) --->  loss = (batch_size, output_dim)
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, probabilities, logits, predictions)


def model_fn_builder(albert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, task_name, hub_module=None,
                     optimizer="adamw",multilabel_classifier=False):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        elif not multilabel_classifier:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
        elif multilabel_classifier:
            is_real_example = tf.ones(shape=(tf.shape(label_ids)[0], 1), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, probabilities, logits, predictions) = \
            create_model(albert_config, is_training, input_ids, input_mask,
                         segment_ids, label_ids, num_labels,
                         use_one_hot_embeddings, task_name, hub_module)

        # # 使用 tf.identity operation 来复制一个节点作为输出节点
        # outputs = tf.identity(probabilities, name="probabilities")

        # 或许需要训练的 变量
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        # 获取 global_step
        global_step = tf.train.get_or_create_global_step()

        # TPUEstimatorSpec.eval_metrics 是 metric_fn 和 tensors 的元组，其中tensors可以是Tensor s的列表或Tensor s的名称字典。
        # metric_fn 获取 tensors并从度量字符串名称返回dict到调用度量函数的结果，即(metric_tensor, update_op)元组。
        eval_metrics = (metric_fnV2,
                        [per_example_loss, label_ids, logits, is_real_example, num_labels, "macro", 1])

        # 生产 metrics的字典
        metric_dict = metric_fnV2(per_example_loss, label_ids, logits, is_real_example, num_labels, "macro", 1)

        output_spec = None
        # 当Estimator的train方法被调用时，model_fn会使用mode = ModeKeys.TRAIN进行调用。
        # 在这种情况下，model_fn必须返回一个包含loss和一个训练操作 train_op(它会执行一个training step)的EstimatorSpec。
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps,
                use_tpu, optimizer)

            # 要在训练过程中实现输出 loss 日志，我们可以使用 LoggingTensorHook 参数
            train_log = {'global_step': global_step,
                         'train total_loss': total_loss,
                         "train accuracy": metric_dict['accuracy'][1],
                         # "train precision": metric_dict['precision'][1],
                         # "train recall": metric_dict['recall'][1],
                         # "train f1-socre ": metric_dict['f1-score'][1],
                         }
            train_log_hook = tf.train.LoggingTensorHook(tensors=train_log, every_n_iter=100)

            # 要在eval 过程中实现输出 loss 日志，我们可以使用 LoggingTensorHook 参数
            eval_log = {'global_step': global_step,
                        'eval total_loss': total_loss,
                        "eval accuracy": metric_dict['accuracy'][1],
                        # "eval precision": metric_dict['precision'][1],
                        # "eval recall": metric_dict['recall'][1],
                        # "eval f1-socre ": metric_dict['f1-score'][1],
                        }

            eval_log_hook = tf.train.LoggingTensorHook(tensors=eval_log, every_n_iter=100)

            # output_spec = contrib_tpu.TPUEstimatorSpec(
            #     mode=mode,
            #     loss=total_loss,
            #     train_op=train_op,
            #     scaffold_fn=scaffold_fn)

            # 生产一个统一的 TPUEstimatorSpec
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                # 对于mode == ModeKeys.TRAIN：必填字段是loss和train_op.
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                eval_metrics=eval_metrics,  # eval_metrics 是 metric_fn 和tensors 的元组
                predictions={"probabilities": probabilities},
                training_hooks=[train_log_hook],  # , train_summary_hook],
                evaluation_hooks=[eval_log_hook],  # , eval_summary_hook],
                scaffold_fn=scaffold_fn)

        # elif mode == tf.estimator.ModeKeys.EVAL:
        #   if task_name == 'text_classifier':
        #
        #   elif task_name not in ["sts-b", "cola"]:
        #     def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        #       predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        #       accuracy = tf.metrics.accuracy(
        #           labels=label_ids, predictions=predictions,
        #           weights=is_real_example)
        #       loss = tf.metrics.mean(
        #           values=per_example_loss, weights=is_real_example)
        #       return {
        #           "eval_accuracy": accuracy,
        #           "eval_loss": loss,
        #       }
        #   elif task_name == "sts-b":
        #     def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        #       """Compute Pearson correlations for STS-B."""
        #       # Display labels and predictions
        #       concat1 = contrib_metrics.streaming_concat(logits)
        #       concat2 = contrib_metrics.streaming_concat(label_ids)
        #
        #       # Compute Pearson correlation
        #       pearson = contrib_metrics.streaming_pearson_correlation(
        #           logits, label_ids, weights=is_real_example)
        #
        #       # Compute MSE
        #       # mse = tf.metrics.mean(per_example_loss)
        #       mse = tf.metrics.mean_squared_error(
        #           label_ids, logits, weights=is_real_example)
        #
        #       loss = tf.metrics.mean(
        #           values=per_example_loss,
        #           weights=is_real_example)
        #
        #       return {"pred": concat1, "label_ids": concat2, "pearson": pearson,
        #               "MSE": mse, "eval_loss": loss,}
        #   elif task_name == "cola":
        #     def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        #       """Compute Matthew's correlations for STS-B."""
        #       predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        #       # https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        #       tp, tp_op = tf.metrics.true_positives(
        #           predictions, label_ids, weights=is_real_example)
        #       tn, tn_op = tf.metrics.true_negatives(
        #           predictions, label_ids, weights=is_real_example)
        #       fp, fp_op = tf.metrics.false_positives(
        #           predictions, label_ids, weights=is_real_example)
        #       fn, fn_op = tf.metrics.false_negatives(
        #           predictions, label_ids, weights=is_real_example)
        #
        #       # Compute Matthew's correlation
        #       mcc = tf.div_no_nan(
        #           tp * tn - fp * fn,
        #           tf.pow((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.5))
        #
        #       # Compute accuracy
        #       accuracy = tf.metrics.accuracy(
        #           labels=label_ids, predictions=predictions,
        #           weights=is_real_example)
        #
        #       loss = tf.metrics.mean(
        #           values=per_example_loss,
        #           weights=is_real_example)
        #
        #       return {"matthew_corr": (mcc, tf.group(tp_op, tn_op, fp_op, fn_op)),
        #               "eval_accuracy": accuracy, "eval_loss": loss,}
        #
        #   eval_metrics = (metric_fn,
        #                   [per_example_loss, label_ids, logits, is_real_example])
        #   output_spec = contrib_tpu.TPUEstimatorSpec(
        #       mode=mode,
        #       loss=total_loss,
        #       eval_metrics=eval_metrics,
        #       scaffold_fn=scaffold_fn)
        else:
            output_spec = contrib_tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={
                    "probabilities": probabilities,
                    "predictions": predictions
                },
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def get_serving_input_fn(max_seq_length: int, output_size: int):
    def serving_input_fn():
        # 保存模型为SaveModel格式
        # 采用最原始的feature方式，输入是feature Tensors。
        # 如果采用build_parsing_serving_input_receiver_fn，则输入是tf.Examples

        # serving_input_receiver_fn 格式定义：
        label_ids = tf.placeholder(tf.int32, [None, output_size], name='label_ids')
        input_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='input_ids')
        input_mask = tf.placeholder(tf.int32, [None, max_seq_length], name='input_mask')
        segment_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='segment_ids')

        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'label_ids': label_ids,
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
        })()
        return input_fn

    return serving_input_fn
