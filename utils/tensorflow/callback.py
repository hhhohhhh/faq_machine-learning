#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:
回调是传递给模型的对象，用于在训练期间自定义该模型并扩展其行为。您可以编写自定义回调，

tf.keras的回调函数实际上是一个类 tf.keras.callbacks.Callback
一般是在model.fit时作为参数指定，用于控制在训练过程开始或者在训练过程结束，在每个epoch训练开始或者训练结束，在每个batch训练开始或者训练结束时执行一些操作，
例如收集一些日志信息，改变学习率等超参数，提前终止训练过程等等。

同样地，针对model.evaluate或者model.predict也可以指定callbacks参数，用于控制在评估或预测开始或者结束时，在每个batch开始或者结束时执行一些操作，
但这种用法相对少见。

tf.keras.callbacks.ModelCheckpoint：定期保存模型的检查点。
tf.keras.callbacks.LearningRateScheduler：动态更改学习速率。
tf.keras.callbacks.EarlyStopping：在验证效果不再改进时中断训练。
tf.keras.callbacks.TensorBoard：使用 TensorBoard 监控模型的行为。

要使用 tf.keras.callbacks.Callback，请将其传递给模型的 fit 方法：

callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]


所有回调函数都继承至 keras.callbacks.Callbacks基类，拥有params和model这两个属性。
其中 params 是一个dict，记录了 training parameters (eg. verbosity, batch size, number of epochs...).
model即当前关联的模型的引用。


对于回调类中的一些方法:
- on_epoch_begin(self, epoch, logs=None) :
- on_batch_end(self, batch, logs=None)
输入参数 logs, 提供有关当前epoch或者batch的一些信息，并能够记录计算结果，如果model.fit指定了多个回调函数类，这些logs变量将在这些回调函数类的同名函数间依顺序传递。
被回调函数作为参数的 logs 字典，它会含有于当前批量或训练轮相关数据的键。
目前，Sequential 模型类的 .fit() 方法会在传入到回调函数的 logs 里面包含以下的数据：

on_epoch_end: 包括 acc 和 loss 的日志， 也可以选择性的包括 val_loss（如果在 fit 中启用验证），和 val_acc（如果启用验证和监测精确值）。
on_batch_begin: 包括 size 的日志，在当前批量内的样本数量。
on_batch_end: 包括 loss 的日志，也可以选择性的包括 acc（如果启用监测精确值）。

也可以使用包含以下方法的内置 tf.keras.callbacks：



ReduceLROnPlateau
will adjust the learning rate when a plateau in model performance is detected, e.g. no change for a given number of training epochs.
his callback is designed to reduce the learning rate after the model stops improving with the hope of fine-tuning model weights.
    - monitor : the metric to monitor during training
    - factor: value that the learning rate will be multiplied
    - patience:the number of training epochs to wait before triggering the change in learning rate.


@time: 2021/5/20 16:08 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/20 16:08   wangfc      1.0         None
"""

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=g-import-not-at-top
# pylint: disable=g-classes-have-attributes
from utils.common import is_logging_disabled

"""Callbacks: utilities called at certain points during model training."""
import logging
import os
import re
from logging import Logger
from pathlib import Path
from typing import Optional, Dict, Text, Any, List, Tuple
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.python.ops import summary_ops_v2

from tqdm import tqdm

# import transformers.trainer
from utils.tensorflow.save_utils import TF_CKPT_NAME

logger = logging.getLogger(__name__)





def create_common_callbacks(
        epochs: int,
        logger: Optional[Logger] = None,
        tensorboard_log_dir: Optional[Text] = None,
        tensorboard_log_update_freq: Optional[Text] = 100,
        output_dir: Optional[Path] = None,
        # checkpoint_name="model_{epoch}", # # The saved model name will include the current epoch.
        tf_model_name= TF_CKPT_NAME,
        monitor ='val_loss',
        save_best_only=True, # Only save a model if `val_loss` has improved.
        signatures=None,
        verbose=1,**model_config) -> List["Callback"]:
    """Create common callbacks.

    The following callbacks are created:
    - RasaTrainingLogger callback
    - Optional TensorBoard callback
    - Optional RasaModelCheckpoint callback

    Args:
        epochs: the number of epochs to train
        tensorboard_log_dir: optional directory that should be used for tensorboard
        tensorboard_log_level: defines when training metrics for tensorboard should be
                               logged. Valid values: 'epoch' and 'batch'.
        checkpoint_dir: optional directory that should be used for model checkpointing

    Returns:
        A list of callbacks.
    """
    tf_model_name_pattern = "%s_{epoch}" % tf_model_name
    tf_model_filepath = os.path.join(output_dir,tf_model_name_pattern)
    callbacks = [LoggerCallback(epochs=epochs, logger=logger),
                 # ModelCheckpoint 保存的是 saved_model 格式的模型 ?
                 ModelCheckpoint(filepath=tf_model_filepath,
                                 monitor=monitor,
                                 save_best_only=save_best_only,
                                 verbose=verbose)
                 ]

    # callbacks.append(SavedModelCallback(checkpoint_dir=checkpoint_dir, checkpoint_name=checkpoint_name,
    #                                     savedmodel_name=savedmodel_name))
    if tensorboard_log_dir:
        callbacks.append(LearningRateTensorBoard(log_dir=tensorboard_log_dir,
                                                 update_freq=tensorboard_log_update_freq))

    return callbacks



class LoggerCallback(tf.keras.callbacks.Callback):
    """
    @time:  2021/5/20 22:47
    @author:wangfc
    @version: 来源自  RasaTrainingLogger
    @description: Callback for logging the status of training.

    @params:
    @return:
    """

    def __init__(self, logger: Optional[Logger] = None, log_every_steps = 100,
                 epochs: int = None, silent: bool = False) -> None:
        """Initializes the callback.

        Args:
            epochs: Total number of epochs.
            silent: If 'True' the entire progressbar wrapper is disabled.
        """
        # 增加 log 日志写入的频率
        self.log_every_steps = log_every_steps
        super().__init__()

        # disable = silent or rasa.shared.utils.io.is_logging_disabled()
        # self.progress_bar = tqdm(range(epochs), desc="训练Epochs", disable=disable)
        self.logger = logger

    def on_batch_end(self, batch: int, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Updates the logging output on every epoch end.

        Args:
            batch: batch 在每个 epoch 会重置为 0
            logs: The training metrics: ex: {'loss': 5.3495001792907715, 'accuracy': 0.0}
        """
        # self.progress_bar.update(1)
        # self.progress_bar.set_postfix(logs)
        # keras 自带的 epoch 是从 1开始的

        iterations = K.eval(self.model.optimizer._iterations)
        if iterations % self.log_every_steps == 0:
            lr = get_lr(step=iterations, optimizer=self.model.optimizer)
            logs["learning_rate"] = lr
            msg = "steps: %i, %s" % (iterations, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
            if self.logger:
                self.logger.info(msg)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Updates the logging output on every epoch end.

        Args:
            epoch: The current epoch.
            logs: The training metrics.
        """
        # self.progress_bar.update(1)
        # self.progress_bar.set_postfix(logs)
        # keras 自带的 epoch 是从 1开始的
        # if logs is None or "learning_rate" in logs:
        #     return
        # logs["learning_rate"] = self.model.optimizer.lr_t   #get_lr(optimizer=self.model.optimizer)

        msg = "Epoch: %i, %s" % (epoch + 1, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        if self.logger:
            self.logger.info(msg)

    def on_train_end(self, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Closes the progress bar after training.

        Args:
            logs: The training metrics.
        """
        # self.progress_bar.close()
        msg = "训练结束"
        if self.logger:
            self.logger.info(msg)


class LearningRateTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir,
                 update_freq=100, #'batch', 控制写入 tensorboard 的频率
                 write_graph=True,
                 write_images=False,
                 histogram_freq=10, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, update_freq=update_freq, write_graph=write_graph, write_images=write_images,
                         histogram_freq=histogram_freq,
                         **kwargs)

    def on_train_batch_end(self, batch, logs=None):
        """
        1. 使用 on_train_batch_end，而不是 on_batch_end()  Todo： why？
        2. 编写 _log_learning_rate() 函数 写入 learning_rate 到 filewriter
        3. 使用 tf.keras.optimizers.schedules.LearningRateSchedule  API 编写的 warmup 来建立 learning_rate_schedule，在新建 optimizer的时候传入，
           可以使用 learning_rate_schedule(step) 动态获取 learning_rate
           也可以使用  tf.keras.callback.LearningRateScheduler() API 来控制 learning_rate，此时 会在每个 epoch结束的的时候（.on_epoch_end() ）将 learning_rate 写入 log 中，从而可以直接使用原始的
           tf.keras.callbacks.TensorBoard 就可以记录 log 中的 learning_rate (epoch)
        4. batch 在每个 epoch 会重置为 0
        """

        iterations = K.eval(self.model.optimizer._iterations)
        if iterations % self.update_freq==0:
            lr = get_lr(step=iterations,optimizer=self.model.optimizer)
            # 写入 learning_rate
            self._log_learning_rate(step=iterations,learning_rate=lr)
            super().on_train_batch_end(batch=iterations,logs=logs)


    def _log_learning_rate(self, step, learning_rate):
        """Writes epoch metrics out as scalar summaries.

        Arguments:
            epoch: Int. The global step to use for TensorBoard.
            logs: Dict. Keys are scalar summary names, values are scalars.
        """
        with summary_ops_v2.always_record_summaries():
            with self._train_writer.as_default():
                summary_ops_v2.scalar('learning_rate', learning_rate, step=step)




# @tf.function
def get_lr(step,optimizer:tf.keras.optimizers):
    """
    @time:  2021/7/8 9:15
    @author:wangfc
    @version:
    @description:


    @params:
    @return:
    """
    # 获取 optimizer.lr 属性
    learning_rate = optimizer.lr
    if isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
        # 对于使用 optimizer + LearningRateSchedule 来调整 learning_rate的方式，可以通过这个方法获取
        learning_rate = learning_rate(step=step)
    # 转换 为 numpy
    learning_rate = K.eval(learning_rate)
    return learning_rate


    # # 使用 SGD(learning_rate=0.005, decay=0.01)
    # _initial_decay = optimizer._initial_decay
    #
    # if _initial_decay > 0:
    #     learning_rate = optimizer._decayed_lr('float32')
    #
    # lr_schedule = getattr(optimizer, 'lr', None)
    # if isinstance(lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
    #     # 使用 tf.keras.optimizers.schedules
    #     # tf.keras.callbacks.TensorBoard 会判断 lr_schedule 是否 LearningRateSchedule
    #     learning_rate = lr_schedule(optimizer.iterations)
    #
    # if isinstance(optimizer, AdamW):
    #     learning_rate = optimizer.lr_t

    return learning_rate






class ValidateCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_accuracy') > 1.0):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True




class SavedModelCallback(tf.keras.callbacks.Callback):
    """Callback for saving intermediate model checkpoints."""

    def __init__(self, checkpoint_dir: Path,
                 checkpoint_name="checkpoint.tf_model", keys='all', save_weights_only=False,
                 save_model=True, savedmodel_name='tf_model', save_format='tf',
                 signatures=None,
                 include_optimizer=False) -> None:
        """Initializes the callback.

        Args:
            checkpoint_dir: Directory to store checkpoints to.

            keys=[val_loss,val_accuracy]
        """
        super().__init__()

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.keys = keys  # 模型评估指标
        self.best_metrics_so_far: Optional[Dict[Text, Any]] = {}
        self.save_weights_only = save_weights_only
        self.save_model = save_model
        self.savedmodel_name = savedmodel_name
        self.save_format = save_format
        self.signatures = signatures
        self.include_optimizer = include_optimizer

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Save the model on epoch end if the model has improved.

        只保存权重weights：
        1） model.save_weights API；
        2） callbacks： ModelCheckpoint，但是参数save_weights_only 要设置成True

        Args:
            epoch: The current epoch.
            logs: The training metrics.
            {'loss': 0.6612747311592102,
             'accuracy': 0.6090149879455566,
             'val_loss': 0.5485597252845764,
             'val_accuracy': 0.7643013596534729,
             'learning_rate': <tf.Variable 'Adam/learning_rate:0' shape=() dtype=float32, numpy=1e-05>}
        """
        # 判断是否针对 评估的 key 有改进：
        all_improved = self._does_model_improve(logs)
        if all_improved:
            # logger.info(
            #     f"Creating model checkpoint at epoch={epoch + 1} ,best_metrics_so_far ={self.best_metrics_so_far}")
            # checkpoint_file = os.path.join(self.checkpoint_dir, f"{self.checkpoint_name}_epoch_{epoch}")
            #
            # self.model.save_weights(
            #     checkpoint_file, overwrite=True, save_format=self.save_format
            # )
            if self.save_model:
                """
                需要注意的是，这种方式依然会保存模型的所有信息，即“网络结构、权重、配置、优化器状态”四个信息，所以可以接着训练。"""
                saved_model_file = os.path.join(self.checkpoint_dir, f"{self.savedmodel_name}_epoch_{epoch}")
                # overwrite：如果有已经存在的model，是否覆盖它
                # include_optimizer：是否将优化器optimizer的状态一起保存到模型中
                # save_format：是保存成"tf"格式，还是"h5"格式；#在2.x中默认是"tf","tf"格式也就是.pb格式的文件
                self.model.save(saved_model_file, save_format=self.save_format,
                                include_optimizer=self.include_optimizer,
                                signatures=self.signatures)

    def _does_model_improve(self, current_results: Dict[Text, Any]) -> bool:
        # Initialize best_metrics_so_far with the first results
        if not self.best_metrics_so_far:
            # 获取 比较的 key
            if self.keys == 'all':
                keys = filter(
                    lambda k: k.startswith("val")
                              and (k.endswith("_loss") or k.endswith("_accuracy")),
                    current_results.keys(),
                )
            elif self.keys=='accuracy':
                keys = filter(
                    lambda k: k.startswith("val") and k.endswith("_accuracy") ,
                    current_results.keys())
            elif self.keys=='loss':
                keys = filter(
                    lambda k: k.startswith("val") and k.endswith("_loss") ,
                    current_results.keys())
            else:
                keys = self.keys

            if not keys:
                return False

            for key in keys:
                self.best_metrics_so_far[key] = float(current_results[key])
            return True

        all_improved = all(
            [
                float(current_results[key]) > self.best_metrics_so_far[key]
                if not key.endswith('loss') else float(current_results[key]) < self.best_metrics_so_far[key]
                for key in self.best_metrics_so_far.keys()

            ]
        )

        if all_improved:
            for key in self.best_metrics_so_far.keys():
                self.best_metrics_so_far[key] = float(current_results[key])

        return all_improved


class BestModelSaverCallback(tf.keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_acc = 0.

    def evaluate(self, data):
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = self.model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]
            total += len(y_true)
            right += (y_true == y_pred).sum()
        return right / total

    def on_epoch_end(self, epoch, logs=None):
        val_acc = self.evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights('best_model.weights')
        # test_acc = evaluate(test_generator)
        # print(
        #     u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
        #     (val_acc, self.best_val_acc, test_acc)
        # )





class RasaTrainingLogger(tf.keras.callbacks.Callback):
    """Callback for logging the status of training."""

    def __init__(self, epochs: int, silent: bool,ncols=100, steps_per_epoch:int=None) -> None:
        """Initializes the callback.
        进度百分比|进度条| 当前迭代数/总迭代个数，[消耗时间<剩余时间，迭代的速度]
        Args:
            epochs: Total number of epochs.
            silent: If 'True' the entire progressbar wrapper is disabled.
            ncols: tqdm固定总长度（字符数量）
            desc: 设置前缀 一般为epoch的信息,可以使用  pbar.set_description() 方法设置
            pbar.update() 设置你每一次想让进度条更新的iteration 大小
            pbar.set_postfix() : 设置你想要在本次循环内实时监视的变量  可以作为后缀打印出来
        """
        super().__init__()

        # from rasa.shared.utils.io import is_logging_disabled
        disable = silent or is_logging_disabled()

        self.progress_bar =None
        self.progress_bar = tqdm(range(epochs), desc="Epochs", disable=disable, ncols=ncols)

        self.batch_progress_bar = None
        # if steps_per_epoch:
        #     self.batch_progress_bar = tqdm(range(steps_per_epoch),desc="Batches",disable=disable,ncols=ncols)


    def on_epoch_end(self, epoch: int, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Updates the logging output on every epoch end.

        Args:
            epoch: The current epoch.
            logs: The training metrics.
        """
        if self.batch_progress_bar:
            print(f"epoch={epoch},logs={logs}")
        else:
            self.progress_bar.update(1)
            # short_logs = {key: "{0:.3f}".format(value) for key,value in logs.items()}
            self.progress_bar.set_postfix(logs)

    # def on_batch_end(self, batch, logs=None):
    #     if self.batch_progress_bar:
    #         self.batch_progress_bar.update(1)
    #         self.batch_progress_bar.set_postfix(logs)

    def on_train_end(self, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Closes the progress bar after training.

        Args:
            logs: The training metrics.
        """
        if self.progress_bar:
            self.progress_bar.close()
        if self.batch_progress_bar:
            self.batch_progress_bar.close()






class RasaModelCheckpoint(tf.keras.callbacks.Callback):
    """Callback for saving intermediate model checkpoints."""

    def __init__(self, checkpoint_dir: Path) -> None:
        """Initializes the callback.

        Args:
            checkpoint_dir: Directory to store checkpoints to.
        """
        super().__init__()

        self.checkpoint_file = checkpoint_dir / "checkpoint.tf_model"
        self.best_metrics_so_far: Optional[Dict[Text, Any]] = {}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Save the model on epoch end if the model has improved.

        Args:
            epoch: The current epoch.
            logs: The training metrics.
        """
        if self._does_model_improve(logs):
            logger.debug(f"Creating model checkpoint at epoch={epoch + 1} ...")
            self.model.save_weights(
                self.checkpoint_file, overwrite=True, save_format="tf"
            )

    def _does_model_improve(self, current_results: Dict[Text, Any]) -> bool:
        # Initialize best_metrics_so_far with the first results
        if not self.best_metrics_so_far:
            keys = filter(
                lambda k: k.startswith("val")
                and (k.endswith("_acc") or k.endswith("_f1")),
                current_results.keys(),
            )
            if not keys:
                return False

            for key in keys:
                self.best_metrics_so_far[key] = float(current_results[key])
            return True

        all_improved = all(
            [
                float(current_results[key]) > self.best_metrics_so_far[key]
                for key in self.best_metrics_so_far.keys()
            ]
        )

        if all_improved:
            for key in self.best_metrics_so_far.keys():
                self.best_metrics_so_far[key] = float(current_results[key])

        return all_improved
