#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/5/19 22:08 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/19 22:08   wangfc      1.0         None
"""
import os
import importlib
from pathlib import Path
from typing import Text, Union, Dict, List, Tuple, Optional
import tqdm
import numpy as np
from models.builder import MODELS
import tensorflow as tf

from utils.io import get_files_from_dir, read_yaml_file, get_file_stem

if tf.__version__ >= "2.0":
    from tensorflow.keras.models import Model
    from tensorflow.keras.losses import Loss
    from tensorflow.keras.optimizers.schedules import LearningRateSchedule
    from tensorflow.keras.metrics import Metric
    from tensorflow.python.eager.function import ConcreteFunction
    from data_process.data_generator import KerasTextClassifierDataGenerator
    from models.temp_keras_modules import TmpKerasModel
else:
    Model = None
    Loss = None
    LearningRateSchedule = None
    Metric = None
    ConcreteFunction = None
    KerasTextClassifierDataGenerator = None
    TmpKerasModel = None

from utils.common import init_logger
from utils.constants import TEST_DATATYPE

import logging

from utils.tensorflow.save_utils import get_tf_model_dir_to_epoch_ls, load_tf_model, save_model_to_savedmodel, \
    get_latest_ckpt_dir_to_epoch

logger = logging.getLogger(__name__)


# MODEL_NAME2MODULE = {
#     'BaiscConvolutionModel': 'model.convolution_model_tf',
#     'ThreeConvolutionLayerModelV1': 'model.convolution_model_tf',
#     'ThreeConvolutionLayerModelV2': 'model.convolution_model_tf',
#     'TransferInceptionModel': 'model.inception_model_tf',
#     'EasyResidualModel': 'model.resnet_tf',
#     'MiniResidualModel': 'model.resnet_tf',
# }


class BaseTfTask():
    """
    数据生成
    建立模型
    进行训练
    进行评估
    模型输出
    """

    def __init__(self,
                 debug=False, run_eagerly=False, log_level='info', log_subdir='log',
                 task=None, mode='train',
                 corpus='corpus', dataset=None, data_subdir=None, data_dir=None,
                 test_data_subdir='test_data',
                 test_data_filename=None,
                 output_test_data_filename='test_data.json',
                 output_dir='output',
                 output_model_subdir='models',
                 model_name=None,
                 verbose_model_dirname=False,
                 model_config: Dict[Text, any] = {},
                 model_config_filepath: Union[Text, Path] = None,
                 model_input_keys=None,
                 model_output_logits: bool = None,
                 loss_name=None,
                 metric_names=None,  # ['accuracy'],
                 optimizer_name=None,
                 learning_rate: Union[float, LearningRateSchedule] = None,
                 train_epochs=None, train_batch_size=None, dev_batch_size=None, test_batch_size=None,
                 train_data_lowest_threshold: int = None,
                 train_data_highest_threshold: int = None,
                 filter_source: List[Text] = None,
                 filter_single_label_examples: bool = None,
                 batch_strategy=None,
                 class_weight_strategy: Optional[Text] = None,
                 validation_datatype=TEST_DATATYPE,
                 use_multiprocessing=False,
                 workers=1,
                 savedmodel_name='tf_model_epoch_{epoch}',
                 # checkpoint_name =  "checkpoint.tf_model",
                 tensorboard_log_dirname='tensorboard_log',
                 predict_top_k=3, p_threshold=0.5,
                 *args,
                 **kwargs):

        self.debug = debug
        # model.compile() 的时候 debug 模式，这样　model.fit() 的时候可以进行断点调试
        self.run_eagerly = run_eagerly
        self.log_level = log_level
        self.log_subdir = log_subdir
        self.task = task
        self.mode = mode
        self.corpus = corpus
        self.dataset = dataset
        self.data_dir = data_dir
        self.data_subdir = data_subdir
        self.test_data_subdir = test_data_subdir

        self.model_config = model_config

        self.dataset_dir = os.path.join(self.corpus, self.dataset)
        if self.data_dir is None and self.data_subdir is not None:
            self.data_dir = os.path.join(self.corpus, self.dataset, self.data_subdir)

        if self.test_data_subdir is not None:
            self.test_data_dir = os.path.join(self.corpus, self.dataset, self.test_data_subdir)
            if test_data_filename is None and os.path.exists(self.test_data_dir):
                self.test_data_filename = get_files_from_dir(self.test_data_dir, return_filename=True)[0]
            else:
                self.test_data_filename = test_data_filename

        self.output_test_data_filename = output_test_data_filename
        if output_test_data_filename is None and self.test_data_filename:
            self.output_test_data_filename = get_file_stem(self.test_data_filename) + "_test.xlsx"

        self.output_test_data_path = os.path.join(self.test_data_dir, self.output_test_data_filename)

        self.model_config_filepath = model_config_filepath
        if self.model_config_filepath and os.path.exists(self.model_config_filepath):
            self.model_config = read_yaml_file(self.model_config_filepath)['model_paras']
            # logger.info(f"模型配置参数model_config:{self.model_config}")

        self.train_epochs = self._get_param_or_from_model_config(param='train_epochs', value=train_epochs)
        self.train_batch_size = self._get_param_or_from_model_config(param='train_batch_size', value=train_batch_size)
        self.dev_batch_size = self._get_param_or_from_model_config(param='dev_batch_size', value=dev_batch_size)
        self.test_batch_size = self._get_param_or_from_model_config(param='test_batch_size', value=test_batch_size)
        self.validation_datatype = self._get_param_or_from_model_config(param='validation_datatype',
                                                                        value=validation_datatype)
        self.train_data_lowest_threshold = self._get_param_or_from_model_config(param='train_data_lowest_threshold',
                                                                                value=train_data_lowest_threshold)
        self.train_data_highest_threshold = self._get_param_or_from_model_config(param='train_data_highest_threshold',
                                                                                 value=train_data_highest_threshold)
        self.filter_source = self._get_param_or_from_model_config(param='filter_source', value=filter_source)
        self.filter_single_label_examples = self._get_param_or_from_model_config(param='filter_single_label_examples',
                                                                                 value=filter_single_label_examples)

        self.use_multiprocessing = use_multiprocessing
        self.workers = workers

        # self.model = self._get_model(model_name=self.model_name)
        # self.loss = self._get_loss()
        self.model_name = self._get_param_or_from_model_config(param='model_name', value=model_name)
        self.model_input_keys = model_input_keys
        self.model_output_logits = self._get_param_or_from_model_config(param='model_output_logits',
                                                                        value=model_output_logits)
        self.loss_name = self._get_param_or_from_model_config(param='loss_name', value=loss_name)
        self.optimizer_name = self._get_param_or_from_model_config(param='optimizer_name', value=optimizer_name)
        self.learning_rate = self._get_param_or_from_model_config(param='learning_rate', value=learning_rate)
        self.metric_names = self._get_param_or_from_model_config(param='metric_names', value=metric_names)
        self.batch_strategy = self._get_param_or_from_model_config(param='batch_strategy', value=batch_strategy)
        self.class_weight_strategy = self._get_param_or_from_model_config(param="class_weight_strategy",
                                                                          value=class_weight_strategy)

        self.output_dir = output_dir
        self.output_model_subdir = output_model_subdir

        if verbose_model_dirname:
            # 根据 model_name +optimizer_name + learning_rate 来设置 output_model_subdir
            model_dirname = f"{self.model_name}_{self.loss_name}_{self.optimizer_name}_{self.learning_rate}" \
                            f"_{self.train_batch_size}_{self.batch_strategy}_class_weight_strategy{self.class_weight_strategy}"

            if self.train_data_highest_threshold:
                model_dirname = f"{model_dirname}_threshold{self.train_data_highest_threshold}"
            if self.filter_source:
                model_dirname = "{0}_{1}".format(model_dirname, "_".join(self.filter_source))
            self.model_dirname = model_dirname
        else:
            self.model_dirname = self.model_name

        self.tensorboard_log_dirname = tensorboard_log_dirname

        if self.model_dirname:
            self.output_model_dir = os.path.join(self.output_dir, self.output_model_subdir, self.model_dirname)
            self.log_filename = f"{self.mode}_{self.model_dirname}"
            # output/log/model_dir 下建立不同的
            self.tensorboard_log_dir = os.path.join(self.output_dir, self.tensorboard_log_dirname, self.model_dirname)
        else:
            self.output_model_dir = os.path.join(self.output_dir, self.output_model_subdir)
            self.log_filename = f"{self.mode}"
            self.tensorboard_log_dir = os.path.join(self.output_dir, self.tensorboard_log_dirname)

        # self.checkpoint_name =  checkpoint_name
        self.savedmodel_name = savedmodel_name

        self.predict_top_k = predict_top_k
        self.p_threshold = p_threshold

    def _get_param_or_from_model_config(self, param, value):
        if value is None and self.model_config:
            value = self.model_config[param]
        return value

    def train(self):
        raise NotImplementedError

    def _prepare_data(self):
        raise NotImplementedError

    def _load_data(self):
        raise NotImplementedError

    # def _build_model(self,model_name:Text) -> Model:
    #     raise NotImplementedError

    def _get_model_class(self, model_name: Text) -> TmpKerasModel:
        # module_name = MODEL_NAME2MODULE[model_name]
        # module = importlib.import_module(name=module_name)
        module = MODELS.get(model_name)
        # model = getattr(module,model_name)
        return module

    def _instance_model_class(self):
        raise NotImplementedError

    def _build_optimizer(self, optimizer_name: str = 'adam',
                         learning_rate: float = 1e-3,
                         train_steps: int = None, train_step_per_epoch: int = None,
                         warmup_steps: int = None, warmup_epoch=None,
                         # min_lr_ratio=0.0,power: float = 1.0,
                         # clipnorm=1.0,use_cosine_annealing=False,
                         **model_config):
        """
        @time:  2021/6/27 22:43
        @author:wangfc
        @version:
        @description:
        learning_rate :
        Creates an optimizer with a learning rate schedule using a warmup phase followed by a linear decay.
        可以是某个值
                        或者 tf.keras.optimizers.schedules.LearningRateSchedule 类（ TensorboardCallback 可以自动记录 变换的 learning_rate ）

        @params:
        @return:
        """
        from utils.tensorflow.optimizer import create_optimizer

        if warmup_steps is None and warmup_epoch is not None:
            warmup_steps = train_step_per_epoch * warmup_epoch
        self.warmup_steps = warmup_steps
        # optimizer_name = self.optimizer_name.lower()
        # optimizer_name=optimizer_name,learning_rate=self.learning_rate, min_lr_ratio=min_lr_ratio,power=power,use_cosine_annealing=use_cosine_annealing,
        optimizer, lr_schedule = create_optimizer(
            optimizer_name=optimizer_name, learning_rate=learning_rate,
            num_train_steps=train_steps, num_warmup_steps=self.warmup_steps,
            model=self.model, train_step_per_epoch=train_step_per_epoch,
            **model_config)

        logger.info(f"初始化 optimizer= {optimizer},learning_rate= {self.learning_rate},"
                    f"train_step_per_epoch={train_step_per_epoch},"
                    f"warmup_steps={self.warmup_steps}")

        # elif self.optimizer_name.lower() == 'adamw':
        #     Adam =  extend_with_decoupled_weight_decay()

        return optimizer

    def _build_loss(self) -> Loss:
        # if loss_name is None:
        if self.loss_name.lower() == "binary_crossentropy":
            # binary_crossentropy (二元交叉熵，用于二分类，类实现形式为 BinaryCrossentropy)
            # multi-label由于假设每个标签的输出是相互独立的，因此常用配置是sigmoid+BCE， 其中每个类别输出对应一个sigmoid。
            # updated : 增加 from_logits 参数的设置，因为 albert 的输出是 logits
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=self.model_output_logits)
        elif self.loss_name.lower() == 'categorical_crossentropy':
            # categorical_crossentropy(类别交叉熵，用于多分类，要求label为 onehot 编码，类实现形式为 CategoricalCrossentropy)
            # 多分类其实也可以使用BCE
            loss = tf.keras.losses.CategoricalCrossentropy()
        elif self.loss_name.lower() == 'sparse_categorical_crossentropy':
            # sparse_categorical_crossentropy(稀疏类别交叉熵，用于多分类，要求label为序号编码形式，类实现形式为 SparseCategoricalCrossentropy)
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
        logger.info(f"使用 loss ={loss}")
        return loss

    @staticmethod
    def _get_metrics(metric_names: List[Text]) -> List[Metric]:
        """
        在tensorflow2.x 中我们进行模型编译的时候，会看到其中有一个参数是metrics
        model.compile(metrics = )
        - 直接使用字符串
        - 使用tf.keras.metrics下的类创建的实例化对象或者函数
        """
        from utils.tensorflow.metric import get_keras_metric

        metrics = []
        for metric_name in metric_names:
            metric = get_keras_metric(metric_name)
            metrics.append(metric)
        return metrics

    def _get_callbacks(self):
        from utils.tensorflow.callback import create_common_callbacks
        callbacks = create_common_callbacks(epochs=self.train_epochs, logger=logger,
                                            tensorboard_log_dir=self.tensorboard_log_dir,
                                            output_dir=self.output_model_dir,
                                            savedmodel_name=self.savedmodel_name,
                                            **self.model_config
                                            )
        return callbacks

    def evaluate(self):
        pass

    def _get_tf_model_dir_to_epoch_ls(self, from_ckpt=False, from_savedmodel=True) -> List[Tuple[Text, int]]:
        tf_model_dir_to_epoch_ls = get_tf_model_dir_to_epoch_ls(output_model_dir=self.output_model_dir,
                                                                from_ckpt=from_ckpt,
                                                                from_savedmodel=from_savedmodel)
        return tf_model_dir_to_epoch_ls

    def persist_model(self, model: Model, filepath: Text, to_savedmodel=True) -> Text:
        if to_savedmodel:
            savedmodel_filepath = save_model_to_savedmodel(model=model, filepath=filepath)
        return savedmodel_filepath

    def load_model(self, filepath, from_savedmodel=True, from_ckpt=False, model: Model = None):
        model = load_tf_model(file_path=filepath, from_savedmodel=from_savedmodel, from_ckpt=from_ckpt, model=model)
        return model

    def _load_latest_ckpt(self, model):
        latest_ckpt_dir, restored_epoch = get_latest_ckpt_dir_to_epoch(output_model_dir=self.output_model_dir)
        if latest_ckpt_dir is not None:
            model = self.load_model(filepath=latest_ckpt_dir, from_ckpt=True, model=model)
        return model, restored_epoch

    def _predict(self, model: Union[Model, ConcreteFunction],
                 test_dataset: KerasTextClassifierDataGenerator,
                 output_logits_key: Text = 'logits',
                 model_input_keys: List[Text] = None,
                 output_probabilities=True) -> Tuple[np.ndarray]:
        logits_ls = []
        labels_ls = []
        batch_num = 0
        for batch_data in tqdm.tqdm(test_dataset, total=test_dataset.__len__(), desc="预测测试数据"):
            batch_inputs, batch_labels = batch_data
            output = self._infer(model=model, model_inputs=batch_inputs, model_input_keys=model_input_keys)
            # transforms albert out:  TFSequenceClassifierOutput
            batch_logits = output[output_logits_key]
            logits_ls.append(batch_logits)
            labels_ls.append(batch_labels)
            # if batch_num % 100 == 0:
            #     logger.info(f"完成第 {batch_num} batch的预测")
            batch_num += 1
            if self.debug and batch_num > 1:
                break

        logits = np.concatenate(logits_ls, axis=0)
        labels = np.concatenate(labels_ls, axis=0)
        if output_probabilities:
            # 转换为 概率
            probabilities = tf.sigmoid(logits).numpy()
            return probabilities, labels
        else:
            return logits, labels

    def _infer(self, model: Union[Model, ConcreteFunction], model_inputs: List[np.ndarray],
               model_input_keys: List[Text] = None) \
            -> Dict[Text, np.ndarray]:
        if isinstance(model, tf.keras.models.Model):
            output = model(model_inputs)
        else:
            output = self._infer_with_serving_model(infer_model=model, model_input_keys=model_input_keys,
                                                    model_inputs=model_inputs)
        return output

    def _infer_with_serving_model(self, infer_model, model_inputs_dict: Dict[Text, Union[np.ndarray, tf.Tensor]] = None,
                                  model_input_keys: List[Text] = None,
                                  model_inputs: List[Union[np.ndarray, tf.Tensor]] = None):
        if model_inputs_dict is None and model_input_keys is not None and model_inputs is not None:
            model_inputs_dict = {key: tf.convert_to_tensor(value) for key, value in
                                 zip(model_input_keys, model_inputs)}
        output = infer_model(**model_inputs_dict)
        return output

    def _logger_model_summary(self, model, logger):
        """
        @time:  2021/5/25 14:17
        @author:wangfc
        @version:
        @description: 将 model summary 输出的 logger 日志中

        @params:
        @return:
        """
        stringlist = ["Model summary"]
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        logger.info(short_model_summary)
