#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file:
@version:
@desc:
@time: 2021/5/19 21:50

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/19 21:50   wangfc      1.0         None
"""
import os
import re
from typing import Text, Union, Tuple, Dict, Counter, Optional
from pathlib import Path
import tensorflow as tf



if tf.__version__ >="2.0":
    from tensorflow.keras.models import Model
    from data_process.data_generator import KerasTextClassifierDataGenerator
    from models import HFAlbertClassifierModel, AlbertClassifierModel
else:
    Model =None
    KerasTextClassifierDataGenerator = None
    HFAlbertClassifierModel, AlbertClassifierModel=None,None


import pandas as pd
import numpy as np

from albert.tokenization import FullTokenizer
from data_process.dataset.intent_classifier_dataset import IntentClassifierProcessor
from data_process.dataset.text_classifier_dataset import TextClassifierProcessor


from models.model_contants import TRANSFORMERS_ALBERT_INPUT_KEYS
from tasks.builder import TASKS
from data_process.data_processor import  SUPPORT_TYPE_INDEX, SUPPORT_TYPE_COUNT
from evaluation.event_evaluate import EventEvaluation
from models.builder import MODELS
from models.train_utils import ClassWeight
from tasks.tf_task import BaseTfTask
from utils.constants import TRAIN_DATATYPE, EVAL_DATATYPE, TEST_DATATYPE


import logging
logger = logging.getLogger(__name__)


class ClassifierTask(BaseTfTask):
    def __init__(self, total_support_types_dict: Dict[Text, Dict[Text, int]] = None,
                 total_class_num: int = None,
                 multilabel_classifier: bool = None,
                 *args, **kwargs):
        super(ClassifierTask, self).__init__(*args, **kwargs)
        # support_types_dict : 区分为 total_support_types_dict  vs model_support_types_dict
        self.total_support_types_dict = total_support_types_dict
        self.total_class_num = total_class_num
        # 后面需要获取的属性
        self.model_support_types_dict: Dict[Text, Dict[Text, int]]

        self.multilabel_classifier = self._get_param_or_from_model_config(param="multilabel_classifier",
                                                                          value=multilabel_classifier)



    def _get_label_count(self):
        """
        support_types_dict: 包含 label 对应的 index 和 count 信息
        """
        return {v[SUPPORT_TYPE_INDEX]: v[SUPPORT_TYPE_COUNT]
                for key, v in self.model_support_types_dict.items()}

    def _get_class_weights(self):
        """
        根据 class_weight_strategy 获取 class_weights
        """
        label_counter = self._get_label_count()
        class_weight = ClassWeight(class_weight_strategy=self.class_weight_strategy,
                                   label_counter=label_counter,
                                   )
        class_weights = class_weight.get_class_weights()
        return class_weights


class TextClassifierTask(ClassifierTask):
    def __init__(self,
                 tokenizer_type=None,
                 label_column='sub_intent',
                 stopwords_path=None,
                 max_seq_length=None,
                 *args, **kwargs
                 ):
        super(TextClassifierTask, self).__init__(*args, **kwargs)
        self.tokenizer_type = tokenizer_type
        self.label_column = label_column
        self.stopwords_path = stopwords_path
        self.max_seq_length = self._get_param_or_from_model_config(param="max_seq_length", value=max_seq_length)

        # 后面需要设置的属性
        self.data_processor : TextClassifierProcessor
        if self.model_config:
            self.pretrained_model_dir = self.model_config['pretrained_model_dir']
            self.vocab_filename = self.model_config['vocab_filename']
            self.do_lower_case = self.model_config['do_lower_case']
            self.vocab_filepath = os.path.join(self.pretrained_model_dir, self.vocab_filename)
            self.pretrained_model_config_filename = self.model_config['model_config_filename']
            self.pretrained_model_config_filepath = os.path.join(self.pretrained_model_dir,self.pretrained_model_config_filename)

            if tf.__version__<"2.0":
                # 获取 albert_config 配置
                self.albert_config = self._get_albert_config(albert_config_file=self.pretrained_model_config_filepath,
                                                             max_seq_length=self.max_seq_length)
                # 获取 预训练模型的ckpt
                # bert_model.ckpt*，这是预训练好的模型的checkpoint，我们的Fine-Tuning模型的初始值就是来自于这些文件，然后根据不同的任务进行Fine-Tuning。
                # BERR_BASE_MODEL + '_model.ckpt'
                self.pretrained_model_init_checkpoint_filename = self.model_config['ckpt_name']
                self.pretrained_model_init_checkpoint_file = os.path.join(self.pretrained_model_dir,
                                                                          self.pretrained_model_init_checkpoint_filename)
            # for key,value in self.model_config.items():
            #     self.__setattr__(key,value)

            # output/model_dir 下建立不同的
            if self.model_config.get('restore_epoch'):
                self.restore_savedmodel_dir = os.path.join(self.output_model_dir,
                                                           f"{self.savedmodel_name}_epoch_{self.restore_epoch}")


    def _get_albert_config(self, albert_config_file, max_seq_length,
                           albert_hub_module_handle=None):
        from albert.modeling import AlbertConfig
        if not albert_config_file and not albert_hub_module_handle:
            raise ValueError("At least one of `--albert_config_file` and "
                             "`--albert_hub_module_handle` must be set")

        if albert_config_file:
            albert_config = AlbertConfig.from_json_file(albert_config_file)
            if max_seq_length > albert_config.max_position_embeddings:
                raise ValueError(
                    "Cannot use sequence length %d because the ALBERT model "
                    "was only trained up to sequence length %d" %
                    (max_seq_length, albert_config.max_position_embeddings))
        else:
            albert_config = None  # Get the config from TF-Hub.
        return albert_config


    def _prepare_data(self):
        # 使用 data_processor 来加载数据，进行数据预处理
        self.data_processor = self._create_data_processor()
        self.train_dataset, self.eval_dataset, self.test_dataset = self._build_model_data()
        self.class_num = self.data_processor.class_num

    def _build_model_data(self) -> Tuple[KerasTextClassifierDataGenerator]:
        """
        建立输入到模型中的数据
        """
        # if not self.debug:
        if self.mode == 'train':
            train_dataset = self.data_processor.prepare_model_data(data_type=TRAIN_DATATYPE)
            eval_dataset = self.data_processor.prepare_model_data(data_type=EVAL_DATATYPE)
            test_dataset = self.data_processor.prepare_model_data(data_type=TEST_DATATYPE)
        else:
            train_dataset = None
            eval_dataset = None
            test_dataset = self.data_processor.prepare_model_data(data_type=TEST_DATATYPE)
        return train_dataset, eval_dataset, test_dataset

    def _create_data_processor(self) -> TextClassifierProcessor:
        if self.task == 'intent_classification':
            data_processor = IntentClassifierProcessor(corpus=self.corpus, dataset=self.dataset_name,
                                                       label_column=self.label_column,
                                                       output_data_subdir=self.data_subdir,
                                                       output_dir=self.output_dir,
                                                       stopwords_path=self.stopwords_path,
                                                       max_seq_length=self.max_seq_length,
                                                       vocab_file=self.vocab_file,
                                                       batch_strategy=self.batch_strategy,
                                                       train_batch_size=self.train_batch_size,
                                                       test_batch_size=self.test_batch_size)
        elif self.task == 'clue_classification':
            from data_process.dataset.clue_dataset import CLUEClassifierProcessor
            data_processor = CLUEClassifierProcessor(corpus=self.corpus, dataset=self.dataset_name,
                                                     label_column=self.label_column,
                                                     output_data_subdir=self.data_subdir,
                                                     output_dir=self.output_dir,
                                                     stopwords_path=self.stopwords_path,
                                                     max_seq_length=self.max_seq_length,
                                                     vocab_file=self.vocab_file,
                                                     train_batch_size=self.train_batch_size,
                                                     test_batch_size=self.test_batch_size
                                                     )
        return data_processor

    def _create_tokenizer(self, tokenizer_type='bert'):
        raise NotImplementedError

    def _prepare_model(self):

        model = self._build_model()
        model, restored_epoch = self._load_latest_ckpt(model)

        self.model = model
        self.restored_epoch = restored_epoch
        self.loss = self._build_loss()

        train_step_per_epoch = self.train_dataset.__len__()
        self.train_steps = train_step_per_epoch * self.train_epochs
        self.train_step_per_epoch = train_step_per_epoch
        self.optimizer = self._build_optimizer(train_steps=self.train_steps,
                                               train_step_per_epoch=train_step_per_epoch, **self.model_config)
        self.metrics = self._get_metrics(metric_names=self.metric_names)
        # model.compile has nothing to do with graph mode, it defines the loss, optimizer, and metrics,
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metric_names,
                           run_eagerly=self.run_eagerly)

        # 获取 class_weights
        self.class_weights = self._get_class_weights()
        # self.class_weights = self._get_class_weights(class_weight_strategy=self.class_weight_strategy,
        #                                              label_counter=self.data_processor.train_data_analysis.label_counter)

    def _build_model(self, if_initialize_parameters=True) -> HFAlbertClassifierModel:
        # 获取 model 类
        model_class = self._get_model_class(model_name=self.model_name)
        # 实例化： 初始化 model 参数, 使用 model_support_types_dict 动态的 output_size
        model = model_class(output_size=self.data_processor.model_support_types_dict.__len__(),
                            **self.model_config)
        # 构建自定义的模型： build layer + init_checkpoint
        model.build_model(if_initialize_parameters=if_initialize_parameters)

        if isinstance(model, AlbertClassifierModel):
            # 提取 keras API 格式的 model
            model = model.model
        elif isinstance(model, Model):
            # 直接自定义的 model
            model = model

        logger.info(
            f"创建模型model_name = {model.__class__},model.count_params={model.count_params()}")
        # model.summary(print_fn=logger.info)
        return model

    def _get_model_class(self, model_name: Text) -> Union[HFAlbertClassifierModel, AlbertClassifierModel]:
        """
        model_type 映射到 model
        """
        model_class = MODELS.get(model_name)
        return model_class

    def train(self):
        """
        模型的训练主要有内置fit方法、内置tran_on_batch方法、自定义训练循环。
        1. fit 方法
        model.fit(trainX, trainY, batch_size=32, epochs=50)
        1) The call to .fit is making two primary assumptions here:
        2) Our entire training set can fit into RAM
        There is no data augmentation going on (i.e., there is no need for Keras generators)

        steps_per_epoch ?
        Keep in mind that a Keras data generator is meant to loop infinitely — it should never return or exit.


        2. 内置train_on_batch方法
        该内置方法相比较fit方法更加灵活，可以不通过回调函数而直接在批次层次上更加精细地控制训练的过程
        You’ll typically use the .train_on_batch function when you have very explicit reasons for wanting to maintain your own training data iterator,
        such as the data iteration process being extremely complex and requiring custom code.


        3. 自定义训练循环
        自定义训练循环无需编译模型，直接利用优化器根据损失函数反向传播迭代参数，拥有最高的灵活性。
        """

        """
        .fit()

        x： 输入数据，可以是以下形式 
            Numpy数组，或者Numpy数组列表
            eager tensors，或者张量列表
            tf.data.Datasets 数据集，返回(inputs,targets)或者(inputs,targets,sample_weights)
            python generator或者 keras.utils.Sequence, 返回(inputs,targets)或者(inputs,targets,sample_weights)
            Pandas dataframes

            In general, we recommend that you use:
            NumPy input data if your data is small and fits in memory
            Dataset objects if you have large datasets and you need to do distributed training
            Sequence objects if you have large datasets and you need to do a lot of custom Python-side processing that cannot be done in TensorFlow (e.g. if you rely on external libraries for data loading or preprocessing).


        steps_per_epoch: 如果需要具体设定每个 epoch 计算的步数。
            If you do this, the dataset is not reset at the end of each epoch, instead we just keep drawing the next batches. The dataset will eventually run out of data (unless it is an infinitely-looping dataset).
        max_queue_size： int类型，最大队列大小，仅用于输入为generator 或者 keras.utils.Sequence时。
        validation_freq： 仅validation_data不为空时有效，取值可以是整数或者集合。取值为整数时，表示运行多少次epoch后，执行一次验证，取值为集合时，表示在哪些epoch运行后，执行一次验证
        workers：int类型，工作线程数量，仅用于输入为generator 或者 keras.utils.Sequence时。
        use_multiprocessing：Boolean类型，是否使用多线程，仅用于输入为generator 或者 keras.utils.Sequence时。


        """
        # history object holds a record of the loss values and metric values during training:
        # 使用不同的 validation_data
        if self.validation_datatype == EVAL_DATATYPE:
            validation_data = self.eval_dataset
        elif self.validation_datatype == TEST_DATATYPE:
            validation_data = self.test_dataset

        history = self.model.fit(
            self.train_dataset,  # .forfit(datatype=TRAIN_DATATYPE)
            epochs=self.train_epochs,
            initial_epoch=self.restored_epoch + 1,
            steps_per_epoch=self.train_step_per_epoch,
            # 使用 eval_dataset 作为 dev 数据集，random=False 不做重排，而不是 test_dataset
            validation_data=validation_data,  # .forfit(datatype=EVAL_DATATYPE)
            validation_steps=validation_data.__len__(),
            callbacks=self._get_callbacks(),
            class_weight=self.class_weights,
            use_multiprocessing=self.use_multiprocessing,
            workers=self.workers,

        )
        return history

        # self.train_model(epochs=self.train_epochs,
        #     ds_train=self.train_dataset.forfit(),ds_valid=self.test_dataset.forfit(),
        #                  )

        # 训练之后进行 predict
        # self.predict()

    def evaluate(self, from_ckpt=True, eval_last_k_model=-4):
        test_dataset = self.test_dataset
        tf_model_dir_to_epoch_ls = self._get_tf_model_dir_to_epoch_ls(from_ckpt=from_ckpt)
        model = self._build_model(if_initialize_parameters=False)

        # 获取测试的原始数据
        test_examples = test_dataset.examples
        raw_test_df = pd.DataFrame([example.to_dict(with_model_input=False) for example in test_examples])

        for tf_model_dir, epoch in tf_model_dir_to_epoch_ls[-1:eval_last_k_model:-1]:
            model = self.load_model(filepath=tf_model_dir, from_ckpt=from_ckpt, model=model)
            # model.summary()
            predict_probabilities, labels = self._predict(model=model, test_dataset=test_dataset,
                                                          output_probabilities=True)

            output_predict_dir = os.path.join(self.output_dir, 'test_result', self.model_dirname,
                                              Path(tf_model_dir).stem)
            # f'p_threshold_{self.p_threshold}')
            event_evaluation = EventEvaluation(predict_probabilities_matrix=predict_probabilities,
                                               labels=labels,
                                               raw_test_df=raw_test_df,
                                               output_predict_dir=output_predict_dir,
                                               support_types_dict=self.data_processor.support_types_dict,
                                               predict_top_k=self.predict_top_k,
                                               p_threshold=self.p_threshold,
                                               event_label_column=self.label_column)
            event_evaluation.evaluate()


# def get_class_weights(self,class_weight_strategy=None,label_counter=None):
#     """
#     根据 class_weight_strategy 获取 class_weights
#     """
#     class_weight = ClassWeight(class_weight_strategy=class_weight_strategy, label_counter=label_counter,
#                                )
#     class_weights = class_weight.get_class_weights()
#     return class_weights
#
# def train(self):
#     raise NotImplementedError


@TASKS.register_module()
class TransformerClassifierTask(TextClassifierTask):
    """
    自定义的 使用 albert 的分类任务
    """

    def __init__(self, model_input_keys=TRANSFORMERS_ALBERT_INPUT_KEYS,
                 tokenizer_type=None, model_name=None ,#'HFAlbertClassifierModel',
                 *args, **kwargs):
        super(TransformerClassifierTask, self).__init__(model_input_keys=model_input_keys,
                                                        tokenizer_type=tokenizer_type,
                                                        model_name=model_name,
                                                        *args,
                                                        **kwargs)

    def _create_tokenizer(self, tokenizer_type='bert') -> "FullTokenizer":
        if tokenizer_type == 'bert':
            spm_model_file = self.model_config.get("spm_model_file")
            tokenizer = FullTokenizer.from_scratch(vocab_file=self.vocab_filepath,
                                                   do_lower_case=self.do_lower_case,
                                                   spm_model_file=spm_model_file)

        return tokenizer

    # def train_model(self,ds_train, ds_valid, epochs):
    #     """
    #     @time:  2021/6/29 22:32
    #     @author:wangfc
    #     @version:
    #     @description:
    #     自定义训练过程： 重写 model.train_on_batch() 函数
    #
    #     @params:
    #     @return:
    #     """
    #     for epoch in tf.range(epochs):
    #         self.model.reset_metrics()
    #         # 在后期降低学习率
    #         if epoch == 5:
    #             self.model.optimizer.lr.assign(self.model.optimizer.lr / 2.0)
    #             tf.print("Lowering optimizer Learning Rate...\n\n")
    #         for x, y in ds_train:
    #             train_result = self.model.train_on_batch(x, y)
    #         for x, y in ds_valid:
    #             valid_result = self.model.test_on_batch(x, y, reset_metrics=False)
    #         if epoch % 1 == 0:
    #             # printbar()
    #             tf.print("epoch = ", epoch)
    #             print("train:", dict(zip(self.model.metrics_names, train_result)))
    #             print("valid:", dict(zip(self.model.metrics_names, valid_result)))
    #             print("")
    #
    #
    #
    # def train_and_eval_model(self,model, ds_train, ds_valid, epochs):
    #     # optimizer = optimizers.Nadam()
    #     # loss_func = losses.SparseCategoricalCrossentropy()
    #     train_loss = keras.metrics.Mean(name='train_loss')
    #     train_metric = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    #     valid_loss = keras.metrics.Mean(name='valid_loss')
    #     valid_metric = keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
    #
    #     @tf.function
    #     def train_step(model, loss_func, features, labels):
    #         with tf.GradientTape() as tape:
    #             # forward pass
    #             logits = model(features, training=True)
    #             # Loss value for this batch.
    #             loss_value = loss_func(labels, logits)
    #         # Get gradients of loss wrt the weights.
    #         gradients = tape.gradient(loss_value, model.trainable_variables)
    #         # Update the weights of the model.
    #         # An Operation that applies the specified gradients. The iterations will be automatically increased by 1.
    #         self.optimizer.apply_gradients(grads_and_vars= zip(gradients, model.trainable_variables))
    #
    #         train_loss.update_state(loss_value)
    #         train_metric.update_state(labels, logits)
    #
    #     @tf.function
    #     def valid_step(model,loss_func, features, labels):
    #         predictions = model(features)
    #         batch_loss = loss_func(labels, predictions)
    #         valid_loss.update_state(batch_loss)
    #         valid_metric.update_state(labels, predictions)
    #
    #
    #     for epoch in tf.range(1, epochs + 1):
    #         for features, labels in ds_train:
    #             self.train_step(model, features, labels)
    #         for features, labels in ds_valid:
    #             self.valid_step(model, features, labels)
    #         logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'
    #         if epoch % 1 == 0:
    #             # printbar()
    #             tf.print(tf.strings.format(logs,
    #                                        (epoch, train_loss.result(), train_metric.result(), valid_loss.result(),
    #                                         valid_metric.result())))
    #             tf.print("")
    #         train_loss.reset_states()
    #         valid_loss.reset_states()
    #         train_metric.reset_states()
    #         valid_metric.reset_states()
    #


# if __name__ == '__main__':
#     from conf.config_parser import *
#
#     RUN_EAGERLY = True
#     DEBUG = True
#     classifier_task = TransformerClassifierTask(task=TASK_NAME, corpus=CORPUS, dataset_name=DATASET_NAME,
#                                                 data_subdir=DATA_SUBDIR, data_dir=DATA_DIR, output_dir=OUTPUT_DIR,
#                                                 max_seq_length=MAX_SEQ_LENGTH, model_type=MODEL_NAME,
#                                                 pretrained_model_dir=PRETRAINED_MODEL_DIR,
#                                                 vocab_filename=VOCAB_FILENAME,
#                                                 learning_rate=LEARNING_RATE,
#                                                 debug=DEBUG, run_eagerly=RUN_EAGERLY
#                                                 )
