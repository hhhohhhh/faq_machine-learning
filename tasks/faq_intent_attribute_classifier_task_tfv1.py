#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/11 15:02 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/11 15:02   wangfc      1.0         None
"""
import math
import os
from typing import Text, List, Callable, Dict

# tf_version ==1.13.1
import tensorflow as tf
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu
from utils.constants import TRAIN_DATATYPE, EVAL_DATATYPE
from models.tf_v1.albert_classifier_utils import model_fn_builder, get_serving_input_fn
from data_process.data_feature import file_based_convert_examples_to_features, file_based_input_fn_builder
from albert.classifier_utils import InputExample, input_fn_builder

from tasks.rasa_task import RasaTask
from utils.io import read_json_file


class FaqIntentAttributeClassifierTaskTfV1(RasaTask):
    """
    继承 原来的 RasaTask 中的数据处理
    train： 不使用 rasa的 train 方式
    evaluate:
    """

    def __init__(self,
                 do_intent_classifier=False,
                 do_intent_corresponding_attribute_classifier=True,
                 do_intent_attribute_hierarchical_classifier=False,
                 intent_attribute_classification_subdir="haitong_intent_attribute_classification_data",
                 intent_attribute_config_subdir='intent_config',
                 intent_to_index_mapping_filename="label2id.json",
                 train_intent_to_attributes_filename="train_intent_to_attributes.json",
                 exporter_compare_key='accuracy',
                 *args, **kwargs):
        super(FaqIntentAttributeClassifierTaskTfV1, self).__init__(*args, **kwargs)
        # 训练三种不同的模式：
        # 单独训练意图模型
        self.do_intent_classifier = do_intent_classifier
        # 单独训练各个意图对应的属性模型
        self.do_intent_corresponding_attribute_classifier = do_intent_corresponding_attribute_classifier
        # 综合训练意图和属性模型
        self.do_intent_attribute_hierarchical_classifier = do_intent_attribute_hierarchical_classifier
        # 属性和意图的数据
        self.intent_attribute_classification_subdir = intent_attribute_classification_subdir
        self.intent_attribute_config_subdir = intent_attribute_config_subdir
        self.intent_to_index_mapping_filename = intent_to_index_mapping_filename
        self.train_intent_to_attributes_filename = train_intent_to_attributes_filename
        self.intent_attribute_config_dir = os.path.join(self.corpus, self.dataset,
                                                        self.intent_attribute_classification_subdir,
                                                        self.intent_attribute_config_subdir)
        self.intent_to_index_mapping_filepath = os.path.join(self.intent_attribute_config_dir,
                                                             self.intent_to_index_mapping_filename)
        self.train_intent_to_attributes_filepath = os.path.join(self.intent_attribute_config_dir,
                                                                self.train_intent_to_attributes_filename)

        self.exporter_compare_key = self._get_param_or_from_model_config(param="exporter_compare_key",
                                                                         value=exporter_compare_key)

        # 分离 intent + attribute 格式的数据
        # self._prepare_intent_attribute_data()

        # self._prepare_model()

    def _prepare_intent_attribute_data(self, if_get_train_data=True, if_get_test_data=False, if_split_data=False):
        """
        继承父类的 _prepare_data() 方法，分离 intent + attribute 格式的数据
        """
        # 加载需要训练的 意图
        self.intent_to_index_mapping = read_json_file(json_path=self.intent_to_index_mapping_filepath)

        # 加载需要训练的属性
        self.train_intent_to_attributes_mapping = read_json_file(self.train_intent_to_attributes_filepath)

        # 使用智霖的生成数据
        self.data = self._load_haitong_product_data_from_zhilin()

        self.tokenizer = self._create_tokenizer()

        # 增加 标注的数据
        # if if_get_train_data:
        #     # 读取标准问和扩展问
        #     self.standard_to_extend_data_reader = self._get_standard_to_extend_data_reader()
        #     # 获取 standard_to_extend_data 对象
        #     standard_to_extend_data = self.standard_to_extend_data_reader.standard_to_extend_data

        # 创建 data_processor
        # 转换为 examples

        # return standard_to_extend_data

    def _load_haitong_product_data_from_zhilin(self) -> Dict[Text, Dict[Text, InputExample]]:
        from run_script_tf_v1.intent_classify import SentencePairClassificationProcessor
        intent_data_dir = os.path.join(self.data_dir, "processed_data")
        attribute_data_dir = os.path.join(self.data_dir, "attribute_data")
        data_processor = SentencePairClassificationProcessor()

        # 智霖的意图数据中的 label 保存为 数字了，需要转换过来
        index_to_intent_mapping = {str(v): k for k, v in self.intent_to_index_mapping.items()}

        intent_train_examples = data_processor.get_train_examples(data_dir=intent_data_dir,
                                                                  index_to_label_mapping=index_to_intent_mapping)
        intent_dev_examples = data_processor.get_dev_examples(data_dir=intent_data_dir,
                                                              index_to_label_mapping=index_to_intent_mapping)

        intent_to_attribute_examples = {"intent": {TRAIN_DATATYPE: intent_train_examples,
                                                   EVAL_DATATYPE: intent_dev_examples}}

        for intent in self.train_intent_to_attributes_mapping.keys():
            attribute_train_examples = data_processor.get_train_examples(data_dir=attribute_data_dir, intent=intent)
            attribute_dev_examples = data_processor.get_dev_examples(data_dir=attribute_data_dir, intent=intent)
            intent_to_attribute_examples.update({intent: {TRAIN_DATATYPE: attribute_train_examples,
                                                          EVAL_DATATYPE: attribute_dev_examples}})
        return intent_to_attribute_examples

    def _prepare_model(self):
        if self.do_intent_classifier:
            raise NotImplementedError
        elif self.do_intent_corresponding_attribute_classifier:
            self._prepare_intent_corresponding_attribute_classifier_model()
        elif self.do_intent_attribute_hierarchical_classifier:
            raise NotImplementedError

    def _prepare_intent_corresponding_attribute_classifier_model(self):
        for intent, attributes in self.train_intent_to_attributes_mapping:
            self._prepare_text_classifier_model(labels=attributes)

    def _prepare_text_classifier_model(self, labels, tf_version='1.3.1'):
        """
        创建 tf_version='1.3.1' 的 模型
        """
        pass

    def train(self, do_intent_classifier=True,
              do_intent_corresponding_attribute_classifier=True,
              do_intent_attribute_hierarchical_classifier=False):

        if do_intent_classifier:
            self._train_intent_classifier()
        if do_intent_corresponding_attribute_classifier:
            self._train_intent_corresponding_attribute_classifier()
        if do_intent_attribute_hierarchical_classifier:
            self._train_intent_atttribute_classifier()

    def _train_intent_classifier(self):
        # 获取 任务名称
        task_name = f"intent_classifier"
        model_dir = self._get_model_dir()
        intent_list = list(self.intent_to_index_mapping.keys())
        self._train_text_classifier(task_name=task_name,
                                    label_list=intent_list,
                                    tokenizer=self.tokenizer,
                                    max_seq_length=self.max_seq_length,
                                    train_batch_size=self.train_batch_size,
                                    eval_batch_size=self.dev_batch_size,
                                    model_dir=model_dir,
                                    albert_config=self.albert_config,
                                    pretrained_model_init_checkpoint_file=self.pretrained_model_init_checkpoint_file,

                                    )

    def _train_intent_corresponding_attribute_classifier(self):
        """
        针对每个需要训练的意图，需要对应的属性分类模型
        """
        for intent, attribute_list in self.train_intent_to_attributes_mapping.items():
            # 获取 任务名称
            task_name = f"{intent}_attributes_classifier"

            model_dir = self._get_model_dir(intent)
            # 获取属性的列表
            self._train_text_classifier(task_name=task_name,
                                        label_list=attribute_list,
                                        tokenizer=self.tokenizer,
                                        max_seq_length=self.max_seq_length,
                                        train_batch_size=self.train_batch_size,
                                        eval_batch_size=self.dev_batch_size,
                                        model_dir=model_dir,
                                        albert_config=self.albert_config,
                                        pretrained_model_init_checkpoint_file=self.pretrained_model_init_checkpoint_file,
                                        )

    def _train_text_classifier(self,
                               task_name,
                               label_list: List[Text],
                               tokenizer,
                               max_seq_length,
                               train_batch_size,
                               eval_batch_size,
                               model_dir,
                               albert_config,
                               pretrained_model_init_checkpoint_file,
                               intent: Text = None,
                               data_type='train',
                               data_output_file=None,
                               ):

        # 获取意图对应属性分类的训练数据 : raw_data -> example -> feature -> input_fn
        train_steps, train_input_fn = self._get_classifier_input_fn(data_type=TRAIN_DATATYPE,
                                                                    intent=intent,
                                                                    task_name=task_name,
                                                                    label_list=label_list,
                                                                    tokenizer=tokenizer,
                                                                    max_seq_length=max_seq_length,
                                                                    batch_size=train_batch_size,
                                                                    is_training=True,
                                                                    output_file=data_output_file)

        eval_steps, eval_input_fn = self._get_classifier_input_fn(data_type=EVAL_DATATYPE,
                                                                  intent=intent,
                                                                  task_name=task_name,
                                                                  label_list=label_list,
                                                                  tokenizer=tokenizer,
                                                                  max_seq_length=max_seq_length,
                                                                  batch_size=eval_batch_size,
                                                                  is_training=False,
                                                                  output_file=data_output_file
                                                                  )

        warmup_steps = train_steps

        # 构建模型
        model_fn = self._create_model_fn(
            task_name=task_name,
            label_list=label_list,
            albert_config=albert_config,
            init_checkpoint=pretrained_model_init_checkpoint_file,
            optimizer_name=self.optimizer_name,
            learning_rate=self.learning_rate,
            train_steps=train_steps,
            warmup_steps=warmup_steps,
        )

        # 构建 estimator
        estimator = self._create_estimator(model_dir=model_dir, model_fn=model_fn)

        # 构建 early_stopping
        early_stopping_epoch = 1
        early_stopping_hook = self._create_early_stopping_hook(estimator=estimator,
                                                               train_steps=train_steps,
                                                               early_stopping_epoch=early_stopping_epoch)

        # train_spec and  eval_spec
        # 构建 train_spec
        # https://stackoverflow.com/questions/52641737/tensorflow-1-10-custom-estimator-early-stopping-with-train-and-evaluate/52642619#52642619
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn,
            max_steps=train_steps,
            hooks=[early_stopping_hook]
        )

        # 构建 exporter
        serving_input_fn = get_serving_input_fn(max_seq_length=self.max_seq_length,
                                                output_size=label_list.__len__())
        exporter = self._create_exporter(estimator=estimator,
                                         serving_input_fn=serving_input_fn,
                                         exporter_compare_key=self.exporter_compare_key)

        # 构建 eval_spec
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            # the number of steps of each evaluation,
            # set steps to FLAGS.save_checkpoints_steps so we can evaluate each checkpoint saved while training.
            steps=eval_steps,  # FLAGS.save_checkpoints_steps,
            exporters=exporter,
            start_delay_secs=60,
            throttle_secs=0,  # the interval in seconds between each evaluation
            # Name of the evaluation if user needs to run multiple evaluations on different data sets.
            # Metrics for different evaluations are saved in separate folders, and appear separately in tensorboard.
            # 当 name=None, eval 数据会默认保存在 output_dir/eval
            # 当 name='validation', eval 数据会默认保存在 eval_validation String.
            # 而 early_stopping 的对于 eval_meitrics 会从该文件夹下自动读取。
            # 但是如果上面 model_fn 中 通过 eval_hooks 中自定义了 tf.summary.hook() 当中定义了 summary 的不同路径，可能
            # 造成 early_stopping 读取错误
            # 因此 必须确保 tf.summary.hook() 路径 == name 定义路径 保持一致
            # name='validation',
        )

        tf.logging.info(f"***** 正在运行 {intent} 属性 train_and_evaluate *****")
        tf.estimator.train_and_evaluate(
            estimator,
            train_spec,
            eval_spec
        )

    def _get_classifier_input_fn(self, data_type,
                                 task_name, label_list,
                                 tokenizer,
                                 max_seq_length=32,
                                 batch_size=32,
                                 is_training=True, drop_remainder=False,
                                 output_file=None,
                                 use_tpu=False,
                                 intent=None,
                                 ):
        """
        获取意图对应属性分类的训练数据 : raw_data -> example -> feature -> input_fn
        """
        # 获取 属性的原始数据 + 将原始数据 转换为 example
        examples = self._get_classifier_examples(data_type, intent)

        # 转换 examples_to_features
        label_to_id_mapping = {label: index for index, label in enumerate(label_list)}
        features = file_based_convert_examples_to_features(examples=examples,
                                                           label_to_id_mapping=label_to_id_mapping,
                                                           max_seq_length=max_seq_length,
                                                           tokenizer=tokenizer, output_file=output_file)
        # 建立 input_fn
        input_fn = self._create_input_fn(task_name=task_name, label_list=label_list,
                                         output_file=output_file, features=features,
                                         max_seq_length=max_seq_length,
                                         batch_size=batch_size, is_training=is_training, drop_remainder=drop_remainder,
                                         use_tpu=use_tpu
                                         )
        if drop_remainder:
            steps = math.floor(features.__len__() / batch_size)
        else:
            steps = math.ceil(features.__len__() / batch_size)
        return steps, input_fn

    def _get_classifier_examples(self, data_type, intent=None) -> List[InputExample]:
        if intent:
            examples = self.data.get(intent).get(data_type)
        else:
            examples = self.data.get("intent").get(data_type)
        return examples

    def _create_input_fn(self, task_name, label_list, max_seq_length, batch_size,
                         output_file=None, features=None,
                         is_training=True, drop_remainder=True, use_tpu=False):
        if output_file:
            # 读取 tf_record 格式的 features 转换为  input_fn
            input_fn = file_based_input_fn_builder(
                is_training=is_training,
                task_name=task_name,
                multilabel_length=len(label_list),
                input_file=output_file,
                seq_length=max_seq_length,
                bsz=batch_size,
                drop_remainder=drop_remainder,
                use_tpu=use_tpu,
            )
        elif features:
            input_fn = input_fn_builder(features=features,
                                        seq_length=max_seq_length,
                                        is_training=is_training,
                                        drop_remainder=drop_remainder)
        return input_fn

    def _create_model_fn(self, task_name, label_list,
                         albert_config, init_checkpoint,
                         train_steps, warmup_steps,
                         optimizer_name, learning_rate,
                         use_tpu=False, albert_hub_module_handle=None):
        """
        init_checkpoint : Initial checkpoint (usually from a pre-trained BERT model).
        albert_hub_module_handle : If set, the ALBERT hub module to use.
        """

        model_fn = model_fn_builder(
            albert_config=albert_config,
            num_labels=len(label_list),
            init_checkpoint=init_checkpoint,
            learning_rate=learning_rate,
            num_train_steps=train_steps,
            num_warmup_steps=warmup_steps,
            use_tpu=use_tpu,
            use_one_hot_embeddings=use_tpu,
            task_name=task_name,
            hub_module=albert_hub_module_handle,
            optimizer=optimizer_name)
        return model_fn

    def _get_model_dir(self, intent: Text = None) -> Text:
        if intent:
            model_dir = os.path.join(self.output_model_dir, intent)
        else:
            model_dir = os.path.join(self.output_model_dir, "intent")
        return model_dir

    def _create_estimator(self, model_dir, model_fn,
                          save_checkpoints_steps=1000, keep_checkpoint_max=10,
                          use_tpu=False, tpu_name=None, tpu_zone=None, gcp_project=None,
                          tpu_cluster_resolver=None, master=None, do_train=False,
                          iterations_per_loop=1000, num_tpu_cores=8, ):
        """
        save_checkpoints_steps: How often to save the model checkpoint.
        keep_checkpoint_max: How many checkpoints to keep.

        use_tpu: Whether to use TPU or GPU/CPU.
        tpu_name: The Cloud TPU to use for training. This should be either the name used when creating the Cloud TPU,
                or a grpc://ip.address.of.tpu:8470 url.
        tpu_zone: [Optional] GCE zone where the Cloud TPU is located in. If not specified, we will attempt to
                    automatically detect the GCE project from metadata."
        gcp_project: [Optional] Project name for the Cloud TPU-enabled project. If not specified, we will attempt to
                automatically detect the GCE project from metadata
        master: [Optional] TensorFlow master URL.

        iterations_per_loop : How many steps to make in each estimator call.
        num_tpu_cores: Only used if `use_tpu` is True. Total number of TPU cores to use.
        """
        if use_tpu and tpu_name:
            tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
                tpu_name, zone=tpu_zone, project=gcp_project)

        is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
        if do_train:
            iterations_per_loop = int(min(iterations_per_loop, save_checkpoints_steps))
        else:
            iterations_per_loop = iterations_per_loop

        run_config = contrib_tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=master,
            model_dir=model_dir,
            save_checkpoints_steps=int(save_checkpoints_steps),
            keep_checkpoint_max=keep_checkpoint_max,
            tpu_config=contrib_tpu.TPUConfig(
                iterations_per_loop=iterations_per_loop,
                num_shards=num_tpu_cores,
                per_host_input_for_training=is_per_host))

        # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
        estimator = contrib_tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.dev_batch_size,
            predict_batch_size=self.test_batch_size,
            export_to_tpu=False)  # http://yaqs/4707241341091840

        return estimator

    def _create_early_stopping_hook(self, estimator, early_stopping_steps=None, early_stopping_epoch=1,
                                    train_steps: int = None, metric_name='loss',
                                    min_steps=100
                                    ):
        if early_stopping_steps is None:
            early_stopping_steps = int(train_steps * early_stopping_epoch)  # FLAGS.num_train_epochs

        # tf.logging.info('early_stopping_steps={},eval_dir = estimator.eval_dir()'.format(early_stopping_steps,
        #                                                                                  estimator.eval_dir()))
        early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
            estimator,
            metric_name=metric_name,  # 获取 eval_results 中 metrics
            max_steps_without_decrease=early_stopping_steps,
            # eval_dir = estimator.eval_dir(),
            min_steps=min_steps,
        )
        return early_stopping_hook

    def _create_exporter(self, estimator, serving_input_fn: Callable, exports_to_keep=10,
                         exporter_compare_key="f1-score"):

        # exporters : Iterable of Exporters, or a single one, or None. exporters will be invoked after each evaluation.
        # export 输出的时候比较的评估标准：loss, accuracy, precision, recall ,f1-score
        from utils.tensorflow_v1.metrics import get_exporter_compare_fn
        exporter = tf.estimator.BestExporter(
            name="best_exporter",
            serving_input_receiver_fn=serving_input_fn,
            exports_to_keep=exports_to_keep,
            compare_fn=get_exporter_compare_fn(compare_key=exporter_compare_key)
        )
        # best_eval_result cannot be empty or no loss is found in it.
        estimator._export_to_tpu = False
        return exporter

    def _train_intent_attribute_hierarchical_classifier(self):
        pass

    def evaluate(self,
                 url="http://10.20.33.3:8015/hsnlp/faq_intent_attribute_classifier/",
                 test_result_file_suffix='result',
                 ):
        from apps.intent_attribute_classifier_apps.intent_attribute_classifier import IntentAttributeClassifierTfv1
        # 加载预测数据
        from evaluation.intent_classifier_evaluate import IntentAttributeEvaluator

        self._prepare_data(if_get_train_data=False, if_get_test_data=True)

        # self._prepare_intent_attribute_data()

        # 初始化模型
        intent_attribute_classifier = IntentAttributeClassifierTfv1()

        intent_attribute_evaluator = IntentAttributeEvaluator(url=url,
                                                              raw_test_data_filename=self.test_data_filenames[0],
                                                              test_data_dir=self.data_dir,
                                                              test_data_filename=self.output_test_data_filename,
                                                              intent_attribute_classifier=intent_attribute_classifier,
                                                              support_intent_list=intent_attribute_classifier.intent_function.label2id.keys(),
                                                              test_result_file_suffix=test_result_file_suffix)
        intent_attribute_evaluator.evaluate(threshold=0.9)
