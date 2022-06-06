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
# tf_version >=2.3.0

# from rasa.shared.nlu.training_data.training_data import TrainingData
# from rasa.nlu.tokenizers.hf_transfromers_zh_tokizer import HFTransformersTokenizer
# from rasa.nlu.featurizers.dense_featurizer.lm_featurizer import LanguageModelFeaturizer
# from rasa.shared.utils.cli import print_info
# from rasa.nlu.train import create_persistor
# from models.model import Interpreter, Trainer

import os
from typing import List, Any

from apps.intent_attribute_classifier_apps.intent_attribute_classifier_app import IntentAttributeClassifierSanicApp
from models.featurizers.dense_featurizer.lm_featurizer import LanguageModelFeaturizer
from tokenizations.hf_transformers_zh_tokenizer import HFTransformersTokenizer
from models.classifiers.diet_classifier import DIETClassifier, HierarchicalDIETClassifier
from cli.cli_utils import print_success, print_info

from utils.constants import QUESTION
from utils.io import json_to_string, dataframe_to_file

from tasks.rasa_task import RasaTask
import logging

logger = logging.getLogger(__name__)


class FaqIntentAttributeClassifierTaskTfV2(RasaTask):
    """
    继承 原来的 RasaTask 中的数据处理
    train： 不使用 rasa 的 train 方式
    run:
    evaluate:

    """

    def __init__(self,
                 do_intent_classifier=True,
                 do_attribute_classifier=True,
                 do_entity_extractor=True,
                 *args, **kwargs):
        super(FaqIntentAttributeClassifierTaskTfV2, self).__init__(*args, **kwargs)
        # 与智霖提供的tfv1 版本不同，我们希望  在同一个模型中同时训练 意图+属性+实体
        # 训练三种不同的模式：
        # 训练意图模型

        self.do_intent_classifier = do_intent_classifier
        # 训练属性模型
        self.do_attribute_classifier = do_attribute_classifier
        # 训练实体识别模型
        self.do_entity_extractor = do_entity_extractor

        tensorboard_log_directory = os.path.join(self.output_dir, 'tf_log', )
        self.config = {
            "language": 'zh',
            "pipeline":
                [{"name": "HFTransformersTokenizer"},
                 # "featurizer":
                 # {"name":"CountVectorsFeaturizer",
                 #  "min_df": 3,
                 #  "min_ngram": 1,
                 #  "max_ngram": 3,
                 #  "max_features": 100
                 #  },

                 {"name": "LanguageModelFeaturizer",
                  "language": "zh",
                  "model_name": "albert",
                  "model_weights": "/home/wangfc/faq/pretrained_model/albert-chinese-tiny",
                  "from_pt": True},
                 # "classifier":
                 {"name": "HierarchicalDIETClassifier",  # "DIETClassifier",
                  "epochs": 50,
                  "sub_intent_classification": True,
                  "entity_recognition": True,
                  "constrain_similarities": True,
                  "model_confidence": "softmax",
                  "tensorboard_log_directory": tensorboard_log_directory,
                  "run_eagerly": self.run_eagerly},
                 {"name": "IntentAttributeMapper",
                  "intent_attribute_mapping_dir": "/home/wangfc/faq/corpus/hsnlp_kbqa_faq_data",
                  "intent_attribute_mapping_filename": self.intent_attribute_mapping_filename
                  }
                 ]
        }

    def train(self, do_intent_classifier=True,
              do_intent_corresponding_attribute_classifier=True,
              do_intent_attribute_hierarchical_classifier=False,
              **kwargs: Any):
        """
        from rasa.nlu.train import train
        from rasa.nlu.model import Trainer
        0. 准备训练数据 self._prepare_data()
        1. 加载训练数据获取 training_data
        2. 加载训练配置参数，构建 trainer 对象
        3. 调用 self.train() 进行训练,返回 interpreter 对象
        4. 调用 self.persist() 进行保存
        """
        from models.nlu.nlu_config import NLUModelConfig
        from models.model import Trainer

        # 0. 准备训练数据 self._prepare_data()
        self._prepare_data(if_transform_to_rasa_data=True)

        # 1. 从 yaml 文件中读取数据
        self.training_data = self._load_rasa_yml_format_data()

        # 2. 加载训练配置参数
        nlu_model_config = NLUModelConfig(configuration_values=self.config)

        # 3. 构建 trainer 对象
        trainer = Trainer(cfg=nlu_model_config)

        # context = kwargs
        # # data gets modified internally during the training - hence the copy
        # working_data: TrainingData = copy.deepcopy(self.training_data)
        #
        # self.pipeline = self._create_model(config=self.config)
        # for i, component in enumerate(self.pipeline):
        #     logger.info(f"Starting to train component {component.name}")
        #     component.prepare_partial_processing(self.pipeline[:i], context)
        #     component.train(working_data, self.config, **context)
        #     logger.info("Finished training component.")

        # 4. 调用 self.train() 进行训练
        interpreter = trainer.train(data=self.training_data)

        # from rasa.nlu.train import create_persistor
        # persistor = create_persistor(persistor=None)

        # 5. 保存模型
        persisted_path = trainer.persist(path=self.output_model_dir,
                                         persistor=None,
                                         fixed_model_name=None,
                                         persist_nlu_training_data=False)
        return trainer, interpreter, persisted_path



    def _prepare_data(self, if_get_train_data=True, if_postprocess_data=True,
                      if_get_chat_data=True, if_add_new_ivr_data=True,if_add_new_data=True,
                      if_enhance_data=True,
                      if_get_test_data=False, if_split_data=False,
                      check_labeled_data=True,if_drop_duplicates=True,
                      if_transform_to_rasa_data=False):
        """
        增加读取 闲聊数据
        """
        # 获取意图属性实体的知识定义体系
        self.knowledge_definition_data = self._get_knowledge_definition_data()

        # 读取实体的近义词和正则表达式等信息
        self.entity_data = self._get_entity_data()

        # 获取 标准问的定义信息：一级知识点，二级知识点，意图，first_intent（大意图）等
        # self.standard_question_info = self._get_standard_question_info()
        self.standard_question_knowledge_data = self._get_standard_question_knowledge_data()
        # 意图和属性的映射数据
        self.intent_attribute_mapping, self.intent_to_attribute_dict, self.output_intents, self.output_attributes = \
            self._load_intent_attribute_mapping()

        # self.label_to_sub_label_mapping = {key:list(value.keys()) for key,value in self.intent_attribute_mapping.items()}

        train_data, test_data = None, None
        train_data_path = os.path.join(self.output_dir,'train_data.json')
        if if_get_train_data and os.path.exists(train_data_path):
            # 加载训练数据
            train_data = dataframe_to_file(path=train_data_path,mode='r')
        elif if_get_train_data and not os.path.exists(train_data_path):
            # 1. 读取标注的标准问和扩展问数据
            self.standard_to_extend_data_reader = self._get_standard_to_extend_data_reader()

            # 2. 获取 standard_to_extend_data 对象
            standard_to_extend_data = self.standard_to_extend_data_reader.standard_to_extend_data

            # 转换为 含有 intent 的数据
            faq_intent_dataset = self._transform_to_intent_data(standard_to_extend_data)
            # faq_intent_data = faq_intent_dataset.data.copy()

            if if_get_chat_data:
                # 读取闲聊数据
                chat_intent_dataset = self._get_new_data(data_type='chat', output_filename="hsnlp_chat_data.json")
                # 合并 闲聊数据
                faq_intent_dataset.merge(chat_intent_dataset)

            # 读取新增海通ivr数据
            if if_add_new_ivr_data:
                new_faq_intent_data = self._get_new_data(data_type='new_haitong_ivr',
                                                         output_filename="new_haitong_ivr_data.json")
                faq_intent_dataset.merge(new_faq_intent_data)

            # 读取新增的数据
            if if_add_new_data:
                new_intent_data = self._get_new_data(data_type='new_data',output_filename="new_intent_data.json")
                faq_intent_dataset.merge(new_intent_data)


            if if_postprocess_data:
                # 是否进行后处理
                data = faq_intent_dataset.postprocess_data(faq_intent_dataset.data, if_mapping_attribute=True,
                                                           intent_attribute_mapping=self.intent_attribute_mapping)
                faq_intent_dataset.data = data

            if if_enhance_data:
                # 是否进行数据增强
                faq_intent_dataset.enhance_data()

            if if_drop_duplicates:
                # 是否去除重复数据
                faq_intent_dataset.drop_duplicates()
            faq_intent_dataset.save_intent_attribute_info(filename=self.train_data_intent_attribute_info_path)
            train_data = faq_intent_dataset.data
            dataframe_to_file(data=train_data,path=train_data_path,mode='w')

        # # postprocess
        # if train_data is not None and if_postprocess_data:
        #     train_data = self._postprocess_data(train_data)

        if if_get_test_data:
            # 获取测试数据
            test_data = self._get_test_data()
            # 分割数据集: 可能在去除测试数据后，训练数据中的某个类型数据偏少的情况
            if if_split_data:
                data = train_data
                test_questions = test_data.loc[:, QUESTION]
                is_test_data = data.loc[:, QUESTION].isin(test_questions)
                train_data = data.loc[~is_test_data].copy()
                test_data = data.loc[is_test_data].copy()
                logger.info(f"分割数据为训练集 {train_data.shape}和测试集 {test_data.shape}")

        if if_transform_to_rasa_data:
            # 是否转换为rasa的数据格式
            rasa_train_data_dir, rasa_test_data_dir, intent_to_attribute_mapping = self._transform_to_rasa_data(
                train_data=train_data,
                test_data=test_data,
                entity_data=self.entity_data,
                standard_question_knowledge_data=self.standard_question_knowledge_data,
                output_dir=self.output_dir,
                rasa_train_data_subdirname=self.rasa_train_data_subdirname,
                rasa_test_data_subdirname=self.rasa_test_data_subdirname,
                rasa_config_filename=self.rasa_config_filename)
            self.rasa_train_data_dir, self.rasa_test_data_dir = rasa_train_data_dir, rasa_test_data_dir
            self.train_intent_to_attribute_mapping = intent_to_attribute_mapping

    def _create_model(self, config) -> List["Component"]:
        """
        参考 rasa\nlu\model\Trainer._build_pipeline()
        借鉴 rasa的训练模式 将 model 转换为多个 component 的组合的 pipeline：

        tokenizer + featurizer + classifier:
        tokenizer : HFTransformersTokenizer
        featurizer : CountVectorsFeaturizer + LanguageModelFeaturizer
        classifier: DIETClassifier
        """
        # from rasa import __main__
        # from rasa.api import train
        # from rasa.nlu.train import train
        # from rasa.nlu.model import Trainer
        # from rasa.nlu.registry import create_component_by_config

        tokenizer = HFTransformersTokenizer()

        component_config = config.get("featurizer", {})
        featurizer = LanguageModelFeaturizer(component_config)

        component_config = config.get("classifier", {})
        classifier_name = component_config.get("name")
        if classifier_name == "DIETClassifier":
            classifier = DIETClassifier(component_config)
        elif classifier_name == 'HierarchicalDIETClassifier':
            classifier = HierarchicalDIETClassifier(component_config)
        pipeline = [tokenizer, featurizer, classifier]
        return pipeline

    def _load_model(self, config) -> List["Component"]:
        """
        参考 rasa\nlu\model\Trainer._build_pipeline()
        借鉴 rasa的训练模式 将 model 转换为多个 component 的组合的 pipeline：

        tokenizer + featurizer + classifier:
        tokenizer : HFTransformersTokenizer
        featurizer : CountVectorsFeaturizer + LanguageModelFeaturizer
        classifier: DIETClassifier
        """
        # from rasa import __main__
        # from rasa.api import run
        # from rasa.nlu.run import run_cmdline
        # from rasa.nlu.model import Interpreter
        # from rasa.nlu.registry import load_component_by_meta

        tokenizer = HFTransformersTokenizer()

        component_config = config.get("featurizer", {})
        featurizer = LanguageModelFeaturizer(component_config)

        component_config = config.get("classifier", {})
        classifier_name = component_config.get("name")
        if classifier_name == "DIETClassifier":
            classifier = DIETClassifier(component_config)
        elif classifier_name == 'HierarchicalDIETClassifier':
            classifier = HierarchicalDIETClassifier(component_config)
        pipeline = [tokenizer, featurizer, classifier]
        return pipeline

    def run(self, run_nlu_app=False, run_shell_nlu=False,
            port=8015, num_processes=1, gpu_memory_config="0:1024",
            with_preprocess_component=True,
            use_intent_attribute_regex_classifier=True):
        if run_nlu_app:
            # 运行 nlu 为 app
            self.run_nlu_app(
                port=port, num_processes=num_processes, gpu_memory_config=gpu_memory_config,
                with_preprocess_component=with_preprocess_component,
                use_intent_attribute_regex_classifier=use_intent_attribute_regex_classifier
            )
        elif run_shell_nlu:
            self.run_shell_nlu(with_preprocess_component, use_intent_attribute_regex_classifier)

    def run_nlu_app(self,
                    port=8015, num_processes=1, gpu_memory_config="0:1024",
                    with_preprocess_component=True,
                    use_intent_attribute_regex_classifier=True):
        app = self._build_app(
            port=port, num_processes=num_processes, gpu_memory_config=gpu_memory_config,
            with_preprocess_component=with_preprocess_component,
            use_intent_attribute_regex_classifier=use_intent_attribute_regex_classifier)
        app.run()

    def _build_app(self, port=8015, num_processes=1, gpu_memory_config="0:1024",
                   with_preprocess_component=True,
                   hsnlp_word_segment_url="http://10.20.33.3:8017/hsnlp/faq/sentenceProcess",
                   use_intent_attribute_regex_classifier=True) -> IntentAttributeClassifierSanicApp:

        pattern_data_dir = os.path.join('corpus', 'hsnlp_kbqa_faq_data', "regex_data")
        app = IntentAttributeClassifierSanicApp(gpu_memory_per_worker=gpu_memory_config,
                                                num_of_workers=num_processes,
                                                port=port,
                                                with_preprocess_component=with_preprocess_component,
                                                hsnlp_word_segment_url=hsnlp_word_segment_url,
                                                from_tfv1=False,
                                                model_path=self.output_model_dir,
                                                use_intent_attribute_regex_classifier=use_intent_attribute_regex_classifier,
                                                pattern_data_dir=pattern_data_dir,
                                                )
        return app

    def run_shell_nlu(self, use_compressed_model: bool = False):
        """
        借鉴 run_cmdline 生成 Interpreter
        from rasa.cli.shell import shell_nlu
        from rasa.nlu.run import run_cmdline
        model_path = None
        interpreter = Interpreter.load(model_path)
        #
        model_metadata

        手动加载模型 pipeline -> interpreter
        1. 构建 pipeline
        2. 构建 interpreter
        """
        # from rasa.cli.shell import shell_nlu
        # from rasa.cli.utils import get_validated_path
        # from rasa.model import get_model
        # from rasa.model import get_model_subdirectories

        from models.model import load_interpreter_from_model
        interpreter = load_interpreter_from_model(model_dir=self.output_model_dir, use_compressed_model=False)

        while True:
            print_success("请输入问句 message:")
            try:
                message = input().strip()
            except (EOFError, KeyboardInterrupt):
                print_info("Wrapping up command line chat...")
                break
            result = interpreter.parse(message)
            print(json_to_string(result))

    def evaluate(self,
                 url="http://10.20.33.3:8015/hsnlp/faq_intent_attribute_classifier",
                 test_result_file_suffix='result',
                 ):
        # 加载预测数据
        from evaluation.intent_classifier_evaluate import IntentAttributeEvaluator
        from data_process.dataset.intent_classifier_dataset import RasaIntentDataset

        self._prepare_data(if_get_train_data=False, if_get_test_data=True)

        train_intent_count, train_attribute_count, train_intent_to_attribute_mapping = \
            RasaIntentDataset.load_intent_attribute_info(filename=self.train_data_intent_attribute_info_path)

        intent_attribute_evaluator = IntentAttributeEvaluator(url=url,
                                                              raw_test_data_filename=self.test_data_filename,
                                                              test_data_dir=self.test_data_dir,
                                                              test_data_filename=self.output_test_data_filename,
                                                              train_intent_to_attribute_mapping=train_intent_to_attribute_mapping,
                                                              intent_to_attribute_dict=self.intent_to_attribute_dict,
                                                              # intent_attribute_classifier=intent_attribute_classifier,
                                                              # support_intent_list=intent_attribute_classifier.intent_function.label2id.keys(),
                                                              test_result_file_suffix=test_result_file_suffix)
        intent_attribute_evaluator.evaluate(threshold=0.9)
