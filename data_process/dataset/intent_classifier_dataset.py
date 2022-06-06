#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/8 9:25 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/8 9:25   wangfc      1.0         None
"""
import os
from copy import deepcopy
from pathlib import Path
from typing import Text, Set, List, Dict, Union
import tqdm
import numpy as np
from collections import Counter
import pandas as pd
import re

from data_process.dataset.hsnlp_faq_dataset import StandardToExtendQuestionDataReader, Dataset, FAQSentenceExample
from data_process.dataset.hsnlp_faq_knowledge_dataset import ATTRIBUTE, ENTITY, ENTITY_VALUES, \
    ATTRIBUTE_ZH  # FIRST_KNOWLEDGE, SECOND_KNOWLEDGE,FIRST_INTENT, \
from hsnlp_faq_utils.preprocess.word_segement import HSNLPWordSegmentApi
from rasa.shared.nlu.constants import INTENT

from data_process.data_analysis import TextClassifierDataAnalysis
from data_process.data_generator import KerasTextClassifierDataGenerator
from data_process.data_processing import percent_in_vocab, is_security_code
from data_process.dataset.finacial_knowledge_base import FinancialTerms
from data_process.dataset.regx_dataset import RegxDataSet
from data_process.dataset.text_classifier_dataset import TextClassifierProcessor

from tokenizations import tokenization
from tokenizations.tokenization import preprocess_text, FullTokenizer
from data_process.data_processor import RawDataProcessor
from hsnlp_faq_similarity.evaluation_lzl import get

from utils.constants import ID, QUESTION, QUESTION_LENGTH, INTENT, SUB_INTENT, \
    SOURCE, RESPONSE, INTENT_CATEGORY, FUNCTION_NAME, FAQ, CHAT, \
    STANDARD_QUESTION, EXTEND_QUESTION, SCENARIO, TRAIN_DATATYPE, EVAL_DATATYPE, TEST_DATATYPE, RANDOM_BATCH_STRATEGY, \
    SEQUENCE_BATCH_STRATEGY, BALANCED_BATCH_STRATEGY, INTENT_TO_ATTRIBUTE_MAPPING

from utils.io import read_json_file, dump_obj_as_json_to_file, write_to_yaml, dataframe_to_file, \
    get_matched_filenames, get_file_stem

import logging

logger = logging.getLogger(__name__)


class RawIntentDasaset(Dataset):
    # 将 dataset 属性作为 类属性
    def __init__(self):
        super(RawIntentDasaset, self).__init__()
        # self.id_column = ID
        self.question_column = QUESTION
        self.question_length_column = QUESTION_LENGTH
        self.intent_column = INTENT
        self.sub_intent_column = SUB_INTENT
        self.response_column = RESPONSE
        self.source_column = SOURCE

    @property
    def required_columns(self):
        return [self.question_column, self.question_length_column, self.intent_column,
                self.sub_intent_column, self.response_column, self.source_column]


class RasaIntentDataset(RawIntentDasaset):
    def __init__(self):
        super(RasaIntentDataset, self).__init__()

        self.attribute_key = ATTRIBUTE

        # depreciated: 因为这两个字段都是合成的
        # self.first_knowledge_key = FIRST_KNOWLEDGE
        # self.first_intent_key = FIRST_INTENT
        # self.secondary_knowledge_key = SECOND_KNOWLEDGE  # 作为 intent + entity 返回的意图

        self.scenario_column = SCENARIO
        self.intent_category_column = INTENT_CATEGORY
        self.standard_question_column = STANDARD_QUESTION
        # self.entity_column = ENTITY
        # self.entity_values_column = ENTITY_VALUES

    @property
    def required_columns(self):
        # expand_columns = [self.scenario_column,self.intent_category_column,
        #                   self.standard_question_column,
        #                   self.first_knowledge_key,self.attribute_key]
        #                   # self.first_intent_key, self.secondary_knowledge_key,
        #                   # self.entity_column,self.entity_values_column]
        # output_columns = RawIntentDasaset().required_columns
        # output_columns.extend(expand_columns)

        return [self.standard_question_column, self.question_column, self.question_length_column,
                self.intent_column, self.attribute_key, self.sub_intent_column, self.response_column
                # self.first_knowledge_key,
                ]
        # self.scenario_column, self.intent_category_column,self.source_column]

    def merge(self, other: RawIntentDasaset):
        if other and isinstance(other, RawIntentDasaset):
            self.data = pd.concat([self.data, other.data], ignore_index=True)

    def _get_intent_count(self) -> Dict[Text, int]:
        intent_count = self.data.loc[:, INTENT].value_counts().to_dict()
        return intent_count

    def _get_attribute_count(self) -> Dict[Text, int]:
        attribute_count = self.data.loc[:, ATTRIBUTE].value_counts().to_dict()
        return attribute_count

    def _get_intent_to_attribute_count(self) -> pd.DataFrame:
        grouped = self.data.loc[:, [INTENT, ATTRIBUTE]].groupby([INTENT, ATTRIBUTE])
        intent_to_attribute_count = grouped.size().reset_index(name='counts')
        return intent_to_attribute_count

    def save_intent_attribute_info(self, filename):
        intent_count = self._get_intent_count()
        attribute_count = self._get_attribute_count()
        intent_to_attribute_count = self._get_intent_to_attribute_count()
        intent_to_attribute_mapping = intent_to_attribute_count.loc[:, [INTENT, ATTRIBUTE]].groupby(INTENT) \
                                          .agg({ATTRIBUTE: lambda x: sorted(x)}).loc[:, ATTRIBUTE].to_dict()
        intent_attribute_info = {INTENT: intent_count, ATTRIBUTE: attribute_count,
                                 INTENT_TO_ATTRIBUTE_MAPPING: intent_to_attribute_mapping}
        dump_obj_as_json_to_file(filename=filename, obj=intent_attribute_info)

    @classmethod
    def load_intent_attribute_info(cls, filename) -> [Dict[Text, int], Dict[Text, int]]:
        intent_attribute_info = read_json_file(filename)
        intent_count = intent_attribute_info.get(INTENT)
        attribute_count = intent_attribute_info.get(ATTRIBUTE)
        intent_to_attribute_mapping = intent_attribute_info.get(INTENT_TO_ATTRIBUTE_MAPPING)
        return intent_count, attribute_count,intent_to_attribute_mapping

    @classmethod
    def postprocess_data(self,data, if_mapping_attribute=True,intent_attribute_mapping=None):
        """
        对于 海通 ivr 数据：
        1. 查询权限 和 开户操作 暂时不加区分，开户操作属性包含 : 开户操作/开户条件/开户材料/查询开户渠道/开户所需时间/开户费用
        2. 忘记帐号和密码 和 忘记密码

        3. 转人工 对应的属性，默认为 转接人工
        4. 其他 对应的属性，默认为其他
        5. 最后输出结果的时候，对于识别为非支持的属性，我们默认为其他
        """
        # 过滤默写意图的数据

        # TODO : 查询权限 和 开户操作 暂时不加区分
        # logger.warning(f"暂时不加区分属性： 查询权限 和 开户操作 ")

        # TODO :  忘记帐号和密码 和 忘记密码
        # logger.warning(f"暂时不加区分属性： 忘记帐号和密码 和 忘记密码 ")
        # # 我们暂时把 标注为查询权限的数据 转换为 开户操作 or 我们修改规则使得其可以适配 同一个标准问下，对应多个属性，因此正则表达式多个条件
        # is_forget_account_and_password = data.loc[:, ATTRIBUTE] == "忘记帐号和密码"
        # # data.loc[is_forget_account_and_password].shape
        # data.loc[is_forget_account_and_password, ATTRIBUTE] = '忘记密码'

        # intent = "手机软件操作"  # "账户密码" #"银证转账" # '开户'
        # is_intent = data.loc[:, INTENT] == intent
        # data.loc[is_intent].loc[:, ATTRIBUTE].value_counts()

        if if_mapping_attribute:
            # 是否对 属性的标签进行映射
            # train_attributes = set()
            output_attribute_mapping = intent_attribute_mapping["attribute"]
            for intent, attribute_mapping in output_attribute_mapping.items():
                for target_attribute, source_attributes in attribute_mapping.items():
                    # train_attributes.add(mapping_attribute)
                    if source_attributes:
                        # 选取 attribute 存在对应关系的  attributes 的数据
                        is_attribute = data.loc[:, ATTRIBUTE].isin(source_attributes)
                        # data.loc[is_attribute].shape
                        data.loc[is_attribute, ATTRIBUTE] = target_attribute
                        # train_attributes.update(set(attributes))


        # 转人工 对应的属性，默认为 转接人工
        is_intent_transfer_to_human = data.loc[:, INTENT] == '转人工'
        # data.loc[is_intent_transfer_to_human].loc[:,ATTRIBUTE].value_counts()
        data.loc[is_intent_transfer_to_human, ATTRIBUTE] = '转接人工'

        # 其他 对应的属性，默认为其他
        is_intent_other = data.loc[:, INTENT] == '其他'
        # data.loc[is_intent_other].loc[:,ATTRIBUTE].value_counts()
        data.loc[is_intent_other, ATTRIBUTE] = '其他'

        # 对于非 训练 attributes,设置为 None，后面模型中不加训练
        # is_ivr_train_attributes = data.loc[:, ATTRIBUTE].isin(train_attributes)
        # data.loc[~is_ivr_train_attributes].loc[:, ATTRIBUTE].value_counts()
        # data.loc[~is_ivr_train_attributes, ATTRIBUTE] = None

        # 确定每个 意图和属性都有值
        is_intent_na = data.loc[:, INTENT].isna()
        is_attribute_na = data.loc[:, ATTRIBUTE].isna()
        assert is_intent_na.sum() == 0
        assert is_attribute_na.sum() == 0
        return data


    def enhance_data(self):
        """
        数据增强的一种方式： 使用预处理的方法进行处理后，新增数据
        """
        hsnlp_word_segment_url="http://10.20.33.3:8017/hsnlp/faq/sentenceProcess"
        hsnlp_word_segment_api = HSNLPWordSegmentApi(url=hsnlp_word_segment_url)
        new_rows = []
        for index in tqdm.trange(self.data.__len__(),desc="预处理方法进行数据增强"):
            row = self.data.iloc[index].copy()
            question = row.loc[self.question_column]
            processed_text = hsnlp_word_segment_api.word_segment(text=question)
            if processed_text !="" and processed_text != question:
                row.loc[self.question_column] = processed_text
                new_rows.append(row)
        new_data = pd.DataFrame(new_rows)
        data = pd.concat([self.data,new_data],ignore_index=True)
        logger.info(f"预处理的方法进行数据增强后，原有数据共{self.data},新增数据共{new_data},当前数据共{data.shape}")
        self.data =data


    def drop_duplicates(self):
        original_hsape = self.data.shape
        self.data.drop_duplicates(subset=[self.question_column],inplace=True)
        logger.info(f"原有数据共{original_hsape},去除重复后共{self.data.shape}")


class RawIntentClassifierDataProcessor(RawDataProcessor):
    """
    对原始数据进行处理，分割
    """

    def __init__(self, corpus='corpus',
                 dataset_dir='intent_classification',
                 new_added_data_subdir=None,
                 output_data_subdir='intent_classifier_data_20210809',
                 id_column=ID,
                 label_column=INTENT,
                 text_column=QUESTION,
                 raw_data_format='xlsx',
                 stopwords_path=None,
                 support_labels=None, min_threshold=100,
                 sample_size_for_label=1000, random_state=111,
                 intent_classifier_data_filename='raw_intent_data.csv',
                 port=21327,
                 ):
        if new_added_data_subdir is None:
            new_added_data_subdir = output_data_subdir

        super().__init__(corpus=corpus, dataset_dir=dataset_dir,
                         random_state=random_state,
                         output_data_subdir=output_data_subdir,
                         id_column=id_column,
                         label_column=label_column,
                         text_column=text_column,
                         raw_data_format=raw_data_format,
                         stopwords_path=stopwords_path,
                         new_added_data_subdir=new_added_data_subdir,
                         new_added_data_filename=intent_classifier_data_filename,
                         support_labels=support_labels,
                         min_threshold=min_threshold,
                         sample_size_for_label=sample_size_for_label)

        self.PORT = port

    def read_raw_data(self, path: Path) -> pd.DataFrame:
        # self.faq_log_data = self.filter_chat_data_from_faq_log()
        intent_classifier_raw_data = dataframe_to_file(path=path, mode='r')

        logger.info(f"读取原始的意图标注数据共{intent_classifier_raw_data.shape}，"
                    f"columns={intent_classifier_raw_data.columns.tolist()},"
                    f"\n按照意图的数据分布为：\n{intent_classifier_raw_data.intent.value_counts()}")

        return intent_classifier_raw_data

    def get_similar_sentence_with_post(self):
        """
        使用 hsnlp faq 工具匹配相似问题
        """
        filter_faq_log_data = self.faq_log_data[self.faq_log_data.CHANNEL_ID == 2]
        recall_question_ls = []
        for index in filter_faq_log_data.index:
            question = self.faq_log_data.loc[index]['TEXT']
            print(f"{'*' * 100}\nindex ={index},question = {question}")
            try:
                result = get(q=question, port=self.PORT)
                # for key,value in result.items():
                #     print(f"{key}:{value}")
                recall_question = result['recall_question']
            except Exception as e:
                recall_question = []
            recall_question_ls.append(recall_question)
        filter_faq_log_data.loc['recall_question'] = recall_question_ls
        path = './data/domain_classification/gyzq_sms_0628_with_recall_question.xlsx'
        filter_faq_log_data.to_excel(path, index=False, engine='xlsxwriter')


class IntentClassifierProcessor(RawIntentClassifierDataProcessor, TextClassifierProcessor):
    """
    该类继承 RawIntentClassifierDataProcessor,TextClassifierProcessor
    RawIntentClassifierDataProcessor 主要负责对原始语料进行读取，预处理，分割等操作
    TextClassifierProcessor： 主要负责 tf 模型训练的时候提供数据

    重写 get_train_examples() 函数
    重写  create_generator() 函数创建 data generator
    """

    def __init__(self,
                 corpus='corpus',
                 dataset_dir='intent_classification',
                 output_data_subdir='gf_intent_classifier_data_20210721',
                 output_dir='output/intent_classifier_02',
                 intent_classifier_data_filename='raw_intent_data.csv',
                 id_column='id',
                 label_column='intent',
                 text_column='question',
                 raw_data_format='csv',
                 stopwords_path=None,
                 vocab_file=None, support_labels=None, min_threshold=100,
                 sample_size_for_label=2000,
                 max_seq_length=512,
                 batch_strategy=None,
                 train_batch_size=32, test_batch_size=32,
                 if_transform_to_rasa_nlu_data=True,
                 use_retrieve_intent=True,
                 use_intent_column_as_label=True
                 ):

        # 合并数据
        # combined_raw_intent_dataset = CombinedRawIntentDataset(
        #     corpus=corpus, dataset=dataset,
        #     output_data_subdir=output_data_subdir,
        #     intent_classifier_data_filename=intent_classifier_data_filename,
        #     vocab_file=vocab_file)

        # 对原始数据进行处理，分割为 train, dev,test 三个数据集
        RawIntentClassifierDataProcessor.__init__(self,
                                                  dataset_dir=dataset_dir,
                                                  output_data_subdir=output_data_subdir,
                                                  id_column=id_column,
                                                  label_column=label_column,
                                                  text_column=text_column,
                                                  intent_classifier_data_filename=intent_classifier_data_filename,
                                                  raw_data_format=raw_data_format,
                                                  support_labels=support_labels,
                                                  min_threshold=min_threshold,
                                                  stopwords_path=stopwords_path,
                                                  sample_size_for_label=sample_size_for_label
                                                  )
        # 生成 tensorflow 训练的 data_processor
        # TextClassifierProcessor.__init__(self, corpus=corpus, dataset_dir=dataset_dir,
        #                                  output_data_subdir=output_data_subdir,
        #                                  output_dir=output_dir,
        #                                  vocab_file=vocab_file,
        #                                  max_seq_length=max_seq_length,  # 生成 feature 的时候使用
        #                                  batch_strategy=batch_strategy,
        #                                  train_batch_size=train_batch_size,
        #                                  test_batch_size=test_batch_size)
        # 生成训练中的数据
        self.get_new_version_data()
        self.data_analysis = TextClassifierDataAnalysis(
            label_column=label_column,
            text_column=text_column,
            new_version_data_dict=self.new_version_data_dict,
            output_dir=self.output_data_dir,
            stopwords=self.stopwords)

        # rasa_output_path = os.path.join(output_dir, 'data', 'nlu.yml')
        # if if_transform_to_rasa_nlu_data:
        #     if not os.path.exists(rasa_output_path):
        #         train_data = self.new_version_data_dict.get(TRAIN_DATATYPE)
        #         transform_to_rasa_nlu_data(data=train_data, path=rasa_output_path, dataset=dataset,
        #                                    use_retrieve_intent=use_retrieve_intent,
        #                                    use_intent_column_as_label=use_intent_column_as_label
        #                                    )
        #     else:
        #         train_data = read_from_yaml(path=rasa_output_path)

    def get_train_examples(self) -> List[FAQSentenceExample]:
        """See base class."""

        examples = self._create_examples(raw_data_path=self.raw_train_file,
                                         set_type=TRAIN_DATATYPE)

        return examples

    def _create_examples(self, raw_data_path, set_type):
        """Creates examples for the training and dev sets."""
        raw_examples = self.new_version_data_dict.get(set_type)
        examples = []
        for i in range(raw_examples.__len__()):
            try:
                raw_example = raw_examples.iloc[i]
                guid = '%s-%s' % (set_type, i)
                id = raw_example.get(self.id_column)
                intent = raw_example.get(self.label_column)
                sentence = raw_example.get(self.text_column)
                sentence = tokenization.convert_to_unicode(sentence)
                # 获取 对应的 label_id
                intent_id = self.support_types_dict[intent.lower()]
                examples.append(
                    FAQSentenceExample(guid=guid, id=id, sentence=sentence, intent=intent, intent_id=intent_id))
            except Exception as e:
                raise e
        return examples

    def get_dev_examples(self):
        """See base class."""
        examples = self._create_examples(raw_data_path=self.raw_eval_file,
                                         set_type=EVAL_DATATYPE)

        return examples

    def get_test_examples(self):
        """See base class."""

        examples = self._create_examples(raw_data_path=self.raw_test_file,
                                         set_type=TEST_DATATYPE)
        return examples

    def create_generator(self, data_type, data, batch_size) -> KerasTextClassifierDataGenerator:
        # examples = self.get_examples(data_type=data_type)
        # 使用 bert4keras 的方式生成 DataGenerator
        # data_generator = IntentClassifierDataGenerator(data_type=data_type,
        #                                   data=data, tokenizer=self.tokenizer,
        #                                   max_seq_length=self.max_seq_length,
        #                                   batch_size=batch_size
        #                                   )
        if data_type == TRAIN_DATATYPE and self.batch_strategy and self.batch_strategy == BALANCED_BATCH_STRATEGY:
            batch_strategy = BALANCED_BATCH_STRATEGY
        elif data_type == TRAIN_DATATYPE and self.batch_strategy is None:
            batch_strategy = RANDOM_BATCH_STRATEGY
        else:
            batch_strategy = SEQUENCE_BATCH_STRATEGY

        data_generator = KerasTextClassifierDataGenerator(data_type=data_type,
                                                          data=data, tokenizer=self.tokenizer,
                                                          max_seq_length=self.max_seq_length,
                                                          batch_size=batch_size,
                                                          batch_strategy=batch_strategy,
                                                          data_analysis=self.train_data_analysis
                                                          )
        return data_generator


class RawIntentDataReader():
    def __init__(self, corpus='corpus', dataset='ivr_data',
                 dataset_dir=None,
                 subdataset_name=None,
                 output_data_subdir=None,
                 output_dir=None,
                 use_intent_column_as_label=True,
                 intent_classifier_data_filename='raw_intent_data.csv',
                 filter_intent_filename=f'filter_intent_data.csv',
                 use_regx=False, regx_filename='intent_regx.yml', vocab_file=None,
                 ):
        self.corpus = corpus
        self.dataset = dataset
        if dataset_dir is None:
            self.dataset_dir = os.path.join(self.corpus, self.dataset)
        else:
            self.dataset_dir = dataset_dir
        self.subdataset_name = subdataset_name
        self.output_data_subdir = output_data_subdir
        self.use_intent_column_as_label = use_intent_column_as_label

        self.intent_classifier_data_filename = intent_classifier_data_filename
        self.id_column = ID
        self.question_column = QUESTION
        self.question_length_column = QUESTION_LENGTH
        self.intent_column = INTENT
        self.sub_intent_column = SUB_INTENT
        self.response_column = RESPONSE
        self.source_column = SOURCE
        # 输出的列
        self.output_columns = [self.id_column, self.question_column, self.question_length_column, self.intent_column,
                               self.sub_intent_column, self.response_column, self.source_column]

        # 正则表达式
        if use_regx:
            self.regx_filename = regx_filename
            self.regx_path = os.path.join(self.data_dir, self.regx_filename)
            self.regx_dataset = RegxDataSet(regx_path=self.regx_path)

        if vocab_file:
            self.vocab_file = vocab_file
            self.vocab = self.get_vocab()

        # 金融专业词汇
        self.financial_dataset = 'finance',
        self.financial_term_filename = 'financial_term.yml'

        if self.subdataset_name:
            # 如果存在 subdataset_name
            self.data_dir = os.path.join(self.dataset_dir, self.subdataset_name)
        else:
            self.data_dir = os.path.join(self.dataset_dir)

        if self.output_data_subdir:
            self.output_data_dir = os.path.join(self.data_dir, self.output_data_subdir)
        else:
            self.output_data_dir = output_dir

        # 过滤意图的数据路径
        self.filter_intent_filename = filter_intent_filename

        self.raw_intent_data_path = None
        self.filter_intent_data_path = None
        if self.output_data_dir:
            # 综合的 intent 数据
            self.raw_intent_data_path = os.path.join(self.output_data_dir,
                                                     self.intent_classifier_data_filename)
            self.filter_intent_data_path = os.path.join(self.output_data_dir, self.filter_intent_filename)

    def get_raw_intent_data(self) -> pd.DataFrame:
        raw_intent_data = None
        if os.path.exists(self.raw_intent_data_path):
            raw_intent_data = dataframe_to_file(path=self.raw_intent_data_path, mode='r')
            logger.info(f"读取原始数据 {raw_intent_data.shape} from {self.raw_intent_data_path}")
        elif not os.path.exists(self.raw_intent_data_path) and self.if_build_raw_intent_data:
            raw_intent_data = self._build_raw_intent_data()
            if self.output_data_dir:
                self._dump_raw_intent_data(raw_intent_data)
        return raw_intent_data

    def _dump_raw_intent_data(self, raw_intent_data):
        dataframe_to_file(path=self.raw_intent_data_path, data=raw_intent_data, mode='w')

    def _build_raw_intent_data(self):
        raise NotImplementedError

    def _read_raw_intent_data(self):
        """
        读取原始数据为 标准格式
        """
        raise NotImplementedError

    def _build_filter_intent_data(self):
        filter_intents = self._get_filter_intents()
        filter_intent_data = self.raw_intent_data.loc[
            self.raw_intent_data.loc[:, self.intent_column].isin(filter_intents)]
        logger.info(f"过滤后的filter_intents 数据共 {filter_intent_data.shape} 条")

        # 合并意图 + 增加 entity 信息
        # 1. 合并 科创板开通 和 创业板开通 到 开户查询，并且标注 科创板和创业板为的实体属性为 block
        # 2. 合并 查个股行情 和 查指数行情 到 查行情，并且标注 stock 和 zsstock
        # 3. 变换 手机软件下载 和 电脑软件下载 为检索式标签 软件下载/手机软件下载 和 软件下载/电脑软件下载，使用检索模型进行分类
        # def change_label(label):
        #     if label in ['科创板开通','创业板开通']:
        #         new_label = '开户咨询'
        #     elif label in ['查个股行情','查指数行情']:
        #         new_label = '查行情'
        #     elif label in ['手机软件下载','电脑软件下载']:
        #         new_label = f"软件下载/{label}"
        #     else:
        #         new_label = label
        #     return new_label

        # new_texts = filter_intent_data.loc[:,self.question_column].apply(lambda text: entity_labeling(text,self.entity_attribute_to_value_dict))
        # filter_intent_data.loc[:,self.question_column]= new_texts
        # new_labels = filter_intent_data.loc[:,self.sub_intent_column].apply(change_label)
        # filter_intent_data.loc[:,self.sub_intent_column] = new_labels

        dataframe_to_file(path=self.filter_intent_data_path, data=filter_intent_data)
        return filter_intents, filter_intent_data

    def get_filter_intent_data(self):
        """
        对意图数据进行过滤
        """
        if os.path.exists(self.filter_intent_data_path):
            filter_intent_data = dataframe_to_file(path=self.filter_intent_data_path, mode='r')
            filter_intents = filter_intent_data.loc[:, self.intent_column].value_counts().keys().tolist()
        else:
            filter_intents, filter_intent_data = self._build_filter_intent_data()

        return filter_intents, filter_intent_data


class CombinedRawIntentDataReader(RawIntentDataReader):
    """
    从以下数据中提取 意图识别数据：
    从 hsnlp 在原来faq数据标注意图的数据 中 获取 faq的数据
    从 广发的意图数据 中获取 advice 和 chat 数据
    从 haitong kms 数据中验证 和获取 faq，advice，chat数据

    转换后数据格式：
    id, question, intent,sub_intent, source,
    其中 sub_intent: faq/查询费用, 使用 /分割


     1. hsnlp 在原来faq数据标注意图的数据 （高峰等人标注）:
     hsnlp_faq_intent_filename="baselib_data.xlsx"

     2. 广发的意图数据
     gf_intent_classifier_data_filename='gf_label_intent_recognition.csv'

     3. haitong kms faq xiaoi 问答日志数据 和 单月的日志数据
     ht_kms_faq_intent_data_filename = "kms_intention_record.csv"
     ht_mks_intent_data_filename = "kms_intention_record.csv"

     4. 国元证券的log日志数据
     faq_log_filname="gyzq_sms_0628.csv",

    """

    def __init__(self, output_data_subdir='intent_classifier_data_20210809',
                 # corpus = 'corpus',
                 dataset='intent_classification',
                 # regx_filename='intent_regx.yml',
                 # intent_classifier_data_filename='raw_intent_data.csv',
                 vocab_file=None,
                 ):

        super(CombinedRawIntentDataReader, self).__init__(dataset=dataset, output_data_subdir=output_data_subdir)
        # self.corpus = corpus
        # self.dataset = dataset
        # self.data_dir = os.path.join(CombinedRawIntentDataset.corpus, self.dataset)
        # self.id_column = 'id'
        # self.question_column = 'question'
        # self.question_length_column = 'question_length'
        # self.intent_column = 'intent'
        # self.sub_intent_column = 'sub_intent'
        # self.response_column = 'response'
        # self.source_column = 'source'
        # self.output_columns = [self.id_column, self.question_column, self.intent_column, self.sub_intent_column,
        #                        self.response_column, self.source_column]

        # # 金融专业词汇
        # self.financial_dataset = 'finance',
        # self.financial_term_filename = 'financial_term.yml',

        # hsnlp 在原来faq数据标注意图的数据
        hsnlp_faq_intent_filename = "baselib_data.xlsx"
        # 广发的意图数据
        gf_intent_classifier_data_filename = 'gf_label_intent_recognition.csv'
        hationg_intent_dataset = "haitong_intent_classification_data"
        # haitong 证券 kms 平台导出数据（数据没有验证）
        ht_mks_faq_intent_data_filename = "kms_intention_record.csv"
        ht_mks_intent_data_filename = "20210804.csv"
        # 国元证券的log日志数据
        faq_log_filname = "gyzq_sms_0628.csv",

        # self.regx_filename = regx_filename
        # self.regx_path = os.path.join(self.corpus, hationg_intent_dataset, self.regx_filename)
        # self.regx_dataset = RegxDataSet(regx_path=self.regx_path)

        # self.vocab_file = vocab_file
        # self.vocab = self.get_vocab()

        self.hsnlp_faq_intent_data_path = os.path.join(self.data_dir, hsnlp_faq_intent_filename)

        self.gf_intent_classifier_data_path = os.path.join(self.data_dir, gf_intent_classifier_data_filename)
        # 过滤 gf 的advice数据
        self.new_advice_data_path = os.path.join(self.data_dir, 'gf_intent_classifier_data_20210809',
                                                 'advice_data.xlsx')

        self.ht_mks_faq_intent_data_path = os.path.join(CombinedRawIntentDataReader.corpus, hationg_intent_dataset,
                                                        ht_mks_faq_intent_data_filename)
        self.ht_mks_intent_data_path = os.path.join(CombinedRawIntentDataReader.corpus, hationg_intent_dataset,
                                                    ht_mks_intent_data_filename)

        self.ht_select_faq_data_dir = os.path.join(CombinedRawIntentDataReader.corpus, hationg_intent_dataset,
                                                   'haitong_preprocessed_intent_data')
        self.ht_select_faq_data_filename_pattern = 'select_faq_data_\d{3}.yml'

        # 综合的 intent 数据
        self.raw_intent_data_path = os.path.join(self.data_dir, output_data_subdir,
                                                 CombinedRawIntentDataReader.intent_classifier_data_filename)
        self.raw_intent_data = self.read_raw_intent_data()

        # self.financial_terms_path = os.path.join(self.corpus, self.financial_dataset, self.financial_term_filename)
        # self.financial_terms_set = read_financial_terms(financial_terms_path=self.financial_terms_path)

    def read_raw_intent_data(self):
        # # 读取 hsnlp 的faq 数据
        self.hsnlp_faq_intent_data = self.read_hsnlp_faq_intent_data(path=self.hsnlp_faq_intent_data_path)
        # 读取 广发的意图数据并进行筛选
        self.gf_intent_classifier_data = self.read_gf_raw_intent_data(path=self.gf_intent_classifier_data_path)
        # 读取海通过滤的数据
        self.ht_select_faq_data = self.read_ht_select_faq_data(data_dir=self.ht_select_faq_data_dir,
                                                               filename_pattern=self.ht_select_faq_data_filename_pattern)
        raw_intent_data = pd.concat(
            [self.hsnlp_faq_intent_data, self.gf_intent_classifier_data, self.ht_select_faq_data], ignore_index=True)
        # 进行排序
        raw_intent_data.sort_values(
            by=[self.source_column, self.intent_column, self.sub_intent_column, self.question_length_column,
                self.question_column],
            inplace=True)
        # self.ht_raw_intent_data = self.read_haitong_kms_data(path=self.ht_mks_intent_data_path)
        dataframe_to_file(path=self.raw_intent_data_path, data=raw_intent_data)
        return raw_intent_data

    def get_vocab(self):
        tokenizer = FullTokenizer.from_scratch(vocab_file=self.vocab_file, do_lower_case=True,
                                               spm_model_file=None)
        vocab = set([k for k, v in tokenizer.vocab.items()])
        return vocab

    def read_ht_select_faq_data(self, data_dir, filename_pattern, transform_intent_to_chat=True):
        filenames = get_matched_filenames(data_dir, filename_pattern)
        data_ls = []
        for filename in filenames:
            data_path = os.path.join(data_dir, filename)
            data = dataframe_to_file(mode='r', path=data_path)
            data_ls.append(data)
        data = pd.concat(data_ls)
        logger.info(f"读取数据共 {data.shape} from {data_dir} and {filenames}")
        # 新增 句子长度列
        question_length = data.loc[:, self.question_column].apply(len)
        data.loc[:, self.question_length_column] = question_length
        # 将 原本的 intent 转换为 chat
        if transform_intent_to_chat:
            logger.info(f"将ht_select_faq_data 的意图转换为 chat ")
            data.loc[:, self.intent_column] = 'chat'
        return data

    def read_hsnlp_faq_intent_data(self, path, sheet_name='数据',
                                   standard_question_column='标准问',
                                   category_column='类别',
                                   intent='faq',
                                   source='hsnlp'
                                   ):
        """
        读取 hsnlp_faq_intent_data,去掉营业部的数据,因为营业部地址都是地址，不具有语义的信息
        """
        hsnlp_faq_intent_data = pd.read_excel(path, sheet_name=sheet_name, dtype=str)
        logger.info(f"读取 hsnlp_faq_intent_data 共 {hsnlp_faq_intent_data.shape} 条,"
                    f"columns={hsnlp_faq_intent_data.columns.tolist()}"
                    f"\n按照 {category_column} 的数据分布为：\n{hsnlp_faq_intent_data.loc[:, category_column].value_counts()}")

        # 去掉营业部的数据
        is_address = hsnlp_faq_intent_data.loc[:, category_column] == '营业部'
        hsnlp_faq_intent_data = hsnlp_faq_intent_data.loc[~is_address].copy()

        # is_new_stock = hsnlp_faq_intent_data.loc[:,category_column] == '新股'
        # hsnlp_faq_intent_data.loc[is_new_stock].head()

        def combine_id(row, source, question_id_column='问题id', standard_id_column='标准问id'):
            question_id = row.get(question_id_column)
            standard_id = row.get(standard_id_column)
            if isinstance(question_id, str) and isinstance(standard_id, str):
                new_id = f"{source}_{str(question_id)}_{str(standard_id)}"
            else:
                new_id = f"{source}_{str(row.name)}"
            return new_id

        new_id = hsnlp_faq_intent_data.apply(lambda row: combine_id(row, source=source), axis=1)
        assert new_id.unique().__len__() == new_id.__len__()
        hsnlp_faq_intent_data.loc[:, 'new_id'] = new_id

        selected_hsnlp_faq_intent_data = hsnlp_faq_intent_data.loc[:,
                                         ['new_id', standard_question_column, category_column]].copy()
        selected_hsnlp_faq_intent_data.columns = [self.id_column, self.question_column, self.sub_intent_column]
        selected_hsnlp_faq_intent_data.loc[:, self.intent_column] = intent
        selected_hsnlp_faq_intent_data.loc[:, self.source_column] = source
        # 增加问句长度
        length_series = selected_hsnlp_faq_intent_data.loc[:, self.question_column].str.len()
        selected_hsnlp_faq_intent_data.loc[:, self.question_length_column] = length_series

        # selected_hsnlp_faq_intent_data = selected_hsnlp_faq_intent_data.loc[:, self.output_columns].copy()
        logger.info(f"最后的hsnlp faq intent 标注数据共{selected_hsnlp_faq_intent_data.shape}，"
                    f"columns={selected_hsnlp_faq_intent_data.columns.tolist()}"
                    f"\n按照意图的数据分布为：\n{selected_hsnlp_faq_intent_data.loc[:, self.intent_column].value_counts()}"
                    f"\n按照sub_intent 的数据分布为：\n{selected_hsnlp_faq_intent_data.loc[:, self.sub_intent_column].value_counts()}")
        return selected_hsnlp_faq_intent_data

    def read_gf_raw_intent_data(self, path: Path,
                                id_column='id',
                                sentence_column='sentence',
                                label_column='intent',
                                source='gf') -> pd.DataFrame:
        """
        读取 gf 的意图数据
        """
        # self.faq_log_data = self.filter_chat_data_from_faq_log()
        intent_classifier_raw_data = pd.read_csv(path, dtype=str)
        logger.info(f"读取gf原始的意图标注数据共{intent_classifier_raw_data.shape}，"
                    f"columns={intent_classifier_raw_data.columns.tolist()},"
                    f"\n按照意图的数据分布为：\n{intent_classifier_raw_data.intent.value_counts()}")

        # 去除空行
        isna_counts = intent_classifier_raw_data.isna().sum()
        gf_intent_classifier_raw_data_dropna = intent_classifier_raw_data.dropna(how='any').copy()
        # 对意图的文本进行预处理
        new_sentences = gf_intent_classifier_raw_data_dropna.sentence.apply(lambda x: preprocess_text(x))
        gf_intent_classifier_raw_data_dropna.loc[:, sentence_column] = new_sentences
        gf_intent_classifier_raw_data_dropdups = gf_intent_classifier_raw_data_dropna.drop_duplicates(
            subset=[sentence_column]).copy()
        #  将 label 转换为 小写
        labels_lower = gf_intent_classifier_raw_data_dropdups.intent.apply(lambda x: str.lower(x))
        gf_intent_classifier_raw_data_dropdups.loc[:, label_column] = labels_lower

        # 去除句子中异常字符过多的句子
        if_percent_in_vocab = gf_intent_classifier_raw_data_dropdups.sentence.apply(
            lambda sentence: percent_in_vocab(sentence=sentence, vocab=self.vocab))
        gf_intent_classifier_raw_data_dropdups = gf_intent_classifier_raw_data_dropdups.loc[if_percent_in_vocab].copy()

        logger.info(f"去除空行+去除重复数据+去除句子中异常字符过多的句子 的广发意图标注数据共{gf_intent_classifier_raw_data_dropdups.shape}，"
                    f"\n按照意图的数据分布为：\n{gf_intent_classifier_raw_data_dropdups.intent.value_counts()}")

        # 分析 product 数据
        is_product = gf_intent_classifier_raw_data_dropdups.loc[:, label_column] == 'product'
        product_length = 6
        product_data = gf_intent_classifier_raw_data_dropdups.loc[is_product].copy()
        is_short_product_name = product_data.loc[:, sentence_column].apply(lambda x: len(x) < product_length)
        product_value_counts = product_data.loc[is_short_product_name].value_counts()

        # 去除 product 的数据
        gf_intent_classifier_raw_data = gf_intent_classifier_raw_data_dropdups[~is_product].copy()
        logger.info(f"去除product的广发意图标注数据共{gf_intent_classifier_raw_data.shape}，"
                    f"\n按照意图的数据分布为：\n{gf_intent_classifier_raw_data.intent.value_counts()}")

        gf_intent_classifier_raw_data.columns = [self.id_column, self.intent_column, self.question_column]
        # new_questions = gf_intent_classifier_raw_data.loc[:,self.question_column].apply(lambda x: remove_punctuation(x) )

        is_advice_data = gf_intent_classifier_raw_data.loc[:, self.intent_column] == 'advice'
        advice_raw_data = gf_intent_classifier_raw_data.loc[is_advice_data].copy()
        if os.path.exists(self.new_advice_data_path):
            new_gf_advice_data = dataframe_to_file(path=self.new_advice_data_path, mode='r')
            # 删除手动判断为 delete的数据
            new_gf_advice_data = new_gf_advice_data.loc[new_gf_advice_data.loc[:, 'comment'] != 'delete'].copy()
            new_gf_advice_data = new_gf_advice_data.drop(columns=['comment']).copy()
            # new_gf_advice_data.columns= ['id', 'intent', 'question', 'question_length', 'source', 'comment']
        else:
            new_gf_advice_data = self.filtering_gf_advice_data(advice_raw_data=advice_raw_data)

        gf_intent_classifier_raw_data = self.transfer_gf_raw_data(gf_raw_data=new_gf_advice_data, id_column=id_column,
                                                                  source=source)

        # gf_intent_classifier_raw_data = gf_intent_classifier_raw_data.loc[:, self.output_columns].copy()
        logger.info(f"最后的广发意图标注数据共{gf_intent_classifier_raw_data.shape}，"
                    f"columns={gf_intent_classifier_raw_data.columns.tolist()}"
                    f"\n按照意图的数据分布为：\n{gf_intent_classifier_raw_data.loc[:, self.intent_column].value_counts()}")

        return gf_intent_classifier_raw_data

    # 转换为统一的格式
    def transfer_gf_raw_data(self, gf_raw_data, id_column, source):
        # 设置新的id
        new_id = gf_raw_data.loc[:, id_column].apply(lambda x: f"{source}_{x}")
        gf_raw_data.loc[:, self.id_column] = new_id

        # 设置 sub_intent
        sub_intents = gf_raw_data.loc[:, self.source_column].apply(
            lambda x: x if x is not None and x in self.regx_dataset.advice_regx_data.keys() else None)
        gf_raw_data.loc[:, self.sub_intent_column] = sub_intents

        # 设置新的 source
        sources = gf_raw_data.loc[:, self.source_column].apply(lambda x: source if x is None else f"{source}_{x}")
        gf_raw_data.loc[:, self.source_column] = sources

        return gf_raw_data

    def filtering_gf_advice_data(self, advice_raw_data, length_threshold=40, short_advice_length_threshold=5,
                                 source='gf', source_column='source', seed=1234, sample_threshold=100
                                 ):
        filter_gf_advice_data_ls = []

        # 筛选 security_code
        is_securitycode = advice_raw_data.loc[:, self.question_column].apply(
            lambda x: is_security_code(x, only_digit_pattern=True))
        security_code_data = advice_raw_data.loc[is_securitycode].copy()
        security_code_data.loc[:, source_column] = f'security_code'
        logger.info(f" only_digit_pattern 的数据共 {security_code_data.shape}")

        # 过滤 security_code 的数据
        advice_raw_data = advice_raw_data.loc[~is_securitycode].copy()

        # 统计长度分布,过滤特别长的句子
        question_length_series = advice_raw_data.loc[:, self.question_column].apply(lambda x: len(x))
        advice_raw_data.loc[:, self.question_length_column] = question_length_series

        advice_raw_data = advice_raw_data.loc[question_length_series < length_threshold].copy()
        logger.info(f"问句长度小于 {length_threshold} 的数据共 {advice_raw_data.shape}")

        # advice_raw_data = advice_raw_data.loc[:,[self.id_column,self.intent_column,self.question_column]].copy()
        is_short_question = advice_raw_data.loc[:, self.question_column].apply(
            lambda x: len(x) < short_advice_length_threshold)
        short_advice_raw_data = advice_raw_data.loc[is_short_question].copy()
        notshort_advice_raw_data = advice_raw_data.loc[~is_short_question].copy()

        short_advice_raw_data.sort_values(by=[self.question_length_column, self.question_column],
                                          ascending=[True, True],
                                          inplace=True)
        short_advice_raw_data.loc[:, source_column] = f'short_advice'
        logger.info(f"问句长度小于 {short_advice_length_threshold} 的数据共 {short_advice_raw_data.shape}"
                    f"\n{short_advice_raw_data.head(10)}")
        filter_gf_advice_data_ls.append(short_advice_raw_data)

        # 使用 正则表达式进行过滤
        # quotation_regx = '(请?问?).{2,6}(的?行情|怎么样|走势|今天下跌因何原因|怎么一直不涨|近期走势如何|该股怎么走呢可介入吗)'
        # stock_funds = '(请?问?).{2,6}的资金流?向?'
        # stock_trend =  '(请?问?).{2,6}的?市场表现'
        # news_common = '(请?问?).{2,6}的?(最新动态|利好公告)'
        # f10_common = '.{2,6}的?(盈利预测|高管|评级|所属概念|市盈率|经营范围|前十大股东)'
        # ipo_stock = '(打新股|最近可以打哪些新股)'
        # condition_list = '(市盈率小于\d|业绩预增|BOLL突破上轨|高管增持|MACD金叉|多头排列|总市值大于\d{1,10}[亿|万]|天然气相关|industy_name相关|机构关注|\d日内创历史新高)的股票'
        # diagnose = '(分析一下.{2,6}|.{2,6}(可以买吗|该不该持有|怎么操作))'
        # market_vane = '(当下行情怎么操作|大盘风向)'
        # predict_price = '.{2,6}(会不会涨|股价预测)'
        # predict_kline = '.{2,6}的?(k线走势|后市如何)'
        # stock_distribution = '.{2,6}的?筹码分布'
        # smart_invest = '(震荡行情如何选股|如何识别断线好股票)'
        # voice_transition = '\d{1,4}元(买入|卖出)\d{1,8}股'

        for advice_reason, advice_info in self.regx_dataset.advice_regx_data.items():
            regx = advice_info.get('examples')[0]
            regx = regx.replace('stock_name', '\d{2,6}')
            regx_raw_data = notshort_advice_raw_data.loc[
                advice_raw_data.loc[:, self.question_column].str.match(regx)].copy()
            if regx_raw_data.__len__() > 0:
                regx_raw_data.loc[:, source_column] = f"{advice_reason}"
                regx_raw_data.sort_values(by=[self.question_length_column, self.question_column],
                                          ascending=[True, True],
                                          inplace=True)
                logger.info(f"match 正则表达式: {regx} 的数据共\n {regx_raw_data.shape}\n{regx_raw_data.head()}")
                filter_gf_advice_data_ls.append(regx_raw_data)

        filter_gf_advice_data = pd.concat(objs=filter_gf_advice_data_ls)
        logger.info(f"共筛选 {filter_gf_advice_data.shape}条adivice 数据")

        # 按照 句子长度进行抽样
        left_advice_raw_data = advice_raw_data.loc[
            pd.Index(advice_raw_data.index).difference(filter_gf_advice_data.index)].copy()

        sample_data_ls = []
        length_counts = left_advice_raw_data.loc[:, self.question_length_column].value_counts().sort_index()
        for length in length_counts.index:
            is_length = left_advice_raw_data.loc[:, self.question_length_column] == length
            is_length_data = left_advice_raw_data.loc[is_length].copy()
            if is_length_data.shape[0] > sample_threshold:
                is_length_data = is_length_data.sample(sample_threshold, random_state=seed).copy()
            is_length_data.sort_values(by=[self.question_column], ascending=True, inplace=True)
            logger.info(f"在剩余的数据中抽样 {is_length_data.shape[0]}条数据")
            sample_data_ls.append(is_length_data)
        sample_advice_data = pd.concat(sample_data_ls)
        sample_advice_data.loc[:, source_column] = f'sampled'
        logger.info(f"在剩余的数据中共 抽样 {sample_advice_data.shape[0]}条数据")

        new_advice_data = pd.concat([filter_gf_advice_data, sample_advice_data])
        logger.info(f"在advice的数据中共  {new_advice_data.shape[0]}条数据")

        dataframe_to_file(path=self.new_advice_data_path, data=new_advice_data)

        return new_advice_data

    def read_haitong_kms_data(self, path, output_data_dir=None,
                              question_column='QUESTION',
                              out_intention="OUT_INTENTION", recog_intention="RECOG_INTENTION",
                              answer_column='REPLY_TITLE',
                              save_frequency=None,
                              source='haitong_kms'
                              ):
        """
        读取 haitong_kms_faq_data ，共两个字段
        """
        raw_data = pd.read_csv(path, dtype=str)
        raw_data.dropna(inplace=True)
        logger.info(f"读取原始数据共{raw_data.shape},columns={raw_data.columns.tolist()}")
        questions = raw_data.loc[:, question_column].apply(lambda x: preprocess_text(x))
        raw_data.loc[:, question_column] = questions
        answers = raw_data.loc[:, answer_column].apply(lambda x: preprocess_text(x))
        raw_data.loc[:, answer_column] = answers
        raw_data.drop_duplicates(ignore_index=False, inplace=True)
        logger.info(f"对原始数据的 {question_column}进行预处理并去除重复数据后，数据共 {raw_data.shape}")

        is_security_code_data = raw_data.loc[:, question_column].apply(lambda x: is_security_code(x))
        security_code_data = raw_data.loc[is_security_code_data].copy()
        logger.info(f"{question_column} 为六位的数字的数据共{security_code_data.shape}")

        none_security_code_data = raw_data.loc[~is_security_code_data].copy()

        # answer_to_questions_dict = self.convert_to_answer_to_questions_dict(none_security_code_data,
        #                                                                     output_data_dir=output_data_dir,
        #                                                                     save_frequency=save_frequency)

        # 从数据中筛选出部分 chitchat 数据
        select_faq_answer_to_intent_dict = {
            # '你好': 'greet', '在吗': "greet", '忙吗': 'greet', '周末愉快': 'greet', '节日快乐': 'greet',
            # '谢谢': 'thanks', '再见': 'goodbye',
            # '嗯': 'affirm',  '没有': 'deny',
            # '不知道': 'not_understand',
            # '没解决': 'unsolved', '还有问题': 'unsolved',
            # "解决了":'solved',
            # '我先试试': 'try_and_wait', '稍等': 'wait',
            # '对不起': 'sorry',
            # '呵呵': 'hehe', '哈哈哈哈': 'hehe',
            # '好评': 'approving',
            # '去你妈的': 'mood_unhappy', '太气人': 'mood_unhappy', '我勒个去': 'mood_unhappy',
            # '你好聪明': 'you_are_smart',
            # "加油": 'come_on',
            # "你是机器人吗": 'bot_challenge',
            # "你叫什么名字": 'what_is_your_name',
            # "怎么回事":'what_is_wrong',
            # "怎么办":'how_to_do'
            '转人工': 'transfer_to_human'
        }
        # 显示 answer 较短的数据
        answer_length = 5
        # for answer, questions_info in answer_to_questions_dict.items():
        #     if answer.__len__() < answer_length:
        #         logger.info(f"answer_length < {answer_length}, answer = {answer}")

        # 转换为
        # id,question,intent,sub_intent,response,source
        is_select_faq = none_security_code_data.loc[:, answer_column].isin(select_faq_answer_to_intent_dict)
        select_faq_data = none_security_code_data.loc[is_select_faq].loc[:,
                          [question_column, answer_column]].sort_values(by=answer_column)
        select_faq_data.columns = [self.question_column, self.response_column]
        indies = select_faq_data.index.tolist()
        new_id = [f"{source}_{index}" for index in indies]
        select_faq_data.loc[:, self.id_column] = new_id
        select_faq_data.loc[:, self.intent_column] = 'faq'
        # 根据 answer 匹配到的 sub_intent
        sub_intents = select_faq_data.loc[:, self.response_column].apply(lambda x: select_faq_answer_to_intent_dict[x])
        select_faq_data.loc[:, self.sub_intent_column] = sub_intents
        select_faq_data.loc[:, self.source_column] = source
        # 进行排序
        select_faq_data.sort_values(by=[self.intent_column, self.sub_intent_column, self.question_column], inplace=True)
        select_faq_data = select_faq_data.loc[:, self.output_columns].copy()

        select_faq_data_path = os.path.join(os.path.dirname(path), 'haitong_preprocessed_intent_data',
                                            'select_faq_data.yml')
        dataframe_to_file(path=select_faq_data_path, data=select_faq_data, format='yaml')
        logger.info(f"筛选部分 faq 数据 共 {select_faq_data.shape},"
                    f"\n按照 response 分为：\n{select_faq_data.loc[:, self.response_column].value_counts()}")
        return select_faq_data

    def convert_to_answer_to_questions_dict(self, none_security_code_data,
                                            output_data_dir=None, save_frequency=None,
                                            question_column='QUESTION',
                                            out_intention="OUT_INTENTION", recog_intention="RECOG_INTENTION",
                                            answer_column='REPLY_TITLE',
                                            ):
        # 转换为 yaml 格式的数据
        """
        answer: 共对应多少个questions
            examples:
                - question
                -
        """
        counts_groupby_question = none_security_code_data.groupby(answer_column).count().sort_values(by=question_column,
                                                                                                     ascending=False)
        counts_groupby_question.head(20)

        file_index = 1
        answer_to_questions_dict = {}
        for answer_index in tqdm.trange(counts_groupby_question.__len__()):
            answer = counts_groupby_question.index[answer_index]
            answer_to_data = none_security_code_data.loc[none_security_code_data.loc[:, answer_column] == answer].copy()
            answer_to_data.sort_values(by=question_column, inplace=True)
            # logger.info(f"answer = {answer},数据共 {answer_to_data.shape}")
            examples = answer_to_data.loc[:, question_column].tolist()
            answer_to_questions_dict.update({answer: {'statistics': f"对应{answer_to_data.__len__()}个questions",
                                                      'examples': examples}})

            if save_frequency and (answer_index != 0 and answer_index % save_frequency == 0 or
                                   answer_index == counts_groupby_question.__len__() - 1):
                yaml_path = os.path.join(output_data_dir, f'answer_to_questions_{file_index}.yml')
                write_to_yaml(data=answer_to_questions_dict, path=yaml_path)
                answer_to_questions_dict = {}
                save_frequency = 4000
                file_index += 1
        return answer_to_questions_dict

    def modify_data_label(self, data: pd.DataFrame, ori_column='intent', changed_column='changed_intent'):
        """
        我们可能临时需要修改一些标注，修改后更新原来的标注数据
        """

        def change_intent(row, support_labels, ori_column='intent', changed_column='changed_intent'):
            ori_intent = row.get(ori_column)
            changed_intent = row.get(changed_column)
            if isinstance(changed_intent, float) and np.isnan(changed_intent):
                intent = ori_intent
            elif isinstance(changed_intent, str) and changed_intent.lower() == 'not_sure':
                intent = None
            elif changed_intent is None:
                intent = None
            elif isinstance(changed_intent, str) and changed_intent in support_labels:
                intent = changed_intent
            else:
                raise ValueError(f"changed_intent={changed_intent}存在错误")
            return intent

        # 生成新的 intent
        new_intents = data.apply(
            lambda row: change_intent(row, support_labels=self.support_labels, ori_column=ori_column,
                                      changed_column=changed_column), axis=1)
        data.loc[:, self.label_column] = new_intents
        # 去除 数据中 intent 为空的数据
        data.dropna(subset=[self.label_column], inplace=True)
        return data

    def filter_chat_data_from_faq_log(self):
        """
        在 faq 日志文件中 过滤出 闲聊的问题
        """
        path = './data/domain_classification/gyzq_sms_0628_none_financial_term_data_dropdups.xlsx'
        if os.path.exists(path):
            none_financial_term_data_dropdups = pd.read_excel(path)
        else:
            faq_log_data = pd.read_csv(self.faq_log_data_path, parse_dates=['BEGIN_TIME', 'END_TIME'])
            print(
                f"读取数据共 {faq_log_data.shape} 条 from {self.faq_log_data_path},columns= {faq_log_data.columns.tolist()}")

            # data_df.iloc[0]
            # num_id_isnull = data_df.loc[:,'DETAIL_ID'].isnull().sum()
            # id_value_counts = data_df.loc[:, 'DETAIL_ID'].value_counts(sort=True)
            # type(id_value_counts)
            # id_value_counts.sort_index(ascending=False)
            # data_df.loc[:, 'AGENT_ID'].value_counts(sort=True)

            def has_financial_term(text: Text, financial_term_set: Set):
                text = text.lower()
                for term in financial_term_set:
                    if term in text:
                        return True
                return False

            having_financial_term = faq_log_data.loc[:, 'TEXT'].apply(
                lambda text: has_financial_term(text, self.financial_terms_set))
            having_financial_term_data = faq_log_data[having_financial_term]
            none_financial_term_data = faq_log_data[~having_financial_term]
            none_financial_term_data_dropdups = none_financial_term_data.drop_duplicates(subset=['TEXT'])
            print(
                f"不包含提取的金融词汇的数据共 {none_financial_term_data.shape},none_financial_term_data_dropdups={none_financial_term_data_dropdups.shape}")

            none_financial_term_data_dropdups.to_excel(path, index=False, engine='xlsxwriter')  # 'openpyxl')
        print(
            f"none_financial_term_data_dropdups={none_financial_term_data_dropdups.shape}")
        return none_financial_term_data_dropdups

    def extract_terms(self, sentence):
        """
        从 六家券商中 抽取专业金融词汇
        """
        faq_sentence_data = pd.read_excel(self.faq_sentence_data_path, engine='openpyxl', sheet_name='数据')
        filter_data = faq_sentence_data[faq_sentence_data.loc[:, '类别'] == '定义']

        def extract():
            # sentence = '什么是涨跌幅？'
            matched = re.match(pattern=r'^什么是(.*?)？?$', string=sentence)
            if matched:
                term = matched.groups()[0]
                return term

        term_series = filter_data.loc[:, '标准问'].apply(lambda x: extract(x))
        term_series = term_series[~term_series.isna()]
        term_ls = term_series.values.tolist()
        # for t in term_ls:
        #     print(t)
        return term_ls


class IVRRawDataReader(RawIntentDataReader, StandardToExtendQuestionDataReader):
    """
    读取 shaoaojiaqi 和  linjinshu 提供的原始 海通 IVR 数据
    继承 StandardToExtendQuestionDataReader： 读取标准问于扩展问的对应
    """

    # dataset = 'ivr_data'

    def __init__(self,
                 # dataset='ivr_data',
                 # dataset_dir=None,
                 ivr_navigation_filename='ivr导航.xls',
                 # output_data_subdir=None,
                 # use_intent_column_as_label=True,
                 if_get_raw_intent_data=False,
                 if_get_function_name_information_dict=False,
                 if_get_filter_data=False,
                 if_build_raw_intent_data=False,
                 **kwargs
                 ):
        RawIntentDataReader.__init__(self,
                                     # dataset=dataset,
                                     # dataset_dir=dataset_dir,
                                     # output_data_subdir=output_data_subdir,
                                     # use_intent_column_as_label=use_intent_column_as_label,
                                     **kwargs)
        StandardToExtendQuestionDataReader.__init__(self,
                                                    **kwargs
                                                    # output_dir=kwargs.get('output_dir'),
                                                    # output_filename='ivr_service_standard_to_extend_question_data.json'
                                                    )

        self.intent_category_column = INTENT_CATEGORY
        self.function_name_column = FUNCTION_NAME
        # self.response_intent_key = RESPONSE_INTENT_KEY
        # self.first_knowledge_key = FIRST_KNOWLEDGE
        # self.secondary_knowledge_key = SECOND_KNOWLEDGE  # 作为 intent + entity 返回的意图
        # self.first_intent_key = FIRST_INTENT

        self.attribute_key = ATTRIBUTE

        self.entity_column = ENTITY
        self.entity_values_column = ENTITY_VALUES

        # 针对 ivr 数据增加 4列
        self.output_columns.extend([self.intent_category_column,
                                    self.function_name_column,
                                    self.attribute_key,
                                    # self.first_knowledge_key,
                                    # self.secondary_knowledge_key,
                                    # self.first_intent_key
                                    ])

        # shaojiaqi 数据
        self.ivr_navigation_filename = ivr_navigation_filename
        self.ivr_assess_filename_pattern = r"评估集.*.xlsx"
        self.ivr_navigation_data_path = os.path.join(self.data_dir, self.ivr_navigation_filename)

        # jinshu 数据
        self.ivr_function_list_filename = "证券综合语音服务系统产品功能清单.xlsx"
        self.citiao_data_filename = '2.0词条导入-ivr业务(1).xlsx'

        # jinshu ivr new data （20211028）

        # 生成字典的时候的key
        self.function_index_key = 'function_index'
        self.function_code_key = 'function_code'
        self.function_name_key = 'function_name'  # 将其作为 意图的 label
        self.standard_question_key = 'standard_question'
        self.extend_questions_key = 'extend_question'
        self.vocabulary_key = "vocabulary"

        # 区分 faq vs chat category ，而不是将其作为 意图
        self.intent_category = INTENT_CATEGORY
        self.default_faq_category = FAQ
        self.default_chat_category = CHAT
        # 对于 ivr 数据我们保存一个 function_name_to_code 的字典
        self.function_name_information_dict_filename = "function_name_information.json"

        self.if_get_raw_intent_data = if_get_raw_intent_data
        self.if_get_function_name_information_dict = if_get_function_name_information_dict
        self.if_get_filter_data = if_get_filter_data
        self.if_build_raw_intent_data = if_build_raw_intent_data

        if hasattr(self, "output_data_dir"):
            self.function_name_information_dict_path = os.path.join(self.output_data_dir,
                                                                    self.function_name_information_dict_filename)

        # if if_get_raw_intent_data:
        #     self.raw_intent_data = self.get_raw_intent_data()
        #
        # if if_get_filter_data:
        #     # 暂时不支持的类型
        #     self.unsupported_intents = ['（stock）', '（zsstock）', '查指数行情', '查个股行情']
        #     # 选取重点识别的意图并将其转换为 rasa 的 yaml 格式，以方便标注
        #     self.filter_intent_sheetname = 'IVR对应关系'  # 需要先抽取的意图
        #
        #     self.entity_attribute_to_value_dict = self.get_entity_attribute_to_value_dict()
        #     self.support_intents = self.get_support_intents()
        #     self.filter_intents, self.filter_intent_data = self.get_filter_intent_data()
        #     self.get_test_data()

    def get_entity_attribute_to_value_dict(self):
        self.financial_terms = FinancialTerms()
        entity_attribute_to_value_dict = {"block": self.financial_terms.get_block_synonyms(),
                                          "stock": ['海通证券'],
                                          "zsstock": ["恒生指数"]}
        return entity_attribute_to_value_dict

    def _dump_function_name_information_dict(self, function_name_information_dict):
        dump_obj_as_json_to_file(obj=function_name_information_dict, filename=self.function_name_information_dict_path)

    def _load_function_name_information_dict(self):
        if os.path.exists(self.function_name_information_dict_path):
            function_name_information_dict = read_json_file(self.function_name_information_dict_path)
            return function_name_information_dict

    def get_support_function_name_or_intents(self, if_get_intent=False):
        if if_get_intent:
            key = self.intent_column
        else:
            key = self.function_name_key
        function_name_information_dict = self._load_function_name_information_dict()
        # self.vocabulary = self.get_vocabulary()
        support_intents = set()
        for function_name, function_name_info in function_name_information_dict.items():
            # 获取  function_name_info 中对应的意图
            intent = function_name_info.get(key)
            if intent:
                support_intents.add(intent)

        if hasattr(self, "unsupported_intents"):
            support_intents = list(set(support_intents).difference(set(self.unsupported_intents)))
        else:
            support_intents = set(support_intents)
        # logger.info(f"support_labels共{self.support_labels.__len__()}:{self.support_labels}")
        return support_intents

    def _get_filter_intents(self):
        """
        读取 'ivr导航.xls' 表格中的 ，获取重点先做的意图
        """
        path = os.path.join(self.data_dir, self.ivr_navigation_filename)
        data_df = dataframe_to_file(path=path, mode='r', sheet_name=self.filter_intent_sheetname)
        filter_intents = data_df.loc[data_df.loc[:, '是否通用'] == 'Y'].loc[:, '海通IVR-功能名称'].tolist()
        logger.info(f"读取 filter_intents 共 {filter_intents.__len__()}:{filter_intents}")
        return filter_intents

    def _build_raw_intent_data(self) -> pd.DataFrame:
        # 读取原始数据
        raw_intent_data, function_name_information_dict = self._read_raw_intent_data()
        # 增加其他字段转换为标准格式的原始数据
        new_raw_intent_data = self._transform_raw_intent_data(raw_intent_data, function_name_information_dict)
        # 保存 function_name_information_dict
        self._dump_function_name_information_dict(function_name_information_dict)

        return new_raw_intent_data

    def _read_raw_intent_data(self) -> [pd.DataFrame, Dict[Text, Dict[Text, Text]]]:
        """
        读取原始数据为 标准格式
        """
        # function_code 对应的数据字典和 对应关系
        navigation_data_df, function_name_to_code_dict = self._read_ivr_navigation_data(
            path=self.ivr_navigation_data_path, intent_category=self.default_faq_category)
        #  读取 ivr_assess_data 并且更新 function_name_to_code_dict
        assess_data_df, function_name_to_code_dict = self._read_ivr_assess_data(function_name_to_code_dict)

        citiao_data_df = self._read_citiao_data(function_name_to_code_dict)

        # 对 三个数据源进行合并并去重
        raw_intent_data = pd.concat([navigation_data_df, assess_data_df, citiao_data_df])

        return raw_intent_data, function_name_to_code_dict

    # 获取 response:
    def _get_response(self, row, function_name_to_code_dict):
        function_name = row[self.function_name_column]
        intent_category = row[self.intent_category]
        intent = row[self.intent_column]
        # sub_intent = row[self.sub_intent_column]
        response = function_name_to_code_dict[function_name].get(self.response_column)
        if response is None and intent_category == self.default_chat_category:
            # 对于闲聊的语料，拼接一个 response
            response = f"{intent}"
            logger.info(f"intent={intent}的 response 为空，intent_category={self.default_chat_category},response设置为 intent")
        elif response is None and intent_category != self.default_chat_category:
            logger.error(f"intent= {intent} 没有匹配到response ")
            response = intent
            logger.info(f"intent={intent}的 response 为空，intent_category={self.default_chat_category},response设置为 intent")
        return response

    def _transform_raw_intent_data(self, raw_intent_data, function_name_information_dict) -> pd.DataFrame:
        """
        转换为标注的 意图标注格式数据
        """
        # 过滤 function_name_information_dict 中存在意图的数据
        if_have_intent = raw_intent_data.loc[:, self.function_name_column] \
            .apply(lambda function_name: True if function_name_information_dict[function_name].get(
            self.intent_column) else False)

        raw_intent_data = raw_intent_data.loc[if_have_intent].copy()

        # 创建 意图：从 function_name_information_dict 读取 function_name -> intent
        intents = raw_intent_data.loc[:, self.function_name_column] \
            .apply(lambda function_name: function_name_information_dict[function_name][self.intent_column])
        raw_intent_data[self.intent_column] = intents

        # 增加一级知识点
        for key in [self.attribute_key,
                    # self.first_knowledge_key, self.secondary_knowledge_key, self.first_intent_key
                    ]:
            values = raw_intent_data.loc[:, self.function_name_column] \
                .apply(lambda function_name: function_name_information_dict[function_name][key])
            raw_intent_data[key] = values

        responses = raw_intent_data.apply(lambda row: self._get_response(row, function_name_information_dict), axis=1)
        raw_intent_data[self.response_column] = responses

        question_lengths = raw_intent_data.loc[:, self.question_column].apply(len)
        raw_intent_data[self.question_length_column] = question_lengths

        raw_intent_data.loc[:, self.sub_intent_column] = ''

        raw_intent_data = raw_intent_data.loc[:, self.output_columns]

        if hasattr(self, 'unsupported_intents'):
            # 去除暂时不支持的几个类型
            raw_intent_data = raw_intent_data.loc[
                ~raw_intent_data.loc[:, self.intent_column].isin(self.unsupported_intents)]

        logger.info(f"共获取数据 {raw_intent_data.shape} 条")

        return raw_intent_data

    def _read_ivr_navigation_data(self, path,
                                  sheet_name='ivr业务',
                                  intent_category='faq',
                                  function_index_column='功能序号',
                                  function_code_column='功能代码',
                                  function_column='功能名称',
                                  voice_content_column='语音内容',
                                  extend_questions_column='扩展问',
                                  vocabulary_column='词分布',
                                  split_char='###',
                                  id_prefix='navigation_data',
                                  source='ivr导航.xls'
                                  ) -> [pd.DataFrame, Dict[Text, Dict[Text, Text]]]:
        data_df = dataframe_to_file(path=path, mode='r', sheet_name=sheet_name)

        source = get_file_stem(path, full_name=True)
        logger.info(f"读取数据共 shape={data_df.shape},column = {data_df.columns.tolist()}")

        # 确保每个 function_name 作为 key是唯一的
        assert data_df.loc[:, function_column].drop_duplicates().__len__() == data_df.__len__()
        function_name_information_dict = self._crete_function_name_information_dict(data_df)

        extract_data_df = self._expand_extend_questions(data_df, function_index_column, function_column,
                                                        extend_questions_column,
                                                        self.id_column, self.function_name_column,
                                                        self.question_column,
                                                        intent_category=intent_category,
                                                        split_char=split_char,
                                                        id_prefix=id_prefix,
                                                        source=source)
        # 输出的 column ： id, question,intent, sub_intent, source
        logger.info(f"共提取数据 {extract_data_df.shape} 条，columns={extract_data_df.columns.tolist()} from {path} ")
        return extract_data_df, function_name_information_dict

    def _crete_function_name_information_dict(self, data_df,
                                              function_code_column='功能代码',
                                              function_column='功能名称',
                                              voice_content_column='语音内容',
                                              extend_questions_column='扩展问',
                                              vocabulary_column='词分布',
                                              ) -> Dict[Text, Dict[Text, Text]]:
        # 从 ivr_navigation_data 中创建 function_name_information_dict
        function_name_information_dict = {data_df.iloc[i][function_column].strip():
                                              {self.intent_category: self.default_faq_category,
                                               # 作为 意图的类别，默认为faq
                                               self.function_code_key: data_df.iloc[i][function_code_column].strip(),
                                               # 作为 function_code
                                               self.response_column: data_df.iloc[i][voice_content_column].strip(),
                                               # 作为 response
                                               self.vocabulary_key: data_df.iloc[i][vocabulary_column].strip(),
                                               }
                                          for i in range(data_df.__len__())}
        return function_name_information_dict

    def _expand_extend_questions(self, data_df, function_index_column, function_column,
                                 extend_questions_column,
                                 id_column, function_name_column, question_column,
                                 split_char="###", id_prefix='ivr', source='ivr', intent_category=None):
        """
        将 同一个cell中 扩展问 展开
        """
        if extend_questions_column not in data_df.columns:
            extend_questions_column = "普通扩展问"
        # 将原始数据中扩展问拆分后 转换为 dataframe
        select_columns = [function_index_column, function_column, extend_questions_column]

        select_data_df = data_df.loc[:, select_columns].copy()

        new_rows = []
        for index in range(select_data_df.__len__()):
            row = select_data_df.iloc[index].copy()
            function_name = row.get(function_column)
            extend_questions_str = row.get(extend_questions_column)
            # questions = 去重重复的 extend_questions + function_name
            if extend_questions_str and isinstance(extend_questions_str, str):
                extend_questions_set = set([str(ex).strip() for ex in extend_questions_str.split(split_char)])
                questions_ls = list(extend_questions_set.union({function_name}))
            else:
                questions_ls = []
            # sorted(questions_ls,key=lambda x:(len(x),x))

            # row = row.drop(extend_questions_column)
            function_index = row.get(function_index_column)

            if questions_ls:
                for ex_index, ex in enumerate(questions_ls):
                    id = id_prefix + "_" + str(function_index).zfill(3) + "_" + str(ex_index).zfill(5)
                    row[id_column] = id
                    row[question_column] = ex
                    new_row = [id, function_name, ex]
                    new_rows.append(new_row)
            else:
                # 对于空的行
                new_row = [id, function_name, function_name]
                new_rows.append(new_row)

        # 保存 标准问 function_name_key 字段
        extract_data_df = pd.DataFrame(new_rows, columns=[id_column, function_name_column, question_column])

        # intent_category 使用 默认的 faq
        # extract_data_df.loc[:, self.intent_column] = self.default_faq_intent
        extract_data_df.loc[:, self.intent_category] = intent_category

        # source
        extract_data_df.loc[:, self.source_column] = source

        # 去除重复
        extract_data_df.drop_duplicates(subset=[self.question_column], inplace=True, ignore_index=True)

        return extract_data_df

    def _read_ivr_assess_data(self, function_name_information_dict,
                              assess_question_column="评估问",
                              standard_question_column='标准问',  # 这里的标注问其实是 功能名称
                              intent_category_column='评估意图',
                              assess_time='评估时间',
                              ivr_business="IVR业务",
                              ivr_chat="IVR闲聊",
                              unknown_standard_question='未命中'
                              ):
        data_df_ls = []
        matched_filenames = get_matched_filenames(data_dir=self.data_dir,
                                                  filename_pattern=self.ivr_assess_filename_pattern)
        for filename in matched_filenames:
            path = os.path.join(self.data_dir, filename)
            data_df = dataframe_to_file(path=path, mode='r')
            data_df_ls.append(data_df)
        data_df = pd.concat(data_df_ls)
        logger.info(
            f"读取数据共 shape={data_df.shape},column = {data_df.columns.tolist()} from {self.ivr_assess_filename_pattern}")

        data_df.drop_duplicates(inplace=True)
        data_df = data_df.loc[~(data_df.loc[:, standard_question_column] == unknown_standard_question)]
        logger.info(f"去除重复数据后 共 {data_df.shape}")

        function_name_to_intent_category_df = data_df.loc[:, [standard_question_column,
                                                              intent_category_column]].copy().drop_duplicates()

        function_name_information_dict, unsupport_function_names = self._check_and_update_function_name_information_dict(
            function_name_information_dict,
            function_name_to_intent_category_df)

        # 过滤 unsupport_function_names
        if unsupport_function_names.__len__() > 0:
            data_df = data_df.loc[~data_df.loc[:, standard_question_column].isin(unsupport_function_names)]

        def get_intent(intent_category):
            if intent_category == ivr_business:
                return self.default_faq_intent
            elif intent_category == ivr_chat:
                return self.default_chat_intent
            else:
                logger.error(f"intent_category={intent_category}不合理")

        ids = data_df.apply(lambda row: f"{row[assess_time]}_{str(row.name).zfill(3)}", axis=1)
        intents = data_df.loc[:, intent_category_column].apply(lambda intent_category: get_intent(intent_category))

        data_df = data_df.loc[:, [assess_question_column, standard_question_column]]
        data_df.columns = [self.question_column, self.sub_intent_column]
        data_df[self.id_column] = ids
        data_df[self.intent_column] = intents
        data_df[self.source_column] = self.ivr_assess_filename_pattern

        return data_df, function_name_information_dict

    def _check_and_update_function_name_information_dict(self, function_name_information_dict,
                                                         function_name_to_intent_category_df,
                                                         assess_question_column="评估问",
                                                         standard_question_column='标准问',  # 这里的标注问其实是 功能名称
                                                         intent_category_column='评估意图',
                                                         assess_time='评估时间',
                                                         ivr_business="IVR业务",
                                                         ivr_chat="IVR闲聊",
                                                         unknown_standard_question='未命中'
                                                         ):
        unsupport_function_names = set()
        for i in range(function_name_to_intent_category_df.__len__()):
            """
            这里的标注问其实是 功能名称，如果功能名称不在 function_name_to_code_dict 字典中
            """
            row = function_name_to_intent_category_df.iloc[i]
            function_name = row[standard_question_column]
            function_name_info = function_name_information_dict.get(function_name)
            intent_category = row[intent_category_column]

            if function_name_info is not None and intent_category == ivr_chat:
                logger.error(
                    f"function_name={function_name} 应该是属于 {self.default_faq_category},但是 {intent_category_column}={ivr_chat}")
                unsupport_function_names.add(function_name)
            elif function_name_info is None and intent_category == ivr_business:
                logger.error(
                    f"function_name={function_name} 是 {intent_category_column}={ivr_business},但是没有对应的 function_name_info")
                unsupport_function_names.add(function_name)
            elif function_name_info is None and intent_category == ivr_chat:
                # 更新 function_name_to_code_dict, 键入  闲聊的function_name
                function_name_information_dict.update(
                    {function_name: {self.intent_category: self.default_chat_category}})
                logger.info(f"更新 function_name= {function_name} 对应 {self.intent_category}={self.default_chat_category}")
        return function_name_information_dict, unsupport_function_names

    def _read_citiao_data(self, function_name_to_code_dict,
                          citiao_index_column='词条ID',
                          citiao_title_column='词条标题',
                          citiao_content_column='词条内容',
                          extend_questions_column='扩展问',
                          ):
        citiao_data_path = os.path.join(self.data_dir, self.citiao_data_filename)
        data_df = dataframe_to_file(path=citiao_data_path, mode='r')
        logger.info(f"读取数据共 shape={data_df.shape},column = {data_df.columns.tolist()}")

        # 验证词条标题是否在 function_name_to_code_dict 中
        citiao_titles = data_df.loc[:, citiao_title_column].drop_duplicates()
        is_titles_in_dict = all([title in function_name_to_code_dict for title in citiao_titles])
        assert is_titles_in_dict == True

        extract_data_df = self.expand_extend_questions(data_df, function_index_column=citiao_index_column,
                                                       extend_questions_column=extend_questions_column,
                                                       id_column=self.id_column, question_column=self.question_column,
                                                       source='citiao_data')
        extract_data_df.drop_duplicates(inplace=True)
        intents = extract_data_df.loc[:, citiao_title_column].apply(
            lambda title: function_name_to_code_dict[title][self.intent_column])

        extract_data_df = extract_data_df.loc[:, [self.id_column, citiao_title_column, self.question_column]]
        extract_data_df.columns = [self.id_column, self.sub_intent_column, self.question_column]
        extract_data_df[self.intent_column] = intents
        extract_data_df[self.source_column] = self.citiao_data_filename
        return extract_data_df

    def get_test_data(self, path, output_test_path, filter_intents=None):
        """
        读取高峰提供的测试数据，将其转换为指定的格式
        filter_intents: 对 数据按照 filter_intents 进行过滤
        """
        # 新增测试数据
        # raw_test_filename = '3.4测试集.xlsx'
        # output_test_filename = 'test.csv'
        # test_file_path = os.path.join(self.data_dir, raw_test_filename)
        # if os.path.exists(test_file_path):
        #     output_test_path = os.path.join(self.output_data_dir, output_test_filename)
        #     self.get_test_data(test_file_path, output_test_path,filter_intents=self.filter_intents)

        if not os.path.exists(output_test_path):
            assess_question_column = "评估问"
            standard_question_column = '标准问'  # 这里的标注问其实是 功能名称
            data_df = dataframe_to_file(path=path, mode='r')

            # 去除 na 的数据
            data_df = data_df.loc[~data_df.loc[:, standard_question_column].isin(['na'])].copy()
            # 扩展问作为 sub_intent_column
            data_df.columns = [self.question_column, self.sub_intent_column]
            value_counts = data_df.loc[:, self.sub_intent_column].value_counts()
            logger.info(f"测试数据统计:\n {value_counts}")
            data_df[self.id_column] = data_df.apply(lambda row: f"test3.4_{row.name}", axis=1)
            data_df[self.intent_column] = self.default_faq_intent
            data_df[self.question_length_column] = data_df.loc[:, self.question_column].apply(len)
            data_df[self.response_column] = ''
            data_df[self.source_column] = 'test3.4'
            data_df = data_df.loc[:, self.output_columns]

            if filter_intents:
                data_df = data_df.loc[data_df.loc[:, self.sub_intent_column].isin(filter_intents)].copy()
            dataframe_to_file(path=output_test_path, data=data_df)

    def get_vocabulary(self, filter_intents: Union[List[Text], Dict[Text, int]] = None):
        """
        从 self.function_name_to_code_dict 中获取 vocabulary
        """
        vocabulary = set()
        for function_name, function_name_info in self.function_name_information_dict.items():
            intent = function_name_info.get(self.intent_column)

            if filter_intents is None or \
                    (filter_intents is not None and intent is not None and intent in filter_intents):
                function_name_vocabulary = function_name_info.get("vocabulary")
                if function_name_vocabulary is not None:
                    words = [word_and_counts.split('|')[0].lower() for word_and_counts in
                             function_name_vocabulary.split(',')]
                    vocabulary = vocabulary.union(set(words))
        vocabulary = sorted(list(vocabulary), key=lambda x: (len(x), x))
        return vocabulary

    def read_ivr_function_list(self):
        """
        将 功能说明 作为 key，生成字典
        """
        path = os.path.join(self.data_dir, self.ivr_function_list_filename)
        data_df = dataframe_to_file(path=path, mode='r', sheet_name='证券详细功能清单')
        data_df.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
        title_index = 0
        column_index = 1
        last_index = data_df.__len__() - 1
        columns = data_df.iloc[column_index].tolist()
        data_df.columns = columns
        data_df.drop(index=[title_index, column_index, last_index], inplace=True)

        # 统计表中的所有值
        rows_values = data_df.apply(lambda row: row.value_counts().keys().tolist(), axis=1)
        value_counter = Counter()
        for index in range(rows_values.__len__()):
            value_counter.update(rows_values.iloc[index])
        function_values_set = set(value_counter.keys())
        return function_values_set


class IVRRawDataReaderV2(IVRRawDataReader):
    """
    读取  ivr_data_haitong_linjinshu_20211028 提供的 ivr 数据
    新股、新债缴款规则 去掉 顿号输出 新股新债缴款规则

    输出的 raw_intent_data
    增加 2 个字段：
    1. intent_category （faq vs chat）
    2. function_name： 用于映射业务中定义的 标准问

    修改 2个字段：
    intent_column : 训练使用的意图
    sub_intent_column: 暂时为空

    输出的 function_name_information_dict 改变：
    1. 增加  intent_category
    2. 增加 对应的 intent

    """

    def __init__(self,
                 intent_structure_sheetname='intent&entity-20211104',
                 **kwargs):
        super(IVRRawDataReaderV2, self).__init__(**kwargs)
        self.ivr_new_data_filename = 'ivr意图数据.xlsx'
        self.ivr_new_data_path = os.path.join(self.data_dir, self.ivr_new_data_filename)
        self.data_sheetname = 'ivr业务'
        self.ivr_new_data_extend_questions_column = '普通扩展问'
        # 设计的 意图（包括 大意图+精确意图） 和 实体
        self.intent_structure_sheetname = intent_structure_sheetname  # 'intent&entity-20211027'

        # 将业务上的知识点和 模型上的意图分开
        self.function_name_column_in_intent_structure = '标准问'
        self.first_knowledge_column = "一级知识点"
        self.secondary_knowledge_column = '二级知识点'
        self.attribute_column = '属性'
        self.first_intent_column = "first_intent"
        self.intent_column = INTENT
        self.response_column = RESPONSE

        self.raw_intent_data = None
        if self.if_get_raw_intent_data:
            self.raw_intent_data = self.get_raw_intent_data()

    def get_function_name_information_dict(self):
        if os.path.exists(self.function_name_information_dict_path):
            function_name_information_dict = self._load_function_name_information_dict()
        else:
            function_name_information_dict = self._build_function_name_information_dict()

        return function_name_information_dict

    def _build_function_name_information_dict(self):
        ivr_new_data_df, function_name_information_dict = self._read_raw_intent_data()
        return function_name_information_dict

    def _build_data(self) -> pd.DataFrame:
        """
        筛选出 标准问和扩展问
        """
        ivr_new_data_df, function_name_information_dict = self._read_raw_data()
        select_columns = [self.function_name_column, self.question_column, self.intent_category, self.source_column]
        new_columns = [self.standard_question_key, self.extend_questions_key, self.intent_category, self.source_column]
        ivr_standard_to_extend_data = ivr_new_data_df.loc[:, select_columns].copy()
        ivr_standard_to_extend_data.columns = new_columns
        return ivr_standard_to_extend_data

    def _read_raw_data(self):
        # function_code 对应的数据字典和 对应关系
        ivr_new_data_df, function_name_information_dict = self._read_ivr_navigation_data(
            self.ivr_new_data_path,
            extend_questions_column=self.ivr_new_data_extend_questions_column,
            split_char='###',
            id_prefix='ivr_new_data', source=self.ivr_new_data_filename
        )
        return ivr_new_data_df, function_name_information_dict

    def _read_raw_intent_data(self) -> [pd.DataFrame, Dict[Text, Dict[Text, Text]]]:
        """
        读取原始数据为 标准格式
        """
        ivr_new_data_df, function_name_information_dict = self._read_raw_data()

        function_names = ivr_new_data_df.loc[:, self.function_name_column].value_counts().index.tolist()

        # 读取我们定义的意图体系的信息，将其转换为 function_name_to_intent_dict
        support_function_name_to_intent_entity_info_dict = self._get_support_function_name_to_intent_dict(
            function_names=function_names,
            function_name_column=self.function_name_column_in_intent_structure,
            first_knowledge_column=self.first_knowledge_column,
            secondary_knowledge_column=self.secondary_knowledge_column,
            attribute_column=self.attribute_column,
            first_intent_column=self.first_intent_column,
            entity_column=self.entity_column,
            entity_values_column=self.entity_values_column
        )

        # 使用  function_name 对应的 intent 更新 function_name_to_intent_dict
        for function_name, function_name_to_intent_entity_info in support_function_name_to_intent_entity_info_dict.items():
            function_name_info = function_name_information_dict.get(function_name)
            if function_name_info:
                # 在 function_name_information_dict 增加 entity 信息 + 二级知识点信息
                function_name_info.update(function_name_to_intent_entity_info)
            else:
                # 如果 function_name_info 为空, 将信息加入到
                function_name_information_dict.update({function_name: function_name_to_intent_entity_info})

            # function_name_info.update({self.intent_column: intent})

        return ivr_new_data_df, function_name_information_dict

    def _get_support_function_name_to_intent_dict(self, function_name_column='haitong-功能名称',
                                                  first_knowledge_column='一级知识点',
                                                  secondary_knowledge_column='二级知识点',
                                                  attribute_column='属性',
                                                  first_intent_column='fist_intent',
                                                  intent_column=INTENT,
                                                  entity_column=ENTITY,
                                                  entity_values_column=ENTITY_VALUES,
                                                  response_column=RESPONSE,
                                                  function_names=None,
                                                  valid_lines=135):
        """
        @time:  2021/11/19 0:03
        @author:wangfc
        @version:
        @description: 将业务定义的知识点 和 意图&实体等模型信息 分开

        @params:
        @return:
        """
        # 读取原始的 意图结构数据
        intent_structure_data = dataframe_to_file(path=self.ivr_new_data_path,
                                                  sheet_name=self.intent_structure_sheetname, mode='r')
        # 选取对应的列
        function_column_to_info_columns = [first_knowledge_column, secondary_knowledge_column,
                                           attribute_column, first_intent_column, intent_column,
                                           response_column,
                                           entity_column, entity_values_column, ]
        selected_columns = [function_name_column] + function_column_to_info_columns
        selected_intent_structure_data = intent_structure_data.loc[:, selected_columns].dropna(how='all').iloc[
                                         :valid_lines]

        # 对于合并的行向前进行填充，确保 二级知识点 + intent 非空
        filled_columns = function_column_to_info_columns[:-2]
        filled = selected_intent_structure_data.loc[:, filled_columns].fillna(method='ffill')
        for filled_column in filled_columns:
            selected_intent_structure_data.loc[:, filled_column] = filled.loc[:, filled_column]

        # 确定没有空行
        assert selected_intent_structure_data.loc[:, [first_knowledge_column, secondary_knowledge_column,
                                                      attribute_column, first_intent_column, intent_column]] \
                   .isna().sum().sum() == 0

        # 去除 二级知识点 标记为 N 的 行
        selected_intent_structure_data = selected_intent_structure_data.loc[
            selected_intent_structure_data.loc[:, intent_column] != 'N'].copy()

        # TODO: 对 haitong_function_column 中含有 / 符号的标准问进行分割
        new_rows = []
        for i in range(selected_intent_structure_data.__len__()):
            row = selected_intent_structure_data.iloc[i]
            function_names = row.loc[function_name_column].strip()
            function_names = [name for name in function_names.split('/') if name.strip() != '']
            print(f"row_index={i},function_names={function_names}")
            if function_names.__len__() > 1:
                for function_name in function_names:
                    new_row = deepcopy(row)
                    new_row.loc[function_name_column] = function_name
                    new_rows.append(new_row)
            else:
                new_rows.append(row)
        selected_intent_structure_data = pd.DataFrame(new_rows).drop_duplicates()

        # 验证 haitong_function_column 值的唯一的
        assert selected_intent_structure_data.loc[:,
               function_name_column].value_counts().__len__() == selected_intent_structure_data.__len__()

        # 获取 function_name_to_intent_dict 并且验证 support_function_names 全部在 原始数据当中
        function_name_to_intent_entity_info_dict = {}
        for i in range(selected_intent_structure_data.shape[0]):
            row = selected_intent_structure_data.iloc[i]
            function_name = row.loc[function_name_column]
            intent_entity_info_series = row.loc[function_column_to_info_columns]
            intent_entity_info_dict = intent_entity_info_series.to_dict()

            info_dict = intent_entity_info_dict.copy()
            for key, value in intent_entity_info_dict.items():
                if pd.isna(value):
                    # nan 转换 None
                    info_dict.update({key: None})
                elif key == entity_values_column:
                    # entity_values 转换为list
                    entity_values = value.strip('[]').split(',')
                    assert isinstance(entity_values, list)
                    info_dict.update({key: entity_values})
                elif key == first_knowledge_column:
                    key = self.first_knowledge_key
                    info_dict.update({key: value})
                    info_dict.pop(first_knowledge_column)
                elif key == secondary_knowledge_column:
                    # 增加 二级知识点的 字段名称 为 seccodary_intent,表示模型将要输出的 意图给下游任务
                    key = self.secondary_knowledge_key
                    info_dict.update({key: value})
                    info_dict.pop(secondary_knowledge_column)
                elif key == attribute_column:
                    # 增加 二级知识点的 字段名称 为 seccodary_intent,表示模型将要输出的 意图给下游任务
                    key = self.attribute_key
                    info_dict.update({key: value})
                    info_dict.pop(attribute_column)
                # elif key == first_intent_column:
                #     key = self.first_intent_key
                #     info_dict.update({key: value})
                #     info_dict.pop(first_intent_column)

            function_name_to_intent_entity_info_dict.update({function_name: info_dict})

        # function_name_to_intent_dict = {selected_intent_structure_data.iloc[i][haitong_function_column]:
        #                                     selected_intent_structure_data.iloc[i][secondary_intent_column]
        #                                 for i in range(selected_intent_structure_data.shape[0])}

        # support_function_names = set(function_name_to_intent_entity_info_dict.keys())
        # assert support_function_names.issubset(set(function_names))
        return function_name_to_intent_entity_info_dict

    def check_intersection_data(self):
        """
        1) 去除重复情况

        2) 验证数据是否存在交叉的情况
        标准问对应的扩展问中可能存在交叉的情况
        """
        self.raw_intent_data.drop_duplicates(subset=[self.function_name_column, self.question_column], inplace=True)

        grouped_data = self.raw_intent_data.groupby(by=[self.function_name_column])
        group_keys = set(grouped_data.groups.keys())
        for function_name, group in grouped_data:
            # logger.info(f"group_key={function_name}")
            examples = set(group.loc[:, self.question_column].tolist())
            group_keys.difference_update({function_name})
            for other_function_name in group_keys:
                another_group = grouped_data.get_group(other_function_name)
                other_examples = set(another_group.loc[:, self.question_column].tolist())
                intersection_examples = examples.intersection(other_examples)
                if intersection_examples.__len__() > 0:
                    intersection_examples_str = '\n'.join(list(intersection_examples))
                    logger.error(
                        f"{function_name} 和 {other_function_name} 存在着数据交集共{intersection_examples.__len__()}条：\n"
                        f"{intersection_examples_str}")

    def get_chat_data(self, sheet_name):
        # function_code 对应的数据字典和 对应关系
        navigation_data_df, function_name_to_code_dict = self._read_ivr_navigation_data(
            path=self.ivr_navigation_data_path, sheet_name=sheet_name,
            intent_category=self.default_chat_category
        )
        return navigation_data_df

    # def _transform_raw_intent_data(self, raw_intent_data, function_name_information_dict) -> pd.DataFrame:
    #     # 计算问题的长度
    #     question_lengths = raw_intent_data.loc[:, self.question_column].apply(len)
    #     raw_intent_data[self.question_length_column] = question_lengths
    #
    #     # 创建 intent_categories：从 function_name_information_dict 读取
    #     intent_categories = raw_intent_data.loc[:, self.function_name_column].apply(
    #         lambda function_name:function_name_information_dict[function_name][self.intent_category])
    #     raw_intent_data[self.intent_category] = intent_categories
    #
    #     # 创建 responses
    #     responses = raw_intent_data.apply(lambda row: self._get_response(row, function_name_information_dict), axis=1)
    #     raw_intent_data[self.response_column] = responses
    #
    #     # 按照 output_columns 排序输出
    #     raw_intent_data = raw_intent_data.loc[:, self.output_columns]
    #
    #     if hasattr(self, 'unsupported_intents'):
    #         # 去除暂时不支持的几个类型
    #         raw_intent_data = raw_intent_data.loc[
    #             ~raw_intent_data.loc[:, self.intent_column].isin(self.unsupported_intents)]
    #
    #     logger.info(f"共获取数据 {raw_intent_data.shape} 条")
    #
    #     return raw_intent_data


class IVRStandardToExtendDataReader(RawIntentDataReader, StandardToExtendQuestionDataReader):
    """
    处理 gaofeng 和 liaozhilin 提供的 IVR 海通数据，该数据按照 标准问和扩展问的方式提供
    """
    STANDARD_QUESTION_COLUMN_NAME = "标准问"
    EXTEND_QUESTION_COLUMN_NAME = "扩展问"
    ASSESS_QUESTION_COLUMN_NAME = "评估问"

    INTENT_CATEGORY = INTENT_CATEGORY
    DEFAULT_FAQ_CATEGORY = FAQ
    DEFAULT_CHAT_CATEGORY = CHAT

    def __init__(self,
                 dataset='ivr_data',
                 dataset_dir=None,
                 output_data_subdir=None,
                 filter_intents=None, function_name_information_dict=None,
                 if_build_raw_intent_data=True, **kwargs):

        RawIntentDataReader.__init__(self,
                                     dataset=dataset,
                                     dataset_dir=dataset_dir,
                                     output_data_subdir=output_data_subdir,
                                     **kwargs)
        StandardToExtendQuestionDataReader.__init__(self, output_dir=kwargs.get('output_dir'),
                                                    output_filename='ivr_standard_to_extend_question_data.json')

        self.intent_category_column = INTENT_CATEGORY
        self.function_name_column = FUNCTION_NAME
        # self.response_intent_key = RESPONSE_INTENT_KEY

        self.attribute_column = ATTRIBUTE_ZH
        self.attribute_key = ATTRIBUTE
        # 将业务上的知识点和 模型上的意图分开
        # self.first_knowledge_column = "一级知识点"
        # self.secondary_knowledge_column = '二级知识点'
        # self.first_intent_column = "first_intent"
        # self.first_knowledge_key = FIRST_KNOWLEDGE
        # self.secondary_knowledge_key = SECOND_KNOWLEDGE  # 作为 intent + entity 返回的意图
        # self.first_intent_key = FIRST_INTENT

        self.entity_column = ENTITY
        self.entity_values_column = ENTITY_VALUES

        self.if_build_raw_intent_data = if_build_raw_intent_data
        # 新增 3 列
        self.output_columns.extend([self.intent_category_column, self.function_name_column, self.attribute_key,
                                    # self.first_knowledge_key, self.secondary_knowledge_key,self.first_intent_key
                                    ])

        self.intent_list_todo_filename = '意图列表.xlsx'

        self.filter_intents = filter_intents
        self.function_name_information_dict = function_name_information_dict

        # 允许的列名： 标准问 + 扩展问/评估问
        self.allowed_raw_column_names = {self.STANDARD_QUESTION_COLUMN_NAME, self.EXTEND_QUESTION_COLUMN_NAME,
                                         self.ASSESS_QUESTION_COLUMN_NAME}
        self.raw_column_mapping = {self.ASSESS_QUESTION_COLUMN_NAME: self.EXTEND_QUESTION_COLUMN_NAME}

        # self.raw_intent_data = self.get_raw_intent_data()
        # self.filter_intents, self.filter_intent_data = self.get_filter_intent_data()

    def _build_data(self) -> pd.DataFrame:
        data = self._read_raw_data()
        select_columns = [self.STANDARD_QUESTION_COLUMN_NAME, self.EXTEND_QUESTION_COLUMN_NAME,
                          self.intent_category_column, self.source_column]
        new_columns = [STANDARD_QUESTION, EXTEND_QUESTION, self.intent_category_column, self.source_column]
        data = data.loc[:, select_columns].copy()
        data.columns = new_columns
        return data

    def _read_raw_data(self):
        """
         读取所有的标准问和扩展问
         """
        filenames = os.listdir(self.data_dir)
        data_filenames = list(set(filenames).difference({self.intent_list_todo_filename}))
        data_filenames = [data_filename for data_filename in data_filenames if
                          os.path.splitext(data_filename)[-1].split('.')[-1] in ['xlsx', 'xls']]
        data_df_ls = []
        for filename in data_filenames:
            path = os.path.join(self.data_dir, filename)
            data_df = dataframe_to_file(path=path, mode='r')
            # 筛选数据列：
            mapping_data_df = self._mapping_new_column_names(data_df)
            # 增加 id 列
            nosuffux_filename, suffix = os.path.splitext(filename)
            ids = mapping_data_df.apply(lambda row: f"{nosuffux_filename}_{row.name}", axis=1)
            mapping_data_df.loc[:, self.id_column] = ids
            # 增加 source 列，标记数据来源
            mapping_data_df.loc[:, self.source_column] = filename
            logger.debug(mapping_data_df.head(2))
            data_df_ls.append(mapping_data_df)

        data = pd.concat(data_df_ls)
        data.drop_duplicates(subset=[self.STANDARD_QUESTION_COLUMN_NAME, self.EXTEND_QUESTION_COLUMN_NAME],
                             inplace=True)
        logger.info(f"共读取原始数据 {data.shape} columns={data.columns.tolist()} from {self.data_dir}")

        # standard_question_set = set(data.loc[:, self.STANDARD_QUESTION_COLUMN_NAME].value_counts().index.tolist())
        # support_function_names_set = self._get_support_function_names_set()
        # standard_question_set.difference(support_function_names_set)
        # 根据 support_function_names_set 过滤对应的数据
        # data = data.loc[data.loc[:, self.STANDARD_QUESTION_COLUMN_NAME].isin(support_function_names_set)].copy()

        # 转换为标准格式 self.output_columns
        # 增加 2列 :intent_category_column
        data.loc[:, self.INTENT_CATEGORY] = self.DEFAULT_FAQ_CATEGORY
        return data

    def _build_raw_intent_data(self):
        data = self._read_raw_data()
        # function_name_column: 标准问
        data.loc[:, self.function_name_column] = data.loc[:, self.STANDARD_QUESTION_COLUMN_NAME]
        # 去除 sub_intent_column
        # data.loc[:, self.sub_intent_column] = data.loc[:, self.STANDARD_QUESTION_COLUMN_NAME]
        data.loc[:, self.sub_intent_column] = ''
        data.loc[:, self.question_column] = data.loc[:, self.EXTEND_QUESTION_COLUMN_NAME]
        data.loc[:, self.question_length_column] = data.loc[:, self.question_column].apply(len)
        data.loc[:, self.response_column] = data.loc[:, self.STANDARD_QUESTION_COLUMN_NAME]

        # 意图： 使用 function_name_information_dict 将 STANDARD_QUESTION （function_name）来匹配 意图来匹配
        # STANDARD_QUESTION -> intent
        # data.loc[:, self.intent_column] = data.loc[:, self.STANDARD_QUESTION_COLUMN_NAME] \
        #     .apply(lambda standard_question: self.function_name_information_dict.get(standard_question).get(
        #     self.intent_column))
        #
        # # 新增 first_intent_key 列
        # data.loc[:, self.first_intent_key] = data.loc[:, self.STANDARD_QUESTION_COLUMN_NAME] \
        #     .apply(lambda standard_question: self.function_name_information_dict.get(standard_question).get(
        #     self.first_intent_key))
        #
        # # 新增 first_intent_key 列
        # data.loc[:, self.secondary_intent_key] = data.loc[:, self.STANDARD_QUESTION_COLUMN_NAME] \
        #     .apply(lambda standard_question: self.function_name_information_dict.get(standard_question).get(
        #     self.secondary_intent_key))

        # 增加一级知识点
        for key in [self.intent_column, self.attribute_key,
                    # self.first_knowledge_key, self.secondary_knowledge_key, self.first_intent_key
                    ]:
            values = data.loc[:, self.STANDARD_QUESTION_COLUMN_NAME] \
                .apply(lambda function_name: self.function_name_information_dict[function_name][key])
            data[key] = values

        data = data.loc[:, self.output_columns].copy()
        logger.info(f"格式标准化的原始数据 {data.shape} columns={data.columns.tolist()} from {self.data_dir}")
        return data

    def _get_filter_intents(self, topK=11, intent_column='标准问'):
        """
        读取 intent_list_todo_filename 中将要做的意图类型,并选取topK 个
        """
        if self.filter_intents is None:
            filter_intents = self.filter_intents()
        else:
            # 计划要实现的意图分类
            unsupported_intents = {'查其他账号', '查手续费', '降低佣金'}

            path = os.path.join(self.data_dir, self.intent_list_todo_filename)
            intents_todo_df = dataframe_to_file(path=path, mode='r')
            filter_intents = intents_todo_df.loc[:topK, intent_column].values.tolist()
            filter_intents = [intent for intent in filter_intents if intent not in unsupported_intents]
            logger.info(f"过滤的标准问类型共 {filter_intents.__len__()} 个：{filter_intents}")
        return filter_intents

    def _mapping_new_column_names(self, data_df):
        # 转换为对应的列名
        column_names = data_df.columns.tolist()

        allowed_column_names = []
        new_column_names = []
        for column_name in column_names:
            if column_name in self.allowed_raw_column_names:
                allowed_column_names.append(column_name)
                mapping_name = self.raw_column_mapping.get(column_name)
                if mapping_name:
                    new_column_names.append(mapping_name)
                else:
                    new_column_names.append(column_name)
        assert allowed_column_names.__len__() == new_column_names.__len__() == 2
        # 筛选数据列 + 更换列名
        mapping_data_df = data_df.loc[:, allowed_column_names].copy()
        mapping_data_df.columns = new_column_names
        return mapping_data_df

    def _get_support_function_names_set(self) -> Dict[Text, Text]:
        support_function_names_set = set()
        # intent_to_function_name_mapping = {}
        for function_name, function_name_info in self.function_name_information_dict.items():

            intent = function_name_info.get(self.intent_column)
            if intent:
                # 如果 intent 是 大意图会重复覆盖掉原来的意图
                # intent_to_function_name_mapping.update({intent: function_name})
                support_function_names_set.add(function_name)
        return support_function_names_set

    def _get_function_name(self, standard_question: Text, intent_to_function_name_mapping):
        function_name = intent_to_function_name_mapping.get(standard_question)
        if function_name is None:
            function_name = standard_question
        return function_name


if __name__ == '__main__':
    from conf.config_parser import VOCAB_FILE

    # raw_indent_classifier_data_processor = RawIntentClassifierDataProcessor()
    # new_version_data = raw_indent_classifier_data_processor.get_new_version_data()

    # 初始化 IntentClassifierProcessor 对象
    output_data_subdir = 'gf_intent_classifier_data_20210728'  # 'gf_intent_classifier_data_20210721',
    support_labels = ['faq', 'advice']
    indent_classifier_processor = IntentClassifierProcessor(
        output_data_subdir=output_data_subdir, support_labels=support_labels,
        vocab_file=VOCAB_FILE)

    # 调用 get_new_version_data
    new_version_data = indent_classifier_processor.get_new_version_data()

    data_generator = indent_classifier_processor.prepare_data(data_type=TRAIN_DATATYPE)
