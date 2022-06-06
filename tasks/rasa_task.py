#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/22 9:51 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/22 9:51   wangfc      1.0         None
"""
import os
import sys
from typing import Text, List, Dict
import pandas as pd

from data_process.data_labeling import EntityLabeler
from data_process.data_processing import remove_line_break_char
from data_process.dataset.hsnlp_faq_knowledge_dataset import StandardQuestionInfo, QuestionKnowledge, \
    StandardQuestionKnowledgeData, EntityData, get_intent_from_labeled_data, ATTRIBUTE, ENTITY_TYPE, \
    KnowledgeDefinitionData, match_entity_type_columns, DEFAULT_INTENT, DEFAULT_ATTRIBUTE
from tasks.data_labeling_task import get_rasa_nlu_parse_results, set_scenario
from utils.constants import STANDARD_QUESTION, SCENARIO, INTENT, EXTEND_QUESTION, QUESTION

from data_process.dataset.hsnlp_faq_dataset import HSNLPFaqDataReader, StandardToExtendQuestionDataReader, \
    StandardToExtendDataset
from data_process.dataset.intent_classifier_dataset import IVRRawDataReaderV2, IVRStandardToExtendDataReader, \
    RasaIntentDataset
from data_process.dataset.rasa_dataset import HSfaqToRasaDataset
from tasks.text_classifier_task import TransformerClassifierTask
from utils.io import read_json_file, dump_obj_as_json_to_file, dataframe_to_file, get_files_from_dir

from conf.config_parser import GPU_MEMORY_CONFIG
from utils.tensorflow.environment import setup_tf_environment

import logging

logger = logging.getLogger(__name__)


class RasaTask(TransformerClassifierTask):

    def __init__(self,
                 raw_faq_knowledge_definition_filename='hsnlp-基于知识图谱的意图和实体定义体系-20211210.xlsx',
                 # faq_knowledge_definition_sheet_name='一级知识点-属性-实体-20211210',
                 faq_intent_attribute_definition_sheet_name="意图和属性定义-20220207",
                 faq_entity_definition_sheet_name="实体定义-20220207",
                 standard_question_knowledge_sheet_name='haitong-ivr标准问与意图属性实体对应关系',
                 raw_standard_to_extend_data_filename="standard_to_extend_question_data-20211130.xlsx",
                 filter_domains: List[Text] = ['haitong_ivr'],
                 intent_attribute_mapping_filename="haitong_ivr_intent_attribute_mapping.json",
                 standard_to_extend_data_sheet_name='standard_to_extend_data',
                 entity_data_filename="financial_entity.yml",
                 standard_question_info_filename='standard_question_info.json',
                 train_data_intent_attribute_info_filename = 'train_data_intent_attribute_info.json',
                 rasa_config_filename='config.yml',
                 rasa_train_data_subdirname='data',
                 rasa_test_data_subdirname='test',
                 rasa_models_subdirname='models',
                 *args,
                 **kwargs):
        super(RasaTask, self).__init__(*args, **kwargs)

        # 定义体系
        self.raw_faq_knowledge_definition_filename = raw_faq_knowledge_definition_filename
        # self.faq_knowledge_definition_sheet_name = faq_knowledge_definition_sheet_name
        self.faq_intent_attribute_definition_sheet_name = faq_intent_attribute_definition_sheet_name
        self.faq_entity_definition_sheet_name = faq_entity_definition_sheet_name

        self.standard_question_knowledge_sheet_name = standard_question_knowledge_sheet_name
        # 标准问和扩展问的数据
        self.raw_standard_to_extend_data_filename = raw_standard_to_extend_data_filename
        self.standard_to_extend_data_sheet_name = standard_to_extend_data_sheet_name

        # 实体信息数据
        self.entity_data_filename = entity_data_filename

        # self.faq_knowledge_definition_filename = "faq_knowledge_definition.json"
        self.faq_intent_attribute_definition_filename = "faq_intent_attribute_definition.json"
        self.faq_entity_definition_filename = "faq_entity_definition.json"
        self.standard_question_knowledge_data_filename = "standard_question_knowledge_data.json"
        self.standard_to_extend_data_filename = "standard_to_extend_question_data.json"

        # 标准问的信息
        self.standard_question_info_filename = standard_question_info_filename
        self.standard_question_info_path = os.path.join(self.output_dir, self.standard_question_info_filename)

        self.filter_domains = filter_domains
        # 针对ivr数据，根据标准问覆盖的多个属性的情况，需要对部分的属性做归一化映射为同一个属性，方便后面根据意图+属性识别标准问
        self.intent_attribute_mapping_filename = intent_attribute_mapping_filename
        self.intent_attribute_mapping_path = os.path.join(self.dataset_dir, self.intent_attribute_mapping_filename)

        # 训练数据的意图和属性统计
        self.train_data_intent_attribute_info_filename =  train_data_intent_attribute_info_filename
        self.train_data_intent_attribute_info_path = os.path.join(self.data_dir,self.train_data_intent_attribute_info_filename)

        self.rasa_config_filename = rasa_config_filename
        self.rasa_train_data_subdirname = rasa_train_data_subdirname
        self.rasa_test_data_subdirname = rasa_test_data_subdirname
        self.rasa_models_subdirname = rasa_models_subdirname

        self.rasa_train_data_dir = os.path.join(self.output_dir, self.rasa_train_data_subdirname)
        self.rasa_test_data_dir = os.path.join(self.output_dir, self.rasa_train_data_subdirname)

        # self.output_test_data_dir = os.path.join(self.output_dir, self.rasa_test_data_subdirname)
        # self.rasa_test_data_path = os.path.join(self.rasa_test_data_dir, 'test_data.json')

    def _prepare_data_to_label(self):
        """
        准备 标准问-扩展问数据进行 意图实体标注
        """

        # 读取标准问和扩展问
        self.standard_to_extend_data_reader = self._get_standard_to_extend_data_reader()

        # 对数据进行统计和预标注，并转换为excel格式
        standard_to_extend_data = self.standard_to_extend_data_reader.standard_to_extend_data.data

        # 对 standard_to_extend_data 做去重处理
        standard_to_extend_data.drop_duplicates(subset=["standard_question", "extend_question", "scenario"],
                                                inplace=True)

        standard_to_extend_data.loc[:, SCENARIO] = standard_to_extend_data \
                                                       .loc[:, SCENARIO].apply(
            lambda scenario: set_scenario(scenario))

        source_value_count = standard_to_extend_data.source.value_counts()
        scenario_value_count = standard_to_extend_data.scenario.value_counts()
        standard_to_extend_question_num_count = standard_to_extend_data.loc[:, STANDARD_QUESTION].value_counts()

        # 对标注问的数据进行预标注并且排序
        standard_question_prelabel_results_path = os.path.join(self.standard_to_extend_data_reader.output_dir,
                                                               f"standard_question_prelabel_results.json")
        standard_questions = standard_to_extend_question_num_count.index.tolist()
        standard_question_prelabel_results = get_rasa_nlu_parse_results(questions=standard_questions,
                                                                        path=standard_question_prelabel_results_path)

        # 对扩展问的数据进行预标注
        extend_question_prelabel_results_path = os.path.join(self.standard_to_extend_data_reader.output_dir,
                                                             f"extend_question_prelabel_results.json")
        extend_questions = standard_to_extend_data.loc[:, EXTEND_QUESTION].drop_duplicates().values.tolist()
        extend_question_prelabel_results = get_rasa_nlu_parse_results(questions=extend_questions,
                                                                      path=extend_question_prelabel_results_path)

        # 使用新的count 更新 counts
        for result in standard_question_prelabel_results:
            standard_question = result[STANDARD_QUESTION]
            new_count = standard_to_extend_question_num_count.loc[standard_question]
            result.update({"count": new_count})

        columns = ['standard_question', 'count', 'predict_intent', 'attribute',
                   'entity_type_0', 'entity_value_0', 'entity_type_1', 'entity_value_1',
                   'entity_type_2', 'entity_value_2']
        standard_question_prelabel_results_df = pd.DataFrame(standard_question_prelabel_results).loc[:, columns]

        # 增加 scenario 字段，用于区分不同的场景，分阶段标注和 验证覆盖率
        standard_to_scenario_data_ls = []
        # 按照 scenario_value_count 从多到少排序
        for scenario in scenario_value_count.index:
            standard_to_scenario_data = standard_to_extend_data \
                                            .loc[standard_to_extend_data.scenario == scenario] \
                                            .loc[:, [STANDARD_QUESTION, SCENARIO]].copy()
            standard_to_scenario_data_ls.append(standard_to_scenario_data)
        standard_to_scenario_df = pd.concat(standard_to_scenario_data_ls)
        standard_to_scenario_df.drop_duplicates(subset=[STANDARD_QUESTION], inplace=True)

        standard_question_to_scenario = {}
        for i in range(standard_to_scenario_df.__len__()):
            standard_question = standard_to_scenario_df.iloc[i].loc[STANDARD_QUESTION]
            scenario = standard_to_scenario_df.iloc[i].loc[SCENARIO]
            standard_question_to_scenario.update({standard_question: scenario})

        standard_question_prelabel_results_df.loc[:, SCENARIO] = standard_question_prelabel_results_df.loc[:,
                                                                 STANDARD_QUESTION] \
            .apply(lambda x: standard_question_to_scenario[x])

        # 变换列
        new_columns = ['standard_question', SCENARIO, 'count', 'predict_intent', 'attribute',
                       'entity_type_0', 'entity_value_0', 'entity_type_1', 'entity_value_1',
                       'entity_type_2', 'entity_value_2']
        prelabel_results_df = standard_question_prelabel_results_df.loc[:, new_columns].copy()

        # 按照 scenario 排序
        prelabel_results_data_ls = []
        for scenario in scenario_value_count.index:
            prelabel_results_data = prelabel_results_df.loc[prelabel_results_df.scenario == scenario] \
                .sort_values(by=['count'], ascending=False)
            prelabel_results_data_ls.append(prelabel_results_data)
        prelabel_results_df = pd.concat(prelabel_results_data_ls)

        predict_intent_key = "predict_intent"
        extend_question_count_key = "extend_question_count"
        standard_question_length_key = 'standard_question_length'
        extend_question_length_key = 'extend_question_length'
        standard_question_to_intent = {result[STANDARD_QUESTION]:
                                           {predict_intent_key: result[predict_intent_key],
                                            extend_question_count_key: result['count']
                                            }
                                       for result in standard_question_prelabel_results}

        # 对 standard_to_extend_data 进行排序： 预标注意图 + 标准问长度  + 扩展问的数量 + 扩展问长度
        # standard_to_extend_data.loc[:,predict_intent_key] = \
        #     standard_to_extend_data .loc[:, STANDARD_QUESTION].apply(
        #     lambda x: standard_question_to_intent[x][predict_intent_key])
        standard_to_extend_data.loc[:, standard_question_length_key] = \
            standard_to_extend_data.loc[:, STANDARD_QUESTION].apply(
                lambda x: len(x))
        standard_to_extend_data.loc[:, extend_question_count_key] = \
            standard_to_extend_data.loc[:, STANDARD_QUESTION].apply(
                lambda x: standard_question_to_intent[x][extend_question_count_key])
        standard_to_extend_data.loc[:, extend_question_length_key] = \
            standard_to_extend_data.loc[:, EXTEND_QUESTION].apply(lambda x: len(x))

        # 增加扩展问的预标注信息
        extend_to_predict_result = {}
        for result in extend_question_prelabel_results:
            extend_question = result['question']
            extend_to_predict_result.update({extend_question: result})
        new_rows = []
        for i in range(standard_to_extend_data.__len__()):
            row = standard_to_extend_data.iloc[i]
            extend_question = row.loc[EXTEND_QUESTION]
            predict_result = extend_to_predict_result.get(extend_question).copy()
            predict_result.pop('question')
            new_row = row.to_dict()
            if predict_result:
                new_row.update(predict_result)
            new_rows.append(new_row)

        standard_to_extend_data = pd.DataFrame(new_rows)

        # 按照 scenario 排序
        standard_to_extend_data_on_scenario_ls = []
        for scenario in scenario_value_count.index:
            standard_to_extend_data_on_scenario = standard_to_extend_data.loc[
                standard_to_extend_data.scenario == scenario] \
                .sort_values(by=[extend_question_count_key, predict_intent_key, standard_question_length_key,
                                 extend_question_length_key,
                                 extend_question_count_key], ascending=False)

            standard_to_extend_data_on_scenario_ls.append(standard_to_extend_data_on_scenario)
        standard_to_extend_data = pd.concat(standard_to_extend_data_on_scenario_ls)

        standard_to_extend_question_path = os.path.join(self.standard_to_extend_data_reader.output_dir,
                                                        f"standard_to_extend_question_data.xlsx")

        for data, sheet_name in zip([prelabel_results_df,
                                     standard_to_extend_data,
                                     source_value_count,
                                     scenario_value_count],
                                    ["standard_question",
                                     "standard_to_extend_data",
                                     "source_value_count",
                                     "scenario_value_count"]):
            dataframe_to_file(path=standard_to_extend_question_path, data=data, sheet_name=sheet_name, mode='a',
                              over_write_sheet=True)

    def _prepare_data(self, if_get_train_data=True, if_postprocess_data=True,
                      if_get_test_data=False, if_split_data=False,
                      check_labeled_data=True,
                      if_transform_to_rasa_data=False):
        # 获取意图属性实体的知识定义体系
        self.knowledge_definition_data = self._get_knowledge_definition_data()

        # 读取实体的近义词和正则表达式等信息
        self.entity_data = self._get_entity_data()

        # 获取 标准问的定义信息：一级知识点，二级知识点，意图，first_intent（大意图）等
        # self.standard_question_info = self._get_standard_question_info()
        self.standard_question_knowledge_data = self._get_standard_question_knowledge_data()

        self.intent_attribute_mapping,self.intent_to_attribute_dict, self.output_intents, self.output_attributes = \
            self._load_intent_attribute_mapping()

        # self.label_to_sub_label_mapping = {key:list(value.keys()) for key,value in self.intent_attribute_mapping.items()}

        train_data, test_data = None, None
        if if_get_train_data:
            # 读取原始的标准问和扩展问数据
            self.standard_to_extend_data_reader = self._get_standard_to_extend_data_reader()

            # 获取 standard_to_extend_data 对象
            standard_to_extend_data = self.standard_to_extend_data_reader.standard_to_extend_data

            # 转换为 含有 intent 的数据
            rasa_intent_dataset = self._transform_to_intent_data(standard_to_extend_data)
            train_data = rasa_intent_dataset.data.copy()

            # postprocess
        if if_postprocess_data:
            train_data = self._postprocess_data(train_data)

        # groups = train_data.loc[:,[INTENT,ATTRIBUTE]].groupby(by=[INTENT,ATTRIBUTE])

        if if_get_test_data:
            test_data = self._get_test_data()
            # 分割数据集: 可能在去除测试数据后，训练数据中的某个类型数据偏少的情况
            if if_split_data:
                data = rasa_intent_dataset.data
                test_questions = test_data.loc[:, QUESTION]
                is_test_data = data.loc[:, QUESTION].isin(test_questions)
                train_data = data.loc[~is_test_data].copy()
                test_data = data.loc[is_test_data].copy()
                logger.info(f"分割数据为训练集 {train_data.shape}和测试集 {test_data.shape}")

        if if_transform_to_rasa_data:
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

    def _get_entity_data(self):
        """读取 entity data 信息"""
        entity_data_path = os.path.join(self.corpus, self.dataset, self.entity_data_filename)
        entity_data = EntityData(data_path=entity_data_path)
        return entity_data

    def _get_standard_to_extend_data_reader(self, from_labeled_data=True,
                                            from_raw_data=False,
                                            check_labeled_data=True) -> StandardToExtendQuestionDataReader:
        data_reader = StandardToExtendQuestionDataReader(output_dir=self.data_dir,
                                                         output_filename=self.standard_to_extend_data_filename)
        if not os.path.exists(data_reader.output_data_path) and from_labeled_data:
            json_path = data_reader.output_data_path
            output_dir = os.path.join(self.corpus, self.dataset)
            data_reader = StandardToExtendQuestionDataReader(
                output_dir=output_dir,
                output_filename=self.raw_standard_to_extend_data_filename,
                standard_to_extend_question_sheet_name=self.standard_to_extend_data_sheet_name)
            data_reader.get_data(filter_domains=self.filter_domains)
            if check_labeled_data:
                # TODO: 验证 已经标注数据是否符合当前的定义体系： 验证定义的 first_knowledge ,attribute, entity_type 在知识库定义内
                data = data_reader.standard_to_extend_data.data

                # 验证意图是否在faq知识的定义中
                intent_not_na_data = data.dropna(subset=[INTENT]).copy()

                # is_in_first_knowledge = intent_not_na_data.loc[:, INTENT].isin(
                #     self.knowledge_definition_data._first_knowledge)
                # intent_not_na_data.loc[~is_in_first_knowledge].loc[:,INTENT].value_counts()
                # assert intent_not_na_data.loc[~is_in_first_knowledge].shape[0] == 0

                is_in_intents = intent_not_na_data.loc[:, INTENT].isin(
                    self.knowledge_definition_data._intents)
                intent_not_na_data.loc[~is_in_intents].loc[:, INTENT].value_counts()
                assert intent_not_na_data.loc[~is_in_intents].shape[0] == 0

                # 验证 属性是否在faq知识的定义中
                attribute_not_na_data = data.dropna(subset=[ATTRIBUTE]).copy()
                is_in_attributes = attribute_not_na_data.loc[:, ATTRIBUTE].isin(
                    self.knowledge_definition_data._attributes)
                # attribute_not_na_data.loc[~is_in_attributes].loc[:,ATTRIBUTE].value_counts()
                assert attribute_not_na_data.loc[~is_in_attributes].shape[0] == 0

                # 验证 实体类型 是否在faq知识的定义中
                entity_columns = match_entity_type_columns(intent_not_na_data.columns)
                for entity_column in entity_columns:
                    entity_data = intent_not_na_data.dropna(subset=[entity_column]).copy()
                    is_in_entity_types = intent_not_na_data.loc[:, entity_column].isin(
                        self.entity_data._entity_type_zh_to_en_mapping.keys())
                    entity_data.loc[~is_in_entity_types].loc[:, entity_column].value_counts()
                    assert entity_data.loc[~is_in_entity_types].shape[0] == 0

            # 保存为 标准的json格式
            data_reader._save_data(path=json_path)
        elif not os.path.exists(data_reader.output_data_path) and from_raw_data:
            # 读取 std_library_question_keyword_norepeat.xlsx 中的 faq 的标准问 和 扩展问
            hsnlp_faq_data_reader = HSNLPFaqDataReader(output_dir=self.data_dir,
                                                       companies=['all'],
                                                       standard_question_filename='std_library_question_keyword_norepeat.xlsx',
                                                       standard_question_sheetname='Sheet1',
                                                       extend_question_filename='std_library_extend_question_norepeat.xlsx',
                                                       extend_question_sheetname='Sheet1',
                                                       standard_id_column='single_id',
                                                       scenario_column='domain')
            hsnlp_faq_data_reader.get_data(if_processed=True)

            # 读取 ivr 数据集
            # 读取金曙提供的 60+ ivr 标准问的数据
            ivr_new_raw_data_reader = IVRRawDataReaderV2(dataset='ivr_data',
                                                         # dataset_dir=dataset_dir,
                                                         subdataset_name='ivr_data_haitong_linjinshu_20211028',
                                                         intent_structure_sheetname='intent&entity-20211111',
                                                         output_dir=self.data_dir)
            ivr_new_raw_data_reader.get_data(if_processed=True)

            # 读取高峰提供的 ivr 海通数据
            ivr_standard_to_extend_data_reader = IVRStandardToExtendDataReader(
                dataset='ivr_data',
                subdataset_name='ivr_data_haitong_gaofeng_20210915',
                output_dir=self.data_dir
            )

            ivr_standard_to_extend_data_reader.get_data(if_processed=True)

            # 将数据进行融合
            data_reader = StandardToExtendQuestionDataReader(output_dir=self.data_dir)
            data_reader.standard_to_extend_data = hsnlp_faq_data_reader.combine_data(ivr_new_raw_data_reader)
            data_reader.standard_to_extend_data = data_reader.combine_data(ivr_standard_to_extend_data_reader)
            data_reader._save_data()
        else:
            data_reader.get_data()
        return data_reader

    def _get_standard_question_info(self) -> StandardQuestionInfo:
        """
        @time:  2021/12/3 14:27
        @author:wangfc
        @version:
        @description:
        获取 标准问的定义信息：一级知识点，二级知识点，意图，first_intent（大意图）等
        depreciated :  使用 _get_standard_question_knowledge()

        @params:
        @return:
        """
        if os.path.exists(self.standard_question_info_path):
            standard_question_info_dict = read_json_file(json_path=self.standard_question_info_path)
            standard_question_info = StandardQuestionInfo(standard_question_info_dict)
        else:
            standard_question_info = self._build_standard_question_info()
            dump_obj_as_json_to_file(obj=standard_question_info.data,
                                     filename=self.standard_question_info_path)
        return standard_question_info

    def _build_standard_question_info(self) -> StandardQuestionInfo:

        """
        # 读取金曙提供的 60+ ivr 标准问的数据,
        建立标准问的信息，意图，词典等嘻嘻
        """
        ivr_new_raw_data_reader = IVRRawDataReaderV2(dataset='ivr_data',
                                                     subdataset_name='ivr_data_haitong_linjinshu_20211028',
                                                     intent_structure_sheetname='intent&entity-20211111',
                                                     output_dir=self.data_dir,
                                                     if_get_function_name_information_dict=True)
        function_name_information_dict = ivr_new_raw_data_reader.get_function_name_information_dict()
        standard_question_info = StandardQuestionInfo(value=function_name_information_dict)
        return standard_question_info



    def _get_raw_chat_data(self,output_data_path, add_intent_attribute=True) -> pd.DataFrame:
        """
        将闲聊数据转换为 意图数据
        Raw data -> StandardToExtendDataset - > RasaIntentDataset
        """
        # chat_data_filename = "hsnlp_chat_data.json"
        # data_reader = StandardToExtendQuestionDataReader(output_dir=self.data_dir,
        #                                                  output_filename=chat_data_filename)
        # if not os.path.exists(data_reader.output_data_path):
        output_filename = "hsnlp_faq_chat_data.json"
        hsnlp_faq_data_reader = HSNLPFaqDataReader(dataset=self.dataset,
                                                   subdataset_name="hsnlp_faq_data",
                                                   output_dir=self.data_dir,
                                                   output_filename=output_filename)
        hsnlp_faq_chat_data = hsnlp_faq_data_reader.get_data(filter_category=['chat'])

        # 读取 zhiling 提供的 闲聊 的数据
        ivr_new_raw_data_reader = IVRRawDataReaderV2(dataset=self.dataset,
                                                     subdataset_name= 'haitong_intent_attribute_classification_data/raw_data',
                                                     ivr_navigation_filename="ivr闲聊数据.xls",
                                                     output_dir=self.data_dir)
        new_chat_data = ivr_new_raw_data_reader.get_chat_data(sheet_name='ivr闲聊')
        new_chat_data = new_chat_data.loc[:,['function_name', 'question', 'intent_category', 'source']].copy()
        new_chat_data.columns = [ 'standard_question', 'extend_question', 'intent_category', 'source']
        new_chat_data.loc[:,"scenario"] = 'haitong_ivr_chat'

        # 合并
        chat_data = pd.concat([hsnlp_faq_chat_data,new_chat_data],ignore_index=True)

        # 去掉 "怎么改" ,"怎么开"
        filter_questions = ["怎么改" ,"怎么开"]
        is_in_filter = chat_data.loc[:,STANDARD_QUESTION].isin(filter_questions)
        chat_data = chat_data.loc[~is_in_filter]
        if add_intent_attribute:
            """
            一种简单的给闲聊数据增加意图和属性的方法
            """
            chat_data.loc[:, INTENT] = 'chat'
            chat_data.loc[:, ATTRIBUTE] = 'chat'

        dataframe_to_file(data=chat_data,path=output_data_path,mode='w')

        # data_reader.get_data()
        #
        # standard_to_extend_data = data_reader.standard_to_extend_data
        #
        # chat_intent_data = self._transform_to_intent_data(standard_to_extend_data=standard_to_extend_data)
        # return chat_intent_data
        return chat_data


    def _get_new_data(self,data_type='chat',output_filename="hsnlp_chat_data.json")->RasaIntentDataset:
        """
        读取新增的数据 ： chat or 海通IVR意图和属性训练数据20220111.xlsx
        """
        # 统一转换为  StandardToExtendQuestionDataReader
        data_reader = StandardToExtendQuestionDataReader(output_dir=self.data_dir,
                                                         output_filename=output_filename)
        if not os.path.exists(data_reader.output_data_path):
            if data_type=='chat':
                data = self._get_raw_chat_data(output_data_path=data_reader.output_data_path)
            elif data_type == 'new_haitong_ivr':
                data = self._get_raw_new_haitong_ivr_data(output_data_path=data_reader.output_data_path)
            else:
                # 获取错误分析后的新增的数据
                data = self._get_new_intent_data(output_data_path=data_reader.output_data_path)

            standard_to_extend_data = StandardToExtendDataset()
            standard_to_extend_data.data = data
        else:
            # 通过 get_data 方法 读取数据
            data_reader.get_data()

            standard_to_extend_data = data_reader.standard_to_extend_data

        new_intent_data = self._transform_to_intent_data(standard_to_extend_data=standard_to_extend_data)
        return new_intent_data


    def _get_raw_new_haitong_ivr_data(self,output_data_path) -> pd.DataFrame:
        """
        增加智霖提供 ivr 生产数据加入训练数据: 海通IVR意图和属性训练数据20220111.xlsx
        new_ivr_data = "new_haitong_ivr_data.json"
        """
        new_hationg_ivr_filename = "海通IVR意图和属性训练数据20220111.xlsx"

        new_haitong_ivr_raw_data_path = os.path.join(self.dataset_dir,'haitong_intent_attribute_classification_data','raw_data',
                                                 new_hationg_ivr_filename)
        new_haitong_ivr_raw_data = dataframe_to_file(path=new_haitong_ivr_raw_data_path,mode='r',dtype=str)
        new_haitong_ivr_raw_data.columns = [EXTEND_QUESTION, STANDARD_QUESTION, INTENT, ATTRIBUTE]

        # 统计标准问的情况
        standard_question_count = new_haitong_ivr_raw_data.loc[:, STANDARD_QUESTION].value_counts()
        standard_question_set =  set(standard_question_count.index)
        new_standard_question_set = standard_question_set.difference(set(self.standard_question_knowledge_data._standard_questions))
        standard_question_count.loc[new_standard_question_set].sort_values(ascending=False)

        add_new_standard_questions = ["REITS基金权限开通","基金如何交易","股票无法交易",]
        standard_question_name_mapping = {"开通创业板": "创业板开通",
                           "重置密码":"密码重置",
                           "新股中签查询":"如何查询中签"}
                           # "怎么开":"开户咨询"

        # 筛选对应标准的数据
        is_in_standard_question = new_haitong_ivr_raw_data.loc[:,STANDARD_QUESTION].isin(
            self.standard_question_knowledge_data._standard_questions)
        is_in_standard_question_data = new_haitong_ivr_raw_data.loc[is_in_standard_question].copy()
        logger.info(f"属于定义的标准问体系的数据共 {is_in_standard_question_data.shape}")
        # 增加新的标准问，标注为其他
        is_in_new_standard_question_set = new_haitong_ivr_raw_data.loc[:,STANDARD_QUESTION].isin(add_new_standard_questions)
        add_new_standard_question_data = new_haitong_ivr_raw_data.loc[is_in_new_standard_question_set].copy()
        logger.info(f"增加新的标准问{add_new_standard_questions}，标注为其他的数据 共 {add_new_standard_question_data.shape}")
        # 部分标准问改名称
        is_in_change_standard_question_name = new_haitong_ivr_raw_data.loc[:,STANDARD_QUESTION].\
            isin(standard_question_name_mapping.keys())
        change_standard_question_data  =   new_haitong_ivr_raw_data.loc[is_in_change_standard_question_name].copy()
        new_standard_questions = change_standard_question_data.loc[:,STANDARD_QUESTION].apply(lambda standard_question: standard_question_name_mapping[standard_question])
        change_standard_question_data.loc[:,STANDARD_QUESTION] = new_standard_questions
        logger.info(f"部分标准问改名称的数据共 {change_standard_question_data.shape}")
        # 筛选标准为其他的数据（这个标注可能不准确，并且属性是按照除指定外的都是其他的原则）
        # is_intent_other=  new_haitong_ivr_raw_data.loc[:,INTENT] == DEFAULT_INTENT
        # is_intent_other_data = new_haitong_ivr_raw_data.loc[is_intent_other].copy()
        # new_haitong_ivr_raw_data = pd.concat([is_in_standard_question_data,is_intent_other_data ],ignore_index=True)\
        # .loc[:, [STANDARD_QUESTION, EXTEND_QUESTION]]

        # 合并数据
        new_haitong_ivr_raw_data = pd.concat([is_in_standard_question_data,add_new_standard_question_data,
                                              change_standard_question_data],ignore_index=True).loc[:, [STANDARD_QUESTION, EXTEND_QUESTION]]
        logger.info(f"共提取新的数据{new_haitong_ivr_raw_data.shape}")

        # 对扩展问进行预处理

        new_extend_questions = new_haitong_ivr_raw_data.loc[:,EXTEND_QUESTION].apply(remove_line_break_char)
        new_haitong_ivr_raw_data.loc[:, EXTEND_QUESTION] = new_extend_questions

        dataframe_to_file(path=output_data_path,data=new_haitong_ivr_raw_data,mode='w',index=False)
        return new_haitong_ivr_raw_data

    def _get_new_intent_data(self, output_data_path) -> pd.DataFrame:
        """
        读取新增的意图训练数据，其中含有列名 [EXTEND_QUESTION,STANDARD_QUESTION,INTENT,ATTRIBUTE]
        """
        new_data_subdir = 'new_data'
        new_data_filename = 'new_data.xlsx'
        new_data_path = os.path.join(self.dataset_dir,new_data_subdir, new_data_filename)
        new_data = dataframe_to_file(path=new_data_path,mode='r')
        select_columns = [EXTEND_QUESTION,STANDARD_QUESTION,INTENT,ATTRIBUTE]
        new_data  = new_data.loc[:,select_columns].copy()
        dataframe_to_file(path= output_data_path,mode='w',data=new_data)
        return new_data










    def _get_knowledge_definition_data(self) -> KnowledgeDefinitionData:
        """
        读取 知识图谱定义情况
        """
        faq_intent_attribute_definition_data_path = os.path.join(self.data_dir,
                                                                 self.faq_intent_attribute_definition_filename)
        faq_entity_definition_data_path = os.path.join(self.data_dir,
                                                       self.faq_entity_definition_filename)

        if not os.path.exists(faq_intent_attribute_definition_data_path) or \
                os.path.exists(faq_entity_definition_data_path):
            # 读取 excel格式的意图属性实体的定义数据
            excel_path = os.path.join(self.corpus, self.dataset, self.raw_faq_knowledge_definition_filename)
            knowledge_definition_data = KnowledgeDefinitionData(
                raw_knowledge_definition_file_path=excel_path,
                intent_attribute_definition_sheet_name=self.faq_intent_attribute_definition_sheet_name,
                entity_definition_sheet_name=self.faq_entity_definition_sheet_name
            )
            # 在指定的目录下保存为json格式
            dataframe_to_file(mode='w', path=faq_intent_attribute_definition_data_path,
                              data=knowledge_definition_data.intent_attribute_definition_table)
            dataframe_to_file(mode='w', path=faq_entity_definition_data_path,
                              data=knowledge_definition_data.entity_definition_table)
        else:
            knowledge_definition_data = KnowledgeDefinitionData(
                intent_attribute_definition_path=faq_intent_attribute_definition_data_path,
                entity_definition_path=faq_entity_definition_data_path

            )

        return knowledge_definition_data

    def _get_standard_question_knowledge_data(self) \
            -> StandardQuestionKnowledgeData:
        """
        读取 标准问的知识图谱定义情况
        """
        standard_question_knowledge_data_path = os.path.join(self.data_dir,
                                                             self.standard_question_knowledge_data_filename)
        if not os.path.exists(standard_question_knowledge_data_path):
            # 当path 为空的时候，默认 standard_question_knowledge_file_path 在 standard_to_extend_data_filename 中
            excel_path = os.path.join(self.corpus, self.dataset, self.raw_faq_knowledge_definition_filename)
            sheet_name = self.standard_question_knowledge_sheet_name
            standard_question_knowledge_data = StandardQuestionKnowledgeData(
                standard_question_knowledge_file_path=excel_path,
                sheet_name=sheet_name,
                entity_type_zh_to_en_mapping=self.entity_data._entity_type_zh_to_en_mapping,
            )
            # 在指定的目录下保存为json格式
            dataframe_to_file(path=standard_question_knowledge_data_path,
                              data=standard_question_knowledge_data._standard_question_knowledge_table,
                              )

        else:
            standard_question_knowledge_data = StandardQuestionKnowledgeData(
                standard_question_knowledge_file_path=standard_question_knowledge_data_path,
                entity_type_zh_to_en_mapping=self.entity_data._entity_type_zh_to_en_mapping)

        return standard_question_knowledge_data

    def _update_entity_data_with_label_data(self):
        standard_to_extend_data = self.standard_to_extend_data_reader.standard_to_extend_data
        # 获取标注好的 entity_type_to_values_dict
        labeled_entity_type_to_value_dict = standard_to_extend_data._get_entity_type_to_value_dict()
        self.entity_data._update(entity_type_to_value_dict=labeled_entity_type_to_value_dict)

    def _transform_to_intent_data(self, standard_to_extend_data: StandardToExtendDataset,
                                  data_type='train', if_filter_na=True, if_entity_labeling=True) -> RasaIntentDataset:
        """
        标准问相似问数据集 转换为 意图数据
        if_entity_labeling: 是否使用 entity 信息进行标注
        """

        # 获取 standard_to_extend_data中的数据
        data = standard_to_extend_data.data.copy()

        rasa_intent_dataset = RasaIntentDataset()
        columns_mapping = {EXTEND_QUESTION: QUESTION}

        new_columns = [columns_mapping.get(column)
                       if column in columns_mapping else column
                       for column in data.columns
                       ]
        data.columns = new_columns

        if rasa_intent_dataset.question_length_column not in data.columns:
            data.loc[:, rasa_intent_dataset.question_length_column] = data.loc[:,
                                                                      rasa_intent_dataset.question_column].apply(len)
        if rasa_intent_dataset.sub_intent_column not in data.columns:
            data.loc[:, rasa_intent_dataset.sub_intent_column] = ''
        if rasa_intent_dataset.response_column not in data.columns:
            data.loc[:, rasa_intent_dataset.response_column] = ''
        # 使用 standard_question_knowledge_data 替代  standard_question_info, 增加意图的信息
        # if rasa_intent_dataset.first_knowledge_key in data.columns:
        #     # 对于 intent ,如果 first_knowledge_key 已经存在，则说明已经进行人工标注，使用人工标注的数据组织为意图
        #     intents = data.apply(lambda row: get_intent_from_labeled_data(row), axis=1)
        #     data.loc[:, rasa_intent_dataset.intent_column] = intents

        for key in [rasa_intent_dataset.intent_column,
                    # rasa_intent_dataset.first_knowledge_key,
                    rasa_intent_dataset.attribute_key]:
            # rasa_intent_dataset.secondary_knowledge_key,  # 二级知识点： 意图+属性+实体信息 ,depreciated
            # rasa_intent_dataset.first_intent_key]:  # 大意图： 一级知识点， depreciated
            if key in QuestionKnowledge.KNOWLEDGE_KEYS and key not in data.columns:
                # 如果QuestionKnowledge 存在着key，根据 standard_question 获取 key 所对应的信息
                value = data.loc[:, rasa_intent_dataset.standard_question_column] \
                    .apply(lambda standard_question: self._get_standard_question_key_value(
                    standard_question, standard_question_knowledge_data=self.standard_question_knowledge_data,
                    key=key))
                data[key] = value

        # 去除意图为 空的数据 : na 的答非所问数据
        if if_filter_na:
            is_intent_na = data.loc[:, rasa_intent_dataset.intent_column].isna()
            not_na_data = data.loc[~is_intent_na].copy()
            # is_first_knowledge_na = data.loc[:, FIRST_KNOWLEDGE].isna()
            # not_first_knowledge_na_data = data.loc[~is_first_knowledge_na].copy()
            # assert not_na_data.shape[0] == not_first_knowledge_na_data.shape[0]
            is_intent_na = data.loc[:, INTENT].isna()
            not_intent_na_data = data.loc[~is_intent_na].copy()
            assert not_na_data.shape[0] == not_intent_na_data.shape[0]

            # 对于训练数据，选取固定的字段，对于测试数据，不做要求
            not_na_data = not_na_data.loc[:, rasa_intent_dataset.required_columns].copy()
        else:
            not_na_data = data.copy()

        # if if_entity_labeling:
        #     if data_type == 'train':
        #         rasa_labeled_text_format = True
        #         add_entity_extracted_columns = False
        #     else:
        #         rasa_labeled_text_format = False
        #         add_entity_extracted_columns = True
        #     # TODO : 增加 entity 的标注： 首先验证 entity 是否存在于
        #     # 对数据进行实体标注:对 question 字段进行标注，并且获取 entity的信息
        #     entity_labeler = EntityLabeler(entity_data=self.entity_data)
        #     not_na_data = entity_labeler.apply_entity_labeling(data=not_na_data,
        #                                                        text_column=rasa_intent_dataset.question_column,
        #                                                        rasa_labeled_text_format=rasa_labeled_text_format,
        #                                                        add_entity_extracted_columns=add_entity_extracted_columns)

        rasa_intent_dataset.data = not_na_data

        return rasa_intent_dataset



    def _load_intent_attribute_mapping(self) -> Dict[Text, Dict[Text, List[Text]]]:
        """
        intent_attribute_mapping 的结构为 ：  Dict[训练的意图,Dict[训练的属性,List[需要映射的属性]]]
        """
        from models.classifiers.intent_mapper import IntentAttributeMapper

        intent_attribute_mapping, train_intents, train_attributes = None, None, None
        if os.path.exists(self.intent_attribute_mapping_path):
            intent_attribute_mapping = read_json_file(self.intent_attribute_mapping_path)
            intent_to_attribute_dict, intent_mapping, attribute_mapping = IntentAttributeMapper._parse_intent_attribute_mapping(intent_attribute_mapping)
            output_intents = list(intent_to_attribute_dict.keys())
            output_attributes = []
            for intent, attributes in intent_to_attribute_dict.items():
                output_attributes.extend(attributes)
        return intent_attribute_mapping, intent_to_attribute_dict, output_intents, output_attributes

    # def _postprocess_data(self, data,if_mapping_attribute=True):
    #     """
    #     对于 海通 ivr 数据：
    #     1. 查询权限 和 开户操作 暂时不加区分，开户操作属性包含 : 开户操作/开户条件/开户材料/查询开户渠道/开户所需时间/开户费用
    #     2. 忘记帐号和密码 和 忘记密码
    #
    #     3. 转人工 对应的属性，默认为 转接人工
    #     4. 其他 对应的属性，默认为其他
    #     5. 最后输出结果的时候，对于识别为非支持的属性，我们默认为其他
    #     """
    #     # 过滤默写意图的数据
    #
    #     # TODO : 查询权限 和 开户操作 暂时不加区分
    #     # logger.warning(f"暂时不加区分属性： 查询权限 和 开户操作 ")
    #
    #     # TODO :  忘记帐号和密码 和 忘记密码
    #     # logger.warning(f"暂时不加区分属性： 忘记帐号和密码 和 忘记密码 ")
    #     # # 我们暂时把 标注为查询权限的数据 转换为 开户操作 or 我们修改规则使得其可以适配 同一个标准问下，对应多个属性，因此正则表达式多个条件
    #     # is_forget_account_and_password = data.loc[:, ATTRIBUTE] == "忘记帐号和密码"
    #     # # data.loc[is_forget_account_and_password].shape
    #     # data.loc[is_forget_account_and_password, ATTRIBUTE] = '忘记密码'
    #
    #     # intent = "手机软件操作"  # "账户密码" #"银证转账" # '开户'
    #     # is_intent = data.loc[:, INTENT] == intent
    #     # data.loc[is_intent].loc[:, ATTRIBUTE].value_counts()
    #
    #     if if_mapping_attribute:
    #         # 是否对 属性的标签进行映射
    #         # train_attributes = set()
    #         output_attribute_mapping = self.intent_attribute_mapping["attribute"]
    #         for intent, attribute_mapping in output_attribute_mapping.items():
    #             for target_attribute, source_attributes in attribute_mapping.items():
    #                 # train_attributes.add(mapping_attribute)
    #                 if source_attributes:
    #                     # 选取 attribute 存在对应关系的  attributes 的数据
    #                     is_attribute = data.loc[:, ATTRIBUTE].isin(source_attributes)
    #                     # data.loc[is_attribute].shape
    #                     data.loc[is_attribute, ATTRIBUTE] = target_attribute
    #                     # train_attributes.update(set(attributes))
    #
    #
    #     # 转人工 对应的属性，默认为 转接人工
    #     is_intent_transfer_to_human = data.loc[:, INTENT] == '转人工'
    #     # data.loc[is_intent_transfer_to_human].loc[:,ATTRIBUTE].value_counts()
    #     data.loc[is_intent_transfer_to_human, ATTRIBUTE] = '转接人工'
    #
    #     # 其他 对应的属性，默认为其他
    #     is_intent_other = data.loc[:, INTENT] == '其他'
    #     # data.loc[is_intent_other].loc[:,ATTRIBUTE].value_counts()
    #     data.loc[is_intent_other, ATTRIBUTE] = '其他'
    #
    #     # 对于非 训练 attributes,设置为 None，后面模型中不加训练
    #     # is_ivr_train_attributes = data.loc[:, ATTRIBUTE].isin(train_attributes)
    #     # data.loc[~is_ivr_train_attributes].loc[:, ATTRIBUTE].value_counts()
    #     # data.loc[~is_ivr_train_attributes, ATTRIBUTE] = None
    #
    #     # 确定每个 意图和属性都有值
    #     is_intent_na = data.loc[:, INTENT].isna()
    #     is_attribute_na = data.loc[:, ATTRIBUTE].isna()
    #     assert is_intent_na.sum() == 0
    #     assert is_attribute_na.sum() == 0
    #     return data

    def _get_standard_question_key_value(self, standard_question: Text,
                                         standard_question_knowledge_data: StandardQuestionKnowledgeData,
                                         key: Text):
        """
        根据 standard_question_info 获取 standard_question 对应 key的值
        """
        value = None
        # 获取对应的标准问知识，可能标准问还没有知识，则返回为空
        standard_question_knowledge_ls = standard_question_knowledge_data._standard_question_to_knowledge_ls.get(
            standard_question)
        if standard_question_knowledge_ls:
            if standard_question_knowledge_ls.__len__() == 1:
                standard_question_knowledge = standard_question_knowledge_ls[0]
            else:
                # 标准问可以对应多个知识，standard_question_knowledge_ls 有多个，默认选择第一个作为 knowledge
                standard_question_knowledge = standard_question_knowledge_ls[0]
            value = standard_question_knowledge.get(key)
        else:
            if key == INTENT:  # or key == FIRST_KNOWLEDGE:
                value = DEFAULT_INTENT
            if key == ATTRIBUTE:
                value = DEFAULT_ATTRIBUTE

        return value

    def _transform_to_rasa_data(self, train_data: RasaIntentDataset,
                                test_data: RasaIntentDataset = None,
                                entity_data: EntityData = None,
                                standard_question_knowledge_data: StandardQuestionKnowledgeData = None,
                                output_dir=None,
                                if_get_rule_data=False,
                                rasa_train_data_subdirname=None,
                                rasa_test_data_subdirname=None,
                                rasa_config_filename=None,
                                ):
        """
        将 RasaIntentDataset 转换为 rasa 格式的数据
        """
        hsfaq_to_rasa_dataset = HSfaqToRasaDataset(raw_data=train_data,
                                                   vocabulary=None,
                                                   entity_data=entity_data,
                                                   standard_question_knowledge_data=standard_question_knowledge_data,
                                                   output_dir=output_dir,
                                                   subdata_dir=rasa_train_data_subdirname,
                                                   rasa_config_filename=rasa_config_filename
                                                   )
        hsfaq_to_rasa_dataset.transform_to_rasa_data(if_get_rule_data=if_get_rule_data,
                                                     if_get_synonym_regex_data=False,
                                                     if_get_vocabulary_data=False)
        # 增加对应关系的输出
        intent_to_attribute_mapping = hsfaq_to_rasa_dataset.intent_to_attribute_mapping
        rasa_train_data_dir = hsfaq_to_rasa_dataset.output_data_dir
        if test_data is not None:
            test_hsfaq_to_rasa_dataset = HSfaqToRasaDataset(raw_data=test_data,
                                                            vocabulary=None,
                                                            entity_data=entity_data,
                                                            standard_question_knowledge_data=standard_question_knowledge_data,
                                                            output_dir=output_dir,
                                                            subdata_dir=rasa_test_data_subdirname,
                                                            rasa_config_filename=rasa_config_filename
                                                            )
            test_hsfaq_to_rasa_dataset.transform_to_rasa_data()
            rasa_test_data_dir = hsfaq_to_rasa_dataset.output_data_dir
        else:
            rasa_test_data_dir = None

        return rasa_train_data_dir, rasa_test_data_dir, intent_to_attribute_mapping



    def _load_rasa_yml_format_data(self) -> "TrainingData":
        """
        参考 rasa.nlu.train.train() 存在三种方式加载数据：
            training_data_endpoint
            TrainingDataImporter
            load data from file
        """

        from data_process.training_data.loading import load_data
        train_data = load_data(self.rasa_train_data_dir)
        if self.debug:
            # 过滤数据
            train_data.training_examples = train_data.training_examples[:5000]
        return train_data

    def train(self):
        from rasa.__main__ import main as rasa_main

        self._prepare_data(if_get_test_data=True)

        setup_tf_environment(gpu_memory_config=GPU_MEMORY_CONFIG)
        os.chdir(self.output_dir)
        argv = ['train',
                '--config', self.rasa_config_filename,
                '--data', self.rasa_train_data_subdirname,
                '--out', self.rasa_models_subdirname]
        sys.argv.extend(argv)
        rasa_main()

    def test_nlu(self):
        from rasa.__main__ import main as rasa_main

        setup_tf_environment(gpu_memory_config=GPU_MEMORY_CONFIG)
        os.chdir(self.output_dir)
        argv = ['test', 'nlu', '--nlu', self.rasa_test_data_subdirname, '--out', self.rasa_test_data_subdirname,
                '--config', self.rasa_config_filename,
                "--model", self.rasa_models_subdirname
                ]
        sys.argv.extend(argv)
        rasa_main()

    async def evaluate(self, agent_application=True, nlu_application=False, ):
        from apps.rasa_apps.rasa_application import RasaApplication
        from utils.tensorflow.environment import setup_tf_environment
        setup_tf_environment(gpu_memory_config=None)

        test_data = self._get_test_data()

        rasa_test_data = await self._get_rasa_test_data()

        assert test_data.__len__() == rasa_test_data.__len__()
        # # 筛选测试数据
        # text_column = "扩展问"
        # is_in_test = raw_test_data.loc[:, text_column].isin(rasa_test_data.question)
        # test_data = raw_test_data.loc[is_in_test].copy()

        # 新建 interpreter
        rasa_application = RasaApplication(taskbot_dir=self.output_dir,
                                           config_filename=self.rasa_config_filename,
                                           models_subdir=self.rasa_models_subdirname)

        # 对测试数据进行预测
        rasa_application.evaluate(test_data=test_data,
                                  test_data_dir=self.test_data_dir,
                                  # text_column=text_column,
                                  # test_output_subdir=test_output_subdir,
                                  agent_application=True,
                                  nlu_application=False,
                                  debug=False)

    def _get_test_data(self, if_filter_na=False, if_entity_labeling=False) -> pd.DataFrame:
        """
        TODO: 0. 细分属性
              1. 新增标准问的情况
              2. 新增训练数据
              3. 新增实体信息
              4. 新增意图

        0. 验证测试集的中的标准问是否当前训练的数据中
        1. 转换为 标准问和扩展问的格式
        2. 根据标准问获取意图和属性 :
        TODO: 标准问对应多个 属性 的时候，属性默认选取第一个
              标准问不在知识库中的时候，意图默认为其他
        """
        if os.path.exists(path=self.output_test_data_path):
            test_data = dataframe_to_file(path=self.output_test_data_path, mode='r')
        else:
            # 使用智霖提供的800+ 海通测试集：
            # 读取标准问和扩展问
            raw_test_data = self._get_raw_test_data()
            # 转换为 standard_to_extend_data
            test_standard_to_extend_data = StandardToExtendDataset()
            # 筛选需要的列
            selected_columns = test_standard_to_extend_data._is_columns_belong(raw_test_data.columns.tolist())
            # selected_columns = ['standard_question', 'extend_question', 'scenario','intent_category', 'source']
            test_standard_to_extend_data.data = raw_test_data.loc[:, selected_columns].copy()

            test_intent_data = self._transform_to_intent_data(test_standard_to_extend_data, data_type='test',
                                                              if_filter_na=if_filter_na,
                                                              if_entity_labeling=if_entity_labeling)

            test_data = test_intent_data.data
            # 保存数据
            dataframe_to_file(path=self.output_test_data_path, data=test_data)

            # 验证测试数据中哪些标注问
            # standard_question_count = raw_test_data.loc[:,STANDARD_QUESTION].value_counts()
            # standard_questions = standard_question_count.index.tolist()
            # difference_standard_questions =  set(standard_questions).difference(set(self.standard_question_knowledge_data._standard_questions))

        return test_data

    def _get_raw_test_data(self,data_source ='ivr_v2_test_data' ) -> pd.DataFrame:
        # 获取测试数据

        # scenario_value_counts = rasa_intent_dataset.data.loc[:, rasa_intent_dataset.scenario_column].value_counts()
        # source_value_counts = rasa_intent_dataset.data.loc[:, rasa_intent_dataset.source_column].value_counts()
        # logger.info(f'scenario_value_counts:\n{scenario_value_counts},source_value_counts:\n{source_value_counts}')
        # 将 高峰的一个评估集作为测试集，其他作为训练集
        # test_data_source = '3.6测试集.xlsx'
        # is_test_data = data.loc[:, rasa_intent_dataset.source_column] == test_data_source

        # test_filename = 'test_data_20111113_02.xlsx'
        # test_filename = 'none_text_querys.xlsx'
        # test_filename = "标准问映射意图实验数据.xlsx"
        # test_data_path = os.path.join(self.data, test_filename)
        # test_data = dataframe_to_file(path=test_data_path, mode='r', sheet_name='扩展问意图预测结果')
        # test_data = test_data.loc[:, ["三大意图", "标准问", "扩展问"]].copy()

        # 'test_data','test_data_20211220')
        # test_filenames = ["评估集4-16字符.xlsx", "ivr短句和扩展问无限制_评估集_20211216.xlsx"]

        if data_source ==  'liaozhiling_ivr_test_data':
            test_data_columns = ["评估问", "标准问", "评估意图"]
            new_columns = [EXTEND_QUESTION, STANDARD_QUESTION, "评估意图"]
        elif data_source =='ivr_v2_test_data':
            test_data_columns = ["评估问", "标准问", "意图","属性"]
            new_columns = [EXTEND_QUESTION, STANDARD_QUESTION, INTENT,ATTRIBUTE]

        test_data_path = os.path.join(self.test_data_dir, self.test_data_filename)
        if os.path.exists(test_data_path):
            test_data = dataframe_to_file(path=test_data_path, mode='r')

        # data01 = test_data_ls[0]
        # data02 =test_data_ls[1]
        # questions01 = set(data01.loc[:,'评估问'].drop_duplicates().tolist())
        # questions02 = set(data02.loc[:, '评估问'].drop_duplicates().tolist())
        # questions01.issubset(questions02)

        test_data.drop_duplicates(inplace=True)
        logger.info(f"读取数据共{test_data.shape},columns={test_data.columns}")
        # assert test_data.columns.tolist() == test_data_column
        test_data = test_data.loc[:,test_data_columns].copy()

        test_data.columns = new_columns

        def replace(cell):
            if isinstance(cell, str):
                new_cell = cell.strip().replace('\n', '')
            else:
                new_cell = cell
            return new_cell

        test_data = test_data.applymap(lambda x: replace(x))
        return test_data

    async def _get_rasa_test_data(self) -> pd.DataFrame:
        """
        读取保存在 test目录下的 yml格式的 测试数据
        """
        from rasa.shared.constants import DEFAULT_DATA_PATH
        from rasa.shared.importers.importer import TrainingDataImporter
        import rasa

        pwd = os.getcwd()
        os.chdir(self.output_dir)
        data_path = rasa.cli.utils.get_validated_path(self.rasa_test_data_subdirname, "nlu", DEFAULT_DATA_PATH)
        test_data_importer = TrainingDataImporter.load_from_dict(
            training_data_paths=[data_path]
        )
        nlu_data = await test_data_importer.get_nlu_data()
        question_ls = []
        for example in nlu_data.nlu_examples:
            question_ls.append(example.data['text'])
        data = pd.DataFrame(question_ls, columns=['question'])
        os.chdir(pwd)
        return data

    def _get_eval_results(self, test_data,
                          agent_application=True, nlu_application=False):
        pass
