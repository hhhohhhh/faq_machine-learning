#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/8/2 10:45 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/2 10:45   wangfc      1.0         None
"""
import os
from collections import defaultdict
from pathlib import Path
from typing import Text, Union, List,Dict
from enum import unique, Enum
import pandas as pd
import re
import tqdm
from models.trie_tree import Entity


from data_process.data_example import InputExample
from data_process.dataset.hsnlp_faq_knowledge_dataset  import  ATTRIBUTE, ENTITY_TYPE, ENTITY_VALUE # FIRST_KNOWLEDGE
from data_process.dataset.hsnlp_faq_knowledge_dataset import  match_entity_type_columns
from tokenizations.tokenization import preprocess_text

from utils.constants import STANDARD_QUESTION, EXTEND_QUESTION, SOURCE, SCENARIO, CHAT, FAQ, \
    INTENT_CATEGORY, INTENT
from utils.io import dump_obj_as_json_to_file, dataframe_to_file

import logging
logger = logging.getLogger(__name__)


@unique
class FAQQuestionType(Enum):
    is_standard = '1'
    is_extend = '0'


class FAQSentenceExample(InputExample):
    def __init__(self, guid=None, id: str = None,
                 sentence: str = None, text_b=None, intent: str = None,
                 intent_id=None,
                 is_standard=FAQQuestionType.is_standard, standard_id=None):
        self.guid = guid
        # id 号
        self.id = id
        # 问句的文本
        self.text_a = sentence
        self.text_b = text_b
        # 意图
        self.label = intent
        # 意图的编码
        self.label_id = intent_id
        # 是否是标准问
        self.is_standard = is_standard
        # 如果是扩展问，对应的 标准问id
        if standard_id is None:
            self.standard_id = id
        # 编辑距离
        self.edit_distance = None

    def __repr__(self):
        return f"class={self.__class__.__name__}," \
               f"guid_id={self.guid},id={self.id}," \
               f"text_a={self.text_a}," \
               f"label={self.label},label_id={self.label_id}" \
               f"edit_distance={self.edit_distance}"



class FAQIntentAttributeExample():
    """
    faq 中意图分类的数据
    """
    def __init__(self,id:str=None, question:Text=None,standard_question:Text=None,
                 intent:Text=None,attribute:Text=None,entities:List[Entity]=None,
                 source:Text=None,
                 guid: int=None,
                 predict_intent =None,predict_intent_confidence=None,
                 predict_attribute= None, predict_attribute_confidence=None,
                 **kwargs
                 ):

        self.id = id
        self.question = question
        self.standard_question = standard_question
        self.intent = intent
        self.attribute = attribute
        self.entities = entities
        self.source = source

        self.guid = guid
        self.predict_intent = predict_intent
        self.predict_intent_confidence = predict_intent_confidence
        self.predict_attribute = predict_attribute
        self.predict_attribute_confidence = predict_attribute_confidence
        self.kwargs =kwargs

    def _update(self,key_value_mapping:Dict[Text,Union[Text,float]]):
        for key,value in key_value_mapping.items():
            if hasattr(self, key):
                setattr(self,key,value)


class Dataset():
    def __init__(self):
        self._data: pd.DataFrame = None

    @property
    def required_columns(self):
        # @depreciate : output_columns
        raise NotImplementedError


    @property
    def data(self):
        return self._data

    @data.setter
    def data(self,value:pd.DataFrame):
        """
        验证 data的列名满足要求
        """
        if value is not None:
            assert set(self.required_columns).issubset(set(value.columns.tolist()))
            self._data = value


class StandardToExtendDataset(Dataset):
    def __init__(self,data=None):
        self.standard_question_column = STANDARD_QUESTION
        self.extend_question_column = EXTEND_QUESTION
        self.scenario_column =  SCENARIO
        self.intent_category_column = INTENT_CATEGORY
        self.source_column = SOURCE

        # assert data.columns.tolist() == self.output_columns
        # self.data = data
        # 增加标注的时候使用的列明
        self.intent = INTENT
        self.attribute = ATTRIBUTE
        self.entity_type = ENTITY_TYPE
        self.entity_value = ENTITY_VALUE

        self.entity_type_pattern = f"{self.entity_type}_(\d)"
        self.entity_value_pattern = f"{self.entity_value}_(\d)"

        self.data = data


    @property
    def required_columns(self):
        #  意图不作为要求的列: 只要求 标准问和 扩展问
        return [self.standard_question_column,self.extend_question_column]
                # self.scenario_column,self.source_column]

    @property
    def columns(self):
        return [self.standard_question_column,self.extend_question_column,
                self.scenario_column,self.source_column,
                self.intent_category_column,
                self.intent,self.attribute,
                self.entity_type]

    def _is_match_entity_column(self,column):
        if re.match(pattern=self.entity_type_pattern,string=column ) or \
            re.match(pattern=self.entity_value_pattern,string =column):
            return True
        return False

    def _is_columns_belong(self,columns):
        belonged_columns = []
        for column in columns:
            if column in self.columns or self._is_match_entity_column(column):
                belonged_columns.append(column)
        return belonged_columns

    def _get_entity_type_to_value_dict(self):
        # 获取 实体标注的列名
        entity_type_columns = match_entity_type_columns(self.data.columns)
        entity_type_and_value_tuple_ls = []
        for entity_type_column in entity_type_columns:
            matched = re.match(pattern=self.entity_type_pattern,string=entity_type_column)
            index = matched.groups()[0]
            entity_value_column = f"{self.entity_value}_{index}"
            entity_type_and_value_tuple_ls.append((entity_type_column,entity_value_column))

        data = self.data.astype(str)

        entity_type_and_value_data_ls= []
        for entity_type_and_value_columns in  entity_type_and_value_tuple_ls:
            entity_type_and_value_data = data.loc[:,entity_type_and_value_columns].dropna().copy()
            entity_type_and_value_data.columns = [self.entity_type,self.entity_value]
            entity_type_and_value_data_ls.append(entity_type_and_value_data)
        entity_type_and_value_data = pd.concat(entity_type_and_value_data_ls)


        def collect_entity_values(group,entity_value_column) -> List[Text]:
            return sorted(set(group.loc[:,entity_value_column].drop_duplicates().tolist()),key=lambda x: len(x))
        # 按照 entity_type 进行分组
        grouped = entity_type_and_value_data.groupby(self.entity_type)\
            .apply(lambda group:collect_entity_values(group,entity_value_column=self.entity_value))
        # 获取 字典
        entity_type_to_values_dict = {}
        for index in grouped.index:
            entity_values = grouped.loc[index]
            new_entity_values = []
            for entity_value in entity_values:
                entity_value = entity_value.strip()
                entity_value_split = re.split(pattern=',|，', string=entity_value)
                for entity in entity_value_split:
                    if entity !='nan':
                        new_entity_values.append(entity)

            new_entity_values = sorted(set(new_entity_values), key=lambda x: len(x))
            entity_type_to_values_dict.update({index: new_entity_values})
        return entity_type_to_values_dict






class HSNLPFaqDataReader():
    """
    gaofeng 给的 FAQ 数据
    """

    def __init__(self, corpus='corpus', dataset='hsnlp_faq_data',
                 subdataset_name=None,
                 output_dir=None,
                 output_filename='hsnlp_faq_standard_to_extend_question_data.json',
                 companies=['国元', '一创', '万联', '中邮', '国盛', '湘财'],
                 standard_question_filename='hsnlp_standard_question.xlsx',
                 standard_question_sheetname="hsnlp_standard_question",
                 extend_question_filename="hsnlp_extend_question.xlsx",
                 extend_question_sheetname="hsnlp_extend_question",
                 question_id_column="question_id",
                 question_column='question',
                 scenario_column=SCENARIO,
                 standard_id_column='standard_id',
                 is_filter=False,
                 extend_questions_num=7, **kwargs
                 ):
        # super(HSNLPFaqDataReader, self).__init__(corpus=corpus, dataset=dataset,
        #                                          output_dir=output_dir, output_filename=output_filename,
        #                                          **kwargs)
        self.corpus = corpus
        self.dataset = dataset
        self.subdataset_name = subdataset_name
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.output_data_path = os.path.join(self.output_dir, self.output_filename)


        self.companies = companies
        self.standard_question_filename = standard_question_filename
        self.extend_question_filename = extend_question_filename
        self.standard_question_sheetname = standard_question_sheetname
        self.extend_question_sheetname = extend_question_sheetname

        self.standard_question_column = STANDARD_QUESTION
        self.extend_question_column = EXTEND_QUESTION
        self.intent_category_column = INTENT_CATEGORY
        self.source_column = SOURCE


        self.data_dir = os.path.join(self.corpus, self.dataset)
        if subdataset_name:
            self.data_dir = os.path.join(self.data_dir,subdataset_name)

        # self.output_data_dir = os.path.join('examples', self.dataset_subdir)
        # self.output_rasa_nlu_data_path = os.path.join('examples', self.dataset_subdir, 'data', 'nlu.yml')

        self.question_id_column = question_id_column
        self.question_column = question_column
        self.scenario_column = scenario_column
        self.standard_id_column = standard_id_column

        self.standard_question_column = STANDARD_QUESTION
        self.extend_question_column = EXTEND_QUESTION
        self.intent_category_column = INTENT_CATEGORY
        self.source_column = SOURCE

        self.filter_scenarios = None  # ['gy']

        # 是否过滤 扩展问大于extend_questions_num的数据
        self.is_filter = is_filter
        self.extend_questions_num = extend_questions_num

        # self.data = self.get_data()
        # self.standard_to_extend_question_dict = self._read_company_raw_data(is_filter=is_filter,extend_questions_num=extend_questions_num)
        # 将 standard_to_extend_question_dict 转换为 rasa的数据格式
        # rasa_nlu_data = transform_to_rasa_nlu_data(
        #     data=self.standard_to_extend_question_dict,data_format='standard_to_extend_question_dict',
        #     path = self.output_rasa_nlu_data_path,topK=5
        # )

    def get_data(self,filter_domains=None,filter_category=None) -> StandardToExtendDataset:
        if os.path.exists(self.output_data_path):
            data = self._load_data(filter_domains=filter_domains)
        else:
            data= self._read_data(filter_category=filter_category)
            # dataframe_to_file(path=self.output_data_path, data=data, mode='w')
        return data


    def _read_data(self,filter_category) -> pd.DataFrame:
        company_data_ls = []
        for company in self.companies:
            company_data = self._read_company_raw_data(company=company)
            company_data_ls.append(company_data)
        data = pd.concat(company_data_ls).drop_duplicates(subset=[self.standard_question_column,self.intent_category_column,self.source_column])

        if filter_category:
            is_in_category = data.loc[:, self.intent_category_column].isin(filter_category)
            data = data.loc[is_in_category].copy()

        data = self._add_standard_to_extend_data(data=data)
        data.reset_index(inplace=True, drop=True)
        return data

    def _load_data(self, filter_domains: List[Text] = None) -> StandardToExtendDataset:
        data = dataframe_to_file(path=self.output_data_path, mode='r', dtype='string')

        if filter_domains:
            data = data.loc[data.loc[:, SCENARIO].isin(filter_domains)].copy()

        return data



    def _read_company_raw_data(self, company='国元', is_filter=False, extend_questions_num=None) -> pd.DataFrame:
        """
        读取标准问和扩展问，
        """
        self.company_data_dir = os.path.join(self.data_dir, company)
        standard_question_file_path = os.path.join(self.company_data_dir, self.standard_question_filename)
        extend_question_file_path = os.path.join(self.company_data_dir, self.extend_question_filename)
        # 读取标准问
        standard_questions_df = self._read_raw_data(path=standard_question_file_path,
                                                    sheet_name=self.standard_question_sheetname)
        # 读取扩展问
        extend_questions_df = self._read_raw_data(path=extend_question_file_path,
                                                  sheet_name=self.extend_question_sheetname,
                                                  data_category='扩展问')

        # # 确定 扩展问中  standard_id 都在 标注问中
        # if self.standard_question_filename == 'std_library_question_keyword_norepeat.xlsx':
        #     # std_library_question_keyword_norepeat.xlsx 中标注问的列为 standard_id_column
        #     standard_id_column = self.standard_id_column
        #     # domain 转换为标准的 SCENARIO
        #     scenario_column = SCENARIO
        standard_id_set = set(standard_questions_df.loc[:, self.question_id_column].tolist())
        extend_standard_id_set = set(extend_questions_df.loc[:, self.standard_id_column].tolist())
        if not extend_standard_id_set.issubset(standard_id_set):
            difference_id_set = extend_standard_id_set.difference(standard_id_set)
            logger.warning(f"{company}扩展问的id不是标准问的子集！"
                           f"difference_id_set/extend_standard_id_set="
                           f"{difference_id_set.__len__()}/{extend_standard_id_set.__len__()}")

        merged_df = pd.merge(left=standard_questions_df, right=extend_questions_df,
                             left_on=self.question_id_column, right_on=self.standard_id_column,
                             how='left')

        select_scenario_column = f"{self.scenario_column}_x"
        select_columns = ['question_x', 'question_y', select_scenario_column]
        new_columns = [self.standard_question_column, self.extend_question_column, self.scenario_column]
        select_df = merged_df.loc[:, select_columns]
        select_df.columns = new_columns

        scenario_value_counts = select_df.loc[:, self.scenario_column].value_counts()
        print(f"{company} scenario_value_counts :\n{scenario_value_counts}")

        categories = select_df.loc[:, self.scenario_column].apply(self._change_scenario_to_intent_category)

        source = f'{company}_faq_questions'
        select_df.loc[:, self.intent_category_column] = categories
        select_df.loc[:, self.source_column] = source
        return select_df

    def _change_scenario_to_intent_category(self, scenario: Text):
        if scenario.endswith('_chat'):
            return CHAT
        else:
            return FAQ

    def _read_company_raw_data_to_dict(self, company='国元', is_filter=False, extend_questions_num=None):
        """
        读取标准问和扩展问，
        """
        self.company_data_dir = os.path.join(self.data_dir, company)
        standard_question_file_path = os.path.join(self.company_data_dir, self.standard_question_filename)
        extend_question_file_path = os.path.join(self.company_data_dir, self.extend_question_filename)
        # 读取标准问
        standard_questions_df = self._read_raw_data(path=standard_question_file_path,
                                                    sheet_name=self.standard_question_sheetname)
        # 读取扩展问
        extend_questions_df = self._read_raw_data(path=extend_question_file_path,
                                                  sheet_name=self.extend_question_sheetname,
                                                  data_category='扩展问')

        # 确定 扩展问中  standard_id 都在 标注问中
        standard_id_set = set(standard_questions_df.loc[:, self.question_id_column].tolist())
        extend_standard_id_set = set(extend_questions_df.loc[:, self.standard_id_column].tolist())
        extend_standard_id_set.issubset(standard_id_set)
        # 转换为 标注问 对应 扩展问的形式
        """
        { standard_id: 
            {"standard_questions":[{'question_id':standard_question}],
             "extend_questions":[{"question_id":extend_question,'standard_id':standard_id}]} }
        """
        # 初始化标准问与扩展问对应的字典
        standard_to_extend_question_dict = defaultdict(dict)
        # 使用 标注问 更新字典
        standard_to_extend_question_dict = self._update_standard_to_extend_question_dict(
            standard_to_extend_question_dict,
            questions_df=standard_questions_df)
        # 使用 扩展问 更新字典
        standard_to_extend_question_dict = self._update_standard_to_extend_question_dict(
            standard_to_extend_question_dict,
            questions_df=extend_questions_df,
            question_category="extend_questions")
        # 过滤 满足 extend_questions_num 条件的数据
        if is_filter and extend_questions_num:
            standard_to_extend_question_dict = self._filter_data(standard_to_extend_question_dict,
                                                                 extend_questions_num)
        return standard_to_extend_question_dict

    def _read_raw_data(self, path, sheet_name, data_category='标准问'):
        # 读取 excel 格式的数据
        questions_df = dataframe_to_file(path=path, mode='r', sheet_name=sheet_name, dtype='string')
        # questions_df = pd.read_excel(path, sheet_name=sheet_name).astype(str)
        # logger.info(f"读取 {data_category}数据共={questions_df.shape},columns={questions_df.columns.tolist()}\n"
        #             f"{questions_df.loc[:, self.scenario_column].value_counts()}")

        # 过滤掉 gy_chat
        if self.filter_scenarios:
            filtered_questions_df = questions_df[
                questions_df.loc[:, self.scenario_column].isin(self.filter_scenarios)].copy()
            logger.info(f"过滤数据，只保留 scenario={self.filter_scenarios}的数据共 {filtered_questions_df.shape}")
        else:
            filtered_questions_df = questions_df.copy()
        # 去除重复
        filtered_questions_df.dropna(inplace=True,subset=[self.question_id_column,self.question_column],)
        preprocess_questions = filtered_questions_df.loc[:, self.question_column].apply(lambda x: preprocess_text(x))
        filtered_questions_df.loc[:, self.question_column] = preprocess_questions
        filtered_questions_df.drop_duplicates(subset=[self.question_column], inplace=True)
        # logger.info(f"去除重复后数据共 {filtered_questions_df.shape}")
        return filtered_questions_df


    def _add_standard_to_extend_data(self,data):
        # 增加标准问到扩展问中
        standard_question_df = data.copy().drop_duplicates(subset=[self.standard_question_column])
        standard_question_df.loc[:,self.extend_question_column] = standard_question_df.loc[:,self.standard_question_column]
        add_standard_question_df = pd.concat([data,standard_question_df])
        # 去除扩展问为空的情况
        add_standard_question_df.dropna(subset=[self.extend_question_column],inplace=True)
        return add_standard_question_df


    def _update_standard_to_extend_question_dict(self, standard_to_extend_question_dict, questions_df,
                                                 question_category="standard_questions"):
        for index in tqdm.trange(questions_df.__len__()):
            # 遍历每条数据
            row = questions_df.iloc[index]
            # 对于标准问 standard_id
            if question_category == "standard_questions":
                standard_id = row.loc[self.question_id_column]
            # 对于扩展问的 standard_id
            elif question_category == 'extend_questions':
                standard_id = row.loc[self.standard_id_column]

            # 获取 standard_id 对应的问题字典
            standard_id_dict = standard_to_extend_question_dict[standard_id]
            # 获取 对应问题类型的 列表
            standard_questions = standard_id_dict.get(question_category, [])
            standard_questions.append(row.to_dict())
            # 更新 standard_id_dict
            standard_id_dict.update({question_category: standard_questions})
        return standard_to_extend_question_dict

    def _filter_data(self, standard_to_extend_question_dict, extend_questions_num=7):
        """
        过滤数据较少的标准问和扩展问
        """
        filter_standard_to_extend_question_dict = {}
        for standard_id, standard_id_dict in standard_to_extend_question_dict.items():
            standard_questions_ls = standard_id_dict.get('standard_questions')
            extend_questions_ls = standard_id_dict.get('extend_questions')
            if (standard_questions_ls is not None and standard_questions_ls.__len__() > 0) and \
                    (extend_questions_ls is not None and extend_questions_ls.__len__() > extend_questions_num - 1):
                filter_standard_to_extend_question_dict.update({standard_id:
                                                                    {"standard_questions": standard_questions_ls,
                                                                     "extend_questions": extend_questions_ls}})
        logger.info(f"标准问与扩展问(>={extend_questions_num})的对应关系共 {filter_standard_to_extend_question_dict.__len__()}")
        output_path = os.path.join('examples', self.dataset, "standard_to_extend_question.json")
        dump_obj_as_json_to_file(obj=filter_standard_to_extend_question_dict, filename=output_path)
        return filter_standard_to_extend_question_dict





class StandardToExtendQuestionDataReader(HSNLPFaqDataReader):
    """
    读取 标准问 与 扩展问的对应关系
    """

    def __init__(self, output_filename='standard_to_extend_question_data.json',
                 standard_to_extend_question_sheet_name ="standard_to_extend_data" ,
                 *args,
                 **kwargs):
        super(StandardToExtendQuestionDataReader,self).__init__(output_filename=output_filename,
                                                                *args,**kwargs)
        self.standard_to_extend_question_sheet_name = standard_to_extend_question_sheet_name





    def get_data(self,filter_domains=None,if_processed=False) -> StandardToExtendDataset:
        if os.path.exists(self.output_data_path):
            self.standard_to_extend_data = self._load_data(filter_domains=filter_domains)
        else:
            data= self._read_raw_data()
            if if_processed:
                # 增加标准问到扩展问当中
                data = self._add_standard_to_extend_data(data=data)
            self.standard_to_extend_data = self._to_standard_to_extend_data(data)

            self._save_data(data=self.standard_to_extend_data.data,path=self.output_data_path)
        return self.standard_to_extend_data


    def _load_data(self,filter_domains:List[Text]=None,filter_fund_data=True) -> StandardToExtendDataset:
        data = dataframe_to_file(path=self.output_data_path,
                                 sheet_name=self.standard_to_extend_question_sheet_name,
                                 mode='r',dtype='string')

        if filter_domains:
            data = data.loc[data.loc[:,SCENARIO].isin(filter_domains)].copy()

        # haitong_ivr 过滤基金数据
        if filter_fund_data and INTENT in data.columns:
            data = data.loc[data.loc[:,INTENT]!="基金"].copy()

        # if if_processed:
        #     data = self._add_standard_to_extend_data(data)
        # 验证
        standard_to_extend_data = StandardToExtendDataset()
        # 筛选需要的列
        selected_columns = standard_to_extend_data._is_columns_belong(data.columns.tolist())
        # selected_columns = ['standard_question', 'extend_question', 'scenario','intent_category', 'source']
        standard_to_extend_data.data = data.loc[:,selected_columns].copy()
        # assert standard_to_extend_data.output_columns == data.columns.tolist()
        return standard_to_extend_data

    def _save_data(self,data:pd.DataFrame=None,path:Union[Text,Path]=None)-> None:
        if data is None:
            data  = self.standard_to_extend_data.data
        if path is None:
            path = self.output_data_path
        dataframe_to_file(path=path,data=data,mode='w')

    # def _build_data(self)-> pd.DataFrame:
    #     raise NotImplementedError

    def _to_standard_to_extend_data(self,data:pd.DataFrame) -> StandardToExtendDataset:
        standard_to_extend_dataset = StandardToExtendDataset()
        standard_to_extend_dataset.data = data
        return standard_to_extend_dataset

    def _to_format_data(self) -> pd.DataFrame:
        """
        转换为统一格式的数据： standard_question, extend_question, source
        """
        raise NotImplementedError



    def combine_data(self, other):
        """
        对两个数据集进行合并
        """
        assert isinstance(other, StandardToExtendQuestionDataReader)
        assert hasattr(self, 'standard_to_extend_data')
        assert hasattr(other,'standard_to_extend_data')
        standard_to_extend_data = pd.concat([self.standard_to_extend_data.data,
                                             other.__getattribute__('standard_to_extend_data').data])
        standard_to_extend_data.drop_duplicates(inplace=True,subset=[self.standard_question_column,self.extend_question_column])
        standard_to_extend_data = self._to_standard_to_extend_data(standard_to_extend_data)
        return standard_to_extend_data



