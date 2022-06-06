#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/28 16:39 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/28 16:39   wangfc      1.0         None
"""
import os
from typing import Dict, Text, List, Set
import typing
from collections import Counter
import numpy as np
import pandas as pd
import re

# 对原始数据进行分析
import tqdm

from data_process.data_example import InputExample
from apps.server_requests import post_data
from utils.constants import TRAIN_DATATYPE,EVAL_DATATYPE,TEST_DATATYPE,RAW_DATATYPE
from data_process.data_processing import  get_tokens_counter
from utils.io import dump_obj_as_json_to_file, read_json_file, dataframe_to_file, is_sheet_exist
from sklearn.metrics import classification_report




class TextClassifierDataAnalysis():
    def __init__(self, data_type=None,
                 data_examples: List[InputExample] = None,label_column='label',
                 output_dir=None, data_analysis_filename = 'data_analysis.json'
                 ):
        self.data_type = data_type
        self.data_examples = data_examples
        self.label_column = label_column
        self.output_dir = output_dir
        self.data_analysis_filename = data_analysis_filename
        if self.output_dir:
            self.output_path = os.path.join(self.output_dir, f"{self.data_type}_{self.data_analysis_filename}")

        indexes, index_to_label_mapping, labels,label_counter, label_to_indexes_mapping, label_to_proportion_mapping=\
            self.get_label_info()
        self.indexes = indexes
        self.index_to_label_mapping = index_to_label_mapping
        self.labels = labels
        self.label_counter = label_counter
        self.label_to_indexes_mapping = label_to_indexes_mapping
        self.label_to_proportion_mapping = label_to_proportion_mapping


    def get_label_info(self)-> [List[int],Dict[int,Text],List[Text], typing.Counter,Dict[Text,Set], Dict[Text,float]]:
        """
        对 单标签的数据 data_examples 进行数据分析,
        return :  indexes,  index_to_label_mapping, labels, label_counter, label_to_indexes_mapping, label_to_proportion_mapping
        """
        data_size = self.data_examples.__len__()
        if isinstance(self.data_examples[0].label,list):
            # 验证 label 是单标签的
            source_count = Counter(example.source for example in self.data_examples)
            label_size_count =  Counter(example.label.__len__() for example in self.data_examples)
            assert label_size_count.__len__() ==1

            # 获取 index to label 的对应关系
            index_to_label_mapping = {index: example.__getattribute__(self.label_column)[0] for index, example in
                                      enumerate(self.data_examples)}
        else:
            # 获取 index to label 的对应关系
            index_to_label_mapping = {index: example.__getattribute__(self.label_column) for index, example in
                                      enumerate(self.data_examples)}

        # 获取所有 indexes
        indexes = [index for index, label in index_to_label_mapping.items()]
        # 获取所有的 label
        labels_ls = [label for index, label in index_to_label_mapping.items()]

        label_counter = Counter(labels_ls)

        label_to_counter_ls = sorted(label_counter.items(),key=lambda x:x[0])

        labels = label_counter.keys()

        label_to_indexes_mapping= {}
        for label_key,label_num in label_to_counter_ls:
            label_indexes = [index for index,label in index_to_label_mapping.items() if label==label_key]
            label_to_indexes_mapping.update({label_key:set(label_indexes)})

        assert np.sum([len(indexes) for label, indexes in  label_to_indexes_mapping.items()]) == data_size

        label_to_proportion_mapping = {label:count/data_size for label,count in label_to_counter_ls}

        return indexes, index_to_label_mapping, labels,label_counter, label_to_indexes_mapping, label_to_proportion_mapping



class IntentClassifierDataAnalysis():
    def __init__(self, new_version_data_dict:Dict[Text,pd.DataFrame]=None,
                 data_type=None,
                 data_examples: List[InputExample] = None,
                 output_dir=None, stopwords=None,
                 label_column='intent', text_column='sentence'
                 ):
        if new_version_data_dict:
            self.new_version_data_dict = new_version_data_dict

            self.output_dir = output_dir
            self.data_analysis_filename = 'data_analysis.json'
            if self.output_dir:
                self.output_path = os.path.join(self.output_dir, f"{self.data_analysis_filename}")
            self.label_column = label_column
            self.text_column = text_column
            self.stopwords = stopwords
            self.most_common_words_num = 1000
            self.data_analysis_report = self.get_data_analysis_report()

        if data_examples:
            # 增加对 data_examples 的统计
            self.data_type = data_type
            self.data_examples = data_examples
            self.label_column = data_examples[0].label_column
            indexes, index_to_label_mapping, labels,label_counter, label_to_index_mapping, label_to_proportion_mapping= self.get_label_info()
            self.indexes = indexes
            self.index_to_label_mapping = index_to_label_mapping
            self.labels = labels
            self.label_counter = label_counter
            self.label_to_index_mapping = label_to_index_mapping
            self.label_to_proportion_mapping = label_to_proportion_mapping


    def get_data_analysis_report(self, most_common_words_num=None) -> Dict:
        data_analysis_report_dict = {}

        if not os.path.exists(self.output_path):
            for data_type in [TRAIN_DATATYPE,EVAL_DATATYPE,TEST_DATATYPE,RAW_DATATYPE]:
                data_analysis_report = self.analysis_data(data_type=data_type,most_common_words_num=most_common_words_num)
                if data_analysis_report is not None:
                    data_analysis_report_dict.update({data_type:data_analysis_report})
            dump_obj_as_json_to_file(obj=data_analysis_report,filename= self.output_path)
        else:
            data_analysis_report_dict = read_json_file(self.output_path)
        return data_analysis_report_dict


    def analysis_data(self,data_type,most_common_words_num=None):
        """
        对 Dataframe 格式数据 进行数据分析
        """
        data = self.new_version_data_dict.get(data_type)
        if data is not None:
            # 对每个 标签的统计
            label_value_counts = data.loc[:, self.label_column].value_counts()
            # 对每句话长度的统计
            text_length_counts = pd.Series.sort_index(
                data.loc[:, self.text_column].apply(lambda x: len(x)).value_counts())
            label_to_text_length_counts_dict = {}
            label_to_text_length_counts_dict.update({'all': text_length_counts.to_dict()})

            tokens_counter = get_tokens_counter(data=data, column=self.text_column, stopwords=self.stopwords)
            most_common_words_num = most_common_words_num if most_common_words_num else self.most_common_words_num
            most_common_words = tokens_counter.most_common(most_common_words_num)

            for label in label_value_counts.index:
                label_data = data.loc[data.loc[:, self.label_column] == label].copy()
                label_data_text_length_counts = pd.Series.sort_index(
                    label_data.loc[:, self.text_column].apply(lambda x: len(x)).value_counts())
                label_to_text_length_counts_dict.update({label: label_data_text_length_counts.to_dict()})

            data_analysis_report = {}
            data_analysis_report.update({'length_analysis': label_to_text_length_counts_dict,
                                         f'{most_common_words_num}_most_common_words': most_common_words}
                                        )
            return data_analysis_report




class IVRDataAnalysis():
    def __init__(self,corpus='corpus',dataset_dir='ivr_data',
                 data_subdir='广发证券20211108',
                 statistics_excel_filename = '广发证券20211108_意图统计.xlsx',
                 question_index_column='序号',
                 guest_query_column = "客户问题",
                 standard_question_column='标准问题',
                 question_count_column = '问题频次',
                 total_question_num_column = "问题总数",
                 question_frequency_column = '问题频率',
                 if_get_predict_intent = True,
                 rasa_rest_server_url='http://10.20.33.3:5005/webhooks/rest/webhook',
                 predict_intent_column='模型预测的意图类型',
                 confidence_column= 'confidence',
                 intent_column = '意图类型',
                 intent_topK_count_column= 'topK点击数量',
                 intent_topK_frequency_column  = 'topK占比',
                 intent_frequency_column = '总体占比'
                 ):
        self.corpus =corpus
        self.dataset_dir = dataset_dir
        self.data_subdir = data_subdir
        self.statistics_excel_filename = statistics_excel_filename

        self.data_dir = os.path.join(corpus,dataset_dir,data_subdir)

        self.statistics_excel_path = os.path.join(self.data_dir,statistics_excel_filename)

        self.question_index_column = question_index_column
        self.guest_query_column = guest_query_column
        self.standard_question_column = standard_question_column
        self.question_count_column = question_count_column
        self.total_question_num_column = total_question_num_column
        self.question_count_frequency = question_frequency_column

        self.if_get_predict_intent = if_get_predict_intent
        self.rasa_rest_server_url = rasa_rest_server_url

        self.predict_intent_column = predict_intent_column
        self.confidence_column = confidence_column
        self.intent_column = intent_column

        self.intent_topK_count_column = intent_topK_count_column
        self.intent_topK_frequency_column = intent_topK_frequency_column
        self.intent_frequency_column = intent_frequency_column

        self.analysis_data()



    def analysis_data(self,debug =False,topK=200):
        raw_data = self._read_raw_data()
        if debug:
            raw_data = raw_data.iloc[:10]

        # 统计 用户问题的频率
        guest_query_statistics_df = self._get_question_statistics(raw_data,question_key=self.guest_query_column)

        # 统计标准问的评率
        standard_question_statistics_df = self._get_question_statistics(raw_data,question_key=self.standard_question_column)

        # 映射  用户问题 已经标注的 意图类型
        standard_question_statistics_df = self._prelabel_for_standard_question_statistics(self.standard_question_column,
                                                                                          guest_query_statistics_df,
                                                                                          standard_question_statistics_df,
                                                                                          topK)
        # 对标注数据进行统计
        self._labeled_data_analysis(question_key=self.guest_query_column,statistics_df=guest_query_statistics_df,topK=topK)

        self._labeled_data_analysis(question_key=self.standard_question_column, statistics_df=standard_question_statistics_df,
                                    topK=topK)

        return guest_query_statistics_df,standard_question_statistics_df

    def _labeled_data_analysis(self, question_key,statistics_df,topK):
        # path = os.path.join(self.data_dir, f"{question_key}_意图统计.xlsx")
        sheet_name = f"{question_key}_意图统计"
        intent_topK_count_column = self.intent_topK_count_column.replace('K', str(topK))
        intent_topK_frequency_column = self.intent_topK_frequency_column.replace('K', str(topK))


        total_question_num = statistics_df.iloc[0].loc[self.total_question_num_column]
        topK_statistics_df =  statistics_df.iloc[:topK]

        topK_totoal_question_num = topK_statistics_df.loc[:,self.question_count_column].sum()

        # 计算模型预测效果
        predict_labels = topK_statistics_df.loc[:,self.predict_intent_column]
        true_labels = topK_statistics_df.loc[:,self.intent_column]
        # 每个问题的占比
        sample_weights = topK_statistics_df.loc[:,self.question_count_column]/topK_totoal_question_num
        assert 0.999 <sample_weights.sum() < 1.001
        classification_report_dict = classification_report(y_true=true_labels,y_pred=predict_labels,sample_weight=sample_weights,output_dict=True)


        # 分组统计
        grouped_data = topK_statistics_df.groupby(by = [self.intent_column])
        intent_to_statistics ={}
        for intent, group in grouped_data:
            intent_count_sum = group.loc[:,self.question_count_column].sum()
            intent_statistics = {intent_topK_count_column : intent_count_sum,
                                 intent_topK_frequency_column: intent_count_sum/topK_totoal_question_num,
                                 self.intent_frequency_column: intent_count_sum/total_question_num}
            intent_predict_result = classification_report_dict.get(intent)
            intent_statistics.update(intent_predict_result)

            intent_to_statistics.update({intent:intent_statistics})

        for statistics_key in ['accuracy','macro avg','weighted avg']:
            statistics_value = classification_report_dict.get(statistics_key)
            intent_to_statistics.update({statistics_key: statistics_value})

        # 创建 dataframe
        intent_statistics_df = pd.DataFrame(intent_to_statistics).T
        # intent_statistics_df.loc[:,intent_topK_count_column] = intent_statistics_df.loc[:,intent_topK_count_column].astype(int)
        intent_statistics_df.sort_values(by=[self.intent_frequency_column],inplace=True,ascending=False)
        dataframe_to_file(mode='a',path=self.statistics_excel_path,data=intent_statistics_df,sheet_name=sheet_name)
        return intent_statistics_df




    def _read_raw_data(self):
        filenames = os.listdir(self.data_dir)
        data_ls = []
        for filename in filenames:
            path = os.path.join(self.data_dir,filename)
            file_data = dataframe_to_file(path=path,mode='r')
            data_ls.append(file_data)
        data = pd.concat(data_ls)
        data.drop_duplicates(inplace=True)
        print(f"共读取数据 {data.shape},columns={data.columns.tolist()} from {self.data_dir}")
        return data


    def _get_question_statistics(self, raw_data, question_key,if_get_predict_intent=True):
        # path = os.path.join(self.data_dir, f"{question_key}_统计.xlsx")
        sheet_name = f"{question_key}_原始数据统计"
        if is_sheet_exist(excel_path= self.statistics_excel_path,sheet_name=sheet_name):
            question_statistics_df = dataframe_to_file(self.statistics_excel_path,mode='r',sheet_name=sheet_name)
        else:
            question_statistics_ls = []
            question_counter = raw_data.loc[:, question_key].value_counts(ascending=False)
            total_query_num = raw_data.__len__()
            for  index in tqdm.trange(question_counter.__len__()):
                question = question_counter.index[index].strip()
                question_count = question_counter.iloc[index]
                question_feq = question_count / total_query_num
                if if_get_predict_intent:
                    predict_label, predict_confidence = self._get_predict_result(index=index,question=question)
                else:
                    predict_label, predict_confidence = None, None


            # question_ls = question_counter.index.tolist()
            # question_index = list(range(question_ls.__len__()))
            # question_count = question_counter.values.tolist()
            # question_feq = (question_counter / total_query_num).values.tolist()
                question_statistics_data_dict = {
                    # self.question_index_column:question_index,
                    question_key: question,
                    self.question_count_column: question_count,
                    self.total_question_num_column:total_query_num,
                    self.question_count_frequency: question_feq,
                    self.predict_intent_column: predict_label,
                    self.confidence_column: predict_confidence,
                    self.intent_column:None
                }
                question_statistics_ls.append(question_statistics_data_dict)
            question_statistics_df = pd.DataFrame(question_statistics_ls)
            dataframe_to_file(mode='a',path=self.statistics_excel_path,data=question_statistics_df,sheet_name=sheet_name)
        return question_statistics_df


    def _get_predict_result(self,index,question,
                             intent_key='intent',
                             predict_intent_key='output_intent',
                             confidence_key = 'confidence'
                             ):

        data = {
            "sender": f"test_{index}",
            "message": question
        }
        results = post_data(url=self.rasa_rest_server_url, json_data=data)

        pred_intents = [result.get(intent_key).get(predict_intent_key) for result in results]
        assert pred_intents.__len__() == 1
        pred_intent = pred_intents[0]
        confidence = [result.get(intent_key).get(confidence_key) for result in results][0]

        return pred_intent,confidence



    def _get_predict_results(self,question_key,question_ls,
                             intent_key='intent',
                             predict_intent_key='output_intent',
                             confidence_key = 'confidence'
                             ):
        results_ls_path = os.path.join(self.data_dir,f"{question_key}_predict_results.json")
        if os.path.exists(results_ls_path):
            results_ls = read_json_file(json_path=results_ls_path)
        else:
            results_ls = []
            for index in tqdm.trange(question_ls.__len__()):
                question = question_ls[index]
                if index == 123:
                    print(f"index={index},question={question}")
                data = {
                    "sender": f"test_{index}",
                    "message": question
                }
                results = post_data(url=self.rasa_rest_server_url, json_data=data)
                result_dict = dict(index= index,question=question,results=results)
                results_ls.append(result_dict)

            dump_obj_as_json_to_file(obj=results_ls,filename=results_ls_path)

        questions = []
        predict_labels = []
        predict_confidences = []
        for result_dict in results_ls:
            question= result_dict['question']
            results = result_dict['results']
            pred_intents = [result.get(intent_key).get(predict_intent_key) for result in results]
            assert pred_intents.__len__() == 1
            pred_intent = pred_intents[0]
            confidence = [result.get(intent_key).get(confidence_key) for result in results][0]
            questions.append(question)
            predict_labels.append(pred_intent)
            predict_confidences.append(confidence)

        return questions, predict_labels,predict_confidences

    def _prelabel_for_standard_question_statistics(self,question_key,
                                                   guest_query_statistics_df,
                                                   standard_question_statistics_df,
                                                   topK):
        guest_query_intent_labeled_num = guest_query_statistics_df.loc[:, self.intent_column].dropna().__len__()
        standard_question_intent_labeled_num = standard_question_statistics_df.loc[:,
                                               self.intent_column].dropna().__len__()
        if guest_query_intent_labeled_num > topK and standard_question_intent_labeled_num==0:
            query_to_intent_dict = {}
            for i in range(guest_query_intent_labeled_num):
                query = guest_query_statistics_df.iloc[i].loc[self.guest_query_column]
                query = question_match(question=query)
                intent = guest_query_statistics_df.iloc[i].loc[self.intent_column]
                query_to_intent_dict.update({query: intent})

            standard_question_intents = standard_question_statistics_df.loc[:, self.standard_question_column] \
                .apply(lambda x: query_to_intent_dict.get(question_match(x)))
            standard_question_statistics_df.loc[:, self.intent_column] = standard_question_intents

            # path = os.path.join(self.data_dir, f"{self.standard_question_column}_统计.xlsx")
            sheet_name = f"{question_key}_原始数据统计"
            dataframe_to_file(mode='a',path= self.statistics_excel_path, data=standard_question_statistics_df,sheet_name=sheet_name)
        return standard_question_statistics_df


def question_match(question:Text):
    matched = re.match(pattern=r'(.*)[?？!！。.]$', string=question)
    if matched:
        question = matched.groups()[0]
    return question