#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/23 9:47 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/23 9:47   wangfc      1.0         智能问答意图分类模型的评估类
"""
import os
from typing import List, Dict, Text, Union,TYPE_CHECKING

import tqdm
import pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path
import json

import typing
if TYPE_CHECKING:
    from apps.intent_attribute_classifier_apps.intent_attribute_classifier import IntentAttributeClassifierTfv1


from apps.server_requests import post_data
from data_process.dataset.hsnlp_faq_knowledge_dataset import ATTRIBUTE # FIRST_KNOWLEDGE
from utils.constants import QUESTION, STANDARD_QUESTION, INTENT
from utils.io import get_file_stem, dataframe_to_file
import logging
logger = logging.getLogger(__name__)

class ModelEvaluator(object):
    """智能问答意图分类模型的评估类：
    对训练好的模型，使用测试数据集进行评估:
        1. 使用训练的模型实例 model进行初始化       TODO: 实现 model
        2. 调用模型的 model.predict()函数进行预测  TODO: 实现 model的 predict 接口进行预测
        3. 对输出的结果进行评估

    Attributes:
        model： 训练好的模型
        test_data_keys: 输入数据包含的 key: {'id','sentence','intent'}
        pred_data_keys:  输出结果包含的 key: {'id','intent','probability'}
        intents_set:  意图的类型 :{'advice','faq'}

    测试数据格式：test_data_demo.json
        [{"id":"001","sentence":"请帮我开户","intent": "faq"},
            {"id":"002","sentence":"恒生电子收益怎么样？", "intent": "advice"}]

    For example：
    >>> class ModelDemo():
    ...     def predict(self,input_data):
    ...         return [{'id':x['id'],'intent':'faq','probability':0.9} for x in input_data]
    >>> model_demo = ModelDemo()
    >>> model_evaluator = ModelEvaluator(model = model_demo)
    >>> test_data_path = 'test_data_demo.json'
    >>> model_evaluator.evaluate(test_data_path = test_data_path)
    0.3333333333333333
    """

    def __init__(self, model,
                 test_data_keys={'id', 'sentence', 'intent'},
                 pred_data_keys={'id', 'intent', 'probability'},
                 intents_set={'advice', 'faq'}):
        """ 初始化训练好的模型 """
        self.model = model
        self.test_data_keys = test_data_keys
        self.pred_data_keys = pred_data_keys
        self.intents_set = intents_set

    def evaluate(self, test_data_path: Union[Text, Path]) -> float:
        """ 测试接口函数
        输入测试集，输出模型预测的意图类型
        Args:
            test_data_path： 测试数据路径
        Returns:
            macro f1-score:


        Raises:
            TypeError: 如果输入和输出错误会抛出错误
        """
        # 读取测试数据 ：列表型的测试数据，其中每个元素为 key为 id，sentence,intent的字典
        with open(test_data_path, mode='r', encoding='utf8') as f:
            test_data = json.load(f)

        # 验证测试数据的格式
        self.verify_test_data(test_data)

        # 使用 model的 predict接口进行预测: 列表型的预测结果，其中每个元素为 key为 id，intent, probability 的字典
        pred_output = self.model.predict(test_data)

        # 验证预测结果的格式
        self.verify_pred_output(test_data, pred_output)

        y_pred = []
        y_true = []
        for pred, label in zip(pred_output, test_data):
            y_pred.append(pred.get('intent'))
            y_true.append(label.get('intent'))
        # report= classification_report(y_true=y_true,y_pred=y_pred)
        # print(f"classification_report:\n{report}")
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

        return f1

    def verify_test_data(self, test_data: List[Dict]):
        try:
            assert isinstance(test_data, list)
            for x in test_data:
                assert set(x.keys()) == self.test_data_keys
        except Exception as e:
            raise TypeError(f"test_data格式不正确：{e}")

    def verify_pred_output(self, test_data: List[Dict], pred_output: List[Dict]):
        try:
            assert isinstance(pred_output, list)
            assert test_data.__len__() == pred_output.__len__()
            for x, y in zip(test_data, pred_output):
                assert x.get('id') == y.get('id')
                assert self.pred_data_keys.issubset(set(y.keys()))
        except Exception as e:
            raise TypeError(f"pred_output格式不正确：{e}")


class IntentAttributeEvaluator():
    def __init__(self,
                 raw_test_data_filename,
                 test_data_dir, test_data_filename,
                 output_dir=None, test_result_filename=None,
                 important_intent_list=["开户", "销户", "银证转账", "账户密码", "手机软件操作", "转人工"],
                 intent_attribute_classifier: "IntentAttributeClassifierTfv1" = None,
                 support_intent_list=None,
                 train_intent_to_attribute_mapping=None,
                 intent_to_attribute_dict=None,

                 url="http://10.20.33.3:8016/hsnlp/faq_intent_attribute_classifier",

                 test_result_file_suffix='result'):
        self.raw_test_data_filename = raw_test_data_filename
        self.test_data_dir = test_data_dir
        self.test_data_filename = test_data_filename
        if output_dir is None:
            output_dir = test_data_dir
        self.output_dir = output_dir
        self.test_data_filename = test_data_filename
        self.test_data_path = os.path.join(test_data_dir, test_data_filename)
        if test_result_filename is None:
            stem = get_file_stem(test_data_filename)
            test_result_filename = f"{stem}_{test_result_file_suffix}.xlsx"
        self.test_result_filename = test_result_filename
        self.test_data_result_path = os.path.join(test_data_dir, test_result_filename)
        self.url = url

        self.important_intent_list = important_intent_list
        # tfv1 任务时候
        self.support_intent_list = support_intent_list
        self.intent_attribute_classifier = intent_attribute_classifier

        # tfv2 任务的时候
        # 训练数据中的意图和属性的映射关系
        self.train_intent_to_attribute_mapping = train_intent_to_attribute_mapping
        # 人工设置的属性的映射，重点测试的属性
        self.intent_to_attribute_dict = intent_to_attribute_dict
        if intent_to_attribute_dict:
            self.intent_to_attribute_dict = intent_to_attribute_dict
        else:
            self.intent_to_attribute_dict = self.intent_attribute_classifier.att_function.ATT_CLASS

        if self.train_intent_to_attribute_mapping:
            self.support_intent_list = list(self.train_intent_to_attribute_mapping.keys())
        elif support_intent_list is None:
            raise ValueError("support_intent_list 不能为空")
        else:
            self.support_intent_list = support_intent_list


        self.sorted_support_intent_list = important_intent_list + list(
            set(self.support_intent_list).difference(self.important_intent_list))

    def _load_test_data(self):
        pass

    def _get_predict_result_data(self, from_standard_to_extend_question_dataset=True):
        if os.path.exists(self.test_data_result_path):
            test_result_data_df = dataframe_to_file(mode='r', path=self.test_data_result_path)
        else:
            test_data = dataframe_to_file(mode='r', path=self.test_data_path)
            test_data.reset_index(inplace=True, drop=True)
            select_columns = [QUESTION, STANDARD_QUESTION, INTENT, ATTRIBUTE]
            test_data = test_data.loc[:, select_columns].copy()
            data_ls = []
            logger.info(f"访问服务 {self.url}")
            for i in tqdm.tqdm(range(test_data.__len__()), "测试数据进行预测"):
                data = test_data.iloc[i].to_dict()
                # example = FAQIntentAttributeExample(**row.to_dict())
                question = data['question']

                json_data = {"message": question}
                output = post_data(url=self.url, json_data=json_data)
                preprocessed_text = output['preprocessed_text']
                pred_result = output['pred']

                data.update({"preprocessed_text": preprocessed_text})
                data.update(pred_result)
                data_ls.append(data)

            # 保存测试数据
            test_result_data_df = pd.DataFrame(data_ls)

            dataframe_to_file(mode='a', path=self.test_data_result_path, data=test_result_data_df,index=False)
        return test_result_data_df

    def evaluate(self, threshold=0.9,
                 do_intent_evaluate=True, do_attribute_evaluate=True,
                 intent_column='intent', predict_intent_column='predict_intent',
                 predict_intent_confidence_column='predict_intent_confidence',
                 default_intent='其他', default_attribute='其他', default_confidence=1.0,
                 attribute_column='attribute', predict_attribute_column='predict_attribute',
                 predict_attribute_confidence_column="predict_attribute_confidence"
                 ):
        from evaluation.classification_evaluate import get_classification_report

        test_data_df = self._get_predict_result_data()

        if default_attribute and default_confidence:
            test_data_df.loc[:, intent_column] = test_data_df.loc[:, intent_column].fillna(
                default_intent)
            test_data_df.loc[:, attribute_column] = test_data_df.loc[:, attribute_column].fillna(
                default_attribute)

            test_data_df.loc[:, predict_attribute_column] = test_data_df.loc[:, predict_attribute_column].fillna(
                default_attribute)
            test_data_df.loc[:, predict_attribute_confidence_column] = test_data_df.loc[:,
                                                                       predict_attribute_confidence_column].fillna(
                default_confidence)

        test_data_df['intent_tag'] = test_data_df.loc[:, intent_column] == test_data_df.loc[:, predict_intent_column]
        test_data_df['attribute_tag'] = test_data_df.loc[:, attribute_column] == test_data_df.loc[:,
                                                                                 predict_attribute_column]

        dataframe_to_file(mode='a', path=self.test_data_result_path, data=test_data_df,index=False)

        if self.raw_test_data_filename == "评估集4-16字符-new.xlsx":
            # 去除 意图为 个人信息+ 查询营业部 的数据
            drop_data = (test_data_df.loc[:, intent_column] == '个人信息') & (
                        test_data_df.loc[:, attribute_column] == '查询开户营业部')
            test_data_df = test_data_df[~drop_data].copy()

        test_data_statistics = {"测试集数据名称": self.raw_test_data_filename,
                                "测试集数据数量": test_data_df.__len__()}

        # 评估训练的所有意图数据
        if do_intent_evaluate:
            # 对意图数据进行统计
            intent_clf_sheet_name = f'intent_clf'
            intent_classification_report_df, intent_confusion_matrix_df = get_classification_report(test_data_df,
                                                                                                    label_column=intent_column,
                                                                                                    predict_label_column=predict_intent_column,
                                                                                                    labels=self.sorted_support_intent_list,
                                                                                                    output_dict=True)

            (total_test_data_size, total_intent_valid_test_data_size, total_intent_not_valid_test_data_size,
             total_valid_data_ratio), \
            is_intent_confidence_valid = self._get_threshold_valid_size(test_data_df=test_data_df,
                                                                        predict_confidenc_column=predict_intent_confidence_column,
                                                                        threshold=threshold)

            # 对阈值大于的所有意图数据进行统计
            if threshold is not None:
                # 只选取 意图置信度 大于 threshold
                valid_test_data_df = test_data_df.loc[is_intent_confidence_valid].copy()
                valid_intent_clf_sheet_name = f'valid_intent_clf_{threshold}'
                valid_intent_classification_report_df, valid_intent_confusion_matrix_df = get_classification_report(
                    valid_test_data_df,
                    label_column=intent_column,
                    predict_label_column=predict_intent_column,
                    labels=self.sorted_support_intent_list,
                    output_dict=True)

            # 对重点的6个意图进行统计
            if self.important_intent_list:
                # Note: 在筛选 label 或者 predict_label 属于 important_intent_list的数据
                is_intent_in_important_intent = test_data_df.loc[:, intent_column].isin(self.important_intent_list)
                is_predict_intent_in_important_intent = test_data_df.loc[:, predict_intent_column].isin(
                    self.important_intent_list)

                important_intent_test_data = test_data_df.loc[
                    is_intent_in_important_intent | is_predict_intent_in_important_intent].copy()

                (test_data_size, intent_valid_test_data_size, intent_not_valid_test_data_size, intent_valid_data_ratio), \
                is_intent_confidence_valid = self._get_threshold_valid_size(test_data_df=important_intent_test_data,
                                                                            predict_confidenc_column=predict_intent_confidence_column,
                                                                            threshold=threshold)
                important_valid_intent_test_data = important_intent_test_data.loc[is_intent_confidence_valid].copy()

                important_intent_clf_sheet_name = f"important_valid_intent_clf_{threshold}"
                confusion_matrix_sheet_name = f"important_valid_confusion_matrix_{threshold}"
                important_intent_classification_report_df, important_intent_confusion_matrix_df = \
                    get_classification_report(important_valid_intent_test_data,
                                              label_column=intent_column, predict_label_column=predict_intent_column,
                                              labels=self.important_intent_list, output_dict=True,
                                              if_save=True, mode='a', path=self.test_data_result_path,
                                              classification_report_sheet_name=important_intent_clf_sheet_name,
                                              confusion_matrix_sheet_name=confusion_matrix_sheet_name,
                                              )

                if "micro avg" in important_intent_classification_report_df.index:
                    micro_avg_precision = important_intent_classification_report_df.loc["micro avg", "precision"]
                else:
                    micro_avg_precision = important_intent_classification_report_df.loc["accuracy", "precision"]

                test_data_statistics.update({f"重点评估的意图": self.important_intent_list,
                                             f"{self.important_intent_list.__len__()}个意图的评估总数": test_data_size,
                                             f"意图大于阈值{threshold}的个数": intent_valid_test_data_size,
                                             f"意图小于阈值{threshold}的个数": intent_not_valid_test_data_size,
                                             f"意图大于阈值{threshold}的比例": intent_valid_data_ratio,
                                             "意图 micro_avg_precision": micro_avg_precision
                                             })

        if do_attribute_evaluate:
            intent_to_attribute_list = list(self.intent_to_attribute_dict.keys())
            # Note: 筛选有效的属性数据
            is_intent_in_attribute_data = important_valid_intent_test_data.loc[:, intent_column].isin(
                intent_to_attribute_list)
            is_predict_intent_in_attribute_data = important_valid_intent_test_data.loc[:, predict_intent_column].isin(
                intent_to_attribute_list)
            valid_attribute_data = important_valid_intent_test_data.loc[
                is_intent_in_attribute_data | is_predict_intent_in_attribute_data]

            # 统计置信度大于阈值的 数量
            (total_attribute_size, total_attribute_valid_test_data_size, total_attribute_not_valid_test_data_size,
             total_attribute_valid_ratio), \
            is_attribute_confidence_valid = self._get_threshold_valid_size(test_data_df=valid_attribute_data,
                                                                           predict_confidenc_column=predict_attribute_confidence_column,
                                                                           threshold=threshold,
                                                                           default_confidence=default_confidence)

            # 获取 属性的总体评估指标
            attribute_sheet_name = f'attribute_clf'
            confusion_matrix_sheet_name = f'attribute_confusion_matrix'
            total_attributes = []
            for intent, attributes in self.intent_to_attribute_dict.items():
                total_attributes.extend(attributes)
            total_attributes = set(total_attributes)
            if default_attribute in total_attributes:
                total_attributes.remove(default_attribute)
            total_attributes = list(total_attributes)

            attribute_valid_test_data = valid_attribute_data.loc[is_attribute_confidence_valid]
            attribute_classification_report_df, attribute_confusion_matrix_df \
                = get_classification_report(test_data_df=attribute_valid_test_data,
                                            label_column=attribute_column,
                                            predict_label_column=predict_attribute_column,
                                            labels=total_attributes,
                                            fill_na=default_attribute,
                                            if_save=True, mode='a', path=self.test_data_result_path,
                                            classification_report_sheet_name=attribute_sheet_name,
                                            confusion_matrix_sheet_name=confusion_matrix_sheet_name)
            if "micro avg" in attribute_classification_report_df.index:
                micro_avg_precision = attribute_classification_report_df.loc["micro avg", "precision"]
            else:
                micro_avg_precision = attribute_classification_report_df.loc["accuracy", "precision"]

            test_data_statistics.update({
                f"重点评估的属性": total_attributes,
                f"{intent_to_attribute_list.__len__()}个意图的{total_attributes.__len__()}个属性测试数据总数": total_attribute_size,
                f"属性大于阈值{threshold}的个数": total_attribute_valid_test_data_size,
                f"属性小于阈值{threshold}的个数": total_attribute_not_valid_test_data_size,
                f"属性大于阈值{threshold}的比例": total_attribute_valid_ratio,
                f"属性 micro_avg_precision": micro_avg_precision
            })

            for intent, attributes in self.intent_to_attribute_dict.items():
                is_intent_valid = valid_attribute_data.loc[:, intent_column] == intent
                is_predict_intent_valid = valid_attribute_data.loc[:, predict_intent_column] == intent
                attribute_test_data = valid_attribute_data[is_intent_valid | is_predict_intent_valid]

                (attribute_size, attribute_valid_test_data_size, attribute_not_valid_test_data_size,
                 attribute_valid_ratio), \
                is_attribute_confidence_valid = self._get_threshold_valid_size(
                    test_data_df=attribute_test_data,
                    predict_confidenc_column=predict_attribute_confidence_column,
                    threshold=threshold,
                    default_confidence=default_confidence
                )

                attribute_sheet_name = f'{intent}_attribute_clf'
                confusion_matrix_sheet_name = f'{intent}_confusion_matrix'

                # 选取 阈值符合的 属性进行评估
                if default_attribute in attributes:
                    attributes.remove(default_attribute)
                attribute_valid_test_data = attribute_test_data.loc[is_attribute_confidence_valid]
                attribute_classification_report_df, attribute_confusion_matrix_df \
                    = get_classification_report(test_data_df=attribute_valid_test_data,
                                                label_column=attribute_column,
                                                predict_label_column=predict_attribute_column,
                                                labels=attributes,
                                                fill_na=default_attribute,
                                                if_save=True, mode='a', path=self.test_data_result_path,
                                                classification_report_sheet_name=attribute_sheet_name,
                                                confusion_matrix_sheet_name=confusion_matrix_sheet_name)
                if "micro avg" in attribute_classification_report_df.index:
                    micro_avg_precision = attribute_classification_report_df.loc["micro avg", "precision"]
                else:
                    micro_avg_precision = attribute_classification_report_df.loc["accuracy", "precision"]
                test_data_statistics.update({
                    f"{intent}的属性个数": attribute_size,
                    f"{intent}的属性大于阈值{threshold}的个数": attribute_valid_test_data_size,
                    f"{intent}的属性小于阈值{threshold}的个数": attribute_not_valid_test_data_size,
                    f"{intent}的属性大于阈值{threshold}的比例": attribute_valid_ratio,
                    f"{intent}的属性 micro_avg_precision": micro_avg_precision
                })

        test_data_statistics_df = pd.DataFrame(index=test_data_statistics.keys(),
                                               data=test_data_statistics.values())

        dataframe_to_file(mode='a', path=self.test_data_result_path, sheet_name="test_data_statistics",
                          data=test_data_statistics_df)

    def _get_threshold_valid_size(self, test_data_df, predict_confidenc_column, threshold=None,
                                  default_confidence=None):
        total_data_size = test_data_df.__len__()
        if threshold:
            if default_confidence:
                is_threshold_confidence_valid = test_data_df.loc[:, predict_confidenc_column].fillna(
                    default_confidence) >= threshold
            else:
                is_threshold_confidence_valid = test_data_df.loc[:, predict_confidenc_column] >= threshold
            confidence_valid_data_size = test_data_df.loc[is_threshold_confidence_valid].__len__()
            confidence_not_valid_data_size = test_data_df.loc[~is_threshold_confidence_valid].__len__()
            # 只选取 意图置信度 大于 threshold
            # test_data_df = test_data_df.loc[is_intent_confidence_valid].copy()
        else:
            is_threshold_confidence_valid = True
            confidence_valid_data_size = total_data_size
            confidence_not_valid_data_size = 0
        confidence_valid_data_ratio = confidence_valid_data_size / total_data_size
        assert total_data_size == confidence_valid_data_size + confidence_not_valid_data_size
        return (total_data_size, confidence_valid_data_size,
                confidence_not_valid_data_size, confidence_valid_data_ratio), is_threshold_confidence_valid


if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=True)
