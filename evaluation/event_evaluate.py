#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/6/28 9:34 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/28 9:34   wangfc      1.0         None
"""
import os
from typing import Text, Dict

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import logging

logger = logging.getLogger(__name__)


class EventEvaluation():
    def __init__(self, predict_probabilities_matrix:np.ndarray, raw_test_df:pd.DataFrame,
                 output_predict_dir:Text,
                 support_types_dict:Dict[Text,Dict[Text,int]], top_k=1, predict_top_k=3, p_threshold=0.5,
                 event_label_column='event_label_ls'):
        self.predict_probabilities_matrix = predict_probabilities_matrix
        self.raw_test_df = raw_test_df
        self.output_predict_dir = output_predict_dir
        self.support_types_dict = support_types_dict
        self.top_k = top_k
        self.predict_top_k = predict_top_k
        self.p_threshold = p_threshold

        self.event_label_column = event_label_column

    def evaluate(self):
        """
        @author:wangfc27441
        @time:  2019/9/24  21:30
        @desc:
        :param probabilities_matrix:
        :param raw_test_path:
        :param output_predict_dir:
        :param support_types_dict:
        :param top_k:
        :param p_threshold:
        :return:
        """
        Y_pred = self.get_predict_labels(y_predict_prob_comb=self.predict_probabilities_matrix,
                                         support_types_dict=self.support_types_dict,
                                         predict_top_k=self.predict_top_k,
                                         p_threshold=self.p_threshold)

        # Y_test = self.raw_test_df.loc[:,'event_label']
        Y_test = self.raw_test_df.loc[:, self.event_label_column]

        ### 修改为多标签的评估：Evaluation
        first_true_label_ls, predict_result, tag_list, predict_result_whole, weight_result = self.multilabels_evaluation(
            Y_test, Y_pred, top_k=self.top_k, evaluation_result_path=self.output_predict_dir,
            support_types_dict=self.support_types_dict)

        ### Save_predict_results
        predict_results_path = os.path.join(self.output_predict_dir, 'predict_results_top_{0}.json'.format(self.top_k))
        self.save_predict_results(mode='w', predict_results_path=predict_results_path,
                                  test_data=self.raw_test_df, first_true_lable_ls=first_true_label_ls,
                                  predict_result=predict_result, tag_list=tag_list,
                                  predict_result_whole=predict_result_whole,
                                  weight_result=weight_result)

    def get_predict_labels(self, y_predict_prob_comb, support_types_dict, test_types=None,
                           predict_top_k=3, p_threshold=0.5,
                           pred_type='list_of_list'):
        """
        @author:wangfc27441
        @time:  2019/11/6  9:02
        @author:wangfc27441
        @time:  2019/10/22  10:20
        @desc: 根据预测的概率矩阵，获取预测的类别（可以是多标签）
        @edit：当前所有的预测predict_proba<0.5的时候，取前三个,现在这种情况下改为取"其他"
                增加test_types 参数
        @time：20190719 13:30
        :param y_predict_prob_comb: shape = [n_sample, n_class]
        :param support_type: 替换为 support_types_dict
        :param predict_top_k:
        :param p_threshold:
        :return:
        """
        y_predict_prob_comb_df = pd.DataFrame(y_predict_prob_comb, columns=list(support_types_dict.keys()))
        y_pred = y_predict_prob_comb_df.apply(
            lambda row: self.get_topk_label_index_ls(row, predict_top_k=predict_top_k, p_threshold=p_threshold),
            axis=1).values.tolist()

        # 增加对结果的三位小数舍入,并增加 根据 test_type的筛选
        y_pred_round = []
        # do_predict或者不传入 test_types 的时候使用
        if test_types is None:
            for prediction in y_pred:
                prediction_round = self.prediction_round_fn(prediction, test_type=None, pred_type=pred_type)
                y_pred_round.append(prediction_round)
        # 增加 根据 test_type的筛选
        else:
            for prediction, test_type in zip(y_pred, test_types):
                # 当 test_type=None 的时候，多标签情况，不进行筛选
                # 当 test_type! =None 的时候，二分类情况，需要进行筛选
                prediction_round = self.prediction_round_fn(prediction, test_type=test_type, pred_type=pred_type)

                y_pred_round.append(prediction_round)
            logger.info(
                "prediction_round={3} ,Finished multilabel_prediction with  predict_top_{0} and p_threshold={1},and Y_pred of lenth {2}"
                    .format(predict_top_k, p_threshold, len(y_pred_round), prediction_round))
        return y_pred_round

    def get_topk_label_index_ls(self, row, predict_top_k=3, p_threshold=0.5):
        """
        @version：02
        @desc: 获取预测的类别（可以是多标签）
        @edit：当前所有的预测predict_proba<0.5的时候，取前三个,
               1)现在这种情况下改为取"其他" ,  概率值 返回为 '0',代替  math.nan 或者 'NA'
               2)改为先筛选，然后再排序，减少排序的计算复杂度，从而增加速度
        @time：20190719 13:30
        :param row:
        :param predict_top_k:
        :param p_threshold:
        :return:
        """
        # 筛选满足大于p_threshold的值
        above_threshold_series = row[row > p_threshold]
        if len(above_threshold_series) == 0:
            # 当不存在满足大于p_threshold的值的时候
            top_k_label_index = [('聚源其他', '0')]
        else:
            # 选取前predict_top_k或者所有的值
            predict_k = min(len(above_threshold_series), predict_top_k)
            top_k_sorted_label_index_series = above_threshold_series.sort_values(ascending=False)[:predict_k]
            top_k_label_index = [x for x in top_k_sorted_label_index_series.to_dict().items()]

        return top_k_label_index

    def prediction_round_fn(self, prediction, test_type=None, pred_type='list_of_list'):
        """
        @author:wangfc27441
        @time:  2019/11/29  11:27
        @desc:  对于单个样本的预测进程舍入和输出（可能存在多标签）
        :param prediction:
        :param test_type:
        :param pred_type:
        :return:
        """
        prediction_round = []
        # 对每个样本的prediction进行循环
        for (label, probability) in prediction:
            # logger.info("label={}, probability={}".format(label, probability))
            # if classifier == 'event':
            single_label_round = self.single_label_round_fn(label, probability, test_type, pred_type)
            prediction_round.append(single_label_round)

        if pred_type == 'string' and prediction_round != []:
            # 输出为string的时候，需要拼接各个 label
            prediction_round = ','.join(list(set(prediction_round)))

        if prediction_round == []:
            if pred_type == 'list_of_list':
                prediction_round = [('其他', 'NA')]
            elif pred_type == 'list_of_dict':
                prediction_round = [{'其他': 'NA'}]
            elif pred_type == 'string':
                prediction_round = '其他'
        return prediction_round

    def single_label_round_fn(self, label, probability, test_type, pred_type):
        """
        @author:wangfc27441
        @time:  2019/11/29  14:01
        @desc:  单个标签的舍入和格式化

        :param label:
        :param probability:
        :param test_type:
        :param pred_type:
        :return:
        """
        if pred_type == 'list_of_list':
            if type(probability) == float:
                if test_type is None or test_type == label:
                    single_label_round = (label, round(probability, 3))
                else:
                    # 当test_type存在，但是预测的是其他某个类别的时候，我们标记为 ('其他',probability)
                    single_label_round = ('其他', round(probability, 3))
            else:
                # 当预测概率为‘NA’时
                single_label_round = (label, probability)

        elif pred_type == 'list_of_dict':
            if type(probability) == float:
                if test_type is None or test_type == label:
                    single_label_round = {label: round(probability, 3)}
                else:
                    # 当test_type存在，但是预测的是其他某个类别的时候，我们标记为 ('其他',probability)
                    single_label_round = {'其他': round(probability, 3)}
            else:
                single_label_round = {label: probability}

        elif pred_type == 'string':
            if type(probability) == float:
                if test_type is None or test_type == label:
                    single_label_round = label
                else:
                    # 当test_type存在，但是预测的是其他某个类别的时候，我们标记为 ('其他',probability)
                    single_label_round = '其他'
            else:
                single_label_round = label
        return single_label_round

    # 增加top_k 的评估功能：
    def multilabels_evaluation(self, Y_test, Y_pred, top_k=1, save=True, best_pred=False, evaluation_result_path=None,
                               support_types_dict=None):
        """
        :param Y_test:
        :param Y_pred:
        :param top_k:
        :param save:
        :param best_pred:
        :return:
            # 真实数据的首个标签: first_true_lable_ls = []
            # 最佳的预测类别:    predict_result = []
            # 是否判断正确标记:   tag_list = []
            # 预测的所有标签:     predict_result_whole = []
            # 预测的所有标签权重：weight_result= []
        """
        logger.info("Start top_{0} evaluation with Y_test.shape={1} and Y_pred's len={2}."
                    .format(top_k, Y_test.shape, len(Y_pred)))

        if best_pred:  # 也许使用 Y_pred.shape的值更好一些:top_k==k 时，shape = (n,k)
            # 之前单标签的时候
            # 按照预测概率最大，选取预测中最好的值
            Y_pred_best = Y_pred
        else:
            # 按照bond_risk_control_classify.py 评测的规则取最好的那个
            # 真实数据的首个标签
            first_true_lable_ls = []
            # 最佳的预测类别
            predict_result = []
            # 预测标签对应的权重
            weight_result = []
            # 预测的所有标签
            predict_result_whole = []
            # 是否判断正确标记
            tag_list = []

            true_labels_set = set()
            for i in range(len(Y_pred)):
                # logger.info('Y_test[{0}]={1},and split={2},and Y_pred[{0}]={3}'.format(i, Y_test.iloc[i],
                # Y_test.iloc[i].split('@'), Y_pred[i]) )
                # 测试数据的标注类别
                true_labels = Y_test.iloc[i]  # .split('@')
                true_labels_set.update(set(true_labels))
                label = true_labels[0]

                # Y_pred返回[(label,predict_proba), ...]
                event_bonds = Y_pred[i]
                # 预测的标签
                label_predict_all = [x[0] for x in event_bonds]
                # 预测标签对应的权重
                weight_predict_all = [str(x[1]) for x in event_bonds]

                if len(true_labels) == 1 and label in label_predict_all[0:top_k]:
                    # 当 真实标签只有1个并且在预测的top_n标签当中
                    lable_predict = label
                    predict_result_i = '@'.join(label_predict_all)
                    weight_predict = '@'.join(weight_predict_all)
                elif len(true_labels) >= 2 and len(set(true_labels) & set(label_predict_all)) >= 2:
                    # 当 真实标签个数>2个的时候，并且与预测标签相交>=2
                    lable_predict = label
                    predict_result_i = '@'.join(label_predict_all)
                    weight_predict = '@'.join(weight_predict_all)
                else:
                    # 其他情况，预测类别去首个预测类别
                    lable_predict = label_predict_all[0]
                    predict_result_i = '@'.join(label_predict_all)
                    weight_predict = '@'.join(weight_predict_all)

                # 最佳的预测类别
                predict_result.append(lable_predict)
                weight_result.append(weight_predict)
                # 预测的所有标签
                predict_result_whole.append(predict_result_i)
                # 真实数据的首个标签
                first_true_lable_ls.append(label)

                # 判断是否预测准备的标记，但这个可能会存在问题：
                # 如预测类别为3个，实际类别为2个，但只有1个相交
                # 如top_k=1，但是预测值和真实值顺序下相反的时候
                if label in label_predict_all[0:top_k]:
                    tag_list.append(True)
                else:
                    tag_list.append(False)

        pred_label_set = set(predict_result)
        test_label_set = set(first_true_lable_ls)
        combine_label_set = sorted(list(pred_label_set.union(test_label_set)))

        # 因为 Y_pred 是类别，因此不需要再进行转化了
        # # 获得classification_report的index： get the pred and test index set
        # pred_index_set = set(Y_pred)
        # test_index_set = set(Y_test)
        # combine_index_set = pred_index_set.union(test_index_set)
        #
        # index_label_dict = {v: k for k, v in label_index_dict.items()}
        # metric_label_index_dict = {index_label_dict.get(index, '其他'): index for index in combine_index_set}
        # # 按照 dict 的value值排序
        # sorted_metric_label_index = [label + '_' + str(index) for (label, index) in
        #                              sorted(metric_label_index_dict.items(), key=lambda kv: kv[1])]

        # 使用 Y_pred_best 进行测评
        result_classification_report = classification_report(first_true_lable_ls, predict_result, labels=sorted(
            true_labels_set))  # list(support_types_dict.keys()))
        result_confusion_matrix = confusion_matrix(first_true_lable_ls,
                                                   predict_result)  # ,labels= list(support_types_dict.keys()))

        # result_classification_report = classification_report(Y_test, Y_pred_best,
        #                                                      target_names=sorted_metric_label_index)
        # result_confusion_matrix = confusion_matrix(Y_test, Y_pred_best)

        if save:
            classification_report_path = os.path.join(evaluation_result_path,
                                                      'classification_report_top_{0}.csv'.format(top_k))
            self.save_classification_report(result_classification_report, classification_report_path)
            result_confusion_matrix_path = os.path.join(evaluation_result_path,
                                                        'confusion_matrix_top_{0}.csv'.format(top_k))

            self.save_confusion_matrix(result_confusion_matrix, columns=combine_label_set, index=combine_label_set,
                                       result_confusion_matrix_path=result_confusion_matrix_path)
        logger.info("Finished top_{0} evaluation.".format(top_k))
        return first_true_lable_ls, predict_result, tag_list, predict_result_whole, weight_result

    def save_classification_report(self, classification_report, classification_report_path=None):
        classification_report_dir = os.path.dirname(classification_report_path)
        if not os.path.exists(classification_report_path):
            try:
                os.makedirs(classification_report_dir)
            except Exception as e:
                logger.info(e)
        with open(classification_report_path, 'w', encoding='utf_8_sig') as f:
            f.write(classification_report)
            logger.info("Saved the classification report into {0}".format(classification_report_path))

    def save_confusion_matrix(self, result_confusion_matrix, columns=None, index=None,
                              result_confusion_matrix_path=None):
        confusion_df = pd.DataFrame(data=result_confusion_matrix, columns=columns, index=index)
        confusion_df.to_csv(result_confusion_matrix_path, encoding='utf_8_sig')
        logger.info("Saved the result_confusion_matrix into {0}".format(result_confusion_matrix_path))

    def save_predict_results(self, mode, predict_results_path=None, test_data=None, first_true_lable_ls=None,
                             predict_result=None, tag_list=None, predict_result_whole=None,
                             weight_result=None, if_multilabels=True):
        encoding = 'utf_8_sig'
        select_test_data = test_data.copy()
        if mode == "w":
            # 相对于测试文档，多一列 predict_label
            # if 'ids' in test_data.columns.tolist():
            #     selected_columns = ['ids', 'entity_index', 'entity', 'entity_start_position', 'entity_end_position',
            #                         'event_label', 'predict_label', 'event_label_argument',
            #                         'predict_result_whole', 'tag', 'weight_result', 'title', 'content', 'entity_word', ]
            # else:
            #     selected_columns = ['entity', 'event', 'predict_label', 'predict_result_whole', 'tag', 'weight_result',
            #                         'title', 'content']
            if if_multilabels:
                select_test_data.loc[:, 'first_true_label'] = first_true_lable_ls
                select_test_data.loc[:, 'predict_label'] = predict_result
                select_test_data.loc[:, 'tag'] = tag_list
                select_test_data.loc[:, 'predict_result_whole'] = predict_result_whole
                select_test_data.loc[:, 'weight_result'] = weight_result

                # select_test_data = test_data.loc[:, selected_columns]
                # select_test_data.to_csv(predict_results_path, encoding='utf_8_sig',index=None)
                select_test_data.to_json(predict_results_path)
            else:
                test_data.loc[:, 'first_true_label'] = first_true_lable_ls
                test_data.loc[:, 'predict_label'] = predict_result
                test_data.loc[:, 'tag'] = tag_list
                test_data.loc[:, 'weight_result'] = weight_result
                select_test_data = test_data.loc[:,
                                   ['label', 'predict_label', 'tag', 'weight_result', 'title', 'text']]
                select_test_data.to_csv(predict_results_path, encoding=encoding)
            logger.info("Saved predict results to {0}!".format(predict_results_path))
        else:
            # select_test_data = pd.read_csv(predict_results_path,encoding=encoding,engine="python")
            select_test_data = pd.read_json(predict_results_path)
            logger.info("Read predict results.shape={},columns={} from {}!".format(select_test_data.shape,
                                                                                   select_test_data.columns.tolist(),
                                                                                   predict_results_path))
        return select_test_data
