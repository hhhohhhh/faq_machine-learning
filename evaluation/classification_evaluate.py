#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/17 11:13 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/17 11:13   wangfc      1.0         None
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels

from utils.io import dataframe_to_file


def get_classification_report(test_data_df:pd.DataFrame,label_column,predict_label_column,
                              output_dict=True,fill_na=None,labels=None,
                              statistic_keys=['accuracy',"micro avg" ,'macro avg', 'weighted avg'],
                              if_save=False,mode='a',path=None,
                              classification_report_sheet_name=None,confusion_matrix_sheet_name=None
                              )-> pd.DataFrame:
    """
    labels:
    target_names: 显示 label 对应（按照 labe的顺序）的名称
    """


    true_labels = test_data_df.loc[:, label_column].to_list()
    if fill_na:
        # # 存在为空情况
        predict_labels = test_data_df.loc[:, predict_label_column].fillna(fill_na).to_list()
    else:
        predict_labels = test_data_df.loc[:, predict_label_column].to_list()
    
    
    classification_report_dict = classification_report(true_labels, predict_labels,
                                                       labels=labels,
                                                       output_dict=output_dict)
    classification_report_df = pd.DataFrame(classification_report_dict).T

    # if labels:
    #     filter_indexes= labels + statistic_keys
    #     # 输出指定 labels
    #     classification_report_df = classification_report_df.loc[classification_report_df.index.isin(filter_indexes)].copy()

    all_labels = unique_labels(true_labels, predict_labels)
    confusion_matrix_ndarray = confusion_matrix(true_labels,predict_labels)
    confusion_matrix_df = pd.DataFrame(data=confusion_matrix_ndarray,index=all_labels,columns=all_labels)

    if if_save:
        dataframe_to_file(mode=mode, path=path, sheet_name=classification_report_sheet_name,
                          data=classification_report_df)
        dataframe_to_file(mode=mode, path=path, sheet_name=confusion_matrix_sheet_name,
                          data=confusion_matrix_df)

    return classification_report_df,confusion_matrix_df
