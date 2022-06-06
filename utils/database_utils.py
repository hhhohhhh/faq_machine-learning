#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/9 10:03 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/9 10:03   wangfc      1.0         None
"""
from ast import literal_eval
import pandas as pd
from sqlalchemy import create_engine


def dataframe2mysql(table_name, data_df=None, mode='r'):
    # 建立连接，username替换为用户名，passwd替换为密码，test替换为数据库名
    db_connection = create_engine('mysql+pymysql://root:r#dcenter9@10.20.32.187:3308/wangfc', encoding='utf8')
    if mode == 'w':
        # 写入数据，table_name为表名，‘replace’表示如果同名表存在就替换掉
        pd.io.sql.to_sql(data_df.astype(str), table_name, db_connection, if_exists='replace',
                         index=False,
                         index_label='资讯id')
        print("向数据库中写入table_name={},shape={}".format(table_name, data_df.shape))
    else:
        data_df = pd.read_sql('SELECT * FROM {}'.format(table_name), con=db_connection)
        columns = data_df.columns.tolist()
        if 'entity_label_info_ls' in columns:
            data_df['entity_label_info_ls'] = data_df.entity_label_info_ls.apply(literal_eval)
        if 'labels_ls' in columns:
            data_df['labels_ls'] = data_df.labels_ls.apply(literal_eval)
        print("从数据库中读取 table_name={},shape={}".format(table_name, data_df.shape))
        return data_df


def load2mysql(table_name, data_df=None, mode='r'):
    # 建立连接，username替换为用户名，passwd替换为密码，test替换为数据库名
    db_connection = create_engine('mysql+pymysql://root:r#dcenter9@10.20.32.187:3308/wangfc', encoding='utf8')
    if mode == 'w':
        # 写入数据，table_name为表名，‘replace’表示如果同名表存在就替换掉
        pd.io.sql.to_sql(data_df.astype(str), table_name, db_connection, if_exists='replace',
                         index=False,
                         index_label='资讯id')
        print("向数据库中写入table_name={},shape={}".format(table_name, data_df.shape))
    else:
        data_df = pd.read_sql('SELECT * FROM {}'.format(table_name), con=db_connection)
        columns = data_df.columns.tolist()
        if 'entity_label_info_ls' in columns:
            data_df['entity_label_info_ls'] = data_df.entity_label_info_ls.apply(literal_eval)
        if 'labels_ls' in columns:
            data_df['labels_ls'] = data_df.labels_ls.apply(literal_eval)
        print("从数据库中读取 table_name={},shape={}".format(table_name, data_df.shape))
        return data_df