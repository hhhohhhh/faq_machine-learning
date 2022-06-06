# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# @file:
# @version:
# @desc:
# @author: wangfc
# @site: http://www.***
# @time: 2021/3/1 9:31
#
# @Modify Time      @Author    @Version    @Desciption
# ------------      -------    --------    -----------
# 2021/3/1 9:31   wangfc      1.0         None
#
#  * 密级：秘密
#  * 版权所有：*** 2019
#  * 注意：本内容仅限于***内部传阅，禁止外泄以及用于其他的商业目的
#
# """
#
# from utils.utils import default_load_json
#
# if __name__ == '__main__':
#
#     # 蚂蚁金服的数据集
#     data_dir = 'data'
#     dataset1 = 'atec'
#     data_filename1 = 'atec_nlp_sim_train.csv'
#     data_filename2= 'atec_nlp_sim_train_add.csv'
#     data_filenames1 = [data_filename1,data_filename2]
#
#     dataset2 = 'ccks'
#     data_filename1 = 'task3_train.txt'
#     data_filename2= 'task3_dev.txt'
#     data_filenames2 = [data_filename1]
#     # data_df = get_labeled_data(data_dir,dataset1,dataset2,data_filenames1,data_filenames2)
#
#
#     # dstc7
#     dataset ='dstc7'
#     train_data_filename= 'ubuntu_train_subtask_1.json'
#     datapath = os.path.join(os.getcwd(),'data',dataset,train_data_filename)
#     train_data = default_load_json(json_file_path=datapath)
#     index = 100
#     print(train_data[index].keys())
#     print(train_data[index])
#     print(train_data[index]['data-split'])
#
#     print(train_data[index]['messages-so-far'][0].keys())
#     print(train_data[index]['messages-so-far'][0])
#     print(train_data[index]['messages-so-far'][2].keys())
#
#     print(train_data[index]['options-for-correct-answers'].__len__())
#     from collections import Counter
#
#     answers_len_ls = []
#     for i in range(train_data.__len__()):
#         answers_len = train_data[index]['options-for-correct-answers'].__len__()
#         # print(answers_len)
#         answers_len_ls.append(answers_len)
#
#     counter =Counter(answers_len_ls)
#     counter
#
#     print(train_data[index]['options-for-correct-answers'][0].keys())
#     print(train_data[index]['options-for-correct-answers'][0]['candidate-id'])
#     candidate_id = train_data[index]['options-for-correct-answers'][0]['candidate-id']
#
#     print(train_data[index][ 'options-for-next'].__len__())
#     print(train_data[index][ 'options-for-next'][0].keys())
#     candidate_ids = []
#     for option in train_data[index][ 'options-for-next']:
#         candidate_ids.append(option['candidate-id'])
#
#     candidate_id in candidate_ids

from data_process.data_analysis import IVRDataAnalysis
from conf.config_parser import *
from utils.io import dataframe_to_file

if __name__ == '__main__':
    haitong_ivr_new_data_20211209_filename = "标准问映射意图实验数据.xlsx"
    haitong_ivr_new_data_20211209_path = os.path.join(CORPUS,DATASET_NAME,"20211209",haitong_ivr_new_data_20211209_filename)
    haitong_ivr_new_data = dataframe_to_file(haitong_ivr_new_data_20211209_path,mode='r',sheet_name='扩展问意图预测结果')


    data_filename = "standard_to_extend_question_data-20211130.xlsx"
    data_path = os.path.join(CORPUS,DATASET_NAME,data_filename)
    os.path.exists(data_path)
    data =  dataframe_to_file(data_path,mode='r',sheet_name='standard_to_extend_data')
    is_haitong_ivr = data.loc[:,"scenario"]=="haitong_ivr"
    haitong_ivr_data = data.loc[is_haitong_ivr]
    haitong_ivr_extend_questions = haitong_ivr_data.loc[:,'extend_question']

    is_in = haitong_ivr_new_data.loc[:,'扩展问'].isin(haitong_ivr_extend_questions)
    is_in.value_counts()
    not_in_data = haitong_ivr_new_data.loc[~is_in].copy()

    path = os.path.join(CORPUS,DATASET_NAME,"20211209","标准问映射意图实验数据-20211209-filtered.xlsx")
    dataframe_to_file(data=not_in_data,path=path)




