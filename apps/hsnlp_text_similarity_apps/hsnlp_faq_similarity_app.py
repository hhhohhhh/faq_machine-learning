#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/20 11:59 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/20 11:59   wangfc      1.0         None
"""

import json

from typing import List

import pandas as pd
import requests

from data_process import text_preprocess
from data_process import FAQSentenceExample
from evaluation.evaluation_lzl import drop_duplicate_question





class HundsunFAQSimilarityComponent(object):
    """
    自己的 FAQ 相似度模型
    1.首先 upload() 标注问和扩展问
    2. 提交请求，返回结果
    """
    def __init__(self,address="192.168.73.51", port=21327, threshold=0.0):
        self.address = address
        self.port = port
        self.threshold =threshold
        self.url = "http://" + address + ":" + str(port) + "/hsnlp/qa/sentence_similarity/push_data"


    def get_similar_samples(self,sentence: str):
        assert sentence != ""
        # print(q)
        url_sim = "http://" + self.address + ":" + str(self.port) + "/hsnlp/qa/sentence_similarity/similarity"
        content = requests.post(url_sim, data={"query": sentence, "threshold": self.threshold}).content
        # print(content)
        """
        result = ['recall_question', 'recall_std_question', 'recall_score', 'recall_std_id', 
        'bm25_recall', 'bm25_recall_standard', 'bm25_score', 'logits']
        result['recall_question'].__len__()==10
        """
        result = json.loads(content)["result"]
        return result


    def upload_hsnlp_questions(self,path_std_q, path_ext_q=None, address="192.168.73.51", port=20028):
        url = "http://" + address + ":" + str(port) + "/hsnlp/qa/sentence_similarity/push_data"
        print(f'Upload data to {url}')
        std_df = pd.read_excel(path_std_q, sheet_name="hsnlp_standard_question")
        print(f"读取 标注问数据 {std_df.shape}")
        # 去除重复的数据
        std_df = drop_duplicate_question(std_df)

        question_id_std = std_df["question_id"].tolist()
        question_std = std_df["question"].tolist()
        question_id_set = set(question_id_std)
        question_std_set = set(question_std)

        assert question_id_std.__len__() == question_id_set.__len__()
        assert question_std.__len__() == question_std_set.__len__()
        # 推送标准问
        print("开始组装标准问")
        data = []
        for i in range(len(std_df)):
            d = dict()
            d["question_id"] = question_id_std[i]
            d["question"] = "".join(str(question_std[i]).strip().split())
            d["is_standard"] = "1"
            d["standard_id"] = question_id_std[i]
            data.append(d)
            print(f"共组装标准问= {data.__len__()} 条")
        if path_ext_q is not None:
            ext_df = pd.read_excel(path_ext_q, sheet_name="hsnlp_extend_question")
            # data_df = pd.concat([data_df,ext_df])
            print(f"读取 扩展问数据 {ext_df.shape}")
            # 去除重复的数据
            ext_df = drop_duplicate_question(ext_df)

            question_id_ext = ext_df["question_id"].tolist()
            question_ext = ext_df["question"].tolist()
            standard_id = ext_df["standard_id"].tolist()
            print("开始组装扩展问")
            for i in range(len(ext_df)):
                d = dict()
                question_id = question_id_ext[i]
                question = question_ext[i]
                if question_id in question_id_set:
                    print(f"扩展问中 question_id={question_id},question={question} question_id 已经在标准问的 question_id_set 中")
                    continue
                elif question in question_std_set:
                    print(f"扩展问中 question_id={question_id},question={question} question 已经在标准问的 question_id_set 中")
                else:
                    d["question_id"] = question_id_ext[i]
                    d["question"] = question_ext[i]
                    d["is_standard"] = "0"
                    d["standard_id"] = standard_id[i]
                    data.append(d)
            print(f"共组装标准问+扩展问= {data.__len__()} 条")
        cont = requests.post(url, data={"data": json.dumps(data)}).content
        return_content = json.loads(cont)
        print(f"返回内容： {return_content}")
        return return_content

    def upload(self,intent_samples:List[FAQSentenceExample]):
        print(f'Upload data to {self.url}')
        # 推送标准问
        data = []
        for example in intent_samples :
            d = dict()
            d["question_id"] = example.id
            d["question"] = text_preprocess(example.text)
            d["is_standard"] = example.is_standard
            d["standard_id"] = example.standard_id
            data.append(d)
            print(f"共组装标准问= {data.__len__()} 条")
        cont = requests.post(self.url, data={"data": json.dumps(data)}).content
        return_content = json.loads(cont)
        print(f"返回内容： {return_content}")
        return return_content

if __name__ == '__main__':
    faq_similarity_component = HundsunFAQSimilarityComponent()
    sentence = '忘记了怎么办'
    result = faq_similarity_component.get_similar_samples(sentence=sentence)