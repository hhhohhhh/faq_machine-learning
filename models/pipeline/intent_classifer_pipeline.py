#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/19 10:33 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/19 10:33   wangfc      1.0         None
"""
from typing import Text, List, Dict

from data_process import text_preprocess
from data_process import FAQSentenceExample
from models.edit_distance import EditDistanceSimilarityComponent
from models.faq_similarity import HundsunFAQSimilarityComponent


class Pipeline(object):
    def __init__(self):
        """初始化 pipeline的 component:
        """
        pass

    def upload_regular_expression(self, regular_expressions=List[Text]):
        pass


    def upload_vocabulary(self, words=List[Text]):
        pass

    def predict(self, query):
        """
        输入 query，输出对应的 answer
        """

class IntentSamplesDataset():
    def __init__(self):
        self.intent_samples: Dict[Text, FAQSentenceExample] =dict()

    def upload(self, intent_samples =List[FAQSentenceExample],overwrite=False):
        uploaded_intent_sample=[]
        for intent_sample in intent_samples:
            uploaded = False
            uploading_sample_id= str(intent_sample.id)
            uploaded_intent_sample = self.intent_samples.get(uploading_sample_id)
            if not overwrite and uploaded_intent_sample :
                print(f"id= {uploading_sample_id} 数据已经存在，uploaded_intent_sample = {uploaded_intent_sample}，intent_sample={intent_sample}"
                                 )
                continue
            else:
                self.intent_samples.update({uploading_sample_id:intent_sample})
                uploaded_intent_sample.append(intent_sample)
        print(f"成功上传 intent_samples 共{uploaded_intent_sample.__len__()}/{intent_samples.__len__()}")

        return  uploaded_intent_sample

    def get(self,id:Text)->FAQSentenceExample:
        example = self.intent_samples.get(id)
        return example




class IntentClassifierPipeline(Pipeline):
    def __init__(self,similarity_model= 'HundsunFAQSimilarityComponent'):
        """初始化 pipeline的 component:
        0. 预处理
        1. 正则表达式
        2. 分词器 (实体词库)
        3. 编辑距离模型 + 相似度模型 : 返回相似度最高句子对应的标签
        4. 分类模型
        """

        # 存储 意图数据的 IntentSamplesComponent
        self.intent_samples_dataset = IntentSamplesDataset()

        self.re_intent_classifier = None
        self.word_segmentation = None

        self.edit_distance_similarity_component = EditDistanceSimilarityComponent()

        # if similarity_model== 'HundsunFAQSimilarityComponent':
        self.embedding_similarity_component = HundsunFAQSimilarityComponent()
        self.intent_classifier_component = None

    def upload_regular_expression(self, regular_expressions=List[Text]):
        # TODO
        pass

    def upload_vocabulary(self, words=List[Text]):
        # TODO
        pass


    def upload_intent_samples(self, intent_samples=List[FAQSentenceExample]):
        # 更新 存储 意图数据的 IntentSamplesComponent
        uploaded_intent_sample = self.intent_samples_dataset.upload(intent_samples)
        # 更新 编辑模型的数据
        self.edit_distance_similarity_component.upload(uploaded_intent_sample)
        # 更新 FAQ模型的数据
        self.embedding_similarity_component.upload(uploaded_intent_sample)



    def predict(self, query):
        output = None
        query = text_preprocess(query)
        if self.re_intent_classifier:
            output = self.re_intent_classifier.predict(query)
            return output

        if self.edit_distance_similarity_component:
            output = self.edit_distance_similarity_component.predict(query)
            return output

        if self.embedding_similarity_component:
            output = self.embedding_similarity_component.predict(query)
            return output

        if self.intent_classifier_component:
            output = self.intent_classifier_component.predict(query)
        return output