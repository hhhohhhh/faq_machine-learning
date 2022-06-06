#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/11 21:47 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/11 21:47   wangfc      1.0         None
"""
from typing import Text, List, Union
from data_process import FAQSentenceExample
from models.trie_tree import TrieTree,SequenceType

"""
72. 编辑距离
https://leetcode-cn.com/problems/edit-distance/


给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

插入一个字符
删除一个字符
替换一个字符
 

示例 1：
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
示例 2：
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
"""




def edit_distance(word1: str, word2: str):
    """
    https://www.bilibili.com/video/BV15h411Z7Qd
    """
    length_word1 = word1.__len__()
    length_word2 = word2.__len__()

    # 初始化 distance_state 状态矩阵
    distance_state = []
    for i in range(length_word1 + 1):
        distance_state.append([None] * (length_word2 + 1))

    for i in range(length_word1 + 1):
        for j in range(length_word2 + 1):
            # 初始化状态
            if i == 0:
                distance_state[i][j] = j
            elif j == 0:
                distance_state[i][j] = i
            else:
                # 计算 第 i 个字和 第j 个字的编辑距离
                # 判断
                minimum_edit_distance_in_last_state = min(distance_state[i - 1][j], distance_state[i][j - 1],
                                                          distance_state[i - 1][j - 1])
                if word1[i - 1] == word2[j - 1]:
                    # 该状态可能从前面的三个状态过来
                    # 找到前一个状态中的最小编辑句子 +1
                    distance_state[i][j] = minimum_edit_distance_in_last_state
                else:
                    distance_state[i][j] = minimum_edit_distance_in_last_state + 1
    minimum_edit_distance = distance_state[length_word1][length_word2]
    print(f"word1= {word1}, word2= {word2},minimum_edit_distance = {minimum_edit_distance}")
    return minimum_edit_distance



def computer_edit_distance(sentence: Text, searched_sentences: Union[Text, FAQSentenceExample],
                           top_k: int = 10,
                           distance_threshold=None) -> \
List[FAQSentenceExample]:
    """
    @time:  2021/7/19 18:34
    @author:wangfc
    @version:
    @description: 计算 sentence 和所有的 searched_sentences的 编辑距离并且从小到大进行排序

    @params:
    @return:
    """
    searched_sentence_with_distance = []
    for searched_sentence in searched_sentences:
        if isinstance(searched_sentence, str):
            searched_sentence = FAQSentenceExample(text=searched_sentence)
        # 排除自己
        # if searched_sentence.text != sentence:
        distance = edit_distance(word1=sentence, word2=searched_sentence.text)
        # print(f"distance={distance},word1={sentence},word2={searched_sentence.text}")
        searched_sentence.edit_distance = distance
        if distance_threshold and distance < distance_threshold:
            searched_sentence_with_distance.append(searched_sentence)
        elif distance_threshold is None:
            searched_sentence_with_distance.append(searched_sentence)
    sorted_searched_sentence_with_distance = sorted([sample for sample in searched_sentence_with_distance],
                                                    key=lambda x: x.edit_distance)

    top_k_sorted_searched_sentence_with_distance = sorted_searched_sentence_with_distance[:top_k]
    return top_k_sorted_searched_sentence_with_distance



class EditDistanceSimilarityComponent():
    """
    @time:  2021/7/20 11:44
    @author:wangfc
    @version:
    @description:
    1. 使用 trie tree 来建立 句子结构
    2. 使用 编辑句子 计算相似度

    @params:
    @return:
    """
    def __init__(self,prefix_topk_char =1,length_threshold = 3,distance_threshold =3, top_k = 10):
        self.sentence_label_trietree = TrieTree(sequence_type=SequenceType.sentence)
        # 前几个字符作为搜索的 prefix
        self.prefix_topk_char =prefix_topk_char
        # 句子长度相差的长度
        self.length_threshold = length_threshold
        # 筛选编辑距离 前 top_k 个 样本
        self.top_k = top_k
        # 编辑距离的阈值 （小于等于）
        self.distance_threshold = distance_threshold


    def insert(self, sample: FAQSentenceExample):
        self.sentence_label_trietree.insert(sample)

    # TODO: def delete(self,sample:FAQSentenceExample)

    def upload(self, samples: List[FAQSentenceExample]):
        for sample in samples:
            self.sentence_label_trietree.insert(sample)

    def get_similar_samples(self, sentence: Text, prefix_topk_char:int=None,
                            length_threshold=None, distance_threshold=None,top_k=10,
                            forced_sentence_length=5,forced_distance_threshold=2):
        assert isinstance(sentence, str) and sentence.__len__() > 0
        prefix_topk_char = prefix_topk_char if prefix_topk_char is not None else self.prefix_topk_char
        prefix = sentence[:prefix_topk_char]

        length_upper_threshold = len(sentence) + length_threshold if length_threshold is not None else len(
            sentence) + self.length_threshold

        # 当 prefix_topk_char =0 的时候，搜索全部子串，为了减少搜索子串数量，我们设置 长度的下届
        if prefix_topk_char==0:
            length_lower_threshold = len(sentence) - length_threshold if length_threshold is not None else len(
                sentence) - self.length_threshold
            length_lower_threshold = max(0,length_lower_threshold)
        else:
            length_lower_threshold = None

        searched_samples = self.sentence_label_trietree.search(prefix=prefix,
                                                               length_lower_threshold=length_lower_threshold,
                                                               length_upper_threshold=length_upper_threshold)
        if searched_samples.__len__() > 0:
            top_k = top_k if top_k else self.top_k
            # 计算编辑距离的阈值
            if len(sentence)< forced_sentence_length:
                # 当遇到很短文本的时候，我们强制 编辑距离阈值只有 1
                distance_threshold = forced_distance_threshold
            else:
                distance_threshold = distance_threshold if distance_threshold is not None else self.distance_threshold

            sorted_edit_distance_searched_samples = computer_edit_distance(sentence, searched_samples,
                                                                           distance_threshold=distance_threshold,
                                                                           top_k=top_k)
        else:
            sorted_edit_distance_searched_samples = []
        return sorted_edit_distance_searched_samples


if __name__ == '__main__':
    # word1 = "horse"
    # word2 = "ros"
    #
    # word1 = "intention"
    # word2 = "execution"
    # edit_distance(word1=word1, word2=word2)

    import pandas as pd
    path = './data/domain_classification/gyzq_sms_0628_none_financial_term_data_dropdups.xlsx'
    faq_log_data_df = pd.read_excel(path)
    faq_log_data_df = faq_log_data_df[faq_log_data_df.CHANNEL_ID ==2].copy()
    print(faq_log_data_df.columns)
    # head_texts = faq_log_data_df.loc[faq_log_data_df.index[:100],['TEXT',]]
    faq_examples = []
    for index in faq_log_data_df.index[0:100]:
        text = faq_log_data_df.loc[index,'TEXT']
        intent = faq_log_data_df.loc[index,'CUSTRECORD_ID'].astype(str)
        faq_examples.append(FAQSentenceExample(text=text,intent=intent))


    edit_distance_similarity_component = EditDistanceSimilarityComponent()
    edit_distance_similarity_component.upload(faq_examples)

    sentence= '忘记了怎么办'
    intent= '1'
    example = FAQSentenceExample(text=text, intent=intent)
    edit_distance_similarity_component.insert(example)

    sentence = '好的'
    edit_distance_similarity_component.get_similar_samples(sentence=sentence,prefix_topk_char=0,distance_threshold=5)

    root_node = edit_distance_similarity_component.sentence_label_trietree.root
    next_node = root_node.children.get(text[0])
    next_node.children

    edit_distance_similarity_component.get_similar_samples(sentence=sentence, prefix_topk_char=0, distance_threshold=2)
