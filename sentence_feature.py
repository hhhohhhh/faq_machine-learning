#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.***
@time: 2021/3/1 11:10 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/1 11:10   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：***股份有限公司 2019
 * 注意：本内容仅限于***股份有限公司内部传阅，禁止外泄以及用于其他的商业目的
"""
from typing import List

import Levenshtein
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from data_explore import get_labeled_data

"""

文本相似性/相关性度量是NLP和信息检索中非常基础的任务，在搜索引擎，QA系统中有举足轻重的地位，一般的文本相似性匹配，从大的方法来讲，传统方法和深度学习方法。

特征工程方法
传统方法不外乎各种角度的特征工程，我把常用的特征罗列如下，比如

一. 字符串匹配
1.编辑距离
编辑距离（Edit Distance），又称Levenshtein距离，是指两个字串之间，由一个转成另一个所需的最少编辑操作次数。许可的编辑操作包括将一个字符替换成另一个字符，插入一个字符，删除一个字符。一般来说，编辑距离越小，两个串的相似度越大。

比如单纯的进行子串匹配，搜索 A 串中能与 B 串匹配的最大子串作为得分，
亦或者用比较常见的最长公共子序列算法来衡量两个串的相似程度，

2. 集合度量特征
集合度量方式就是把两个句子看成 BOW (bag of words)。然后使用集合相似性度量方法比如 Jaccard 等。这种方法有一个严重问题就是丢失语序。当然基于集合算重合度的度量方法不止Jaccard，也有很多别的，感兴趣的同学可以了解下。

3. 统计特征
比如句子长度，词长度，标点数量，标点种类，以及词性序列上的度量，这也是最直观的度量方法，大家可以发挥自己的想象力把这一部分做丰富一点

4. 向量空间模型: 使用传统方法如tfidf 等拿到句子表示进行度量
在实践中我们发现，使用tfidf值对词向量做一个加权，得到的句子表示也是一种不错的表示。

5 . 使用 LDA等topic model


6.使用预训练的词向量得到句子表示进行度量
词向量是深度学习进入NLP的非常代表性的工作，谷歌的词向量论文引用了3000+次，可谓影响深远了。
使用词向量的一种简单的方法是，BOW求平均，得到两个句子的表示，然后利用余弦相似度度量或者Minkowski，欧几里得距离等等。这是一种非常简单而直观的方法。
在比赛中我们这篇参考了论文， From Word Embeddings To Document Distances 这篇论文提出的一种叫做WMD的度量方法，七级本原理是利用word2vec的特性，
将文本文档表示为一个weighted point cloud of embedded words。两个文档A和B之间的距离定义为A中所有的词移动精确匹配到文档B中点云的最小累积距离。这个思路也非常直觉，实际表现也非常良好‘

"""


def edit_distance(sentence1, sentence2):
    """
    @author:wangfc27441
    @desc:
    编辑距离（Edit Distance），又称 Levenshtein 距离，是指两个字串之间，由一个转成另一个所需的最少编辑操作次数。许可的编辑操作包括将一个字符替换成另一个字符，插入一个字符，删除一个字符。一般来说，编辑距离越小，两个串的相似度越大。
    例如：
      我们有两个字符串： kitten 和 sitting:
      现在我们要将kitten转换成sitting
      我们可以做如下的一些操作；
    k i t t e n –> s i t t e n 将K替换成S
    sitten –> sittin 将 e 替换成i
    sittin –> sitting 添加g
    在这里我们设置每经过一次编辑，也就是变化（插入，删除，替换）我们花费的代价都是1。
    python-Levenshtein
    python-Levenshtein
    FuzzyWuzzy这个python包提供了比较齐全的编辑距离度量方法。
    @version：
    @time:2021/3/1 11:12

    Parameters
    ----------

    Returns
    -------
    """
    pass


def compute_Jaccard_similarity(sentence1, sentence2, if_print=False):
    """
    @author:wangfc27441
    @desc:
    Jaccard 相似系数又称为Jaccard相似性度量（Jaccard系数，Jaccard 指数，Jaccard index）。用于比较有限样本集之间的相似性与差异性。Jaccard系数值越大，样本相似度越高。定义为相交的大小除以样本集合的大小：
    （若A B均为空，那么定义J（A，B）= 1）

    与 Jaccard 相似系数相对的指标是Jaccard 距离（Jaccard distance），定义为 1- Jaccard系数，即：

    @version：
    @time:2021/3/1 11:30

    Parameters
    ----------

    Returns
    -------
    """
    words_set_ls = []
    for sentence in [sentence1, sentence2]:
        words = jieba.lcut(sentence)
        words_set = set(words)
        words_set_ls.append(words_set)
        if if_print:
            print(f"sentence={sentence},words={words},words_set={words_set}")

    words_union = words_set_ls[0].union(words_set_ls[1])
    words_intersection = words_set_ls[0].intersection(words_set_ls[1])

    Jaccard_similarity = words_intersection.__len__() / words_union.__len__()
    if if_print:
        print(
            f"words_union={words_union}\nwords_intersection={words_intersection},Jaccard_similarity={Jaccard_similarity}")
    return Jaccard_similarity


def train_tfidf_model(corpus: List[str], min_df=1, max_df=0.5, token_pattern=r"(?u)\b\w+\b",
                      stop_words=None, preprocessor=None,tokenizer=None,
                      max_features=10000, norm='l2'):
    """
    @author:wangfc27441
    @desc:
    @version：
    @time:2021/3/1 14:20


    Parameters
    ----------
    max_df/min_df: [0.0, 1.0]内浮点数或正整数, 默认值=1.0
    token_pattern这个参数使用正则表达式来分词，其默认参数为r"(?u)\b\w\w+\b"，其中的两个\w决定了其匹配长度至少为2的单词，所以这边减到1个
    norm: 得到 tf-idf 向量进行 l2 normalization

    Returns
    -------
    """
    # 分词
    # jieba.load_userdict()
    document = [' '.join(jieba.lcut(sentence)) for sentence in corpus]
    print(f"开始训练tf-idf模型")
    vectorizer = TfidfVectorizer(
        min_df=min_df, max_df=max_df, norm=norm, smooth_idf=True, use_idf=True, ngram_range=(1, 1),
        token_pattern=token_pattern, max_features=max_features)
    # 用 TF-IDF 类去训练上面同一个 corpus
    vectorizer.fit_transform(document)
    print(f"vocabulary_.size={vectorizer.vocabulary_.__len__()}")
    # print(vectorizer.vocabulary_)
    return vectorizer

def get_tfidf_vec(sentence,tfidf_vectorizer):
    # 分词并按照空格拼接
    words_ls = jieba.lcut(sentence)
    words = " ".join(words_ls)
    # 转换为 matrix
    tfidf_sparse_vec = tfidf_vectorizer.transform([words])
    # 转换为 array
    tfidf_vec = tfidf_sparse_vec.toarray()[0]
    return tfidf_vec





def test_tfidf(sentence,tfidf_vectorizer):
    # 分词并按照空格拼接
    words_ls = jieba.lcut(sentence)
    words = " ".join(words_ls)
    # 转换为 matrix
    tfidf_sparse_vec = tfidf_vectorizer.transform([words])
    # 转换为 array
    tfidf_vec = tfidf_sparse_vec.toarray()[0]
    # 得到非零的index
    indeies = np.where(tfidf_vec > 0)[0]
    tfidf_nonempty_vec = tfidf_vec[indeies]
    print(f"使用 tf-idf模型转换后，按照 vocabulary的顺序得到的向量：{tfidf_nonempty_vec}")

    vocabulary = tfidf_vectorizer.vocabulary_
    index2vocabulary = {index: w for w, index in sorted(vocabulary.items(), key=lambda x: x[1])}

    # 每个词对应的index
    word_index_ls = []
    idf_ls = []
    for index, word in enumerate(words_ls):
        word_index = vocabulary.get(word)
        if word_index is not None:
            # 得到每个 word 对应的 idf 值
            idf = tfidf_vectorizer.idf_[word_index]
            # tfidf = tfidf_vectorizer._tfidf[word_index]
            print(f'word={word},word_index={word_index},idf={idf}')
            word_index_ls.append(word_index)
            idf_ls.append(idf)

    tfidf_vec[word_index_ls]
    # 按照句子中词语的顺序得到 idf_array
    idf_array = np.array(idf_ls)
    # 进行归一化
    l2_size = np.sqrt(np.square(idf_array).sum())
    idf_array_normalized = idf_array / l2_size
    print(f"使用 tf-idf 模型转换后，按照 句子中出现词的顺序得到的向量：{idf_array_normalized}")

def computer_l2_size(vector):
    l2_size = np.sqrt(np.square(vector).sum())
    return l2_size


def compute_cosine_similarity(vector1,vector2):
    """
    @author:wangfc27441
    @desc:  计算 两个向量的 cos 距离 [-1,1]
    @version：
    @time:2021/3/1 16:00

    Parameters
    ----------

    Returns
    -------
    """
    vector1_l1 = computer_l2_size(vector1)
    vector2_l2 = computer_l2_size(vector2)
    cos_theta = np.sum(vector1*vector2)/(vector1_l1*vector2_l2)
    return cos_theta


if __name__ == '__main__':
    sentence1 = '如何得知关闭借呗'
    sentence2 = '想永久关闭借呗'
    Levenshtein.distance(sentence1, sentence2)
    compute_Jaccard_similarity(sentence1, sentence2)

    # 蚂蚁金服的数据集
    data_dir = 'data'
    dataset1 = 'atec'
    data_filename1 = 'atec_nlp_sim_train.csv'
    data_filename2 = 'atec_nlp_sim_train_add.csv'
    data_filenames1 = [data_filename1, data_filename2]

    dataset2 = 'ccks'
    data_filename1 = 'task3_train.txt'
    data_filename2 = 'task3_dev.txt'
    data_filenames2 = [data_filename1]
    data_df = get_labeled_data(data_dir, dataset1, dataset2, data_filenames1, data_filenames2)

    sentences1 = data_df.loc[:, 'sentence1']
    sentences2 = data_df.loc[:, 'sentence2']
    sentences = pd.concat([sentences1, sentences2], ignore_index=True)
    corpus = sentences.tolist()
    tfidf_vectorizer = train_tfidf_model(corpus=corpus)

    sentence_tfidf_vec1 = get_tfidf_vec(sentence=sentence1,tfidf_vectorizer=tfidf_vectorizer)
    sentence_tfidf_vec2 = get_tfidf_vec(sentence=sentence2, tfidf_vectorizer=tfidf_vectorizer)

    # cos 距离
    compute_cosine_similarity(sentence_tfidf_vec1,sentence_tfidf_vec2)


