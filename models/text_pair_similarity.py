#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/3/29 22:54 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/29 22:54   wangfc      1.0         None

"""
import re
from typing import List
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from data_process.sentence_pair_data_reader import SentencePairDataReader
import jieba
import numpy as np
from utils.io import load_stopwords

from conf.config_parser import *
from utils.common import init_logger

logger = init_logger(output_dir = MODEL_OUTPUT_DIR,log_filename=LOG_FILENAME)

class TFIDFSimilarity():
    def __init__(self,corpus,norm=None,tf_as_term_frequency=True):
        self.corpus = corpus
        userdict_path = os.path.join(data_dir, "userdict.txt")
        jieba.load_userdict(userdict_path)
        stopwords_path = os.path.join(data_dir, 'stop_words_zh.txt')

        self.stop_words = load_stopwords(path=stopwords_path)
        # norm: 是否对 tf-idf 做 normalization,这儿我们做文本匹配模型
        self.norm = norm
        # tf_as_term_frequency: 原来的tf-idf 值计算是 频数而不是频率
        self.tf_as_term_frequency = tf_as_term_frequency
        self.counter_vectorizer,self.tf_transformer ,self.counts_matrix,self.tfidf_matrix,self.term_frequency_idf_array,self.vocabulary_= self.fit(corpus)

        # bm25 超参数
        self.bm25_k = 1.2 # By default, k1 has a value of 1.2 in Elasticsearch.
        self.bm25_b = 0.75 # By default, b has a value of 0.75 in Elasticsearch.


    def tokenizer(self,s):
        words = jieba.lcut(s)
        # token之后 去除 全部为数字的词
        new_words = []
        for w in words:
            if re.match(pattern=r'\d{1,3}', string=w) or w == ' ':
                pass
            else:
                new_words.append(w)
        return new_words


    def get_counter_vectorizer(self):
        # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示在i类文本下j词的词频
        counter_vectorizer = CountVectorizer(lowercase=True, stop_words=self.stop_words, tokenizer=self.tokenizer,
                                             ngram_range=(1, 1),
                                             token_pattern=r"(?u)\b\w+\b",
                                             min_df=1,
                                             max_features=None)
        return counter_vectorizer

    def fit(self,corpus):
        """

        """
        logger.info("开始训练 tf-idf 模型")
        counter_vectorizer = self.get_counter_vectorizer()
        counter_vectorizer.fit(raw_documents=corpus)
        # 获取词袋模型中的所有词语
        # feature_names = counter_vectorizer.get_feature_names()
        # print(f"feature_names共{feature_names.__len__()}个：{feature_names}")
        # self.feature_name2index_dict = {feature_name:index for index,feature_name in enumerate(feature_names)}

        # 获取 feature_names 对应的次数
        vocabulary_ = counter_vectorizer.vocabulary_

        counts_matrix = counter_vectorizer.transform(raw_documents=corpus)
        # counts_matrix_array = counts_matrix.toarray()

        # 该类会统计每个词语的tf-idf权值:
        # norm: 是否对 tf-idf 做 normalization
        tf_transformer = TfidfTransformer(norm=self.norm, use_idf=True, smooth_idf=True)
        tf_transformer.fit(X=counts_matrix)
        # 计算 tfidf 值： 其中 tf 为 词的频数
        tfidf_matrix = tf_transformer.transform(X=counts_matrix)
        # tfidf_matrix_array = tfidf_matrix.toarray()

        # 我们改变成 tf 为 词的频率:其中有些是 零个 token : shape = (39346, 1)
        document_tokens_count = counts_matrix.sum(axis=1)
        # document_tokens_count_squeezed = np.squeeze(document_tokens_count)
        # zero_document_tokens_count_index = np.argwhere(document_tokens_count_squeezed ==0)[:,1]
        # 可能出详细 nan的情况
        term_frequency_idf_array = np.divide(tfidf_matrix.toarray() ,np.array(document_tokens_count))
        term_frequency_idf_array_nan_to_num = np.nan_to_num(term_frequency_idf_array)
        assert np.isnan(term_frequency_idf_array_nan_to_num).sum() == 0
        logger.info(f"训练完成 tf-idf 模型: len(vocabulary_) = {len(vocabulary_)}")
        return counter_vectorizer,tf_transformer,counts_matrix,tfidf_matrix, term_frequency_idf_array_nan_to_num,vocabulary_

    def transform(self,sentences):
        if isinstance(sentences,str):
            sentences = [sentences]

        counts_x = self.counter_vectorizer.transform([sentences])
        tfidf_x = self.tf_transformer.transform(counts_x)
        return tfidf_x


    def tf_idf(self,query,document):
        """
        使用训练好的 tf-idf模型，得到 每个 term 的idf值，
        计算 query 与 document 的 tf-idf 值
        """
        # 1. query 进行分词
        terms  = self.tokenizer(query)
        # 当 document 独立输入的时候
        # 2. 计算 term 在 document 中的数量和tf值
        # 3. 获取每个 term 的 idf 值
        counter_vectorizer = self.get_counter_vectorizer()
        count_matrix = counter_vectorizer.fit_transform(raw_documents=[document])
        len_document = len(self.tokenizer(document))
        tfidf = 0
        for term in terms:
            # 对于每个term，计算 其在 document中的数量
            term_index_in_document = counter_vectorizer.vocabulary_.get(term)
            term_count_in_document = count_matrix[0,term_index_in_document]
            term_tfidf = 0
            # term 在 corpus 的位置
            term_index_in_corpus = self.counter_vectorizer.vocabulary_.get(term)
            if term_count_in_document is not None and term_index_in_corpus is not None:
                # 获取 term 的idf 值
                term_idf = self.tf_transformer.idf_[term_index_in_corpus]
                term_tf = term_count_in_document/len_document
                term_tfidf = term_tf * term_idf
            tfidf += term_tfidf
        return tfidf

    def query_to_onehot(self,query):
        """
        将 query 转换为 one-hot 形式
        """
        # 1. query 进行分词
        terms  = self.tokenizer(query)
        # 2. 将 term 转换为 one-hot 形式
        term_encode = np.zeros(shape = (1,self.vocabulary_.__len__()))
        for term in terms:
            term_index_in_corpus = self.vocabulary_.get(term)
            if term_index_in_corpus is not None:
                term_encode[0,term_index_in_corpus]=1
        return term_encode


    def tfidf_match(self,query,topk=10):
        """
        计算单个query 与 corpus的 tf_idf 值
        """
        term_encode = self.query_to_onehot(query=query)

        # 3. 计算 query 对应于corpus的每个文档的 tf-idf
        if self.tf_as_term_frequency:
            tfidfs_array = term_encode * self.term_frequency_idf_array
        else:
            tfidfs_array = term_encode * self.tfidf_matrix.toarray()

        topk_documents = self.get_topk_documents(array=tfidfs_array,topk=topk,name='tfidf')
        return topk_documents

    def get_topk_documents(self,array,topk,name='tfidf'):
        tfidfs = array.sum(axis=1)

        # 4. 按照 tfidfs 进行 排序
        sorted_index = np.argsort(tfidfs)[::-1]
        if topk:
            topk_documents = {num:{name:tfidfs[index],'document': self.corpus[index]} for num,index in enumerate(sorted_index[:topk]) }
        else:
            topk_documents = {num: {name: tfidfs[index], 'document': self.corpus[index]} for num, index in
                              enumerate(sorted_index[:])}
        return topk_documents

    def bm25_match(self,query,topk=10):
        term_encode = self.query_to_onehot(query=query)

        # bm25 主要针对 tf 进行修正 shape = (num_doc,num_term)
        tf_ori =  self.counts_matrix.toarray() * term_encode
        # 每个 文档的长度 shape = (num_doc,1)
        field_length = np.array(self.counts_matrix.sum(axis=1))
        # 所有文档的平均长度 shape =(,)
        average_field_length = field_length.mean()
        # bm25 中改进的 tf: shape = (num_doc,num_term)
        tf_in_bm25 = tf_ori*(self.bm25_k+1)/(tf_ori + self.bm25_k*(1-self.bm25_b + self.bm25_b*(field_length/average_field_length)))

        # 使用tf-idf中idf 计算
        bm25_array = tf_in_bm25 * self.tf_transformer.idf_

        topk_documents = self.get_topk_documents(array = bm25_array,topk=topk,name='bm25')
        return topk_documents



def similarity_jaccard(words1:List[str], words2:List[str]):
    """jaccard相似度:两句子分词后词语的交集中词语数与并集中词语数之比。"""
    s1, s2 = set(words1), set(words2)
    ret1 = s1.intersection(s2)  # 交集
    ret2 = s1.union(s2)  # 并集
    sim = 1.0 * len(ret1) / len(ret2)
    return sim


def minimum_edit_distance(word1, word2,substitution_cost=1):
    """
    编辑距离相似度:
    一个句子转换为另一个句子需要的编辑次数，编辑包括删除、替换、添加，然后使用最长句子的长度归一化得相似度。

    示例 1：
    输入：word1 = "horse", word2 = "ros"
    输出：3
    解释：
    horse -> rorse (将 'h' 替换为 'r')
    rorse -> rose (删除 'r')
    rose -> ros (删除 'e')

    示例 2：

    输入：word1 = "intention", word2 = "execution"
    输出：5
    解释：
    intention -> inention (删除 't')
    inention -> enention (将 'i' 替换为 'e')
    enention -> exention (将 'n' 替换为 'x')
    exention -> exection (将 'n' 替换为 'c')
    exection -> execution (插入 'u')

    D[i,j]： the minimum edit distance between X[1..i] and Y[1..j]
    substitution_cost

    """
    n = len(word1)
    m = len(word2)
    # 初始化距离矩阵
    D = [[None]*(m+1) for i in range(0,n+1)]
    # 对于 D[i,j] 进行迭代
    for i in range(0,n+1):
        for j in range(0,m+1):
            # 初始化 当 i = 0 的时候，D[0,j] = j
            if i == 0:
                D[0][j] = j
            elif j ==0:
                D[i][0] = i
            else:
               # 定义 sub-problem
                # 1) D[i,j] 可能是  D[i-1,j-1] 经过最后一个字符 substitution 变化过来
                if word1[i-1] == word2[j-1]:
                    i_minus_j_minus = D[i-1][j-1]
                else:
                    i_minus_j_minus = D[i - 1][j - 1] + substitution_cost
                # 2） D[i,j] 也可能是 D[i,j-1] 经过 对 word1 删除一个字符
                # 3） D[i,j] 也可能是 D[i-1,j] 经过 对 word1 插入一个字符
                # 定义状态转移方程
                D[i][j] = min(i_minus_j_minus,D[i][j-1]+1,D[i-1][j] +1)
    return D[n][m]



def wmd(s1,s2):
    pass



def simhash(s1, s2):
    """先计算两句子的simhash二进制编码，然后使用海明距离计算，最后使用两句的最大simhash值归一化得相似度。"""



def train_tfidf_similarity():
    # 蚂蚁金服的数据集
    data_dir = 'data'
    dataset1 = 'atec'
    data_filename1 = 'atec_nlp_sim_train.csv'

    sentence_pair_data_reader = SentencePairDataReader(data_dir=data_dir, dataset=dataset1, data_filenames=[data_filename1],
                                                       if_read_data=True,if_overwrite=False,
                                                       if_build_triplet_data=False,if_build_sentence_label_data=True,
                                                       train_size=0.8,dev_size=0.1,test_size=0.1)
    corpus = sentence_pair_data_reader.data_df.sentence1.tolist()

    tfidf_similarity = TFIDFSimilarity(corpus=corpus)
    query = corpus[0]
    document= corpus[0]
    topk_tfidf_match_docs = tfidf_similarity.tfidf_match(query = query)
    topk_bm25_match_docs = tfidf_similarity.bm25_match(query=query)




if __name__ == '__main__':
    word1 = "horse"
    word2 = "ros"
    word1 = "intention"
    word2 = "execution"
    minimum_edit_distance(word1,word2)
