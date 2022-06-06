#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/8/4 8:51 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/4 8:51   wangfc      1.0         None
"""
import pytest
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from tokenizations.tokenization import whitespace_tokenize


@pytest.mark.parametrize(
    "sentence, ngram_range, vocabulary,sequence_vec_expected, sentence_vec_expected",
    [
        ("john guy",(1,1),{'john': 1, 'guy': 0},  [[0, 1],[1, 0]], [[1, 1]]),

        ("john guy",(1,2),{'john': 1, 'guy': 0, 'john guy':2},  [[0, 1, 0],[1, 0, 0]], [[1, 1, 1]]),
    ],
)


def test_countervectorizer(sentence,
                           ngram_range,
                           vocabulary,
                           sequence_vec_expected,
                           sentence_vec_expected):
    """
    每个 token 会生成对应的一个 one-hot 稀疏特征
    当 ngram_range(1,2) 的时候，会将2个token组成一个词， 该词没有对应的token
    """
    # tokenizer 进行 tokenize
    tokens = whitespace_tokenize(sentence)
    vectorizer = CountVectorizer(ngram_range=ngram_range,max_features=None)
    # fit 参数 raw_document 是 iterable对象,
    # 1.对 sentence 外面加 list
    # 2.对 所有的 tokens 进行训练
    vectorizer.fit([sentence])
    print(vectorizer.vocabulary_)
    # assert  vectorizer.vocabulary_ == vocabulary

    # 对sequence vector :对每个 tokens 转换为 one-hot encoding: shape= (# token, # vocabulary)
    # 对整句话的 进行 vector, 生产 multi-hot encoding: shape = (1, # vocabulary)

    def process(raw_documents):
        """
         # transform -> sort_indices -> tocoo > array
        """
        vec = vectorizer.transform(raw_documents)
        vec.sort_indices()
        vec = vec.tocoo()
        assert isinstance(vec,scipy.sparse.coo_matrix)
        vec_array = vec.toarray()
        return vec_array

    sequence_vec_array = process(tokens)
    # assert np.all(sequence_vec_array == sequence_vec_expected)

    sentence= " ".join(tokens)
    sentence_vec_array = process([sentence])
    # assert np.all(sentence_vec_array == sentence_vec_expected)





# if __name__ == '__main__':

#     document = ['john guy','nice guy']
#     vectorizer = CountVectorizer(ngram_range=(1, 1))
#     X = vectorizer.fit_transform(document)
#     vocabulary = vectorizer.vocabulary_
#     print(vocabulary)
#     print(X)
#     X.sort_indices()
#     print(X)
#     X_coor = X.tocoo()
#     print(type(X_coor))

    # # 对一句话进行特征话
    # sentence = document[0]
    # sentence_feature = vectorizer.transform([sentence])
    # print(sentence_feature.shape) # (1, 3)
    # tokens = sentence.split()
    # sequence_feature = vectorizer.transform(tokens)
    # print(sequence_feature.shape) # (2, 3)
