#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/12/10 10:23 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/12/10 10:23   wangfc      1.0         None
"""
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup,TextVectorization
import pprint


def create_integetlookup_layer_with_known_vocabulary(max_values=None):
    vocab = [12, 36, 1138, 42]
    data = tf.constant([[12, 1138, 42], [41, -1, 35]])
    layer = IntegerLookup(max_values=max_values)
    result = layer(data)
    pprint.pprint(result)

    # num_oov_indices 默认值 1，对于不在 vocab 的数字，映射为 index 1
    # oov_value： 所有不在 vocab，默认为 -1
    # mask_value: 默认为 0
    vocabulary = layer.get_vocabulary()
    pprint.pprint(vocabulary)


def test_text_vectorization_layer():
    max_tokens = 100
    embedding_dimension =8
    text_dataset = tf.data.Dataset.from_tensor_slices(["I foo", "you bar", "he baz"])
    text_vectorization_layer = TextVectorization(max_tokens=max_tokens,output_sequence_length=8)
    text_vectorization_layer.adapt(text_dataset.batch(8))

    vocabulary = text_vectorization_layer.get_vocabulary()
    pprint.pprint(vocabulary)

    embedding_layer = tf.keras.layers.Embedding(input_dim=max_tokens,
                                                output_dim=embedding_dimension,
                                                mask_zero=True)
    for batch in text_dataset.batch(8):
        x = text_vectorization_layer(batch)
        pprint.pprint(x)
        embedding = embedding_layer(x)
        pprint.pprint(embedding)







if __name__ == '__main__':
    # create_integetlookup_layer_with_known_vocabulary()
    # create_integetlookup_layer_with_known_vocabulary(max_values=40)
    test_text_vectorization_layer()
