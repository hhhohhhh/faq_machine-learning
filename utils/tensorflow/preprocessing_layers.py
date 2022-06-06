#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/17 14:56 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/17 14:56   wangfc      1.0         None
"""


from typing import List, Text
import tensorflow as tf
from models.features import CategoricalFeatureEmbedding

if tf.__version__ < "2.6.0":
    from tensorflow.keras.layers.experimental.preprocessing import Normalization
    # normalizaton = tf.keras.layers.experimental.preprocessing.Normalization()
    # in tf2.3.2
    from tensorflow.keras.layers.experimental.preprocessing import StringLookup
    from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup
    from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
    from tensorflow.keras.layers.experimental.preprocessing import Discretization
else:
    from tensorflow.keras.layers import StringLookup
    from tensorflow.keras.layers import IntegerLookup
    from tensorflow.keras.layers import Normalization
    from tensorflow.keras.layers import TextVectorization
    from tensorflow.keras.layers import Discretization


def create_lookup_layer(vocabulary: List[Text], mask_token="[MASK]", oov_token="[UNK]") -> StringLookup:
    """
    参考
    # from models.features import CategoricalFeatureEmbedding

    """
    if mask_token:
        #  StringLookup 是否 含有 mask_token
        vocab_size = len(vocabulary) + 2
    else:
        #  StringLookup 默认含有 oov_token="[UNK]"
        vocab_size = len(vocabulary) + 1
    # self.vocab_size = vocab_size

    lookup_layer = StringLookup(max_tokens=vocab_size, mask_token=mask_token, oov_token=oov_token,
                                vocabulary=vocabulary)
    return lookup_layer