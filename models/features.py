#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/12/8 9:04 

feature:
    user_id/item_id/user_zip_code/user_occupation_text :  string categorical feature
    user_gender/bucketized_user_age : integer categorical feature
        be translated into embedding vectors: high-dimensional numerical representations
        that are adjusted during training to help the model predict its objective better.

        将高维 sparse 特征转化为低维的 embedding 方式：
        1.  one-hot index -> embedding层转换为稠密特征
        2.  hash 方式

    timestamp: numerical continuous feature
        be normalized so that their values lie in a small interval around 0.

        embedding方式：
        标准化 -> 分割化 —> embedding 转换为稠密特征

    movie_title/item_description: text feature
        item_description: text feature
        be tokenized (split into smaller parts such as individual words) and translated into embeddings.

        embedding方式:

        1. tokenization -> multihot-index -> embedding -> pooling
        2. tokenization -> multihot-index -> tf-idf feature


多种 feature 设计方法 & 数据输入格式设计 ：
1） 根据 attribute 设计多种不同类型的 feature： String，Integer， Continuous，Text 等等
分布建立对应的 feature embedding layer，
模型计算的时候通过 attribute 获取对应的数据 和 对应的layer，可以计算
Error: 但是在保存的时候无法顺利保存

2) rasa
attribute:  TEXT, LABEL, ENTITIES
首先对 text 进行 sentence 和 sequence 部分的多种特征的 embedding 编码，最后合并为一个 tensor

3） TODO: 如何结合 rasa 模式?
总体思路： 将 input_attribute 最后转换为一个 tensor
attribute







"""
from typing import Dict, Text, Union, List, Optional, Any
import tensorflow as tf

# from rasa.nlu.config import RasaNLUModelConfig
# from rasa.shared.nlu.training_data.training_data import TrainingData
# from rasa.nlu.featurizers.featurizer import DenseFeaturizer
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Layer

FEATURE_TYPE = 'feature_type'
FEATURE_TYPES = 'feature_types'
CATEGORICAL_FEATURE_TYPE = "categorical_feature"
STRING_CATEGORICAL_FEATURE_TYPE = "string_categorical_feature"
INTEGER_CATEGORICAL_FEATURE_TYPE = "integer_categorical_feature"
CONTINUOUS_FEATURE_TYPE = "continuous_feature"
TEXT_FEATURE_TYPE = 'text_feature'

VOCABULARY = "vocabulary"
BINS = "bins"
MASK_VALUE = "mask_value"
MAX_VALUES = "max_values"
OUTPUT_SEQUENCE_LENGTH = "output_sequence_length"
MAX_TOKENS = "max_tokens"

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


class Feature():
    def __init__(self, name: Text, feature_type: Text, embedding_dimension: int = 32,
                 vocabulary: List[Text] = None, mean=None, variance=None, buckets=None, max_tokens=None):
        self.name = name
        # CATEGORICAL_FEATURE_TYPE: 离散的特征
        # CONTINUOUS_FEATURE_TYPE: 连续特征
        self.feature_type = feature_type
        self.embedding_dimension = embedding_dimension
        self.vocabulary = vocabulary

        self.mean = mean
        self.variance = variance
        self.buckets = buckets
        self.max_tokens = max_tokens

    @property
    def parameters_dict(self):
        return {"name": self.name, "feature_type": self.feature_type,
                "embedding_dimension": self.embedding_dimension,
                "vocabulary": self.vocabulary,
                "mean": self.mean, "variance": self.variance, 'buckets': self.buckets,
                "max_tokens": self.max_tokens
                }

    def __repr__(self):
        output_string = f"{self.__class__.__name__}:name={self.name}," \
                        f"feature_type={self.feature_type}," \
                        f"embedding_dimension={self.embedding_dimension}"
        if self.feature_type == CATEGORICAL_FEATURE_TYPE:
            return output_string + "," + f"vocabulary_size = {self.vocabulary.__len__()}"


class CategoricalFeature(Feature):
    """
    user_id/item_id/user_zip_code/user_occupation_text :  string categorical feature
    user_gender/bucketized_user_age : integer categorical feature
    """

    def __init__(self,
                 vocabulary: List[Text] = None, **kwargs):
        super(CategoricalFeature, self).__init__(**kwargs)
        self.vocabulary = vocabulary

    def __repr__(self):
        output_string = f"{self.__class__.__name__}:name={self.name}," \
                        f"feature_type={self.feature_type}," \
                        f"embedding_dimension={self.embedding_dimension}," \
                        f"vocabulary_size = {self.vocabulary.__len__()}"
        return output_string


class StringCategoricalFeature(CategoricalFeature):
    """
    user_id/item_id/user_zip_code/user_occupation_text :  string categorical feature
    user_gender/bucketized_user_age : integer categorical feature
    """
    pass


class IntegerCategoricalFeature(CategoricalFeature):
    """
    user_id/item_id/user_zip_code/user_occupation_text :  string categorical feature
    user_gender/bucketized_user_age : integer categorical feature
    """
    pass


class ContinuousFeature(Feature):
    """
    连续特征
    """
    pass

    # def __init__(self, mean=None, variance=None, buckets=None, **kwargs):
    #     super(ContinuousFeature, self).__init__(**kwargs)
    #     self.mean = mean
    #     self.variance = variance
    #     self.buckets = buckets
    #
    # def __repr__(self):
    #     output_string = f"{self.__class__.__name__}:name={self.name}," \
    #                     f"feature_type={self.feature_type}," \
    #                     f"embedding_dimension={self.embedding_dimension}," \
    #                     f"mean = {self.mean}," \
    #                     f"variance = {self.variance},"
    #     return output_string


class TextFeature(Feature):
    """
    连续特征
    """

    def __init__(self, max_tokens=None, **kwargs):
        super(TextFeature, self).__init__(**kwargs)
        self.max_tokens = max_tokens

    def __repr__(self):
        output_string = f"{self.__class__.__name__}:name={self.name}," \
                        f"feature_type={self.feature_type}," \
                        f"embedding_dimension={self.embedding_dimension}," \
                        f"max_tokens = {self.max_tokens}"
        return output_string


class FeatureEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 name=None,
                 embedding_dimension=32,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self._feature_name = name
        self._embedding_dimension = embedding_dimension
        self.args= args
        self.kwargs = kwargs

    def _prepare_layers(self):
        raise NotImplementedError

    def call(self, inputs, **kwargs):
        raise NotImplementedError


class CategoricalFeatureEmbedding(FeatureEmbedding):
    """

    将 categorical feature 转换为 embedding：
    A categorical feature is a feature that does not express a continuous quantity,
    but rather takes on one of a set of fixed values.
    """

    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super(CategoricalFeatureEmbedding,self).__init__(*args,**kwargs)
        self._vocabulary = self.kwargs["vocabulary"]
        self._vocab_size = len(self._vocabulary) + 1
        self._prepare_layers()

    def _prepare_layers(self):
        """
        对于 categorical features：
        Taking raw categorical features and turning them into embeddings is normally a two-step process:
            Firstly, we need to translate the raw values into a range of contiguous integers, normally by building a mapping (called a "vocabulary") that maps raw values ("Star Wars") to integers (say, 15).
            Secondly, we need to take these integers and turn them into embeddings.

        string_lookup_layer:
        map the raw values of our categorical features to embedding vectors in our models.
        To do that, we need a vocabulary that maps a raw feature value to an integer in a contiguous range:
        this allows us to look up the corresponding embeddings in our embedding tables.
        """
        # in tf2.3.2
        self.lookup_layer = StringLookup(vocabulary=self._vocabulary, mask_token=None)

        # We add an additional embedding to account for unknown  tokens.
        # vocab_size = self.lookup_layer.vocab_size()  # len(self.vocabulary) + 1
        self.embedding_layer = tf.keras.layers.Embedding(self._vocab_size, self._embedding_dimension)

    def call(self, inputs):
        """
        use Keras preprocessing layers to first convert user ids to integers,
        and then convert those to user embeddings via an Embedding layer.
        Note that we use the list of unique user ids we computed earlier as a vocabulary:
        """
        x = self.lookup_layer(inputs)
        x = self.embedding_layer(x)
        return x


class StringCategoricalFeatureEmbedding(CategoricalFeatureEmbedding):
    pass


class IntegerCategoricalFeatureEmbedding(CategoricalFeatureEmbedding):
    # def __init__(self, **kwargs):
    #     super(IntegerCategoricalFeatureEmbedding, self).__init__(**kwargs)

    def _prepare_layers(self):
        """
        对于 categorical features：
        Taking raw categorical features and turning them into embeddings is normally a two-step process:
            Firstly, we need to translate the raw values into a range of contiguous integers, normally by building a mapping (called a "vocabulary") that maps raw values ("Star Wars") to integers (say, 15).
            Secondly, we need to take these integers and turn them into embeddings.

        string_lookup_layer:
        map the raw values of our categorical features to embedding vectors in our models.
        To do that, we need a vocabulary that maps a raw feature value to an integer in a contiguous range:
        this allows us to look up the corresponding embeddings in our embedding tables.
        """
        # in tf2.3.2
        # max_value 必须大于1：任何大于max_value 都算是 oov
        # vocabulary 必须加入或者 使用 .adapt() 方法学习
        # mask_value = None
        #  mask_value 于 vocabulary 冲突怎么办 ？
        self.lookup_layer = IntegerLookup(**self.kwargs)

        # We add an additional embedding to account for unknown  tokens.
        vocab_size = self.lookup_layer.vocab_size()  # len(self.vocabulary) + 1
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, self._embedding_dimension)


class ContinuousFeatureEmbedding(FeatureEmbedding):
    def __init__(self,
                 # name=None,
                 # embedding_dimension=32,
                 *args,**kwargs
                 # mean=None,
                 # variance=None,
                 # buckets=List[float], **kwargs
                 ):
        super(ContinuousFeatureEmbedding,self).__init__(*args,**kwargs)
        self._bins = self.kwargs.pop("bins")
        self._prepare_layers()
        # self._feature_name = name
        # self._embedding_dimension = embedding_dimension
        # self._mean = mean
        # self._variance = variance



    def _prepare_layers(self):
        """
        https://keras.io/guides/preprocessing_layers/#preprocessing-data-before-the-model-or-inside-the-model

        预处理层： 可以放在 in the model 或者 before the model (eg. in dataset)
        Option 1: Make them part of the model, like this:

        inputs = keras.Input(shape=input_shape)
        x = preprocessing_layer(inputs)
        outputs = rest_of_the_model(x)
        model = keras.Model(inputs, outputs)

        Option 2: apply it to your tf.data.Dataset, so as to obtain a dataset that yields batches of preprocessed data
        dataset = dataset.map(lambda x, y: (preprocessing_layer(x), y))

        """
        # error: Normalization Keyword argument not understood:', 'mean'
        # 将 normalization_layer before the model
        self.normalization_layer = Normalization()

        self.discretization_layer = Discretization(bins=self._bins,**self.kwargs)
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=self._bins.__len__() + 1,
                                                         output_dim=self._embedding_dimension)

    def adapt(self, inputs):
        """
        在训练之前对 normalization_layer 进行初始化
        """
        self.normalization_layer.adapt(data=inputs)

    def call(self, inputs):
        # 先对数据进行 normalization
        x = self.normalization_layer(inputs)
        # 做离散化处理
        x = self.discretization_layer(x)
        # 转换为 embedding: [batch_size,1, embedding_size]
        x = self.embedding_layer(x)
        x = tf.squeeze(x)
        return x


class TextFeatureEmbedding(FeatureEmbedding):
    """
    对sparse特征进行embedding:
    1) Encoding text as a dense matrix of ngrams with multi-hot encoding：
       使用 multi-hot 方式转换 sparse特征，embedding之后再做一个简单的average pooling；
    2） Encoding text as a dense matrix of ngrams with TF-IDF weighting：

    """

    def __init__(self,
                 *args,**kwargs,
                 # name=None,
                 # embedding_dimension=32,
                 # max_tokens=10000,
                 ):
        super().__init__(*args,**kwargs)
        # self._feature_name = name
        # self._embedding_dimension = embedding_dimension
        self._max_tokens = self.kwargs["max_tokens"]
        self._prepare_layers()

    def _prepare_layers(self):
        """
        The first transformation we need to apply to text is tokenization
         (splitting into constituent words or word-pieces),
        followed by vocabulary learning, followed by an embedding.
        """
        # Create a TextVectorization layer:
        # Index the vocabulary via `adapt()`,
        # 'Keyword argument not understood:', 'vocabulary'
        # 无法直接使用 vocabulary 参数进行初始化，我们将其放在 _preprocess_data() pipeline中
        self.text_vectorization_layer = TextVectorization(**self.kwargs)

        # create embedding layer
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=self._max_tokens,
                                                         output_dim=self._embedding_dimension,
                                                         mask_zero=True)

        self.avg_pooling_layer = tf.keras.layers.GlobalAveragePooling1D()

    def adapt(self,inputs):
        # for input in inputs.take(1).as_numpy_iterator():
        #     print(input)
        self.text_vectorization_layer.adapt(inputs)

    def call(self, inputs, **kwargs):
        # 对 input string 进行 tokenization，并且 padding 到同一个长度
        x = self.text_vectorization_layer(inputs)
        # 将 tokens 转换为 embedding
        x = self.embedding_layer(x)
        # 进行 ave_pooling 获取整个句子的 embedding
        x = self.avg_pooling_layer(x)
        return x


class AttributeEmbeddingFeaturizer():
    """
    将 attribute 转换为 feature embedding
    """

    def __init__(self, attribute: Text = None, feature_types: List[Text] = None,
                 embedding_dimension: int = 32, *args, **kwargs):
        """
         对 feature 进行初始化，准备初始的参数
        """
        self.attribute = attribute
        # CATEGORICAL_FEATURE_TYPE: 离散的特征
        # CONTINUOUS_FEATURE_TYPE: 连续特征
        self.feature_types = feature_types
        self.embedding_dimension = embedding_dimension
        self.args = args
        self.kwargs = kwargs

    def get_feature_type_to_embedding_layer(self) -> Dict[Text, FeatureEmbedding]:
        # 对不同的 feature_type 生成不同的 embedding_layer
        feature_type_to_embedding_layer = {}
        for feature_type in self.feature_types:
            feature_embedding_layer = self._get_embedding_layer(feature_type)
            feature_type_to_embedding_layer.update({feature_type: feature_embedding_layer})
        return feature_type_to_embedding_layer

    def _get_embedding_layer(self, feature_type=Text) -> FeatureEmbedding:
        # 对不同的 feature_type 获取 embedding layer
        if feature_type == STRING_CATEGORICAL_FEATURE_TYPE:
            feature_embedding_layer_class = StringCategoricalFeatureEmbedding
        elif feature_type == INTEGER_CATEGORICAL_FEATURE_TYPE:
            feature_embedding_layer_class = IntegerCategoricalFeatureEmbedding
        elif feature_type == CONTINUOUS_FEATURE_TYPE:
            feature_embedding_layer_class = ContinuousFeatureEmbedding
        elif feature_type == TEXT_FEATURE_TYPE:
            feature_embedding_layer_class = TextFeatureEmbedding

        # 初始化
        feature_embedding_layer = feature_embedding_layer_class(name=self.attribute,
                                                                embedding_dimension=self.embedding_dimension,
                                                                **self.kwargs)

        return feature_embedding_layer
