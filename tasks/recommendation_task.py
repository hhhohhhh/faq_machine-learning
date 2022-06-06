#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/18 10:29 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/18 10:29   wangfc      1.0         None
"""
import os
import tempfile
from typing import Text, Union, List, Dict, Any, Tuple
from copy import deepcopy
import pprint
import functools
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

# from matplotlib.font_manager import json_dump, json_load

from data_process.dataset.hsnlp_faq_knowledge_dataset import ATTRIBUTE
from models.features import FEATURE_TYPE, STRING_CATEGORICAL_FEATURE_TYPE, INTEGER_CATEGORICAL_FEATURE_TYPE, \
    AttributeEmbeddingFeaturizer, VOCABULARY, MAX_TOKENS, BINS, MAX_VALUES, FEATURE_TYPES, MASK_VALUE, \
    OUTPUT_SEQUENCE_LENGTH

if tf.__version__ < "2.6.0":
    from tensorflow.keras.layers.experimental.preprocessing import Normalization, TextVectorization
else:
    from tensorflow.keras.layers import Normalization

from rasa.utils.io import pickle_dump, pickle_load

import tensorflow_recommenders as tfrs
from models.builder import MODELS
from models.retrieval_models.two_tower_model import TwoTowerRetrievalModel, TwoTowerRankingModel, Feature, \
    CATEGORICAL_FEATURE_TYPE, TEXT_FEATURE_TYPE, CONTINUOUS_FEATURE_TYPE, CategoricalFeature, ContinuousFeature, \
    TextFeature, MultiTaskRecommendationModel
from tasks.tf_task import BaseTfTask
from utils.io import dump_json, load_json
from utils.tensorflow.model_data import ModelData
from utils.tensorflow.tf_basics import calculate_mean_and_variance_from_dataset, calculate_min_and_max_from_dataset
import logging

logger = logging.getLogger(__name__)


class RecommendationTask(BaseTfTask):
    def __init__(self,
                 use_retrieval_task=True,
                 use_ranking_task=True,
                 retrieval_weight=1.0,
                 ranking_weight=1.0,
                 tower_type: Text = "TowerModel",
                 input_data_extract_attributes: List[Text] = None,
                 candidate_data_extract_attributes: List[Text] = None,
                 candidate_to_index_attribute: Text = None,
                 ranking_attribute: Text = None,
                 query_attributes: List[Text] = None,
                 candidate_attributes: List[Text] = None,
                 embedding_dimension=32,
                 hidden_layer_sizes: List[int] = None,
                 *args, **kwargs):
        super(RecommendationTask, self).__init__(*args, **kwargs)
        # 使用多任务模型的时候
        self.use_retrieval_task = use_retrieval_task
        self.use_ranking_task = use_ranking_task
        self.retrieval_weight = retrieval_weight
        self.ranking_weight = ranking_weight

        # tower_type 用于选择不同基础模型
        self.tower_type = tower_type

        # 原始数据中需要提取的属性
        self.input_data_extract_attributes = input_data_extract_attributes
        # candidate data 中提取的属性
        self.candidate_data_extract_attributes = candidate_data_extract_attributes

        # 检索目标的属性名称,用于建立索引
        self.candidate_to_index_attribute = candidate_to_index_attribute
        # ranking目前的属性名称
        self.ranking_attribute = ranking_attribute

        # query 中的属性，可以是多个属性，并且每个属性可以生成多种不同的特征
        self.query_attributes = query_attributes
        # candidate 中的属性
        self.candidate_attributes = candidate_attributes

        # the dimensionality of the query and candidate representations
        self.embedding_dimension = embedding_dimension
        self.hidden_layer_sizes = hidden_layer_sizes

        self.candidate_dataset = None
        # self.input_names = self._get_input_names(**kwargs)

        # 用于 多种特征的输入
        # depreciated:
        # self.query_features: List[Feature] = None
        # self.candidate_features: List[Feature] = None

        # 使用 featurizers 来从 attribute 转换为 embedding
        self.query_featurizers: List[AttributeEmbeddingFeaturizer] = None
        self.candidate_featurizers: List[AttributeEmbeddingFeaturizer] = None

        self._meta_data_path = os.path.join(self.output_dir, 'meta_data.json')
        self._query_vocabulary_path = os.path.join(self.output_dir, 'query_vocabulary.pkl')
        self._candidate_vocabulary_path = os.path.join(self.output_dir, 'candidate_vocabulary.pkl')

        # 多种特征的时候，保存为 features.pkl
        self._query_features_path = os.path.join(self.output_dir, 'query_features.pkl')
        self._candidate_features_path = os.path.join(self.output_dir, 'candidate_features.pkl')

        self._candidate_dir = os.path.join(self.output_dir, 'candidate')
        self._model_dir = os.path.join(self.output_dir, f'{self.task}_model', 'checkpoint')  # 'tf_model.pb')
        self._index_dir = os.path.join(self.output_dir, "index")

    def train(self):
        raise NotImplementedError

    def _get_input_names(self, **kwargs):
        input_names = None
        if kwargs.get('input_names'):
            input_names = kwargs.get('input_names')
            if isinstance(input_names, str):
                input_names = [input_names]
        return input_names

    def _create_features(self, attributes: List[Text],
                         # 使用 attribute_to_feature_parameters 代替 attribute_to_feature_type 和 attribute_to_vocabulary
                         attribute_to_feature_parameters=Dict[Text, Dict[Text, Any]]
                         # attribute_to_feature_type: Dict[Text, Text],
                         # attribute_to_vocabulary: Dict[Text, List[Text]] = None,
                         # embedding_dimension: int = 32
                         ) -> List[Feature]:
        features = []
        # 对于对应的 attributes
        for attribute in attributes:
            # 判断该 attribute 是否在我们要转换的特征字典中
            if attribute in attribute_to_feature_parameters:
                # # 获取对应的 feature_type
                # feature_type = attribute_to_feature_type[attribute]
                # # 获取对应的 vocabulary
                # vocabulary = attribute_to_vocabulary[attribute]
                feature_parameters = attribute_to_feature_parameters[attribute]
                feature = self._create_feature(name=attribute,
                                               **feature_parameters
                                               # feature_type=feature_type,
                                               # vocabulary=vocabulary,
                                               # embedding_dimension=embedding_dimension
                                               )
            features.append(feature)
        return features

    def _create_featuriers(self, attributes: List[Text]) -> List[AttributeEmbeddingFeaturizer]:
        # 获取 feature_type
        featurizers = []
        for attribute in attributes:
            feature_parameters = self.attribute_to_feature_parameters[attribute]
            featurizer = AttributeEmbeddingFeaturizer(**feature_parameters)
            featurizers.append(featurizer)
        return featurizers

    def _create_feature(self, name: Text, feature_type: Text,
                        embedding_dimension: int = 32, **kwargs) -> List[Feature]:
        if feature_type == CATEGORICAL_FEATURE_TYPE:
            feature_class = CategoricalFeature
        elif feature_type == CONTINUOUS_FEATURE_TYPE:
            feature_class = ContinuousFeature
        elif feature_type == TEXT_FEATURE_TYPE:
            feature_class = TextFeature

        feature = feature_class(name=name,
                                feature_type=feature_type,
                                embedding_dimension=embedding_dimension,
                                # vocabulary=vocabulary,  # 需要 attribute 对应的
                                **kwargs)
        return feature

    def _get_model_class(self, model_name: Text) -> Union[TwoTowerRetrievalModel, TwoTowerRankingModel]:
        """
        重写 _get_model() 方法
        """
        module = MODELS.get(model_name)
        return module

    def _instance_model_class(self, model_data: ModelData = None) -> \
            Union[TwoTowerRetrievalModel, TwoTowerRankingModel, MultiTaskRecommendationModel]:
        """
        参考 rasa.nlu.classifier.diet_classifer.DietClassifier._instantiate_model_class() 方法

        1) model._set_inputs() 方法
        2) 在模型初始化的时候提供  data_signature，确保可以保存为 saved_model
        """
        # 获取 model_class
        model_class = self._get_model_class(model_name=self.model_name)

        # 初始化
        model = model_class(
            # data_signuare = model_data.get_signature,
            # query_features=self.query_features,
            # candidate_features=self.candidate_features,
            query_featurizers=self.query_featurizers,
            candidate_featurizers=self.candidate_featurizers,
            hidden_layer_sizes=self.hidden_layer_sizes,
            candidate_dataset=self.candidate_dataset,
            ranking_attribute=self.ranking_attribute,
            **self.model_config)
        return model

    def _set_model_inputs(self, inputs):
        # 使用 model._set_inputs方法设置 saved_model_inputs_spec： 用于保存为 saved_model 和加载模型使用
        self.model.input_names = self.input_names

        # 手动输入 inputs
        # inputs = tf.TensorSpec(shape=(None,), dtype=tf.string, name='user_id')
        self.model._set_inputs(inputs=inputs)

    def _get_vocabulary(self):
        """
        获取 query 和 candidate 的 vocabulary：
        map the raw values of our categorical features to embedding vectors in our models.
        To do that, we need a vocabulary that maps a raw feature value to an integer in a contiguous range:
        this allows us to look up the corresponding embeddings in our embedding tables.
        """
        raise NotImplementedError

    def _split_data(self):
        raise NotImplementedError

    def _get_metrics(self):
        metric = tfrs.metrics.FactorizedTopK
        return metric

    def persist(self):
        if hasattr(self, 'query_vocabulary') and hasattr(self, 'candidate_vocabulary'):
            self._persist_vocabulary(query_vocabulary=self.query_vocabulary,
                                     candidate_vocabulary=self.candidate_vocabulary)
            # 生成一个 测试输入数据
            test_input = tf.constant(["42"])
        if hasattr(self, 'query_features') and hasattr(self, 'candidate_features'):
            self._persist_query_candidate_features()
            # input 为字典形式的
            test_input = {self.user_id_attribute: tf.constant(["42"])}

        self._persist_candidate(candidate_dataset=self.candidate_dataset)
        self._persist_model()
        self.index = self._get_index(candidate_dataset=self.candidate_dataset)
        # 首先input_data 然后保存模型
        self.index(test_input)
        self._persist_index()
        # self._persist_metadata()

    def _persist_vocabulary(self, query_vocabulary, candidate_vocabulary):
        pickle_dump(filename=self._query_vocabulary_path, obj=query_vocabulary)
        pickle_dump(filename=self._candidate_vocabulary_path, obj=candidate_vocabulary)

    def _load_vocabulary(self):
        query_vocabulary = pickle_load(filename=self._query_vocabulary_path)
        candidate_vocabulary = pickle_load(filename=self._candidate_vocabulary_path)
        return query_vocabulary, candidate_vocabulary

    # 对于多种特征，我们应该保存 feature
    def _persist_query_candidate_features(self):
        self._persist_features(feature_type='query', features=self.query_features, path=self._query_features_path)
        self._persist_features(feature_type='candidate', features=self.candidate_features,
                               path=self._candidate_features_path)

    def _persist_features(self, feature_type, features, path):
        pickle_dump(filename=path, obj=features)

    def _load_query_candidate_features(self):
        query_features = self._load_features(feature_type='query', path=self._query_features_path)
        candidate_feature = self._load_features(feature_type='candidate', path=self._candidate_features_path)
        return query_features, candidate_feature

    def _load_features(self, feature_type, path):
        features = pickle_load(filename=path)
        return features

    def _persist_candidate(self, candidate_dataset):
        # pickle_dump(filename=self._candidate_path, obj=candidate_dataset)
        tf.data.experimental.save(candidate_dataset, path=self._candidate_dir)

    def _load_candidate(self):
        # candidate_dataset = pickle_load(filename=self._candidate_path)
        # element_spec = tf.TensorSpec(shape=(None,), dtype=tf.string)
        element_spec = {'movie_title': tf.TensorSpec(shape=(), dtype=tf.string, name=None)}
        candidate_dataset = tf.data.experimental.load(path=self._candidate_dir,
                                                      element_spec=element_spec)
        return candidate_dataset

    def _persist_model(self):
        """
        1. 保存模型的时候，需要设定 input_shape 或者 saved_model_input_spec

        Model cannot be saved because the input shapes have not been set.
        Usually, input shapes are automatically determined from calling `.fit()` or `.predict()`.
        To manually set the shapes, call `model.build(input_shape)`.

        2.
        When subclassing the `Model` class, you should implement a `call` method.
        """
        # 保存为 saved model
        # # tf.saved_model.save(self.model, export_dir=self._export_model_dir)
        # tf.keras.models.save_model(model=self.model, filepath=filepath)

        # 　保存为 checkpoint
        # TODO: 多个特征的时候，如何保存模型呢？
        self.model.save_model(model_filepath=self._model_dir)

    def _load_model(self):
        # model = tf.saved_model.load(export_dir=self._export_model_dir)
        # model  = tf.keras.models.load_model(filepath = self._export_model_dir)

        model_class = self._get_model_class(model_name=self.model_name)
        if hasattr(self, 'query_vocabulary') and hasattr(self, 'candidate_vocabulary'):
            model = model_class.load_model(model_filepath=self._model_dir,
                                           query_vocabulary=self.query_vocabulary,
                                           candidate_vocabulary=self.candidate_vocabulary,
                                           candidate_dataset=self.candidate_dataset,
                                           )
        if hasattr(self, 'query_features') and hasattr(self, 'candidate_features'):
            # 使用 features 进行初始化
            model = model_class.load_model(model_filepath=self._model_dir,
                                           query_features=self.query_features,
                                           candidate_features=self.candidate_features,
                                           candidate_dataset=self.candidate_dataset,
                                           )

        return model

    def _get_index(self, candidate_dataset=None, use_scann_index=False):
        if os.path.exists(self._index_dir):
            index = self._load_index()
        else:
            index = self._build_index(candidate_dataset=candidate_dataset,
                                      use_scann_index=use_scann_index)
        return index

    def _build_index(self, candidate_dataset, use_scann_index=True):
        """
        如何建立 index
        """
        if not use_scann_index:
            # Create a model that takes in raw query features
            index = tfrs.layers.factorized_top_k.BruteForce(self.model.query_model)
            # recommends movies out of the entire movies dataset.
            # 建立索引:  movie_title 与 经过  candidate_model 获取的 embedding 的对应关系
            index.index_from_dataset(
                tf.data.Dataset.zip((  # candidate_dataset.batch(100)
                    candidate_dataset.batch(100).map(lambda x: x[self.candidate_to_index_attribute]),
                    candidate_dataset.batch(100).map(self.model.candidate_model))
                )
            )
        else:
            index = tfrs.layers.factorized_top_k.ScaNN(self.model.query_model)
            index.index_from_dataset(
                tf.data.Dataset.zip((candidate_dataset.batch(100).map(lambda x: x[self.candidate_to_index_attribute]),
                                     candidate_dataset.batch(100).map(self.model.candidate_model)))
            )

        return index

    def _persist_index(self, use_scann_index=True):
        """
        保存 index 的时候，需要设定 input_shape 或者 saved_model_input_spec ？
        """
        # Export the query model.
        # with tempfile.TemporaryDirectory() as tmp:
        # Save the index.
        if not use_scann_index:
            tf.saved_model.save(self.index, self._index_dir)
        else:
            tf.saved_model.save(
                self.index,
                self._index_dir,
                options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
            )

    def _load_index(self):
        # Load it back; can also be done in TensorFlow Serving.
        loaded_index = tf.saved_model.load(self._index_dir)
        return loaded_index

    def load(self):
        if os.path.exists(self._query_vocabulary_path) and \
                os.path.exists(self._candidate_features_path):
            self.query_vocabulary, self.candidate_vocabulary = self._load_vocabulary()
        if os.path.exists(self._query_features_path) and \
                os.path.exists(self._candidate_features_path):
            self.query_features, self.candidate_features = self._load_query_candidate_features()
        self.candidate_dataset = self._load_candidate()
        self.model = self._load_model()
        self.index = self._get_index(candidate_dataset=self.candidate_dataset)


class MovielensRecommendationTask(RecommendationTask):
    def __init__(self,
                 task='retrieval',
                 corpus='corpus',
                 dataset='movielens_100k',
                 model_name='TwoTowerRetrievalModel',
                 tower_type="TowerModel",
                 user_id_attribute='user_id',
                 timestamp_attribute="timestamp",
                 ranking_attribute='user_rating',
                 movie_id_attribute='movie_id',
                 user_gender_attribute='user_gender',
                 user_zip_code_attribute='user_zip_code',
                 user_occupation_text_attribute='user_occupation_text',
                 bucketized_user_age_attribute='bucketized_user_age',
                 movie_title_attribute='movie_title',
                 movie_title_text_attribute='movie_title_text',

                 # 使用多个 特征
                 # query_attributes=['user_id', 'timestamp'],
                 # candidate_attributes=['movie_title'],

                 output_dir='output/movielens_recommendation_02',
                 *args,
                 **kwargs):

        super(MovielensRecommendationTask, self).__init__(
            task=task, model_name=model_name, tower_type=tower_type,
            corpus=corpus, dataset=dataset, output_dir=output_dir,
            candidate_to_index_attribute=movie_title_attribute,
            ranking_attribute=ranking_attribute,
            *args,
            **kwargs)

        # 原始数据中需要提取的属性
        self.input_data_extract_attributes = [user_id_attribute, timestamp_attribute,
                                              movie_id_attribute, user_gender_attribute, user_zip_code_attribute,
                                              user_occupation_text_attribute, bucketized_user_age_attribute,
                                              movie_title_attribute, ranking_attribute
                                              ]

        # candidate data 中提取的属性
        self.candidate_data_extract_attributes = [movie_title_attribute]

        # QUERY 中的属性,增加 movie_title_text_attribute
        self.query_attributes = self.input_data_extract_attributes.copy() + [movie_title_text_attribute]
        # query_attributes 中  movie_title_attribute 不作为一个embedding 属性
        self.query_attributes.remove(movie_title_attribute)
        self.query_attributes.remove(ranking_attribute)

        # candidate 的属性，用于编码 candidate
        self.candidate_attributes = [movie_title_text_attribute]

        # attribute -> Feature
        self.user_id_attribute = user_id_attribute
        self.timestamp_attribute = timestamp_attribute
        self.movie_id_attribute = movie_id_attribute
        self.user_gender_attribute = user_gender_attribute
        self.user_zip_code_attribute = user_zip_code_attribute
        self.user_occupation_text_attribute = user_occupation_text_attribute
        self.bucketized_user_age_attribute = bucketized_user_age_attribute
        self.movie_title_attribute = movie_title_attribute
        # 在预处理的时候，新增一个属性 movie_title_text_attribute
        self.movie_title_text_attribute = movie_title_text_attribute

        # 存储 attribute 对应的feature 参数信息
        self.attribute_to_feature_parameters = \
            {self.user_id_attribute: {FEATURE_TYPES: [STRING_CATEGORICAL_FEATURE_TYPE]},
             self.movie_id_attribute: {FEATURE_TYPES: [STRING_CATEGORICAL_FEATURE_TYPE]},

             # user_zip_code 为字符串
             self.user_zip_code_attribute: {FEATURE_TYPES: [STRING_CATEGORICAL_FEATURE_TYPE]},
             self.user_occupation_text_attribute: {FEATURE_TYPES: [STRING_CATEGORICAL_FEATURE_TYPE]},

             # user_gender 是 bool 类型， bucketized_user_age 是整数类型
             self.user_gender_attribute: {FEATURE_TYPES: [INTEGER_CATEGORICAL_FEATURE_TYPE]},
             self.bucketized_user_age_attribute: {FEATURE_TYPES: [INTEGER_CATEGORICAL_FEATURE_TYPE]},

             self.timestamp_attribute: {FEATURE_TYPES: [CONTINUOUS_FEATURE_TYPE]},
             # movie_title_text_attribute 是 movie_title_attribute 中获取的
             self.movie_title_text_attribute: {FEATURE_TYPES: [TEXT_FEATURE_TYPE]},
             }

    def train(self, debug=False):
        # 预备数据
        self._prepare_data(data_dir=self.data_dir)
        # 加载训练数据并且提取需要的属性
        ratings, movies = self._load_data(data_dir=self.data_dir, data_type='train')

        # 对各个属性建立一种或者多个特征
        # self._build_features(ratings=ratings, movies=movies)
        self._build_featuriers(ratings=ratings, movies=movies)

        # 使用预处理层对数据进行预处理，其中需要增加一个属性 movie_title_text_attribute
        preprocessed_ratings, preprocessed_movies = self._preprocess_data(ratings=ratings, movies=movies)

        self.candidate_dataset = preprocessed_movies
        element_spec = self.candidate_dataset.element_spec

        # 分割数据集
        train, test = self._split_data(preprocessed_ratings)
        cached_train = train.shuffle(100_000).batch(8192).cache()
        cached_test = test.batch(4096).cache()

        self.model = self._instance_model_class()

        # 在训练之前对 preprocessing layer 中 Normalization 进行初始化
        self.model.adapt(inputs=preprocessed_ratings,candidates = preprocessed_movies)

        self.model._prepare_metrics_and_loss()

        # 模型编译: 在训练模型之前，您需要配置学习过程，这是通过 compile 方法完成的。
        self.model.compile(optimizer=self.optimizer, run_eagerly=self.debug)


        self.model.fit(cached_train, epochs=self.train_epochs)
        # validation_data=cached_test)

        # 训练结束后，进行评估
        train_evaluate_result = self.evaluate(dataset=cached_train)
        test_evaluate_result = self.evaluate(dataset=cached_test)

        print(f"Top-100 accuracy (train): {train_evaluate_result['factorized_top_k/top_100_categorical_accuracy']}")
        print(f"Top-100 accuracy (test): {test_evaluate_result['factorized_top_k/top_100_categorical_accuracy']}")
        print(f"test_evaluate_result={test_evaluate_result}")

        # 保存模型
        self.persist()

    def _prepare_data(self, data_dir) -> tf.data.Dataset:
        # Ratings data.
        # The ratings dataset returns a dictionary of movie id, user id, the assigned rating, timestamp, movie information, and user information:
        ratings = tfds.load("movielens/100k-ratings", data_dir=data_dir)
        # Features of all the available movies.
        # The movies dataset contains the movie id, movie title, and data on what genres it belongs to.
        # Note that the genres are encoded with integer labels.
        movies = tfds.load("movielens/100k-movies", data_dir=data_dir)
        return ratings, movies

    def _load_data(self, data_dir, data_type="train"):
        ratings = tfds.load("movielens/100k-ratings", split=data_type, data_dir=data_dir)
        # Features of all the available movies.
        movies = tfds.load("movielens/100k-movies", split=data_type, data_dir=data_dir)

        # keep only the user_id, and movie_title fields in the dataset.
        # ratings dataset 中使用 movie_title 和 user_id 两个属性 : features : Dict[Text,tf.Tensor]
        def extract_function(x, extract_attributes: List[Text]):
            attribute_to_value_mapping = {}
            for attribute in extract_attributes:
                if attribute in [self.user_gender_attribute, self.bucketized_user_age_attribute]:
                    attribute_to_value_mapping.update({attribute: int(x[attribute])})
                else:
                    attribute_to_value_mapping.update({attribute: x[attribute]})
            return attribute_to_value_mapping

        # 构建一个偏函数，extract_attributes 为所需要抽取的属性
        input_data_extract_func = functools.partial(extract_function,
                                                    extract_attributes=self.input_data_extract_attributes)

        ratings = ratings.map(
            lambda x: input_data_extract_func(x))
        # lambda x: {
        #     "movie_title": x["movie_title"],
        #     "user_id": x["user_id"],
        #     "user_rating": x["user_rating"],  # ranking 任务使用
        #     "timestamp": x["timestamp"]
        # }

        for rating in ratings.take(3).as_numpy_iterator():
            pprint.pprint(rating)

        # movies dataset 中只使用一个 属性 ：  movie_tile
        # movies = movies.map(lambda x: x["movie_title"])
        # movies dataset 转换为  features : Dict[Text,tf.Tensor]
        candidate_data_extract_func = functools.partial(extract_function,
                                                        extract_attributes=self.candidate_data_extract_attributes)
        movies = movies.map(candidate_data_extract_func)

        for movie in movies.take(3).as_numpy_iterator():
            pprint.pprint(movie)

        return ratings, movies

    def _build_features(self, ratings, movies) -> Tuple[List[Feature], List[Feature]]:
        # 创建 query 和 candidate feature,创建多种 features，希望使用多个特征
        attribute_to_feature_parameters = self._get_feature_parameters(ratings, movies)

        self._update_attribute_to_feature_parameters(attribute_to_feature_parameters)
        # 创建features
        self.query_features = self._create_features(attributes=self.query_attributes,
                                                    attribute_to_feature_parameters=self.attribute_to_feature_parameters,
                                                    )

        self.candidate_features = self._create_features(attributes=self.candidate_attributes,
                                                        attribute_to_feature_parameters=self.attribute_to_feature_parameters)

        return self.query_features, self.candidate_features

    def _get_feature_parameters(self, ratings, movies):
        attribute_to_feature_parameters_path = os.path.join(self.output_dir, "attribute_to_feature_parameters.pkl")
        if os.path.exists(attribute_to_feature_parameters_path):
            attribute_to_feature_parameters = pickle_load(filename=attribute_to_feature_parameters_path)
        else:
            # 存储 attribute 对应的
            # CATEGORICAL_FEATURE_TYPE 初始化 需要 vocabulary
            self._get_vocabulary(movies=movies, ratings=ratings)

            # 获取  attribute_to_feature_parameters
            attribute_to_feature_parameters = {
                # 对于 StringLookup
                self.user_id_attribute: {VOCABULARY: self.user_id_vocabulary,
                                         MASK_VALUE: None},
                self.movie_id_attribute: {VOCABULARY: self.movie_id_vocabulary,
                                          MASK_VALUE: None},
                self.user_zip_code_attribute: {VOCABULARY: self.user_zip_code_vocabulary,
                                               MASK_VALUE: None},
                self.user_occupation_text_attribute: {VOCABULARY: self.user_occupation_vocabulary,
                                                      MASK_VALUE: None},
                # 对于 IntegerLookup layer 的参数
                self.user_gender_attribute: {VOCABULARY: self.user_gender_vocabulary,
                                             MASK_VALUE: None,
                                             MAX_VALUES: None},
                self.bucketized_user_age_attribute: {VOCABULARY: self.bucketized_user_age_vocabulary,
                                                     MASK_VALUE: None,
                                                     MAX_VALUES: 101},
                # # Discretization layer 参数
                # self.timestamp_attribute: {BINS: timestamp_buckets},
                # # 对于 TextVectorization
                # self.movie_title_attribute: {MAX_TOKENS: movie_title_max_tokens,
                #                              },
                # self.ranking_attribute: {},
            }

            # 获取预处理的参数
            # min_timestamp, max_timestamp, timestamp_buckets, movie_title_max_tokens
            attribute_to_preprocess_layer_init_parameters = \
                self._get_attribute_to_preprocess_layer_init_parameters(ratings)

            attribute_to_feature_parameters.update(attribute_to_preprocess_layer_init_parameters)

            pickle_dump(filename=attribute_to_feature_parameters_path, obj=attribute_to_feature_parameters)
        return attribute_to_feature_parameters

    def _get_vocabulary(self, movies, ratings):
        # 当 movie 使用 map 转换为 features : Dict[Text,tf.Tensor]
        movie_titles = movies.batch(1_000).map(lambda x: x[self.movie_title_attribute])
        movie_ids = ratings.batch(1_000_000).map(lambda x: x[self.movie_id_attribute])
        user_ids = ratings.batch(1_000_000).map(lambda x: x[self.user_id_attribute])

        user_zip_codes = ratings.batch(1_000_000).map(lambda x: x[self.user_zip_code_attribute])
        user_occupations = ratings.batch(1_000_000).map(lambda x: x[self.user_occupation_text_attribute])
        # 对于 IntegerLookup layer ，value需要是整数
        user_genders = ratings.batch(1_000_000).map(lambda x: x[self.user_gender_attribute])
        bucketized_user_ages = ratings.batch(1_000_000).map(lambda x: x[self.bucketized_user_age_attribute])

        self.movie_title_vocabulary = np.unique(np.concatenate(list(movie_titles)))
        self.movie_id_vocabulary = np.unique(np.concatenate(list(movie_ids)))
        self.user_id_vocabulary = np.unique(np.concatenate(list(user_ids)))
        self.user_zip_code_vocabulary = np.unique(np.concatenate(list(user_zip_codes)))
        self.user_occupation_vocabulary = np.unique(np.concatenate(list(user_occupations)))
        self.user_gender_vocabulary = np.unique(np.concatenate(list(user_genders)))
        self.bucketized_user_age_vocabulary = np.unique(np.concatenate(list(bucketized_user_ages)))

    def _get_attribute_to_preprocess_layer_init_parameters(self, ratings, batch_size=8192,
                                                           timestamp_bucket_num=1000,
                                                           movie_title_max_tokens=10000,
                                                           movie_title_text_output_sequence_length=128,

                                                           ) -> Dict[Text, Any]:
        attribute_to_preprocess_layer_init_parameters_path = os.path.join(self.output_dir,
                                                                          'preprocess_layer_parameters.json')
        # if os.path.exists(attribute_to_preprocess_layer_init_parameters_path):
        #     attribute_to_preprocess_layer_init_parameters = load_json(
        #         json_path=attribute_to_preprocess_layer_init_parameters_path)

        # timestamp_mean = timestamp_normalization_parameters["mean"]
        # timestamp_variance = timestamp_normalization_parameters["variance"]
        # min_timestamp = timestamp_normalization_parameters["min"]
        # max_timestamp = timestamp_normalization_parameters["max"]
        # else:
        movie_title_text_vectorization_parameters = {
            # VOCABULARY: movie_title_text_vocabulary,
            OUTPUT_SEQUENCE_LENGTH: movie_title_text_output_sequence_length,
            MAX_TOKENS: movie_title_max_tokens
        }

        # CONTINUOUS_FEATURE_TYPE 初始化需要 mean，std, burkets
        timestamp_dataset = ratings.map(lambda x: x[self.timestamp_attribute])
        # 使用 normalization.adapt() 方法获取
        # timestamp_mean, timestamp_variance = calculate_mean_and_variance_from_dataset(dataset=timestamp_dataset)
        min_timestamp, max_timestamp = calculate_min_and_max_from_dataset(dataset=timestamp_dataset, dtype=tf.int64)

        timestamp_buckets = np.linspace(min_timestamp, max_timestamp, num=timestamp_bucket_num).tolist()

        timestamp_normalization_parameters = {
            # "mean": timestamp_mean.astype(float),
            # 'variance': timestamp_variance.astype(float),
            # 'min': min_timestamp.astype(float),
            # 'max': max_timestamp.astype(float),
            BINS: timestamp_buckets
        }

        attribute_to_preprocess_layer_init_parameters = {
            self.timestamp_attribute: timestamp_normalization_parameters,
            self.movie_title_text_attribute: movie_title_text_vectorization_parameters}

        # dump_json(json_object=attribute_to_preprocess_layer_init_parameters,
        #           json_path=attribute_to_preprocess_layer_init_parameters_path)

        return attribute_to_preprocess_layer_init_parameters

    def _build_featuriers(self, ratings, movies) -> Tuple[
        List[AttributeEmbeddingFeaturizer], List[AttributeEmbeddingFeaturizer]]:

        updated_attribute_to_feature_parameters = self._get_feature_parameters(ratings, movies)

        self._update_attribute_to_feature_parameters(updated_attribute_to_feature_parameters)

        self.query_featurizers = self._create_featuriers(self.query_attributes)

        self.candidate_featurizers = self._create_featuriers(self.candidate_attributes)
        return self.query_featurizers, self.candidate_featurizers

    def _update_attribute_to_feature_parameters(self,
                                                updated_attribute_to_feature_parameters: Dict[Text, Dict[Text, Any]]):
        # attribute, feature_parameter_key_to_value: Dict[Text, Any]):
        """
        attribute_to_feature_parameters = {'user_id': {"feature_type": None,
                                               "vocabulary": self.user_id_vocabulary},
                                   'timestamp': {"feature_type": None, "mean": None, "variance": None,
                                                 "buckets": None},
                                   'movie_title': {"feature_type": None, "max_tokens": None}}
        """
        for attribute, attribute_feature_parameter_key_to_value in updated_attribute_to_feature_parameters.items():
            feature_parameters = self.attribute_to_feature_parameters.get(attribute, {})

            feature_parameters.update(attribute_feature_parameter_key_to_value)

        # 增加 ATTRIBUTE 参数
        for attribute, feature_parameters in self.attribute_to_feature_parameters.items():
            feature_parameters.update({ATTRIBUTE: attribute})

        return self.attribute_to_feature_parameters

    def _preprocess_data(self, ratings, movies, batch_size=8192):
        """
        使用 preprocess layer 进行 预处理
          https://keras.io/guides/preprocessing_layers/#preprocessing-data-before-the-model-or-inside-the-model
          # preprocessing_layer before model
          Option 2: apply it to your tf.data.Dataset, so as to obtain a dataset that yields batches of preprocessed data, like this:
          dataset = dataset.map(lambda x, y: (preprocessing_layer(x), y))
        """
        """
        
        # CONTINUOUS_FEATURE_TYPE 初始化需要 mean，std, burkets
        timestamp_dataset = ratings.map(lambda x: x[self.timestamp_attribute])
        # 初始化 normalization_layer
        timestamp_normalization_layer = Normalization()
        # 使用 normalization.adapt() 方法获取  预处理层的参数

        # 使用 adapt 方法 初始化内部状态 an internal state
        timestamp_normalization_layer.adapt(timestamp_dataset.batch(batch_size=batch_size))

        # 对于 movie_title_text 进行预处理，获取 vocabulary
        # TEXT_FEATURE_TYPE 初始化需要  max_tokens
        movie_title_text_max_tokens = self.attribute_to_feature_parameters[self.movie_title_text_attribute][MAX_TOKENS]
        # movie_title_text_output_sequence_length = self.attribute_to_feature_parameters[self.movie_title_text_attribute][OUTPUT_SEQUENCE_LENGTH]
        # output_sequence_length: 控制输出的tensor为同一个长度
        movie_title_text_vectorization_layer = TextVectorization(max_tokens=movie_title_text_max_tokens,
                                                                 output_sequence_length=128)
        # 获取 movie_title_dataset
        movie_title_dataset = ratings.map(lambda x: x[self.movie_title_attribute])
        movie_title_text_vectorization_layer.adapt(data=movie_title_dataset.batch(batch_size=batch_size))
        # movie_title_text_vocabulary = movie_title_text_vectorization_layer.get_vocabulary()
        """

        timestamp_attribute = self.timestamp_attribute
        movie_title_attribute = self.movie_title_attribute
        movie_title_text_attribute = self.movie_title_text_attribute

        def preprocess_function(x, extract_attributes: List[Text]):
            attribute_to_value_mapping = {}
            for attribute in extract_attributes:
                if attribute == movie_title_attribute:
                    # 对于该属性，保留该原始属性 movie_title_attribute
                    attribute_to_value_mapping.update({attribute: x[attribute]})
                    # 我们增加一个特征 movie_title_text_attribute,用于后面的 text_vectorization 处理
                    attribute_to_value_mapping.update(
                        {movie_title_text_attribute: x[attribute]})
                    # attribute_to_value_mapping.update(
                    #     {movie_title_text_attribute: movie_title_text_vectorization_layer(x[attribute])})
                # elif attribute == timestamp_attribute:
                #     attribute_to_value_mapping.update({attribute: timestamp_normalization_layer(x[attribute])[0, 0]})
                else:
                    attribute_to_value_mapping.update({attribute: x[attribute]})
            return attribute_to_value_mapping

        # 构建一个偏函数，extract_attributes 为所需要抽取的属性
        input_data_preprocess_func = functools.partial(preprocess_function,
                                                       extract_attributes=self.input_data_extract_attributes)
        # 对 ratings 进行预处理
        preprocessed_ratings = ratings.map(lambda x: input_data_preprocess_func(x))

        # 使用相同的预处理层对 candidate 数据进行预处理
        candidate_data_preprocess_func = functools.partial(preprocess_function,
                                                           extract_attributes=self.candidate_data_extract_attributes)
        # 对 movies进行预处理，保留 movie_title 属性，用于建立索引的时候对应的索引值
        preprocessed_movies = movies.map(lambda x: candidate_data_preprocess_func(x))

        for preprocessed_rating in preprocessed_ratings.take(1).as_numpy_iterator():
            pprint.pprint(preprocessed_rating)

        for preprocessed_movie in preprocessed_movies.take(1).as_numpy_iterator():
            pprint.pprint(preprocessed_movie)

        return preprocessed_ratings, preprocessed_movies

    def _split_data(self, dataset):
        tf.random.set_seed(42)
        shuffled = dataset.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

        train = shuffled.take(80_000)
        test = shuffled.skip(80_000).take(20_000)
        return train, test

    # def persist(self):
    #     self._persist_vocabularies(query_vocabulary=self.user_id_vocabulary,
    #                              candidate_vocabulary=self.movie_title_vocabulary)
    #     self._persist_candidate(candidate_dataset=self.candidate_dataset)
    #     self._persist_model()
    #     self.index = self._get_index(candidate_dataset=self.candidate_dataset)
    #     # 首先input_data 然后保存模型
    #     self.index({self.user_id_attribute: tf.constant(["42"])})
    #     self._persist_index()
    #
    # def _persist_vocabularies(self):
    #     """
    #     重写方法，用以保存 多种特征的 vocabulary
    #     """
    #     pickle_dump(filename=self._query_vocabulary_path, obj=query_vocabulary)
    #     pickle_dump(filename=self._candidate_vocabulary_path, obj=candidate_vocabulary)

    def evaluate(self, dataset):
        result = self.model.evaluate(dataset, return_dict=True)
        return result

    def infer(self, inputs='42'):
        if self.task == 'retrieval':
            # Get recommendations.
            # inputs = ['42']
            _, titles = self.index(inputs)
            print(f"Recommendations for user {inputs}: {titles[0, :3]}")
        elif self.task == 'ranking':
            inputs = {"user_id": np.array(["42"]), "movie_title": ["Speed (1994)"]}
            ratings = self.model(inputs)
            print(f"Rank rating for inputs {inputs}: {ratings}")

# class MovielensRankingTask(MovielensRetrievalTask):
#     def __init__(self, **kwargs):
#         super(MovielensRankingTask, self).__init__(**kwargs)
