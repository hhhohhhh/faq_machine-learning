#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file:
@version:
@desc:
@time: 2021/11/9 23:21

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/9 23:21   wangfc      1.0         None
"""
from typing import Dict, Text, Union, List, Optional, Any
import re
import tensorflow as tf
from tensorflow.python.ops.gen_dataset_ops import MapDataset

import tensorflow_recommenders as tfrs
from models.builder import MODELS
from models.temp_keras_modules import TmpKerasModel
from models.features import CATEGORICAL_FEATURE_TYPE, CONTINUOUS_FEATURE_TYPE, TEXT_FEATURE_TYPE, \
    Feature, CategoricalFeature, ContinuousFeature, TextFeature, \
    FeatureEmbedding, CategoricalFeatureEmbedding, ContinuousFeatureEmbedding, TextFeatureEmbedding, \
    AttributeEmbeddingFeaturizer


class TowerModel(TmpKerasModel):
    """
    two tower model 的组成部分：
        query tower:
        candidate tower:

    # 更深的网络结构：
    The final hidden layer does not use any activation function:
    using an activation function would limit the output space of the final embeddings and might negatively
    impact the performance of the model.
    For instance, if ReLUs are used in the projection layer, all components in the output embedding would be non-negative.
    """

    def __init__(self, tower_name,
                 # features: List[Feature]=None,
                 featurizers: List[AttributeEmbeddingFeaturizer] = None,
                 output_dense_layer_name='output_dense',
                 output_embedding_dimension=32,
                 hidden_layer_sizes: List[int] = None,
                 hidden_layer_name: Text = 'hidden',
                 ):
        """Model for encoding user queries.

        Args:
          hidden_layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """

        super(TowerModel, self).__init__(name=tower_name)
        self.tower_name = tower_name
        # self.features = features
        # 使用 featurizers 对象初始化
        self.featurizers = featurizers

        # 增加一层 output_dense,使得模型统一都输出一个 embedding 维度
        self.output_dense_layer_name = output_dense_layer_name
        self.output_embedding_dimension = output_embedding_dimension
        # 其他层的size 和名称前缀
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_name_prefix = f'{self.tower_name}_{hidden_layer_name}'

    def _prepare_layers(self):
        # embedding_layers
        self.embedding_layers_dict = self._prepare_embedding_layers_dict()
        # 增加其他更深的层
        self.hidden_layers_dict = self._prepare_hidden_layers_dict()

    def _prepare_embedding_layers_dict(self) -> Dict[Text, Dict[Text, FeatureEmbedding]]:
        """
        在保存模型的时候会报错：
        ValueError: Unable to save the object {} (a dictionary wrapper constructed automatically on attribute assignment).
        The wrapped dictionary was modified outside the wrapper (its final value was {}, its value when a checkpoint dependency was added was None), which breaks restoration on object creation.
        因为生成一个 embedding_layers_dict ，会出现无法保存模型的情况
        """
        # if hasattr(self,"features") and self.features:
        #     embedding_layers_dict = self._prepare_embedding_layers_with_features()
        # elif hasattr(self,"featurizers") and self.featurizers:
        # 默认使用这个方式构建 FeatureEmbedding
        embedding_layers_dict = self._prepare_embedding_layers_with_featuriers()
        return embedding_layers_dict

    def _prepare_embedding_layers_with_features(self) -> Dict[Text, FeatureEmbedding]:
        """
        当 具有模型的输入多种特征的时候
        """
        embedding_layers_dict = {}
        # 循环所有的 features,构建 feature name 对应 embedding_layer
        for feature in self.features:
            # 对每个 feature 进行 embedding_layer
            feature_embedding_layer = self._build_feature_embedding_layer(feature)

            # 获取 feature name 对应的 embedding_layer
            embedding_layers_dict.update({feature.name: feature_embedding_layer})

        # 增加一层 output_dense
        self.output_dense_layer = tf.keras.layers.Dense(units=self.output_embedding_dimension)
        return embedding_layers_dict

    def _prepare_embedding_layers_with_featuriers(self) -> Dict[Text, Dict[Text, FeatureEmbedding]]:
        embedding_layers_dict = {}
        for featurizer in self.featurizers:
            # 对于不同的 feature_type 产生不同 embedding_layer
            feature_type_to_embedding_layer = featurizer.get_feature_type_to_embedding_layer()
            embedding_layers_dict.update({featurizer.attribute: feature_type_to_embedding_layer})
        # 增加一层 output_dense
        self.output_dense_layer = tf.keras.layers.Dense(units=self.output_embedding_dimension)
        return embedding_layers_dict

    def _build_feature_embedding_layer(self, feature: Feature) -> FeatureEmbedding:
        if feature.feature_type == CATEGORICAL_FEATURE_TYPE:
            embedding_layer_class = CategoricalFeatureEmbedding
        elif feature.feature_type == CONTINUOUS_FEATURE_TYPE:
            embedding_layer_class = ContinuousFeatureEmbedding
        elif feature.feature_type == TEXT_FEATURE_TYPE:
            embedding_layer_class = TextFeatureEmbedding
        return embedding_layer_class(**feature.parameters_dict)

    def _prepare_hidden_layers_dict(self) -> Dict[Text, tf.keras.layers.Layer]:
        """
        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
          self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
          self.dense_layers.add(tf.keras.layers.Dense(layer_size))
        """
        hidden_layers_dict = {}
        for layer_index, layer_size in enumerate(self.hidden_layer_sizes):
            layer_name = f"{self.hidden_layer_name_prefix}_{layer_index}"
            if layer_index == self.hidden_layer_sizes.__len__() - 1:
                hidden_layers_dict.update({layer_name: tf.keras.layers.Dense(units=layer_size)})
            else:
                hidden_layers_dict.update({layer_name: tf.keras.layers.Dense(units=layer_size, activation='relu')})
        return hidden_layers_dict

    def _compute_embedding(self, inputs, training=None, mask=None):
        # if self.features:
        #     # depreciated:
        #     output_embedding = self._compute_embedding_with_features(inputs, training, mask)
        # elif self.featurizers:
        output_embedding = self._compute_embedding_with_featurizers(inputs, training, mask)
        return output_embedding

    def _compute_embedding_with_features(self, inputs, training=None, mask=None):
        # 对于 inputs 的不同的 attribute,使用不同的 embedding_layer 转换为 embedding
        attribute_embedding_ls = []
        for attribute, embedding_layer in self.embedding_layers_dict.items():
            # TODO: 获取 attribute 对应的 输入
            attribute_embedding = embedding_layer(inputs[attribute])
            attribute_embedding_ls.append(attribute_embedding)
        if attribute_embedding_ls.__len__() > 1:
            attribute_embeddings = tf.concat(attribute_embedding_ls, axis=1)
        elif attribute_embedding_ls.__len__() == 1:
            attribute_embeddings = attribute_embedding_ls[0]
        else:
            # attribute_embeddings = attribute_embedding_ls
            raise ValueError(f"attribute_embeddings 不能为空！")

        output_embedding = self.output_dense_layer(attribute_embeddings)
        return output_embedding

    def _compute_embedding_with_featurizers(self, inputs, training=None, mask=None):
        """
        TODO: 如何使用 inputs 中的attribute 与特定的 embedding layer 中进行计算
        """
        # 对于 inputs 的不同的 attribute,使用不同的 embedding_layer 转换为 embedding
        attribute_embedding_ls = []
        # 针对不同的 attribute：
        for attribute, embedding_layer_dict in self.embedding_layers_dict.items():
            # 对同一个attribute 的 不同的 feature_type 进行计算
            for feature_type, embedding_layer in embedding_layer_dict.items():
                attribute_embedding = embedding_layer(inputs[attribute])
                attribute_embedding_ls.append(attribute_embedding)
        # import pprint
        # pprint.pprint(attribute_embedding_ls)
        if attribute_embedding_ls.__len__() > 1:
            attribute_embeddings = tf.concat(attribute_embedding_ls, axis=1)
        elif attribute_embedding_ls.__len__() == 1:
            attribute_embeddings = attribute_embedding_ls[0]
        else:
            # attribute_embeddings = attribute_embedding_ls
            raise ValueError(f"attribute_embeddings 不能为空！")

        output_embedding = self.output_dense_layer(attribute_embeddings)
        return output_embedding

    def call(self, inputs, training=None, mask=None):
        output_embedding = self._compute_embedding(inputs, training, mask)

        x = tf.identity(output_embedding)
        if self.hidden_layers_dict:
            for layer_name, layer in self.hidden_layers_dict.items():
                x = layer(x)
        return x


class DeepCrossNetwork(TowerModel):
    """
     feature crosses:
     The combination of purchased_bananas and purchased_cooking_books is referred to as a feature cross,
     which provides additional interaction information beyond the individual features.

    Deep & Cross Network (DCN) :
    DCN was designed to learn explicit and bounded-degree cross features more effectively.
    It starts with an input layer (typically an embedding layer), followed by a cross network containing
    multiple cross layers that models explicit feature interactions,
    and then combines with a deep network that models implicit feature interactions.


    1. Stacked structure : we could stack a deep network on top of the cross network;
    2. Parallel structure : The inputs are fed in parallel to a cross network and a deep network.
    3. Concatenating cross layers. The inputs are fed in parallel to multiple cross layers to capture complementary feature crosses.

    """

    def __init__(self, use_cross_layer= True,
                 cross_layer_num=1,
                 cross_layer_prefix = 'cross_layer',
                 use_projection_dim=True,
                 input_dim:int=None,
                 projection_dim=32,
                 kernel_initializer='glorot_uniform',
                 **kwargs):

        super(DeepCrossNetwork, self).__init__(**kwargs)
        self.use_cross_layer = use_cross_layer
        self.cross_layer_prefix = cross_layer_prefix
        self.cross_layer_num = cross_layer_num
        self.use_projection_dim = use_projection_dim

        self.input_dim = input_dim
        self.projection_dim = self._set_projection_dim(projection_dim)
        self.kernel_initializer = kernel_initializer

    def _set_projection_dim(self,projection_dim):
        """
        设置 projection_dim
        Low-rank DCN. To reduce the training and serving cost, we leverage low-rank techniques to approximate the DCN weight matrices.
        The rank is passed in through argument projection_dim; a smaller projection_dim results in a lower cost.
        Note that projection_dim needs to be smaller than (input size)/2 to reduce the cost.
        In practice, we've observed using low-rank DCN with rank (input size)/4 consistently preserved the accuracy of a full-rank DCN.
        """
        if self.use_projection_dim is False or projection_dim:
            pass
        elif projection_dim is None and self.input_dim is not None:
            projection_dim = self.input_dim / 4
        else:
            raise ValueError(f"使用 Low-rank DCN，但是 projection_dim 为空，设置错误")
        return projection_dim


    def _prepare_layers(self):
        # embedding_layers
        self.embedding_layers_dict = self._prepare_embedding_layers_dict()
        # 增加其他更深的层
        self.hidden_layers_dict = self._prepare_hidden_layers_dict()

        self.cross_layer_dict =  self._prepare_cross_layers_dict()

    def _prepare_cross_layers_dict(self):
        """
        TODO: cross_layer_num 是一个超参数，表示交互的度，
        cross_layer_num=1， second order
         cross_layer_num=2， 3 order
        """
        cross_layer_dict = {}
        if self.use_cross_layer:
            # 设置多层的 cross_layer
            for cross_layer_index in range(self.cross_layer_num):
                cross_layer = tfrs.layers.dcn.Cross(projection_dim= self.projection_dim,
                                                    kernel_initializer=self.kernel_initializer)
                cross_layer_name = f"{self.cross_layer_prefix}_{cross_layer_index}"
                cross_layer_dict.update({cross_layer_name:cross_layer})
        return cross_layer_dict


    def adapt(self,inputs:MapDataset):
        # 对于 inputs 的不同的 attribute,使用不同的 embedding_layer 进行 adapt
        # 针对不同的 attribute：
        for attribute, embedding_layer_dict in self.embedding_layers_dict.items():
            # 对同一个attribute 的 不同的 feature_type 进行计算
            for feature_type, embedding_layer in embedding_layer_dict.items():
                if hasattr(embedding_layer,'adapt'):
                    # 当 embedding_layer 具有 adapt 属性的时候，我们针对该 attribute调用对应的adapt方法
                    embedding_layer.adapt(inputs.map(lambda x:x[attribute]).batch(batch_size=8192))




    def call(self, inputs, training=None, mask=None):
        # 获取 embedding
        output_embedding = self._compute_embedding(inputs, training, mask)

        # 增加 cross_layer
        if self.use_cross_layer:
            x_0 = tf.identity(output_embedding)
            for cross_layer_name,cross_layer in self.cross_layer_dict.items():
                matched = re.match(pattern=f'{self.cross_layer_prefix}_(\d)', string=cross_layer_name)
                cross_layer_index = int(matched.groups()[0])
                if cross_layer_index ==0:
                    output_embedding = cross_layer(output_embedding)
                else:
                    output_embedding = cross_layer(x_0,output_embedding)

        x = tf.identity(output_embedding)
        if self.hidden_layers_dict:
            for layer_name, layer in self.hidden_layers_dict.items():
                x = layer(x)
        return x


@MODELS.register_module()
class TwoTowerRetrievalModel(tfrs.Model, TmpKerasModel):
    """
    a two-tower retrieval model,
    we can build each tower separately and then combine them in the final model.
    """

    def __init__(self,
                 # query_features: List[Feature]=None,
                 # candidate_features: List[Feature]=None,
                 query_featurizers: List[AttributeEmbeddingFeaturizer] = None,
                 candidate_featurizers: List[AttributeEmbeddingFeaturizer] = None,
                 candidate_dataset: tf.data.Dataset = None,
                 hidden_layer_sizes: List[int] = None,
                 # embedding_dimension=32,
                 loss_name=None, tower_type="TowerModel",
                 if_prepare_model=True, *args, **kwargs):
        super(TwoTowerRetrievalModel, self).__init__(*args, **kwargs)
        # self.query_feature_name = query_feature_name
        # self.query_vocabulary = query_vocabulary
        #
        # self.candidate_vocabulary = candidate_vocabulary
        # self.candidate_feature_name = candidate_feature_name
        # self.embedding_dimension = embedding_dimension

        # 需要对每个 feature 转换为 embedding
        # self.query_features = query_features
        # self.candidate_features = candidate_features
        # # 使用 AttributeEmbeddingFeaturizer 对象初始化模型
        self.query_featurizers = query_featurizers
        self.candidate_featurizers = candidate_featurizers

        self.hidden_layer_sizes = hidden_layer_sizes

        self.candidate_dataset = candidate_dataset
        self.loss_name = loss_name
        # the dimensionality of the query and candidate representations:

        self.kwargs = kwargs

        if if_prepare_model:
            self.prepare_model()

    def prepare_model(self):
        self._prepare_layers()
        #  model.build() : be used for subclassed models, which do not know at instantiation time what their inputs look like.
        # self.build(input_shape=)

        # 设置 saved_model_inputs_spec： 用于保存为 saved_model
        # self._set_inputs()


    def _prepare_layers(self):
        self._prepare_two_tower_embedding_layers()


    def _prepare_metrics_and_loss(self):
        self._create_metrics()
        self._prepare_loss()


    def _prepare_two_tower_embedding_layers(self):
        """
        two tower model:
        建立 query tower 和 candidate tower
        query tower : 负责将 query 进行 embedding
        candidate tower: 复制讲 candidate 进行 embedding
        """
        # se Keras preprocessing layers to first convert user ids to integers,
        # and then convert those to user embeddings via an Embedding layer.

        # # The query tower
        # self.query_model: Model = CategoricalFeatureEmbedding(
        #     categorical_feature_name=self.query_feature_name,
        #     vocabulary=self.query_vocabulary,
        #     embedding_dimension=self.embedding_dimension)
        self.query_model = TowerModel(tower_name='query_tower',
                                      features=self.query_features,
                                      featurizers=self.query_featurizers,
                                      hidden_layer_sizes=self.hidden_layer_sizes)
        self.query_model._prepare_layers()

        # # The candidate tower
        # self.candidate_model: Model = CategoricalFeatureEmbedding(
        #     categorical_feature_name=self.candidate_feature_name,
        #     vocabulary=self.candidate_vocabulary,
        #     embedding_dimension=self.embedding_dimension)
        self.candidate_model = TowerModel(tower_name='candidate_tower',
                                          features=self.candidate_features,
                                          featurizers=self.candidate_featurizers,
                                          hidden_layer_sizes=self.hidden_layer_sizes)
        self.candidate_model._prepare_layers()

    def _create_metrics(self) -> None:
        self.factorized_topk_metrics = self._create_retrieval_metrics()

    def _create_retrieval_metrics(self) -> tfrs.metrics.FactorizedTopK:
        """
        In our training data we have positive (user, movie) pairs.
        To figure out how good our model is, we need to compare the affinity score that the model calculates for
        this pair to the scores of all the other possible candidates:
        if the score for the positive pair is higher than for all other candidates, our model is highly accurate.
        """
        # TODO: candidate_dataset 必须在初始化模型的时候提供吗？
        # for candidate in self.candidate_dataset.take(1).as_numpy_iterator():
        #     print(candidate)
        #     self.candidate_model(candidate)
        factorized_topk_metrics = tfrs.metrics.FactorizedTopK(
            # 将 candidate_dataset 变换为 embedding, 这里使用 candidate_dataset
            candidates= self.candidate_dataset.batch(128).map(self.candidate_model),
            name='factorized_top_k')
        return factorized_topk_metrics

    def _prepare_loss(self):
        self.retrieval_loss_layer = self._prepare_retrieval_loss()

    def _prepare_retrieval_loss(self) -> tfrs.tasks.Retrieval:
        """
        The task itself is a Keras layer that takes the query and candidate embeddings as arguments,
        and returns the computed loss:

        inputs:  query and candidate embeddings
        output:  loss

        """
        # 使用 factorized_topk_metrics 初始化 Retrieval loss layer： 在计算 loss 的时候更新 metrics
        retrieval_loss_layer = tfrs.tasks.Retrieval(
            metrics=self.factorized_topk_metrics
        )
        return retrieval_loss_layer

    def compute_loss(self, features: Dict[Text, tf.Tensor],
                     training=False) -> tf.Tensor:
        """
        tfrs.Model 自定义训练循环：
        在自定义 的 fit() 中 -> 自定义 train_step(): 其中最重要是 自定义 compute_loss() 方法，我们可以定义该方法
        对于 retrieval model，我们没有 model.call() 来调用模型计算loss，而是直接使用各个 layer 进行计算 loss，
        但是，我们因为 TwoToweRetrievalModel 是 tf.keral.Model 的子类，必须要实现 .call() 方法，即使这儿没有调用该方法
        """
        # We pick out the user features and pass them into the user model.
        # query_embeddings = self.query_model(features[self.query_feature_name])
        # And pick out the movie features and pass them into the movie model,getting embeddings back.
        # positive_candidate_embeddings = self.candidate_model(features[self.candidate_feature_name])

        # 当使用多种特征作为输入的时候，我们将 features 作为参数
        query_embeddings = self.query_model(features)
        # 将 features （含有 candidate 属性作为 label）通过 candidate_model 获取 positive_candidate_embeddings
        positive_candidate_embeddings = self.candidate_model(features)

        # The task computes the loss and the metrics.
        return self.loss_layer(query_embeddings=query_embeddings,
                               candidate_embeddings=positive_candidate_embeddings)

    def train_step(self, features: Dict[Text, tf.Tensor]) -> Dict[Text, float]:
        """
        TODO: features 为何是  Dict[Text, tf.Tensor]
        """
        # Set up a gradient tape to record gradients.
        with tf.GradientTape() as tape:
            # Loss computation.
            # user_embeddings = self.user_model(features["user_id"])
            # positive_movie_embeddings = self.movie_model(features["movie_title"])
            # loss = self.loss_layer(user_embeddings, positive_movie_embeddings)

            loss = self.compute_loss(features=features, training=True)

            # Handle regularization losses as well.
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        # Loss computation.
        # user_embeddings = self.user_model(features["user_id"])
        # positive_movie_embeddings = self.movie_model(features["movie_title"])
        # loss = self.task(user_embeddings, positive_movie_embeddings)
        loss = self.compute_loss(features=features, training=False)

        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)
        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def call(
            self,
            inputs: Union[tf.Tensor, List[tf.Tensor]],
            training: Optional[tf.Tensor] = None,
            mask: Optional[tf.Tensor] = None,
    ) -> Union[tf.Tensor, List[tf.Tensor]]:
        """Calls the model on new inputs.

        Arguments:
            inputs: A tensor or list of tensors.
            training: Boolean or boolean scalar tensor, indicating whether to run
              the `Network` in training mode or inference mode.
            mask: A mask or list of masks. A mask can be
                either a tensor or None (no mask).

        Returns:
            A tensor if there is a single output, or
            a list of tensors if there are more than one outputs.
        """
        # This method needs to be implemented, otherwise the super class is raising a
        # NotImplementedError('When subclassing the `Model` class, you should
        #   implement a `call` method.')
        pass

    def save_model(self, model_filepath, overwrite=True):
        """
        自定义 save_model 方法
        """
        # 保存为 saved model
        # # tf.saved_model.save(self.model, export_dir=self._export_model_dir)
        # tf.keras.models.save_model(model=self.model, filepath=filepath)
        # 　保存为 checkpoint
        self.save_weights(filepath=model_filepath, overwrite=overwrite, save_format="tf")

    @classmethod
    def load_model(cls, model_filepath, *args, **kwargs) -> "TwoTowerRetrievalModel":
        # create empty model
        model = cls(*args, **kwargs)
        # 加载参数
        model.load_weights(filepath=model_filepath)
        return model


@MODELS.register_module()
class TwoTowerRankingModel(TwoTowerRetrievalModel):
    """
    from : https://www.tensorflow.org/recommenders/examples/basic_ranking :RankingModel

    """

    def __init__(self, ranking_attribute='user_rating', **kwargs):
        # if_prepare_model =False
        # 控制父类不进行 layer 的初始化
        super(TwoTowerRankingModel, self).__init__(if_prepare_model=False, **kwargs)
        self.ranking_attribute = ranking_attribute
        # if kwargs.get('if_prepare_model') is None or kwargs.get('if_prepare_model') == True:
        #     self.prepare_model()

    def _prepare_layers(self):
        self._prepare_two_tower_embedding_layers()
        self._prepare_ranking_layers()

    def _prepare_ranking_layers(self):
        # se Keras preprocessing layers to first convert user ids to integers,
        # and then convert those to user embeddings via an Embedding layer.
        # # The query tower
        # self.query_model: Model = TowerModel(vocabulary=self.query_vocabulary,
        #                                      embedding_dimension=self.embedding_dimension)
        # # The candidate tower
        # self.candidate_model: Model = TowerModel(vocabulary=self.candidate_vocabulary,
        #                                          embedding_dimension=self.embedding_dimension)
        # Compute predictions.
        self.rating_layer = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs, training=None, mask=None):
        """
        this model takes user ids and movie titles, and outputs a predicted rating:
        loss: 使用回归模型的loss
        """
        query_feature = inputs[self.query_feature_name]
        candidate_feature = inputs[self.candidate_feature_name]
        query_embedding = self.query_model(query_feature)
        candidate_embedding = self.candidate_model(candidate_feature)

        return self.rating_layer(tf.concat([query_embedding, candidate_embedding], axis=1))

    def _create_metrics(self) -> None:
        self.ranking_metrics = self._create_ranking_metrics()

    def _create_ranking_metrics(self):
        """
        主要不要直接使用 self.metrics 属性，该属性是 tf.keras.model 只读属性
        """
        ranking_metrics = [tf.keras.metrics.RootMeanSquaredError()]
        return ranking_metrics

    def _prepare_ranking_loss(self) -> tfrs.tasks.Ranking:
        """
        The task itself is a Keras layer that takes true and predicted as arguments, and returns the computed loss.
        We'll use that to implement the model's training loop.
        """
        ranking_loss_layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=self.ranking_metrics
        )
        return ranking_loss_layer

    def compute_loss(self, features: Dict[Text, tf.Tensor],
                     training=False) -> tf.Tensor:
        # 获取 label
        ratings = features.pop(self.ranking_attribute)

        # 调用 model.call() 方法，预测 rating_predictions
        rating_predictions = self.call(features)

        # The task computes the loss and the metrics.
        return self.loss_layer(labels=ratings, predictions=rating_predictions)


@MODELS.register_module()
class MultiTaskRecommendationModel(TwoTowerRankingModel):
    """
    here are two critical parts to multi-task recommenders:
    1. They optimize for two or more objectives, and so have two or more losses.
    2. They share variables between the tasks, allowing for transfer learning.
    """

    def __init__(self, use_retrieval_task=True, use_ranking_task=True,
                 retrieval_weight=1.0, ranking_weight=1.0, tower_type='DCN',
                 *args,
                 **kwargs):
        super(MultiTaskRecommendationModel, self).__init__(*args,**kwargs)
        # 支持多任务联合训练
        self.use_retrieval_task = use_retrieval_task
        self.use_ranking_task = use_ranking_task
        self.retrieval_weight = retrieval_weight
        self.ranking_weight = ranking_weight

        # 支持不同的 tower_type:
        self.tower_type = tower_type

        self.prepare_model()

    def _prepare_layers(self):
        # 生成 双塔的模型，用于对 query 和 candidate 做embedding
        self._prepare_two_tower_embedding_layers(tower_type=self.tower_type)
        # 对于 ranking 任务需要增加 ranking layer
        if self.use_ranking_task:
            # 增加 ranking 层，用于预测 rating
            self._prepare_ranking_layers()

    def _prepare_two_tower_embedding_layers(self, tower_type='TowerModel'):
        """
        重写该函数 使得支持不同的 tower_type:
        建立 query tower 和 candidate tower
        query tower : 负责将 query 进行 embedding
        candidate tower: 复制讲 candidate 进行 embedding
        """
        if tower_type == 'TowerModel':
            tower_model = TowerModel
        elif tower_type == "DCN":
            # TODO: DeepCrossNetwork 参数： 一层还是多层的
            tower_model = DeepCrossNetwork

        self.query_model = tower_model(tower_name='query_tower',
                                       # features=self.query_features,
                                       featurizers=self.query_featurizers,
                                       hidden_layer_sizes=self.hidden_layer_sizes)
        self.query_model._prepare_layers()

        self.candidate_model = tower_model(tower_name='candidate_tower',
                                           # features=self.candidate_features,
                                           featurizers=self.candidate_featurizers,
                                           hidden_layer_sizes=self.hidden_layer_sizes)
        self.candidate_model._prepare_layers()

    def _create_metrics(self) -> None:
        if self.use_retrieval_task:
            # 创建检索的评估指标
            self.factorized_topk_metrics = self._create_retrieval_metrics()
        if self.use_ranking_task:
            # 创建排序的评估指标
            self.ranking_metrics = self._create_ranking_metrics()

    def _prepare_loss(self):
        """
        The new component here is that - since we have two tasks and two losses -
        we need to decide on how important each loss is. We can do this by giving each of the losses a weight,
        and treating these weights as hyperparameters.
        If we assign a large loss weight to the rating task, our model is going to focus on predicting ratings
         (but still use some information from the retrieval task);
        if we assign a large loss weight to the retrieval task, it will focus on retrieval instead.
        """
        if self.use_ranking_task:
            # 使用 factorized_topk_metrics 初始化 Retrieval loss layer： 在计算 loss 的时候更新 metrics
            self.retrieval_loss_layer = self._prepare_retrieval_loss()
        if self.use_ranking_task:
            # 创建 排序的loss层
            self.ranking_loss_layer = self._prepare_ranking_loss()

    def compute_loss(self, features: Dict[Text, tf.Tensor],
                     training=False) -> tf.Tensor:
        query_embeddings, candidate_embeddings, rating_predictions = self.call(features)

        retrieval_loss = self.retrieval_loss_layer(query_embeddings=query_embeddings,
                                                   candidate_embeddings=candidate_embeddings)
        ranking_loss = self.ranking_loss_layer(labels=features[self.ranking_attribute],
                                               predictions=rating_predictions)
        # loss 的尺度也不相同，需要做 rescale
        return self.retrieval_weight * retrieval_loss + self.ranking_weight * ranking_loss

    def adapt(self, inputs,candidates, training=None, mask=None):
        """
        增加 adapt 方法，用于 preprocess layer 在训练之前初始化
        """
        self.query_model.adapt(inputs)
        self.candidate_model.adapt(candidates)


    def call(self, inputs, training=None, mask=None):
        # 当使用多种特征作为输入的时候，我们将 features 作为参数
        query_embeddings = self.query_model(inputs)
        #  获取
        positive_candidate_embeddings = self.candidate_model(inputs)
        rating_predictions = None
        if self.use_ranking_task:
            rating_predictions = self.rating_layer(tf.concat([query_embeddings, positive_candidate_embeddings], axis=-1))
        return query_embeddings, positive_candidate_embeddings, rating_predictions
