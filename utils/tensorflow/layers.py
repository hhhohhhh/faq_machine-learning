#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:


我们可以使用很多 tf.keras.layers，它们具有一些相同的构造函数参数：
    activation：设置层的激活函数。此参数由内置函数的名称指定，或指定为可调用对象。默认情况下，系统不会应用任何激活函数。
    kernel_initializer 和 bias_initializer：创建层权重（核和偏差）的初始化方案。此参数是一个名称或可调用对象，默认为 "Glorot uniform" 初始化器。
    kernel_regularizer 和 bias_regularizer：应用层权重（核和偏差）的正则化方案，例如 L1 或 L2 正则化。默认情况下，系统不会应用正则化函数。



通过对 tf.keras.layers.Layer 进行子类化并实现以下方法来创建自定义层：
    build：创建层的权重（或者叫 state）。使用 add_weight 方法添加权重 。
    call：定义前向传播。
    compute_output_shape：指定在给定输入形状的情况下如何计算层的输出形状。
    或者，可以通过实现 get_config 方法和 from_config 类方法序列化层。



保存和恢复
仅限权重
使用 tf.keras.Model.save_weights 保存并加载模型的权重：

默认情况下，会以 TensorFlow 检查点文件格式保存模型的权重。
# Save weights to a TensorFlow Checkpoint file
model.save_weights('./weights/my_model')

# Restore the model's state,
# this requires a model with the same architecture.
model.load_weights('./weights/my_model')

权重也可以另存为 Keras HDF5 格式（Keras 多后端实现的默认格式）：
# Save weights to a HDF5 file
model.save_weights('my_model.h5', save_format='h5')

# Restore the model's state
model.load_weights('my_model.h5')



整个模型可以保存到一个文件中，其中包含权重值、模型配置乃至优化器配置。这样，您就可以对模型设置检查点并稍后从完全相同的状态继续训练，而无需访问原始代码。
# Save entire model to a HDF5 file
model.save('my_model.h5')

# Recreate the exact same model, including weights and optimizer.
model = tf.keras.models.load_model('my_model.h5')



@time: 2021/5/20 16:45 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/20 16:45   wangfc      1.0         None
"""

import logging
from typing import List, Optional, Text, Tuple, Callable, Union, Any
import tensorflow as tf
from .exceptions import TFLayerConfigException
from tensorflow import keras
from tensorflow.keras.layers import Lambda,Layer

import tensorflow_addons as tfa
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend as K

from . import crf
# import rasa.utils.tensorflow.crf
# from rasa.utils.tensorflow.constants import (
#     SOFTMAX,
#     MARGIN,
#     COSINE,
#     INNER,
#     LINEAR_NORM,
#     CROSS_ENTROPY,
# )
# from rasa.utils.tensorflow.exceptions import TFLayerConfigException


from utils.tensorflow.constants import *

logger = logging.getLogger(__name__)

# https://github.com/tensorflow/addons#gpu-and-cpu-custom-ops-1
tfa.options.TF_ADDONS_PY_OPS = True


class MyDenseLayer(Layer):


    def __init__(self, units, trainable=True,activation=None,name="MyDenseLayer"):
        super(MyDenseLayer, self).__init__(name=name)
        # 输出的节点
        self.units = units

        self.trainable = trainable
        # 增加激活函数，增加非线性
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        """

        创建 layer 中的参数,add_weight方法为图层添加权重：
        可以初始化权重参数
        在许多情况下，可能事先不知道输入数据的形状，想在实例化图层后再将参数传递给权重，可以先使用build（inputs_shape）方法创建图层权重占位
        请注意，您不必等到调用 build 来创建变量，您也可以在 __init__中创建它们。
        但是，在 build 中创建它们的好处是，它支持根据将要操作的层的输入形状，创建后期变量。
        另一方面，在 __init__ 中创建变量意味着需要明确指定创建变量所需的形状。
        """


        self.kernel = self.add_weight(name="kernel",
                                      shape=[int(input_shape[-1]),self.units],
                                      dtype='float',
                                      initializer=tf.random_normal_initializer,
                                      trainable=self.trainable
                                      )
        self.bias =  self.add_weight(name = 'bias',
                                     shape = [self.units],
                                     dtype='float',
                                     initializer=tf.zeros_initializer,
                                     trainable=self.trainable)

    def call(self, input):
        x = tf.matmul(input, self.kernel) + self.bias
        x = self.activation(x)
        return x





class SparseDropout(tf.keras.layers.Dropout):
    """Applies Dropout to the input.

    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    Arguments:
        rate: Float between 0 and 1; fraction of the input units to drop.
    """

    def call(
        self, inputs: tf.SparseTensor, training: Optional[Union[tf.Tensor, bool]] = None
    ) -> tf.SparseTensor:
        """Apply dropout to sparse inputs.

        Arguments:
            inputs: Input sparse tensor (of any rank).
            training: Python boolean indicating whether the layer should behave in
                training mode (adding dropout) or in inference mode (doing nothing).

        Returns:
            Output of dropout layer.

        Raises:
            A ValueError if inputs is not a sparse tensor
        """

        if not isinstance(inputs, tf.SparseTensor):
            raise ValueError("Input tensor should be sparse.")

        if training is None:
            training = K.learning_phase()

        def dropped_inputs() -> tf.SparseTensor:
            # 生成 values 形状的 均匀分布 tensor： to_retain_prob
            to_retain_prob = tf.random.uniform(
                tf.shape(inputs.values), 0, 1, inputs.values.dtype
            )
            # 于 self.rate 对比，大于 rate为　True,否则为False
            to_retain = tf.greater_equal(to_retain_prob, self.rate)
            # tf.sparse.retain 对 value 进行 保留或者舍弃
            return tf.sparse.retain(inputs, to_retain)
        # smart_cond: 当 training=True的时候，选择 dropped_inputs，当  training=False 的时候，lambda: tf.identity(inputs)
        outputs = tf_utils.smart_cond(
            training, dropped_inputs, lambda: tf.identity(inputs)
        )
        # need to explicitly recreate sparse tensor, because otherwise the shape
        # information will be lost after `retain`
        # noinspection PyProtectedMember
        return tf.SparseTensor(outputs.indices, outputs.values, inputs._dense_shape)


class DenseForSparse(tf.keras.layers.Dense):
    """Dense layer for sparse input tensor.

    Just your regular densely-connected NN layer but for sparse tensors.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: If the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    Arguments:
        units: Positive integer, dimensionality of the output space.
            activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        reg_lambda: Float, regularization factor.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, reg_lambda: float = 0, **kwargs: Any) -> None:
        if reg_lambda > 0:
            regularizer = tf.keras.regularizers.l2(reg_lambda)
        else:
            regularizer = None

        super().__init__(kernel_regularizer=regularizer, **kwargs)

    def call(self, inputs: tf.SparseTensor) -> tf.Tensor:
        """Apply dense layer to sparse inputs.

        Arguments:
            inputs: Input sparse tensor (of any rank).

        Returns:
            Output of dense layer.
        Raises:
            A ValueError if inputs is not a sparse tensor
        """
        if not isinstance(inputs, tf.SparseTensor):
            raise ValueError("Input tensor should be sparse.")
        # outputs will be 2D
        # inputs =TensorShape([64, 28, 1100]) ， self.kernel = TensorShape([1100, 128])
        # inputs先转换为 2D= [1792, 1100] 再和 kernel相乘，outputs = TensorShape([1792, 128])
        outputs = tf.sparse.sparse_dense_matmul(
            tf.sparse.reshape(inputs, [-1, tf.shape(inputs)[-1]]), self.kernel
        )

        if len(inputs.shape) == 3:
            # reshape back
            outputs = tf.reshape(
                outputs, (tf.shape(inputs)[0], tf.shape(inputs)[1], self.units)
            )

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class RandomlyConnectedDense(tf.keras.layers.Dense):
    """Layer with dense ouputs that are connected to a random subset of inputs.

    `RandomlyConnectedDense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    It creates `kernel_mask` to set a fraction of the `kernel` weights to zero.

    Note: If the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    The output is guaranteed to be dense (each output is connected to at least one
    input), and no input is disconnected (each input is connected to at least one
    output).

    At `density = 0.0` the number of trainable weights is `max(input_size, units)`. At
    `density = 1.0` this layer is equivalent to `tf.keras.layers.Dense`.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, density: float = 0.2, **kwargs: Any) -> None:
        """Declares instance variables with default values.

        Args:
            density: Float between 0 and 1. Approximate fraction of trainable weights.
            units: Positive integer, dimensionality of the output space.
            activation: Activation function to use.
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation")..
            kernel_constraint: Constraint function applied to
                the `kernel` weights matrix.
            bias_constraint: Constraint function applied to the bias vector.
        """
        super().__init__(**kwargs)

        if density < 0.0 or density > 1.0:
            # raise TFLayerConfigException("Layer density must be in [0, 1].")
            raise Exception("Layer density must be in [0, 1].")

        self.density = density

    def build(self, input_shape: tf.TensorShape) -> None:
        """Prepares the kernel mask.

        Args:
            input_shape: Shape of the inputs to this layer
        """
        super().build(input_shape)

        if self.density == 1.0:
            self.kernel_mask = None
            return

        # Construct mask with given density and guarantee that every output is
        # connected to at least one input
        kernel_mask = self._minimal_mask() + self._random_mask()

        # We might accidently have added a random connection on top of
        # a fixed connection
        kernel_mask = tf.clip_by_value(kernel_mask, 0, 1)

        self.kernel_mask = tf.Variable(
            initial_value=kernel_mask, trainable=False, name="kernel_mask"
        )

    def _random_mask(self) -> tf.Tensor:
        """Creates a random matrix with `num_ones` 1s and 0s otherwise.

        Returns:
            A random mask matrix
        """
        mask = tf.random.uniform(tf.shape(self.kernel), 0, 1)
        mask = tf.cast(tf.math.less(mask, self.density), self.kernel.dtype)
        return mask

    def _minimal_mask(self) -> tf.Tensor:
        """Creates a matrix with a minimal number of 1s to connect everythinig.

        If num_rows == num_cols, this creates the identity matrix.
        If num_rows > num_cols, this creates
            1 0 0 0
            0 1 0 0
            0 0 1 0
            0 0 0 1
            1 0 0 0
            0 1 0 0
            0 0 1 0
            . . . .
            . . . .
            . . . .
        If num_rows < num_cols, this creates
            1 0 0 1 0 0 1 ...
            0 1 0 0 1 0 0 ...
            0 0 1 0 0 1 0 ...

        Returns:
            A tiled and croped identity matrix.
        """
        kernel_shape = tf.shape(self.kernel)
        num_rows = kernel_shape[0]
        num_cols = kernel_shape[1]
        short_dimension = tf.minimum(num_rows, num_cols)

        mask = tf.tile(
            tf.eye(short_dimension, dtype=self.kernel.dtype),
            [
                tf.math.ceil(num_rows / short_dimension),
                tf.math.ceil(num_cols / short_dimension),
            ],
        )[:num_rows, :num_cols]

        return mask

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Processes the given inputs.

        Args:
            inputs: What goes into this layer

        Returns:
            The processed inputs.
        """
        if self.density < 1.0:
            # Set fraction of the `kernel` weights to zero according to precomputed mask
            self.kernel.assign(self.kernel * self.kernel_mask)
        return super().call(inputs)


class Ffnn(tf.keras.layers.Layer):
    """Feed-forward network layer.

    Arguments:
        layer_sizes: List of integers with dimensionality of the layers.
        dropout_rate: Float between 0 and 1; fraction of the input units to drop.
        reg_lambda: Float, regularization factor.
        density: Float between 0 and 1. Approximate fraction of trainable weights.
        layer_name_suffix: Text added to the name of the layers.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., layer_sizes[-1])`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, layer_sizes[-1])`.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        dropout_rate: float,
        reg_lambda: float,
        density: float,
        layer_name_suffix: Text,
    ) -> None:
        super().__init__(name=f"ffnn_{layer_name_suffix}")

        l2_regularizer = tf.keras.regularizers.l2(reg_lambda)
        self._ffn_layers = []
        for i, layer_size in enumerate(layer_sizes):
            self._ffn_layers.append(
                RandomlyConnectedDense(
                    units=layer_size,
                    density=density,
                    activation=tfa.activations.gelu,
                    kernel_regularizer=l2_regularizer,
                    name=f"hidden_layer_{layer_name_suffix}_{i}",
                )
            )
            self._ffn_layers.append(tf.keras.layers.Dropout(dropout_rate))

    def call(
        self, x: tf.Tensor, training: Optional[Union[tf.Tensor, bool]] = None
    ) -> tf.Tensor:
        for layer in self._ffn_layers:
            x = layer(x, training=training)

        return x


class Embed(tf.keras.layers.Layer):
    """Dense embedding layer.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., embed_dim)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, embed_dim)`.
    """

    def __init__(
        self, embed_dim: int, reg_lambda: float, layer_name_suffix: Text
    ) -> None:
        """Initialize layer.

        Args:
            embed_dim: Dimensionality of the output space.
            reg_lambda: Regularization factor.
            layer_name_suffix: Text added to the name of the layers.
        """
        super().__init__(name=f"embed_{layer_name_suffix}")

        regularizer = tf.keras.regularizers.l2(reg_lambda)
        self._dense = tf.keras.layers.Dense(
            units=embed_dim,
            activation=None,
            kernel_regularizer=regularizer,
            name=f"embed_layer_{layer_name_suffix}",
        )

    # noinspection PyMethodOverriding
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Apply dense layer."""
        x = self._dense(x)
        return x


class InputMask(tf.keras.layers.Layer):
    """The layer that masks 15% of the input.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, input_dim)`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._masking_prob = 0.85
        self._mask_vector_prob = 0.7
        self._random_vector_prob = 0.1

    def build(self, input_shape: tf.TensorShape) -> None:
        self.mask_vector = self.add_weight(
            shape=(1, 1, input_shape[-1]), name="mask_vector"
        )
        self.built = True

    # noinspection PyMethodOverriding
    def call(
        self,
        x: tf.Tensor,
        mask: tf.Tensor,
        training: Optional[Union[tf.Tensor, bool]] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Randomly mask input sequences.

        Arguments:
            x: Input sequence tensor of rank 3.
            mask: A tensor representing sequence mask,
                contains `1` for inputs and `0` for padding.
            training: Python boolean indicating whether the layer should behave in
                training mode (mask inputs) or in inference mode (doing nothing).

        Returns:
            A tuple of masked inputs and boolean mask.
        """

        if training is None:
            training = K.learning_phase()

        lm_mask_prob = tf.random.uniform(tf.shape(mask), 0, 1, mask.dtype) * mask
        lm_mask_bool = tf.greater_equal(lm_mask_prob, self._masking_prob)

        def x_masked() -> tf.Tensor:
            x_random_pad = tf.random.uniform(
                tf.shape(x), tf.reduce_min(x), tf.reduce_max(x), x.dtype
            ) * (1 - mask)
            # shuffle over batch dim
            x_shuffle = tf.random.shuffle(x * mask + x_random_pad)

            # shuffle over sequence dim
            x_shuffle = tf.transpose(x_shuffle, [1, 0, 2])
            x_shuffle = tf.random.shuffle(x_shuffle)
            x_shuffle = tf.transpose(x_shuffle, [1, 0, 2])

            # shuffle doesn't support backprop
            x_shuffle = tf.stop_gradient(x_shuffle)

            mask_vector = tf.tile(self.mask_vector, (tf.shape(x)[0], tf.shape(x)[1], 1))

            other_prob = tf.random.uniform(tf.shape(mask), 0, 1, mask.dtype)
            other_prob = tf.tile(other_prob, (1, 1, x.shape[-1]))
            x_other = tf.where(
                other_prob < self._mask_vector_prob,
                mask_vector,
                tf.where(
                    other_prob < self._mask_vector_prob + self._random_vector_prob,
                    x_shuffle,
                    x,
                ),
            )

            return tf.where(tf.tile(lm_mask_bool, (1, 1, x.shape[-1])), x_other, x)

        return (
            tf_utils.smart_cond(training, x_masked, lambda: tf.identity(x)),
            lm_mask_bool,
        )


def _scale_loss(log_likelihood: tf.Tensor) -> tf.Tensor:
    """Creates scaling loss coefficient depending on the prediction probability.

    Arguments:
        log_likelihood: a tensor, log-likelihood of prediction

    Returns:
        Scaling tensor.
    """

    p = tf.math.exp(log_likelihood)
    # only scale loss if some examples are already learned
    return tf.cond(
        tf.reduce_max(p) > 0.5,
        lambda: tf.stop_gradient(tf.pow((1 - p) / 0.5, 4)),
        lambda: tf.ones_like(p),
    )


class CRF(tf.keras.layers.Layer):
    """CRF layer.

    Arguments:
        num_tags: Positive integer, number of tags.
        reg_lambda: Float; regularization factor.
        name: Optional name of the layer.
    """

    def __init__(
        self,
        num_tags: int,
        reg_lambda: float,
        scale_loss: bool,
        name: Optional[Text] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_tags = num_tags
        self.scale_loss = scale_loss
        self.transition_regularizer = tf.keras.regularizers.l2(reg_lambda)
        self.f1_score_metric = tfa.metrics.F1Score(
            num_classes=num_tags - 1,  # `0` prediction is not a prediction
            average="micro",
        )

    def build(self, input_shape: tf.TensorShape) -> None:
        # the weights should be created in `build` to apply random_seed
        self.transition_params = self.add_weight(
            shape=(self.num_tags, self.num_tags),
            regularizer=self.transition_regularizer,
            name="transitions",
        )
        self.built = True

    # noinspection PyMethodOverriding
    def call(
        self, logits: tf.Tensor, sequence_lengths: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Decodes the highest scoring sequence of tags.
        viterbi_decode 和 crf_decode 实现了相同功能，前者是numpy的实现，后者是 tensor 的实现,是viterbi_decode 的tensorflow版本。
        Arguments:
            logits: A [batch_size, max_seq_len, num_tags] tensor of
                unary potentials.
            sequence_lengths: A [batch_size] vector of true sequence lengths.
        Returns:
            A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
            Contains the highest scoring tag indices.
            A [batch_size, max_seq_len] matrix, with dtype `tf.float32`.
            Contains the confidence values of the highest scoring tag indices.
        """
        # 已知 发射矩阵 和 转移概率矩阵，使用 veterbi 解码获取最优的 tag 和 score
        predicted_ids, scores, _ = crf.crf_decode(
            logits, self.transition_params, sequence_lengths
        )
        # set prediction index for padding to `0`
        mask = tf.sequence_mask(
            sequence_lengths,
            maxlen=tf.shape(predicted_ids)[1],
            dtype=predicted_ids.dtype,
        )

        confidence_values = scores * tf.cast(mask, tf.float32)
        predicted_ids = predicted_ids * mask

        return predicted_ids, confidence_values

    def loss(
        self, logits: tf.Tensor, tag_indices: tf.Tensor, sequence_lengths: tf.Tensor
    ) -> tf.Tensor:
        """Computes the log-likelihood of tag sequences in a CRF.



        Arguments:
            logits: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
                to use as input to the CRF layer.
            tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which
                we compute the log-likelihood.
                每个 token 实际的 tag_index
            sequence_lengths: A [batch_size] vector of true sequence lengths.

        Returns:
            Negative mean log-likelihood of all examples,
            given the sequence of tag indices.


        crf_log_likelihood ->  crf_sequence_score -> crf_log_norm
        crf_sequence_score:
            crf_unary_score:
            crf_binary_score:

        """
        # 计算对应数据和标注之间的对数似然值
        # log_likelihood：对数似然值
        # transition_params: 状态转移矩阵 shape=(tag_num,tag_num)
        log_likelihood, _ = tfa.text.crf.crf_log_likelihood(
            logits, tag_indices, sequence_lengths, self.transition_params
        )
        loss = -log_likelihood
        if self.scale_loss:
            loss *= _scale_loss(log_likelihood)

        return tf.reduce_mean(loss)

    def f1_score(
        self, tag_ids: tf.Tensor, pred_ids: tf.Tensor, mask: tf.Tensor
    ) -> tf.Tensor:
        """Calculates f1 score for train predictions"""

        mask_bool = tf.cast(mask[:, :, 0], tf.bool)

        # pick only non padding values and flatten sequences
        tag_ids_flat = tf.boolean_mask(tag_ids, mask_bool)
        pred_ids_flat = tf.boolean_mask(pred_ids, mask_bool)

        # set `0` prediction to not a prediction
        num_tags = self.num_tags - 1

        tag_ids_flat_one_hot = tf.one_hot(tag_ids_flat - 1, num_tags)
        pred_ids_flat_one_hot = tf.one_hot(pred_ids_flat - 1, num_tags)

        return self.f1_score_metric(tag_ids_flat_one_hot, pred_ids_flat_one_hot)


class DotProductLoss(tf.keras.layers.Layer):
    """Dot-product loss layer."""

    def __init__(
        self,
        num_neg: int,
        loss_type: Text,
        mu_pos: float,
        mu_neg: float,
        use_max_sim_neg: bool,
        neg_lambda: float,
        scale_loss: bool,
        similarity_type: Text,
        name: Optional[Text] = None,
        same_sampling: bool = False,
        constrain_similarities: bool = True,
        model_confidence: Text = SOFTMAX,
    ) -> None:
        """Declare instance variables with default values.

        Args:
            num_neg: Positive integer, the number of incorrect labels; 默认值 20
                the algorithm will minimize their similarity to the input.
            loss_type:The type of the loss function, either 'cross_entropy' or 'margin'.
                默认值 'cross_entropy'

            mu_pos: Float, indicates how similar the algorithm should
                try to make embedding vectors for correct labels;
                should be 0.0 < ... < 1.0
                for 'cosine' similarity type.
                上面可能写错了，  for margin loss only
                正确的相似度的阈值，默认 0.8

            mu_neg: Float, maximum negative similarity for incorrect labels,
                should be -1.0 < ... < 1.0
                for 'cosine' similarity type.
                面可能写错了，  for margin loss only
                不正确的相似度的阈值， 默认 -0.4

            use_max_sim_neg: Boolean, if 'True' the algorithm only minimizes
                maximum similarity over incorrect intent labels,
                used only if 'loss_type' is set to 'margin'.
                默认 True

            neg_lambda: Float, the scale of how important is to minimize
                the maximum similarity between embeddings of different labels,
                used only if 'loss_type' is set to 'margin'.
                默认值 0.8，设置不同的值来调整该项loss 权重

            scale_loss: Boolean, if 'True' scale loss inverse proportionally to
                the confidence of the correct prediction.                 默认值 False

            similarity_type: Similarity measure to use, either 'cosine' or 'inner'.
                默认值 'inner'

            name: Optional name of the layer.
            same_sampling: Boolean, if 'True' sample same negative labels
                for the whole batch.                                      默认值 False
            constrain_similarities: Boolean, if 'True' applies sigmoid on all
                similarity terms and adds to the loss function to
                ensure that similarity values are approximately bounded.
                Used inside _loss_cross_entropy() only.
                默认值 False， 限制 similarity 的值，使得其在  [0,1] 之间
            model_confidence: Model confidence to be returned during inference.
                Possible values - 'softmax' and 'linear_norm'.
                默认值 'softmax'，
                在后续的判断中 model_confidence 对 trigger fallback threshold 很有意义，
                所以我们现在选择 'linear_norm'
                'linear_norm'：sim_i/sum(sim_i) 直接将 sim 转换为 model_confidence

        Raises:
            LayerConfigException: When `similarity_type` is not one of 'cosine' or
                'inner'.
        """
        super().__init__(name=name)
        self.num_neg = num_neg
        self.loss_type = loss_type
        self.mu_pos = mu_pos
        self.mu_neg = mu_neg
        self.use_max_sim_neg = use_max_sim_neg
        self.neg_lambda = neg_lambda
        self.scale_loss = scale_loss
        self.same_sampling = same_sampling
        self.constrain_similarities = constrain_similarities
        self.model_confidence = model_confidence
        self.similarity_type = similarity_type
        if self.similarity_type not in {COSINE, INNER}:
            raise TFLayerConfigException(
                f"Wrong similarity type '{self.similarity_type}', "
                f"should be '{COSINE}' or '{INNER}'."
            )

    @staticmethod
    def _make_flat(x: tf.Tensor) -> tf.Tensor:
        """Make tensor 2D."""

        return tf.reshape(x, (-1, x.shape[-1]))

    def _random_indices(
        self, batch_size: tf.Tensor, total_candidates: tf.Tensor
    ) -> tf.Tensor:
        return tf.random.uniform(
            shape=(batch_size, self.num_neg), maxval=total_candidates, dtype=tf.int32
        )

    @staticmethod
    def _sample_idxs(target_size: tf.Tensor, x: tf.Tensor, idxs: tf.Tensor) -> tf.Tensor:
        """Sample negative examples for given indices
        对于目标的数量 target_size， 每个从总体 total_candidates 中 抽样 num_neg
        target_size : 源码中 batch_size
        x = (total_candidates, emb_size)
        idxs = (target_size, num_neg)
        """
        # 复制所有的被抽样的样本 ：
        # x = (total_candidates, emb_size) ---> (1,total_candidates, emb_size) ---> (target_size,total_candidates, emb_size)
        tiled = tf.tile(tf.expand_dims(x, 0), (target_size, 1, 1))
        # output = (batch_size,neg_num)
        return tf.gather(tiled, idxs, batch_dims=1)

    def _get_bad_mask(
        self, labels: tf.Tensor, target_labels: tf.Tensor, idxs: tf.Tensor
    ) -> tf.Tensor:
        """Calculate bad mask for given indices.

        Checks that input features are different for positive negative samples.
        """
        # 原本的正样本 label ： target_labels = (target_size, 1) ---> (target_size,1, 1)
        pos_labels = tf.expand_dims(target_labels, axis=-2)
        # 抽样得到的 negative label ： neg_labels = （target_size, num_neg, 1）
        neg_labels = self._sample_idxs(tf.shape(target_labels)[0], labels, idxs)
        # 判断 target label 和 抽样的 negative label 是否一致（利用 broadcasting)
        # 如果相同表示是正样本 ，对于 negative sampling 是 bad mask =1
        # (target_size, num_neg)
        return tf.cast(
            tf.reduce_all(tf.equal(neg_labels, pos_labels), axis=-1), pos_labels.dtype
        )

    def _get_negs(
        self, embeds: tf.Tensor, labels: tf.Tensor, target_labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get negative examples from given tensor."""
        # 将 embedding 变换为 二维的tensor
        # batch input  & total labels embeddings ：
        # batch input embeds = [64, 1, 20]         --->  [64, 20]
        # total labels embed = [num_labels, 1, 20] --->  [num_labels, 20]
        embeds_flat = self._make_flat(embeds)
        labels_flat = self._make_flat(labels)

        # 目标 抽样的 labels = [batch_size]
        target_labels_flat = self._make_flat(target_labels)
        # 总样本的数量:
        # 对 input embeds 做抽样的时候 = batch_size，
        # 对 label embeds 做抽样的时候 = num_label
        total_candidates = tf.shape(embeds_flat)[0]

        # 希望抽取的数量: target_size = batch_size
        target_size = tf.shape(target_labels_flat)[0]

        # 从 total_candidates 随机抽样 self.num_neg 个 index,使得每个样本都有 num_neg 对应的 index
        # neg_ids = (target_size, num_neg),
        neg_ids = self._random_indices(target_size, total_candidates)

        # 根据 neg_ids 从 embeds_flat 进行抽样 target_size 次
        # neg_embeds =(target_size, num_neg,emb_dim)
        neg_embeds = self._sample_idxs(target_size, embeds_flat, neg_ids)
        # 从总样本中抽样的 negative label 与  labels 相同的作为 不合格的 neg
        # bad_negs = (target_size, num_neg)
        bad_negs = self._get_bad_mask(labels_flat, target_labels_flat, neg_ids)

        # check if inputs have sequence dimension
        if len(target_labels.shape) == 3:
            # tensors were flattened for sampling, reshape back
            # add sequence dimension if it was present in the inputs
            target_shape = tf.shape(target_labels)
            neg_embeds = tf.reshape(
                neg_embeds, (target_shape[0], target_shape[1], -1, embeds.shape[-1])
            )
            bad_negs = tf.reshape(bad_negs, (target_shape[0], target_shape[1], -1))

        return neg_embeds, bad_negs

    def _sample_negatives(
        self,
        inputs_embed: tf.Tensor,
        labels_embed: tf.Tensor,
        labels: tf.Tensor,
        all_labels_embed: tf.Tensor,
        all_labels: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

        """Sample negative examples."""
        # inputs_embed = [64, 20] --->  pos_inputs_embed = [64, 1, 20]
        pos_inputs_embed = tf.expand_dims(inputs_embed, axis=-2)
        # labels_embed = [64, 20] --->  pos_inputs_embed = [64, 1, 20]
        pos_labels_embed = tf.expand_dims(labels_embed, axis=-2)
        # sample negative inputs：
        # 从 input_embed 中 进行 随机抽取负样本
        # labels = [batch_size,1],
        # neg_inputs_embed = (batch_size, num_neg, emb_dim)
        # inputs_bad_negs =  (batch_size, num_neg)
        neg_inputs_embed, inputs_bad_negs = self._get_negs(inputs_embed, labels, labels)
        # sample negative labels：
        # 从所有的 all_labels_embed 中进行 负抽样:
        # neg_labels_embed = (batch_size, num_neg, emb_dim)
        # labels_bad_negs = (batch_size, num_neg)
        neg_labels_embed, labels_bad_negs = self._get_negs(all_labels_embed, all_labels, labels)

        return (
            pos_inputs_embed,
            pos_labels_embed,
            neg_inputs_embed,
            neg_labels_embed,
            inputs_bad_negs,
            labels_bad_negs,
        )

    def sim(
        self, a: tf.Tensor, b: tf.Tensor, mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Calculate similarity between given tensors.
        a: sentence embedding   (batch_size, 1,embedding_size)
        b: label embedding      (batch_size, label_num,embedding_size)
        sim: 相似度              (batch_size,)

        COSINE similarity : 约接近于 1表示越相似
        INNER product :
        """
        if self.similarity_type == COSINE:
            a = tf.nn.l2_normalize(a, axis=-1)
            b = tf.nn.l2_normalize(b, axis=-1)
        sim = tf.reduce_sum(a * b, axis=-1)
        if mask is not None:
            sim *= tf.expand_dims(mask, 2)

        return sim

    def similarity_confidence_from_embeddings(
        self,
        input_embeddings: tf.Tensor,
        label_embeddings: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Computes similarity.

        Calculates similary between input and label embeddings and model's confidence.
        First compute the similarity from embeddings and then apply an activation
        function if needed to get the confidence.
        Args:
            input_embeddings: Embeddings of input.
            label_embeddings: Embeddings of labels.
            mask: Mask over input and output sequence.

        Returns:
            similarity between input and label embeddings and model's prediction
            confidence for each label.
        """
        # 计算相似度:
        # input_embeddings = [batch_size, input_size,embedding_size] = [1,1,20]
        # label_embeddings= [batch_size,label_num,embedding_size] =[1, 39, 20]
        # similarities = [batch_size, label_num]
        similarities = self.sim(input_embeddings, label_embeddings, mask)
        confidences = similarities

        # 获取模型的 model_confidence
        if self.model_confidence == SOFTMAX:
            confidences = tf.nn.softmax(similarities)
        if self.model_confidence == LINEAR_NORM:
            # Clip negative values to 0 and linearly normalize to bring the predictions
            # in the range [0,1].
            clipped_similarities = tf.nn.relu(similarities)
            confidences = clipped_similarities / tf.reduce_sum(
                clipped_similarities, axis=-1
            )
        return similarities, confidences

    def _train_sim(
        self,
        pos_inputs_embed: tf.Tensor,
        pos_labels_embed: tf.Tensor,
        neg_inputs_embed: tf.Tensor,
        neg_labels_embed: tf.Tensor,
        inputs_bad_negs: tf.Tensor,
        labels_bad_negs: tf.Tensor,
        mask: Optional[tf.Tensor],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Define similarity."""

        # calculate similarity with several embedded actions for the loss
        neg_inf = tf.constant(-1e9)

        # 计算 embedding 和 positive label 之间的相似度 ： (batch_size,1)
        # 使用 原有的 inputs_embed 和 对应的 labels_embed 的相似度
        # pos_inputs_embed = [64, 1, 20], pos_labels_embed = [64, 1, 20] ---> [64, 1]
        sim_pos = self.sim(pos_inputs_embed, pos_labels_embed, mask)

        # 计算 embedding 和 negative label 之间的相似度 ： (batch_size,num_neg), 后一个维度表示多少个负样本的相似度
        # 使用 原有的 inputs_embed 和 neg_labels_embed 计算相似度
        # pos_inputs_embed =[64, 1, 20],  neg_labels_embed = [64, num_neg, 20], sim_neg_il => [64, num_neg]
        # neg_inf * labels_bad_negs: 相当于 mask 作用，在对应的位置加入一个 无限小的数,后面再做 softmax 或者 logit 之后会变成0
        sim_neg_il = (
            self.sim(pos_inputs_embed, neg_labels_embed, mask)
            + neg_inf * labels_bad_negs
        )

        # 计算 标签之间的相似度 ： (batch_size, num_neg)
        #  pos_labels_embed =[64, 1, 20],  neg_labels_embed = [64, num_neg, 20], sim_neg_ll => [64, num_neg]
        sim_neg_ll = (
            self.sim(pos_labels_embed, neg_labels_embed, mask)
            + neg_inf * labels_bad_negs
        )

        # 计算 embedding 和 negative embedding 之间的相似度
        #  pos_inputs_embed =[64, 1, 20],  neg_inputs_embed = [64, num_neg, 20], sim_neg_ii => [64, num_neg]
        sim_neg_ii = (
            self.sim(pos_inputs_embed, neg_inputs_embed, mask)
            + neg_inf * inputs_bad_negs
        )

        # 计算 positive label 和 neg embedding 之间的相似度
        sim_neg_li = (
            self.sim(pos_labels_embed, neg_inputs_embed, mask)
            + neg_inf * inputs_bad_negs
        )

        # output similarities between user input and bot actions
        # and similarities between bot actions and similarities between user inputs
        return sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li

    @staticmethod
    def _calc_accuracy(sim_pos: tf.Tensor, sim_neg: tf.Tensor) -> tf.Tensor:
        """Calculate accuracy.计算准确性：
        通过对比 positive input & label(可以是多标签的) 的 similarity 值  vs  positive input & negative label"""

        # 拼接 positive similarity 和 neg_num 个 neg similarity
        # 选取 input 对应的 label 中计算得到的 similarities 中最大值
        # sim_pos = (batch_size,1) , sim_neg = (batch_size,num_neg) -> (batch_size, 1+ num_neg) -> (batch_size)

        max_all_sim = tf.reduce_max(tf.concat([sim_pos, sim_neg], axis=-1), axis=-1)
        # 当 最大值与 sim_pos 相等的时候为 true,计算平均值
        return tf.reduce_mean(
            tf.cast(
                tf.math.equal(max_all_sim, tf.squeeze(sim_pos, axis=-1)), tf.float32
            )
        )

    def _loss_margin(
        self,
        sim_pos: tf.Tensor,
        sim_neg_il: tf.Tensor,
        sim_neg_ll: tf.Tensor,
        sim_neg_ii: tf.Tensor,
        sim_neg_li: tf.Tensor,
        mask: Optional[tf.Tensor],
    ) -> tf.Tensor:
        """Define max margin loss."""


        # loss for maximizing similarity with correct action
        # 当 sim_pos > mu_pos=0.8 的时候，不惩罚 loss，但 sim_pos < mu_pos,我们认为相似度较低做惩罚
        loss = tf.maximum(0.0, self.mu_pos - tf.squeeze(sim_pos, axis=-1))

        # loss for minimizing similarity with `num_neg` incorrect actions
        if self.use_max_sim_neg:
            # minimize only maximum similarity over incorrect actions
            max_sim_neg_il = tf.reduce_max(sim_neg_il, axis=-1)
            loss += tf.maximum(0.0, self.mu_neg + max_sim_neg_il)
        else:
            # minimize all similarities with incorrect actions
            max_margin = tf.maximum(0.0, self.mu_neg + sim_neg_il)
            loss += tf.reduce_sum(max_margin, axis=-1)

        # penalize max similarity between pos bot and neg bot embeddings
        max_sim_neg_ll = tf.maximum(
            0.0, self.mu_neg + tf.reduce_max(sim_neg_ll, axis=-1)
        )
        loss += max_sim_neg_ll * self.neg_lambda

        # penalize max similarity between pos dial and neg dial embeddings
        max_sim_neg_ii = tf.maximum(
            0.0, self.mu_neg + tf.reduce_max(sim_neg_ii, axis=-1)
        )
        loss += max_sim_neg_ii * self.neg_lambda

        # penalize max similarity between pos bot and neg dial embeddings
        max_sim_neg_li = tf.maximum(
            0.0, self.mu_neg + tf.reduce_max(sim_neg_li, axis=-1)
        )
        loss += max_sim_neg_li * self.neg_lambda

        if mask is not None:
            # mask loss for different length sequences
            loss *= mask
            # average the loss over sequence length
            loss = tf.reduce_sum(loss, axis=-1) / tf.reduce_sum(mask, axis=1)

        # average the loss over the batch
        loss = tf.reduce_mean(loss)

        return loss

    def _loss_cross_entropy(
        self,
        sim_pos: tf.Tensor,
        sim_neg_il: tf.Tensor,
        sim_neg_ll: tf.Tensor,
        sim_neg_ii: tf.Tensor,
        sim_neg_li: tf.Tensor,
        mask: Optional[tf.Tensor],
    ) -> tf.Tensor:
        """Defines cross entropy loss."""
        loss = self._compute_softmax_loss(
            sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li
        )

        if self.constrain_similarities:
            """
            # 是否 使用 sigmoid 对 logits 限制范围
            In order to constrain all similarity values to an approximate range, 
            we applied sigmoid on each of them and added the cross entropy loss over sigmoid probabilities to the final total loss.
            于此同时，在 _compute_softmax_loss() 的时候计算 sim_pos, sim_neg_il, sim_neg_ll 3者的loss之和
            """
            # TODO:在 infere 的时候如何限制 similarity

            loss += self._compute_sigmoid_loss(
                sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li
            )

        if self.scale_loss:
            # in case of cross entropy log_likelihood = -loss
            loss *= _scale_loss(-loss)

        if mask is not None:
            loss *= mask

        if len(loss.shape) == 2:
            # average over the sequence
            if mask is not None:
                loss = tf.reduce_sum(loss, axis=-1) / tf.reduce_sum(mask, axis=-1)
            else:
                loss = tf.reduce_mean(loss, axis=-1)

        # average the loss over the batch
        return tf.reduce_mean(loss)

    @staticmethod
    def _compute_sigmoid_loss(
        sim_pos: tf.Tensor,
        sim_neg_il: tf.Tensor,
        sim_neg_ll: tf.Tensor,
        sim_neg_ii: tf.Tensor,
        sim_neg_li: tf.Tensor,
    ) -> tf.Tensor:
        # Constrain similarity values in a range by applying sigmoid
        # on them individually so that they saturate at extreme values.
        sigmoid_logits = tf.concat(
            [sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li], axis=-1
        )
        sigmoid_labels = tf.concat(
            [
                tf.ones_like(sigmoid_logits[..., :1]),
                tf.zeros_like(sigmoid_logits[..., 1:]),
            ],
            axis=-1,
        )
        sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=sigmoid_labels, logits=sigmoid_logits
        )
        # average over logits axis
        return tf.reduce_mean(sigmoid_loss, axis=-1)

    def _compute_softmax_loss(
        self,
        sim_pos: tf.Tensor,
        sim_neg_il: tf.Tensor,
        sim_neg_ll: tf.Tensor,
        sim_neg_ii: tf.Tensor,
        sim_neg_li: tf.Tensor,
    ) -> tf.Tensor:
        # Similarity terms between input and label should be optimized relative
        # to each other and hence use them as logits for softmax term
        softmax_logits = tf.concat([sim_pos, sim_neg_il, sim_neg_li], axis=-1)
        if not self.constrain_similarities:
            # Concatenate other similarity terms as well. Due to this,
            # similarity values between input and label may not be
            # approximately bounded in a defined range.
            # 如果不需要 constrain_similarities ，训练的时候加入 sim_neg_ii, sim_neg_ll ，ji
            # 加入之后不利于 限制 similarity 的范围
            softmax_logits = tf.concat(
                [softmax_logits, sim_neg_ii, sim_neg_ll], axis=-1
            )

        # create label_ids for softmax
        # 选取最后一位的第一列 的 0 值 作为 positive
        softmax_label_ids = tf.zeros_like(softmax_logits[..., 0], tf.int32)
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=softmax_label_ids, logits=softmax_logits
        )
        return softmax_loss

    @property
    def _chosen_loss(self) -> Callable:
        """Use loss depending on given option."""
        if self.loss_type == MARGIN:
            return self._loss_margin
        elif self.loss_type == CROSS_ENTROPY:
            return self._loss_cross_entropy
        else:
            raise TFLayerConfigException(
                f"Wrong loss type '{self.loss_type}', "
                f"should be '{MARGIN}' or '{CROSS_ENTROPY}'"
            )

    # noinspection PyMethodOverriding
    def call(
        self,
        inputs_embed: tf.Tensor,
        labels_embed: tf.Tensor,
        labels: tf.Tensor,
        all_labels_embed: tf.Tensor,
        all_labels: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calculate loss and accuracy.

        Arguments:
            inputs_embed: Embedding tensor for the batch inputs.
            labels_embed: Embedding tensor for the batch labels.  根据 all_labels_embed 获取的训练数据的 labels_embed
            labels: Tensor representing batch labels.
            all_labels_embed: Embedding tensor for the all labels. 所有的 label 进行编码后的 embedding
            all_labels: Tensor representing all labels.            所有的 label
            mask: Optional tensor representing sequence mask,
                contains `1` for inputs and `0` for padding.

        Returns:
            loss: Total loss.
            accuracy: Training accuracy.
        """
        # _sample_negatives ：sorted sampling strategy.
        # 不是简单的 randomly sampling negatives，
        # 而是 mine hard negatives by choosing those embeddings which have the highest similarity in the pair.
        # 抽样每个样本对应的的 input_embedding 和  label_embedding
        # pos_inputs_embed = pos_labels_embed = (batch_size, 1, emb_size)
        # neg_inputs_embed = neg_labels_embed = (batch_size, num_neg, emb_size)
        (
            pos_inputs_embed,
            pos_labels_embed,
            neg_inputs_embed,
            neg_labels_embed,
            inputs_bad_negs,
            labels_bad_negs,
        ) = self._sample_negatives(
            inputs_embed, labels_embed, labels, all_labels_embed, all_labels
        )

        # calculate similarities
        # 计算 5 中 similarities: sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li
        # sim_pos = [64, 1], sim_neg_il = [64, num_neg]
        sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li = self._train_sim(
            pos_inputs_embed,
            pos_labels_embed,
            neg_inputs_embed,
            neg_labels_embed,
            inputs_bad_negs,
            labels_bad_negs,
            mask,
        )
        # 计算准确性 accuracy = () ： 通过对比 positive input & label 的 similarity 值  vs  positive input & negative label
        accuracy = self._calc_accuracy(sim_pos, sim_neg_il)
        # 使用 similarities  计算损失 loss =(batch_size)：
        loss = self._chosen_loss(
            sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li, mask
        )

        return loss, accuracy



if __name__ == '__main__':
    layer = MyDenseLayer(1)
    x= tf.zeros([2, 3])
    y = layer(x)
    print(f"y.__class__={y.__class__},\ny.shape={y.shape}\ny={y}")

    variables = layer.variables
    non_trainable_variables =layer.non_trainable_variables
    trainable_variables =layer.trainable_variables

    print(f"trainable_variables.__len__={trainable_variables.__len__()}\n"
          f"trainable_variables[0].shape={trainable_variables[0].shape},trainable_variables[1].shape={trainable_variables[1].shape}")
