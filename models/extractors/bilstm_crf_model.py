#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/10 9:31 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/10 9:31   wangfc      1.0         None
"""
import traceback
from typing import Union, Tuple, Dict, Text, List
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from models.temp_keras_modules import RasaModel



class BilstmCrfModel(RasaModel):
    """
    我们继承 RasaModel
    通过自定义 train_step() 来实现模型的训练：
    train_step(): 由父类 RasaModel 提供
    batch_loss(): 需要重载

    inference:
    predict_step(): 由父类 RasaModel 提供
    batch_predict: 需要重载

    """

    def __init__(self, tag_size: int, vocabulary: List[Text] = None,
                 embed_dims: int = 128, lstm_unit: int = 64,
                 *args, **kwargs):
        super(BilstmCrfModel, self).__init__(*args, **kwargs)
        self.tag_size = tag_size
        self.vocabulary = vocabulary
        # 初始化 vocab_size，后面可能因为增加 OOV_TOKEN 和 MASK_TOKEN 而更新
        self.vocab_size = self.vocabulary.__len__()
        self.embed_dims = embed_dims
        self.lstm_unit = lstm_unit
        self._prepare_layers()

    def _prepare_layers(self):
        # 建立 lookup_layer 并更新 vocab_size
        # self.lookup_layer = self._prepare_lookup_layer(vocabulary=self.vocabulary)
        # 建立 embedding layer
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embed_dims, mask_zero=True)
        # 建立 Bidirectional
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_unit, return_sequences=True))
        # 建立 CRF layer
        self.crf = tfa.layers.CRF(self.tag_size)


    # def _prepare_lookup_layer(self, vocabulary: List[Text], mask_token="[MASK]", oov_token="[UNK]") -> StringLookup:
    #     """
    #     参考
    #     # from models.features import CategoricalFeatureEmbedding
    #
    #     """
    #     if mask_token:
    #         #  StringLookup 是否 含有 mask_token
    #         vocab_size = len(vocabulary) + 2
    #     else:
    #         #  StringLookup 默认含有 oov_token="[UNK]"
    #         vocab_size = len(vocabulary) + 1
    #     # self.vocab_size = vocab_size
    #
    #     lookup_layer = StringLookup(max_tokens=vocab_size, mask_token=mask_token, oov_token=oov_token,
    #                                 vocabulary=vocabulary)
    #     return lookup_layer

    # def adapt(self, tokens):
    #     """
    #     模型中的预处理层需要先进行 adapt
    #     """
    #     self.lookup_layer.adapt(tokens=tokens)
    #     print(len(self.lookup_layer.get_vocabulary()))
    #     print(self.lookup_layer.get_vocabulary()[:10])

    def call(self, inputs, training=None, mask=None):
        # x = self.lookup_layer(inputs)
        try:
            x = self.embedding(inputs)
            # The problem is that my input data existing all-zeros sequences.
            # It will cause the input data will be all masked by Masking Layer.
            x = self.bilstm(x)
            x = self.crf(x)
        except Exception as e:
            traceback.print_exc()
            raise e
        return x

    # @tf.function(experimental_relax_shapes=True)
    def crf_loss_func(self, potentials, sequence_length, kernel, y):
        """
        计算 crf 层 的 loss
        """
        # The likelihood measures the probability of real y output from the CRF layer.
        # By using the real y and some internal variables of the CRF layer, you can compute the log likelihood of real y.
        crf_likelihood, _ = tfa.text.crf_log_likelihood(
            potentials, y, sequence_length, kernel
        )
        # likelihood to loss
        # Since the goal is making the model to output a higher likelihood of the real y,
        # the negative of the log likelihood is used as the loss to optimize.
        flat_crf_loss = -1 * crf_likelihood
        crf_loss = tf.reduce_mean(flat_crf_loss)

        return crf_loss

    # def train_step(self, data):
    #     """
    #     Using a custom training loop is a powerful way to customize the model while still leveraging the convenience of fit().
    #     You can find more detailed information about the custom training loop at
    #     https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch.
    #     In this section, you will learn how to use the CRF layer in a custom training loop.
    #     """
    #     with tf.GradientTape() as tape:
    #         decoded_sequence, potentials, sequence_length, kernel = model(x)
    #         crf_loss = crf_loss_func(potentials, sequence_length, kernel, y)
    #         loss = crf_loss + tf.reduce_sum(model.losses)
    #     grads = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #     train_loss(loss)

    def batch_loss(
            self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        """
        输入 batch_in，计算loss
        """
        x = batch_in[0]
        y = batch_in[1]
        # 使用 bilstm +crf 进行前向传播
        decoded_sequence, potentials, sequence_length, kernel = self.call(x)
        # 计算 crf_loss
        crf_loss = self.crf_loss_func(potentials, sequence_length, kernel, y)
        return crf_loss

    def batch_predict(
            self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]:
        pass
