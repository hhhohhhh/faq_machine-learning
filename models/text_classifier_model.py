#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/23 9:38 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/23 9:38   wangfc      1.0         None
"""

import os
import  tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from bert4keras.models import build_transformer_model
from models.temp_keras_modules import TmpKerasModel
from models.builder import MODELS
from tensorflow.keras.layers import Dropout, Dense,Input,Lambda,Bidirectional,LSTM
from transformers.models.albert.configuration_albert import AlbertConfig

import logging
logger = logging.getLogger(__name__)


class RNNClassiferModel(tf.keras.models.Model):
    """
    RNN 层：
    默认情况下，RNN层的输出每个样本包含一个向量。这个向量是与最后一个时间步对应的RNN单元输出，包含关于整个输入序列的信息。这个输出的形状是(batch_size, units)，其中的units对应于传递给层构造函数的units参数。
    return_sequences=True，则RNN层还可以返回每个样本的整个输出序列(每个时间步一个向量)。这个输出的形状是(batch_size、timesteps、units)
    return_state=True,可以返回其最终的内部状态。返回的状态可用于稍后恢复RNN执行，或初始化另一个RNN。该设置通常用于编码器-解码器的顺序-顺序模型，其中编码器的最终状态用作解码器的初始状态。
    注意，LSTM有两个状态张量，而GRU只有一个。

    RNN Cell:
    RNN(LSTMCell(10))产生与LSTM(10)相同的结果。事实上，这一层在TF v1.x中的实现只是创建了相应的RNN单元并将其包装在RNN层中。
    除了内置的RNN层之外，RNN API还提供单元级API 。与处理整批输入序列的RNN层不同，RNN单元只处理单个时间步长。

    Keras里有三中内置的RNN单元，分别对应三种RNN层。
    （1）tf.keras.layers.SimpleRNNCell与SimpleRNN层相对应
    （2）tf.keras.layers.GRUCell与GRU层相对应
    （3）tf.keras.layers.LSTMCell与LSTM层相对应
    抽象单元，以及通用的tf.keras.layers.RNN类，使它非常容易实现自定义的RNN架构为我们的研究使用。

    RNN 参数量：
    state_t = tanh(U * x_t + W * s_t-1)
    output_t = softmax( V * state_t)

    U = hidden_size* input_dim
    W = hidden_size * hidden_size, 如果考虑到 cell_bias，还需加上 hidden_size
    V = hidden_size * output_dim, 如果考虑到 output_bias，还需加上 output_dim


    对于 SimpleRNN，计算其可训练参数时，并未计入到输出层的矩阵 V ，此时训练参数的数目为:
    SimpleRNN weights = U + W + cell_biases
                      = hidden_size * input_dim + hidden_size * hidden_size + hidden_size
                      = hidden_size *  (input_dim +hidden_size) + hidden_size
                      =  32 * 16 + 32*32 + 32

    对于 LSTM weights = 4 * (SimpleRNN weights)
    candidate_c = tanh(U * x_t + W * s_t-1 + bias)
    update_gate = sigmoid(W_u * (x_t,s_t-1) + bias)
    forget_gate = sigmoid(W_f * (x_t,s_t-1) + bias)
    output_gate = sigmoid(W_o * (x_t,s_t-1) + bias)

    c_t = update_gate * candidate_c + forget_gate * c_t-1
    s_t = output_gate * c_t

    GRU:


    """
    def __init__(self,input_size,vocab_size, embedding_dim=32,
                 cell='rnn',hidden_size=32,layers_num=1, bidirectional=False,
                 units=10,output_size=1):
        super(RNNClassifer, self).__init__()
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.cell = cell
        self.hidden_size = hidden_size
        self.layers_num = layers_num
        self.bidirectional = bidirectional

        self.units = units
        self.output_size = output_size

        self.embed = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)



        rnn_layers = []
        for index in range(self.layers_num):
            if index == self.layers_num-1:
                # 最后一层为 return_sequences = False
                # output = [batchsize,hidden_state]
                return_sequences = False
            else:
                # output = [batchsize,timesteps,hidden_state]
                return_sequences = True


            if self.cell == 'rnn':
                # 一个完全连接的RNN，它将前一个时间步的输出反馈给下一个时间步
                rnn_layer = keras.layers.SimpleRNN(units=self.hidden_size,return_sequences=return_sequences)
            elif self.cell =='lstm':
                #
                rnn_layer = keras.layers.LSTM(units=self.hidden_size,return_sequences=return_sequences)

            elif self.cell =='gru':
                rnn_layer = keras.layers.GRU(units=self.hidden_size)

            if self.bidirectional:
                # 是否使用 bidirectional
                rnn_layer = keras.layers.Bidirectional(rnn_layer)

            rnn_layers.append(rnn_layer)

        self.rnn_layers =rnn_layers

        self.dense = keras.layers.Dense(units=self.units,activation='relu')

        self.output_layer = keras.layers.Dense(units=self.output_size, activation='sigmoid')

        self.input_layer = keras.layers.Input(self.input_size[1:])
        self.outputs = self.call(self.input_layer)

        self.build(input_shape=self.input_size)

    def call(self,inputs):
        x = self.embed(inputs)
        for rnn_layer in self.rnn_layers:
            # x 输出 (batch_size, hidden_size)
            x = rnn_layer(x)
        x = self.dense(x)
        x = self.output_layer(x)
        return x



@MODELS.register_module()
class AlbertClassifierModel(tf.keras.models.Model):
    def __init__(self, multilabel_classifier=True,output_size=250,max_seq_length=512,
                 model_config=None, with_pool=True,with_lstm=True,lstm_units=312,
                 dropout_rate=0.1,
                 pretrained_model_dir=None,
                 config_filename='albert_config.json',init_checkpoint=None,
                 freeze_pretrained_weights=False
                 ):
        super().__init__()
        self.multilabel_classifier = multilabel_classifier
        self.output_size = output_size
        self.max_seq_length = max_seq_length

        self.pretrained_model_dir = pretrained_model_dir
        self.config_filename = config_filename
        self.init_checkpoint = init_checkpoint
        if model_config is None:
            config_path = os.path.join(pretrained_model_dir, config_filename)
            model_config = AlbertConfig.from_pretrained(pretrained_model_name_or_path=config_path)
        # 是否输出 pooler 对应的 embedding
        self.with_pool = with_pool
        self.with_lstm = with_lstm
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.config_path = config_path
        self.model_config = model_config

        self.freeze_pretrained_weights = freeze_pretrained_weights
        self.built =False


    @property
    def initializer(self):
        """默认使用截断正态分布初始化
        """
        return keras.initializers.TruncatedNormal(stddev=self.model_config.initializer_range)

    def get_inputs(self):
        # 定义整个模型的输入和输出
        input_ids = Input(shape=(self.max_seq_length,), name='input_ids', dtype='int32')
        token_type_ids = Input(shape=(self.max_seq_length,), name='token_type_ids', dtype='int32')
        inputs = [input_ids,token_type_ids]
        return inputs


    def build(self):
        """
        @time:  2021/6/29 22:34
        @author:wangfc
        @version:
        @description: 自定义模型的构建函数，建立模型需要的 layer

        @params:
        @return:
        """
        if self.built:
            return None

        # 使用 transformers API 初始化模型，但是 albert的 initial ckpt 参数名称不同需要自己做映射
        # self.albert = TFAlbertModel(config=model_config)

        # 使用 bert4keras API：  创建 layer + build 模型 + 加载预训练模型
        self.albert = build_transformer_model(
            config_path=self.config_path,
            checkpoint_path=self.init_checkpoint,
            model='albert',
            with_pool= self.with_pool,
            return_keras_model=False,
            build_model=True, #  构建一下 albert
            init_from_ckpt= False)

        # final layer
        self.albert_model = self.albert.model

        # # 获取 第一个 token [CLS] 对应的tensor
        # self.pooler = Lambda(function=lambda x: x[:,0],name='pooler')
        # self.pooler_dense = Dense(units=self.model_config.hidden_size,activation='tanh',
        #                           kernel_initializer= self.initializer,
        #                           # tf.truncated_normal_initializer(stddev=self.model_config.initializer_range),
        #                           name = 'pooler_dense')
        # # 增加自定义 dropout + dense 层
        # self.classifier_dropout = Dropout(rate=self.model_config.classifier_dropout_rate, name='classifier_dropout')
        if self.multilabel_classifier:
            self.classifier_dense_activation = 'sigmoid'
        else:
            self.classifier_dense_activation = 'softmax'



        if self.with_lstm:
            # self.bilstm_01 =Bidirectional(LSTM(units=self.albert.hidden_size,return_sequences=True),name='bilstm_01')
            self.bilstm =Bidirectional(LSTM(units=self.lstm_units),name='bilstm')


        self.dropout = Dropout(rate= self.dropout_rate, name='classifier_dropout')
        self.classifier_dense = Dense(units=self.output_size, activation=self.classifier_dense_activation,
                                      kernel_initializer=self.initializer,
                                      use_bias=True,bias_initializer='zeros',
                                      name='classifier_dense')

        # build
        inputs = self.get_inputs()

        outputs= self.call(inputs)

        # 使用 function API 建立自定义的模型
        self.model = Model(inputs=inputs,outputs=outputs)

        # 初始化 transformer 部分的参数
        self.albert.load_weights_from_checkpoint(checkpoint=self.init_checkpoint)

        # 是否 固定 ablert 变量
        if self.freeze_pretrained_weights:
            self.albert_model.trainable = False

        self.built =True
        logger.info(f"构建模型 with_pool={self.with_pool} ,dropout_rate={self.dropout_rate}")
        # logger.info(f"model_name={self.model.__class__},model.count_params={self.model.count_params()}\nmodel_summary={self.model.summary()}")


    def call(self, inputs,training = False):
        """
        @time:  2021/6/17 10:40
        @author:wangfc
        @version:
        @description: 自定义的输入 inputs = [input_ids,token_type_ids]

        @params:
        training 参数: which you can use to specify a different behavior in training and inference:
        @return:
        """
        # x = inputs
        # with_pool:  x = [None, 312]
        # not with_pool: x = [None, 512, 312]
        x = self.albert.model(inputs)

        # 获取 CLS token
        # x = self.pooler(x)
        # x = self.pooler_dense(x)
        # x = self.classifier_dropout(x)

        if self.with_lstm:
            x = self.bilstm(x)
            # x = self.bilstm_02(x)

        # 控制 training 和 inference的不同状态
        if training:
            x= self.dropout(x)

        x = self.classifier_dense(x)
        return x