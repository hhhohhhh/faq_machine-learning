#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/14 16:47 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/14 16:47   wangfc      1.0         None
"""


# from models.temp_keras_modules import TmpKerasModel
# from transformers.modeling_tf_utils import TFModelInputType
# from transformers.models.albert import TFAlbertModel
# from transformers.modeling_tf_outputs import TFSequenceClassifierOutput

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras import Input
from models.model_contants import TRANSFORMERS_ALBERT_INPUT_KEYS
from transformers.models.albert import TFAlbertForSequenceClassification
from transformers.models.albert.configuration_albert import AlbertConfig
from models.builder import MODELS
import logging

from utils.tensorflow.utils import is_checkpoint_exist

logger = logging.getLogger(__name__)


class Transformer(tf.keras.models.Model):
    def __init__(self, checkpoint=None, prefix=None):
        super(Transformer,self).__init__()
        self._checkpoint = checkpoint
        self._prefix = prefix or ''

    def variable_mapping(self):
        """构建keras层与checkpoint的变量名之间的映射表
        """
        return {}

    def _prefixed(self, name):
        """给名字加前缀
        """
        if name is not None:
            return self.prefix + name

    def _load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        if isinstance(checkpoint, dict):
            return checkpoint[name]
        else:
            # 提取 checkpoint 中 的 name = 'bert/embeddings/word_embeddings'  的 value= numpy.ndarray(21128, 128)
            return tf.train.load_variable(checkpoint, name)

    def _load_weights_from_checkpoint(self, checkpoint=None, mapping=None):
        """根据mapping从checkpoint加载权重"""

        checkpoint = checkpoint or self.checkpoint
        logger.info(f"开始根据 variable_mapping 从预训练的 tf checkpoint={checkpoint}加载权重:")
        mapping = mapping or self.variable_mapping()
        mapping = {self.prefixed(k): v for k, v in mapping.items()}
        # k: 自定义的层的名称 vs v： transformer 层的名称
        mapping = {k: v for k, v in mapping.items() if k in self.layers}
        weight_value_pairs = []
        for layer, variables in mapping.items():
            # 获取对应 layer
            layer = self.layers[layer]
            # 自定义模型的 weights
            weights, values = [], []
            # 获取 layer 中对应的 trainable_weights
            for w, v in zip(layer.trainable_weights, variables):  # 允许跳过不存在的权重
                try:
                    # 从 ckpt 中提取 v
                    value = self.load_variable(checkpoint, v)
                    values.append(value)
                    weights.append(w)
                    assert value.shape == tuple(w.shape.as_list())
                    logger.info(f"{v},shape={value.shape}")
                except Exception as e:
                    if self.ignore_invalid_weights:
                        print('%s, but ignored.' % e.message)
                    else:
                        raise e

            for i, (w, v) in enumerate(zip(weights, values)):
                if v is not None:
                    w_shape, v_shape = K.int_shape(w), v.shape
                    # if self.autoresize_weights and w_shape != v_shape:
                    #     v = orthogonally_resize(v, w_shape)
                    #     if isinstance(layer, MultiHeadAttention):
                    #         count = 2
                    #         if layer.use_bias:
                    #             count += 2
                    #         if layer.attention_scale and i < count:
                    #             scale = 1.0 * w_shape[-1] / v_shape[-1]
                    #             v = v * scale**0.25
                    #     if isinstance(layer, FeedForward):
                    #         count = 1
                    #         if layer.use_bias:
                    #             count += 1
                    #         if self.hidden_act in ['relu', 'leaky_relu']:
                    #             count -= 2
                    #         if i < count:
                    #             v *= np.sqrt(1.0 * w_shape[-1] / v_shape[-1])
                    #         else:
                    #             v *= np.sqrt(1.0 * v_shape[0] / w_shape[0])

                    weight_value_pairs.append((w, v))

        K.batch_set_value(weight_value_pairs)


class BERT(Transformer):
    def variable_mapping(self):
        """映射到官方BERT权重格式
        """
        mapping = {
            'Embedding-Token': ['bert/embeddings/word_embeddings'],
            'Embedding-Segment': ['bert/embeddings/token_type_embeddings'],
            'Embedding-Position': ['bert/embeddings/position_embeddings'],
            'Embedding-Norm': [
                'bert/embeddings/LayerNorm/beta',
                'bert/embeddings/LayerNorm/gamma',
            ],
            'Embedding-Mapping': [
                'bert/encoder/embedding_hidden_mapping_in/kernel',
                'bert/encoder/embedding_hidden_mapping_in/bias',
            ],
            'Pooler-Dense': [
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ],
            'NSP-Proba': [
                'cls/seq_relationship/output_weights',
                'cls/seq_relationship/output_bias',
            ],
            'MLM-Dense': [
                'cls/predictions/transform/dense/kernel',
                'cls/predictions/transform/dense/bias',
            ],
            'MLM-Norm': [
                'cls/predictions/transform/LayerNorm/beta',
                'cls/predictions/transform/LayerNorm/gamma',
            ],
            'MLM-Bias': ['cls/predictions/output_bias'],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'bert/encoder/layer_%d/' % i
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'attention/self/query/kernel',
                    prefix + 'attention/self/query/bias',
                    prefix + 'attention/self/key/kernel',
                    prefix + 'attention/self/key/bias',
                    prefix + 'attention/self/value/kernel',
                    prefix + 'attention/self/value/bias',
                    prefix + 'attention/output/dense/kernel',
                    prefix + 'attention/output/dense/bias',
                ],
                'Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'attention/output/LayerNorm/beta',
                    prefix + 'attention/output/LayerNorm/gamma',
                ],
                'Transformer-%d-FeedForward' % i: [
                    prefix + 'intermediate/dense/kernel',
                    prefix + 'intermediate/dense/bias',
                    prefix + 'output/dense/kernel',
                    prefix + 'output/dense/bias',
                ],
                'Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'output/LayerNorm/beta',
                    prefix + 'output/LayerNorm/gamma',
                ],
            })
        return mapping


@MODELS.register_module()
class HFAlbertClassifierModel(TFAlbertForSequenceClassification,BERT):
    """
    继承多个父类：
    从 TFAlbertForSequenceClassification 继承 模型等方法，
    从自己写的 BERT 继承 load_variables()等方法 ，希望加载 albert tf1.0 版本的参数到 tf2.0 版本的模型当中
    """
    def __init__(self, output_size, multilabel_classifier=False, max_seq_length=128,
                 pretrained_model_dir=None, model_config_filename='albert_config.json',
                 init_checkpoint=None, checkpoint_filename='albert_model.ckpt',
                 from_pt=False,torch_savedmodel_name=None,
                 input_keys= TRANSFORMERS_ALBERT_INPUT_KEYS, #, "position_ids"],
                 with_pool=False, dropout_rate=0,
                 freeze_pretrained_weights=False,freeze_embedding_weights=False,
                 *args, **kwargs):

        self.output_size = output_size
        self.multilabel_classifier = multilabel_classifier
        self.max_seq_length = max_seq_length
        # 模型配置文件
        self.model_config_filename = model_config_filename

        # 使用 hf transformers 创建模型，并加载中文的预训练参数
        self.pretrained_model_dir = pretrained_model_dir
        self.config_path = os.path.join(self.pretrained_model_dir, self.model_config_filename)

        self.init_checkpoint = init_checkpoint
        # checkpoint 名称
        self.checkpoint_filename = checkpoint_filename
        # 预训练的 checkpoint
        self.pretrained_model_checkpoint_filpath = os.path.join(self.pretrained_model_dir, self.checkpoint_filename)
        self.from_pt = from_pt
        self.torch_savedmodel_name = torch_savedmodel_name

        self.input_keys = input_keys

        # Initializing an ALBERT-base style configuration
        self.albert_model_config = AlbertConfig.from_json_file(json_file=self.config_path)
        self.with_pool = with_pool
        self.dropout_rate = dropout_rate
        self.freeze_pretrained_weights = freeze_pretrained_weights
        self.freeze_embedding_weights = freeze_embedding_weights

        # 在使用 AlbertConfig初始化 transformer 模型的时候，需要更新以下参数：
        # albert_model_config中没有 classifier_dropout_rate
        # self.albert_model_config.update({"classifier_dropout_rate":self.dropout_rate})
        self.albert_model_config.update({"num_labels": self.output_size})

        # 直接使用 HF 的 TFAlbertForSequenceClassification  from_pretrained 方法 从 config 初始化和 pretrained_model_dir 加载模型参数
        # self.model = TFAlbertForSequenceClassification.from_pretrained(self.pretrained_model_dir, from_pt=True,
        #                                                   config=self.albert_model_config, *args, **kwargs)

        #　先 TFAlbertForSequenceClassification.__init__ 初始化模型，后 加载参数的方式
        TFAlbertForSequenceClassification.__init__(self,config=self.albert_model_config)
        # BERT.__init__(self, checkpoint=checkpoint, *args, **kwargs)

        self.args =args
        self.kwargs = kwargs



    def build_model(self,if_initialize_parameters=True):
        """
        @time:  2021/7/15 16:49
        @author:wangfc
        @version:
        @description:

        @params:
        @return:
        """

        # Initializing a model from the ALBERT-base style configuration
        # self.model = TFAlbertModel(self.albert_model_config)
        # input_ids, attention_mask, token_type_ids
        tf_inputs = self._get_transformer_input_tensors(
                max_seq_length=self.max_seq_length,
                input_keys=self.input_keys)

        outputs = self(tf_inputs,training=False)

        if if_initialize_parameters:
            self._initialize_model_parameters(from_pt=self.from_pt,
                                              pretrained_model_dir=self.pretrained_model_dir,
                                              torch_savedmodel_name=self.torch_savedmodel_name,
                                              tf_model =self,
                                              tf_inputs=tf_inputs,
                                              init_checkpoint=self.init_checkpoint)

        # 是否进行 freeze_pretrained_weights
        if self.freeze_pretrained_weights and self.freeze_embedding_weights:
            self.albert.embeddings.trainable = False
            logger.info(f"冻结 albert.embeddings weights")
        elif self.freeze_pretrained_weights:
            self.albert.trainable =False
            logger.info(f"冻结 albert weights")

        self.summary(print_fn=logger.info)
        self._print_trainable_variables()

    def _initialize_model_parameters(self,from_pt=True,
                                     pretrained_model_dir=None,
                                     torch_savedmodel_name= None,
                                     tf_model=None,
                                     tf_inputs=None,
                                     init_checkpoint = None):
        if from_pt:
            # 使用 transformer 的 from_pretrained()的方法从 pytorch_model.bin 中加载参数
            from transformers.modeling_tf_pytorch_utils import load_pytorch_model_in_tf2_model, \
                load_pytorch_checkpoint_in_tf2_model
            torch_savedmodel_path = os.path.join(pretrained_model_dir, torch_savedmodel_name)

            # init_word_embeddings_weights = self.layers[0].trainable_variables[0]
            # tf.reduce_mean(init_word_embeddings_weights,axis=-1)

            # Load from a PyTorch checkpoint
            model = load_pytorch_checkpoint_in_tf2_model(tf_model=tf_model,
                                                         pytorch_checkpoint_path=torch_savedmodel_path,
                                                         tf_inputs=tf_inputs,
                                                         allow_missing_keys=True)

            # word_embeddings_weights = self.layers[0].trainable_variables[0]
            # tf.reduce_mean(word_embeddings_weights, axis=-1)

        else:
            # 使用 bert4keras的 load_weights_from_checkpoint() 方法从 tf checkpoint 加载预训练参数
            if is_checkpoint_exist(init_checkpoint=self.init_checkpoint):
                self._load_weights_from_checkpoint()
            else:
                logger.warning(f"不加载预训练参数")



    # def build(self):
    #     """
    #     @time:  2021/7/15 16:50
    #     @author:wangfc
    #     @version:
    #     @description: 覆盖 父类的 .build(input_shape ) 方法 ,因为我不大理解 这个逻辑
    #
    #     @params:
    #     @return:
    #     """
    #     if self.built:
    #         return None
    #     # 之前 输入的时候 有key = position_ids，但是 generator 是没有的
    #     input_ids, attention_mask, token_type_ids = self.get_transformer_input_tensors(
    #         max_seq_length=self.max_seq_length,
    #         input_keys=self.input_keys)
    #     self.call(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    #
    #     self.built = True

    def _get_transformer_input_tensors(self, max_seq_length, input_keys):
        """
        定义整个模型的输入
        dtype = tf.dtypes.int64: 开始设置为 int32的时候，模型训练都很差，现在好很多
        """
        input_tensors = []
        for key in input_keys:
            input = Input(shape=(max_seq_length,), name=key, dtype=tf.dtypes.int64)
            # shape = (max_seq_length,)
            input_tensors.append(input)
        return input_tensors

    # def call(self,inputs,training = False):
    #     return self.model(inputs)

    def _print_trainable_variables(self):
        for trainable_variable in self.trainable_variables:
            name = trainable_variable.name
            shape = trainable_variable.shape.as_list()
            logger.info(f"{name},shape={shape}")

    def _variable_mapping(self):
        """映射到官方ALBERT权重格式
        """
        prefix = 'bert/encoder/transformer/group_0/inner_group_0/'
        mapping = {
            # embeddings layer
            'albert/embeddings/word_embeddings': 'bert/embeddings/word_embeddings',
            'albert/embeddings/token_type_embeddings': 'bert/embeddings/token_type_embeddings',
            'albert/embeddings/position_embeddings': 'bert/embeddings/position_embeddings',

            # embedding layernorm layer
            'albert/embeddings/LayerNorm/gamma': 'bert/embeddings/LayerNorm/gamma',
            'albert/embeddings/LayerNorm/beta': 'bert/embeddings/LayerNorm/beta',

            # embedding mapping labyer
            'albert/encoder/embedding_hidden_mapping_in/kernel': 'bert/encoder/embedding_hidden_mapping_in/kernel',
            'albert/encoder/embedding_hidden_mapping_in/bias': 'bert/encoder/embedding_hidden_mapping_in/bias',

            # self-attention layer
            'albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/key/kernel': prefix + 'attention_1/self/key/kernel',
            'albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/key/bias': prefix + 'attention_1/self/key/bias',
            'albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/value/kernel': prefix + 'attention_1/self/value/kernel',
            'albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/value/bias': prefix + 'attention_1/self/value/bias',
            'albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/query/kernel': prefix + 'attention_1/self/query/kernel',
            'albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/query/bias': prefix + 'attention_1/self/query/bias',
            # self-attention dense layer
            'albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/dense/kernel': prefix + 'attention_1/output/dense/kernel',
            'albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/dense/bias': prefix + 'attention_1/output/dense/bias',
            # self-attention layernorm layer
            'albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/LayerNorm/gamma': prefix + 'LayerNorm/gamma',
            'albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/LayerNorm/beta': prefix + 'LayerNorm/beta',
            # feedforward layer
            'albert/encoder/albert_layer_groups_._0/albert_layers_._0/ffn/kernel': prefix + 'ffn_1/intermediate/dense/kernel',
            'albert/encoder/albert_layer_groups_._0/albert_layers_._0/ffn/bias': prefix + 'ffn_1/intermediate/dense/bias',
            'albert/encoder/albert_layer_groups_._0/albert_layers_._0/ffn_output/kernel': prefix + 'ffn_1/intermediate/output/dense/kernel',
            'albert/encoder/albert_layer_groups_._0/albert_layers_._0/ffn_output/bias': prefix + 'ffn_1/intermediate/output/dense/bias',
            # feedforward layernorm layer
            'albert/encoder/albert_layer_groups_._0/albert_layers_._0/full_layer_layer_norm/gamma': prefix + 'LayerNorm_1/gamma',
            'albert/encoder/albert_layer_groups_._0/albert_layers_._0/full_layer_layer_norm/beta': prefix + 'LayerNorm_1/beta',

            'albert/pooler/kernel': 'bert/pooler/dense/kernel',
            'albert/pooler/bias': 'bert/pooler/dense/bias'

            # 'NSP-Proba': [
            #     'cls/seq_relationship/output_weights',
            #     'cls/seq_relationship/output_bias',
            # ],
            # 'MLM-Dense': [
            #     'cls/predictions/transform/dense/kernel',
            #     'cls/predictions/transform/dense/bias',
            # ],
            # 'MLM-Norm': [
            #     'cls/predictions/transform/LayerNorm/beta',
            #     'cls/predictions/transform/LayerNorm/gamma',
            # ],
            # 'MLM-Bias': ['cls/predictions/output_bias'],

        }

        return mapping

    def _load_weights_from_checkpoint(self, checkpoint=None, mapping=None):
        """根据mapping从checkpoint加载权重"""
        checkpoint = checkpoint or self.init_checkpoint
        logger.info(f"开始根据 variable_mapping 从预训练的 tf checkpoint={checkpoint}加载权重:")

        logger.debug(f"提取原本albert tf1.x 预训练参数 from {checkpoint}")
        list_variables = tf.train.list_variables(checkpoint)
        for v,s in list_variables:
            logger.debug(f"{v}:{s}")

        logger.debug(f"显示 albert tf2.x 中参数")
        for w in self.trainable_variables:
            logger.debug(f"{w.name}:{w.shape.as_list()}")

        mapping = mapping or self._variable_mapping()
        weight_value_pairs = []

        for w in self.trainable_variables:
            try:
                # 根据 w 的 name 匹配 ckpt 中 名称
                name = w.name
                matched_layer_names = []
                for k, v in mapping.items():
                    if k in name:
                        matched_layer_names.append(v)

                assert matched_layer_names.__len__() == 1
                matched_name = matched_layer_names[0]
                value = self.load_variable(checkpoint, matched_name)
                # 确保初始化参数 和 提取的参数的 shape 和 dtype 相同
                assert value.shape == tuple(w.shape.as_list())
                assert w.dtype == value.dtype
                weight_value_pairs.append((w, value))
                logger.debug(f"load weight from {matched_name} to {name}")
                logger.info(f"load weight {name},shape={value.shape}")
            except Exception as e:
                logger.info(f"加载参数 {w.name}的时候出错：{e}")

        K.batch_set_value(weight_value_pairs)
        assert np.all(np.all(self.trainable_variables[index].numpy() == weight_value_pairs[index][1]) for index in
                      range(weight_value_pairs.__len__()))
        logger.info(f"成功完成 load_weights_from_checkpoint 共有weight_value_pairs={weight_value_pairs.__len__()} ")


if __name__ == '__main__':
    from datasets import load_dataset
    # from transformers import load_data
    albert_model = HFAlbertClassifierModel(pretrained_model_dir=PRETRAINED_MODEL_DIR)
    albert_model.build_model()
    albert_model._load_weights_from_checkpoint()

    # # Accessing the model configuration
    # configuration = model.config
