#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/5/28 9:27 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/28 9:27   wangfc      1.0         None
"""
import collections
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Text, Union,Optional

import six
import tensorflow as tf
from tensorflow.keras.models import Model, load_model


import logging



logger = logging.getLogger(__name__)

TF_CKPT_NAME = "tf_model_epoch"  # 'tf_ckpt'
TF_SAVEDMODEL_NAME = 'tf_savedmodel'


def save_model_to_savedmodel(model: Model, filepath: Text, version: int = None,
                             save_format="tf", include_optimizer=False) -> Text:
    """
    @time:  2021/6/22 11:22
    @author:wangfc
    @version:
    @description:
    https://www.tensorflow.org/guide/keras/save_and_serialize#%E4%BF%9D%E5%AD%98%E6%9E%B6%E6%9E%84

    一个常见的 Keras 模型通常由以下几个部分组成：
        模型的结构或配置：它指定了模型中包含的层以及层与层之间的连接方式。
        一系列权重矩阵的值：用于记录模型的状态。
        优化器：用于优化损失函数，可以在模型编译 (compile) 时指定。
        一系列损失以及指标：用于调控训练过程，既包括在编译时指定的损失和指标，也包括通过 add_loss() 以及 add_metric() 方法添加的损失和指标。

    使用 Keras API 可以将上述模型组成部分全部保存到磁盘，或者保存其中的部分。 Keras API 提供了以下三种选择：
        保存模型的全部内容：通常以 TensorFlow SavedModel 格式或者 Keras H5 格式进行保存，这也是最为常用的模型保存方式。
        仅保存模型的结构：通常以 json 文件的形式进行保存。
        仅保存模型的权重矩阵的值：通常以 numpy 数组的形式进行保存，一般会在模型再训练或迁移学习时使用这种保存方式。


    tf2 保存 模型的全部信息 有四种方式:
    1）saved_model下的save API；tf.saved_model.save
    2）callbacks:tf.keras.callbacks.ModelCheckpoint ，要注意save_weights_only 参数要设为 False；
    3) keras API； model.save 或者  tf.keras.models.save_model()

    .
    ├── assets
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index

    其中 assets 是一个可选的目录，用于存放模型运行所需的辅助文件，比如字典文件等。
    variables 目录下存放的是模型权重的检查点文件，模型的所有权重信息均保存在该目录下。
    saved_model.pb 文件中包含了模型的结构以及训练的配置信息如优化器，损失以及指标等信息。



    只保存权重weights：
    1） model.save_weights API；
    2） callbacks： ModelCheckpoint，但是参数save_weights_only 要设置成True

    仅保存模型结构
    有时我们可能只对模型的结构感兴趣，而不想保存模型的权重值以及优化器的状态等信息。在这种情况下，我们可以借助于模型的配置方法 (config) 来对模型的结构进行保存和重建。

    Sequential/Functional
    对于 Sequential 模型和 Functional API 模型，因为它们大多是由预定义的层 (layers) 组成的，
    所以它们的配置信息都是以结构化的形式存在的，可以很方便地执行模型结构的保存操作。
    我们可以使用模型的 get_config() 方法来获取模型的配置，然后通过 from_config(config) 方法来重建模型。

    @params:
    @return:
    """
    if version:
        filepath = os.path.join(filepath, str(version))

    if os.path.isdir(filepath):
        logger.info('Already saved a model, cleaning up')
        shutil.rmtree(filepath)

    model.save(filepath, save_format=save_format, include_optimizer=include_optimizer)
    logger.info('保存模型为 saved_model 格式到 {}'.format(filepath))
    return filepath


def export_savedmodel(model, version, path='prod_models'):
    """
    tf1 的保存模型方式，
    在 tf2 中可能会报错： RuntimeError: build_tensor_info is not supported in Eager mode.
    可以试验以下方法：
    if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

    saved_model_cli show --dir SavedModel路径 --all 得到类似以下的结果

    MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:
    signature_def['serving_default']:
      The given SavedModel SignatureDef contains the following input(s):
        inputs['input_ids'] tensor_info:
            dtype: DT_INT32
            shape: (-1, 128)
            name: input_ids:0
        inputs['input_mask'] tensor_info:
            dtype: DT_INT32
            shape: (-1, 128)
            name: input_mask:0
        inputs['label_ids'] tensor_info:
            dtype: DT_INT32
            shape: (-1)
            name: label_ids:0
        inputs['segment_ids'] tensor_info:
            dtype: DT_INT32
            shape: (-1, 128)
            name: segment_ids:0
      The given SavedModel SignatureDef contains the following output(s):
        outputs['probabilities'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, 7)
            name: loss/pred_prob:0
      Method name is: tensorflow/serving/predict



    MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

    signature_def['__saved_model_init_op']:
      The given SavedModel SignatureDef contains the following input(s):
      The given SavedModel SignatureDef contains the following output(s):
        outputs['__saved_model_init_op'] tensor_info:
            dtype: DT_INVALID
            shape: unknown_rank
            name: NoOp
      Method name is:

    signature_def['serving_default']:
      The given SavedModel SignatureDef contains the following input(s):
        inputs['input_ids'] tensor_info:
            dtype: DT_INT32
            shape: (-1, 64)
            name: serving_default_input_ids:0
        inputs['token_type_ids'] tensor_info:
            dtype: DT_INT32
            shape: (-1, 64)
            name: serving_default_token_type_ids:0
      The given SavedModel SignatureDef contains the following output(s):
        outputs['classifier_dense'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, 2)
            name: StatefulPartitionedCall:0
      Method name is: tensorflow/serving/predict

    Defined Functions:
      Function Name: '__call__'
        Option #1
          Callable with:
            Argument #1
              DType: list
              Value: [TensorSpec(shape=(None, 64), dtype=tf.int32, name='inputs/0'), TensorSpec(shape=(None, 64), dtype=tf.int32, name='inputs/1')]
            Argument #2
              DType: bool
              Value: False
            Argument #3
              DType: NoneType
              Value: None
        Option #2
          Callable with:
            Argument #1
              DType: list
              Value: [TensorSpec(shape=(None, 64), dtype=tf.int32, name='input_ids'), TensorSpec(shape=(None, 64), dtype=tf.int32, name='token_type_ids')]
            Argument #2
              DType: bool
              Value: False
            Argument #3
              DType: NoneType
              Value: None
        Option #3
          Callable with:
            Argument #1
              DType: list
              Value: [TensorSpec(shape=(None, 64), dtype=tf.int32, name='input_ids'), TensorSpec(shape=(None, 64), dtype=tf.int32, name='token_type_ids')]
            Argument #2
              DType: bool
              Value: True
            Argument #3
              DType: NoneType
              Value: None
        Option #4
          Callable with:
            Argument #1
              DType: list
              Value: [TensorSpec(shape=(None, 64), dtype=tf.int32, name='inputs/0'), TensorSpec(shape=(None, 64), dtype=tf.int32, name='inputs/1')]
            Argument #2
              DType: bool
              Value: True
            Argument #3
              DType: NoneType
              Value: None

      Function Name: '_default_save_signature'
        Option #1
          Callable with:
            Argument #1
              DType: list
              Value: [TensorSpec(shape=(None, 64), dtype=tf.int32, name='input_ids'), TensorSpec(shape=(None, 64), dtype=tf.int32, name='token_type_ids')]

      Function Name: 'call_and_return_all_conditional_losses'
        Option #1
          Callable with:
            Argument #1
              DType: list
              Value: [TensorSpec(shape=(None, 64), dtype=tf.int32, name='input_ids'), TensorSpec(shape=(None, 64), dtype=tf.int32, name='token_type_ids')]
            Argument #2
              DType: bool
              Value: False
            Argument #3
              DType: NoneType
              Value: None
        Option #2
          Callable with:
            Argument #1
              DType: list
              Value: [TensorSpec(shape=(None, 64), dtype=tf.int32, name='input_ids'), TensorSpec(shape=(None, 64), dtype=tf.int32, name='token_type_ids')]
            Argument #2
              DType: bool
              Value: True
            Argument #3
              DType: NoneType
              Value: None
        Option #3
          Callable with:
            Argument #1
              DType: list
              Value: [TensorSpec(shape=(None, 64), dtype=tf.int32, name='inputs/0'), TensorSpec(shape=(None, 64), dtype=tf.int32, name='inputs/1')]
            Argument #2
              DType: bool
              Value: True
            Argument #3
              DType: NoneType
              Value: None
        Option #4
          Callable with:
            Argument #1
              DType: list
              Value: [TensorSpec(shape=(None, 64), dtype=tf.int32, name='inputs/0'), TensorSpec(shape=(None, 64), dtype=tf.int32, name='inputs/1')]
            Argument #2
              DType: bool
              Value: False
            Argument #3
              DType: NoneType
              Value: None








    """

    tf.keras.backend.set_learning_phase(1)
    if not os.path.exists(path):
        os.mkdir(path)
    export_path = os.path.join(
        tf.compat.as_bytes(path),
        tf.compat.as_bytes(version))

    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)

    model_input = tf.compat.v1.saved_model.utils.build_tensor_info(model.input)
    model_output = tf.compat.v1.saved_model.utils.build_tensor_info(model.output)

    prediction_signature = (
        tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
            inputs={'inputs': model_input},
            outputs={'output': model_output},
            method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME))

    with tf.keras.backend.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={'predict': prediction_signature})

        builder.save()





def get_tf_model_dir_to_epoch_ls(output_model_dir,
                                 from_ckpt=False, from_savedmodel=True,
                                 tf_ckpt_name=TF_CKPT_NAME,
                                 tf_savedmodel_name=TF_SAVEDMODEL_NAME
                                 ) -> List[Tuple[Text, int]]:
    if from_ckpt:
        tf_model_name_pattern = r"(%s)_(\d{1,5})" % tf_ckpt_name

    elif from_savedmodel:
        tf_model_name_pattern = r"(%s)_(\d{1,5})" % tf_savedmodel_name

    tf_model_dir_to_epoch_ls = []
    if not os.path.exists(output_model_dir):
        return tf_model_dir_to_epoch_ls
    filenames = os.listdir(output_model_dir)

    tf_model_dir_to_epoch_dict = {}
    for filename in filenames:
        matched = re.match(pattern=tf_model_name_pattern, string=filename)
        if matched:
            model_name = matched.groups()[0]
            epoch = int(matched.groups()[1])
            tf_model_name = f"{model_name}_{epoch}"
            tf_model_dir = os.path.join(output_model_dir, tf_model_name)
            tf_model_dir_to_epoch_dict.update({tf_model_dir: epoch})

    if tf_model_dir_to_epoch_dict:
        tf_model_dir_to_epoch_ls = sorted(tf_model_dir_to_epoch_dict.items(), key=lambda x: x[1], reverse=False)
    return tf_model_dir_to_epoch_ls


def get_latest_ckpt_dir_to_epoch(output_model_dir)-> [Optional[Text],int]:
    tf_model_dir_to_epoch = get_tf_model_dir_to_epoch_ls(output_model_dir,from_ckpt=True)
    if tf_model_dir_to_epoch:
        latest_tf_model_dir_to_epoch = tf_model_dir_to_epoch[-1]
        return latest_tf_model_dir_to_epoch
    #-1 表示没有可以提取 ckpt, init_epoch = restored_epoch+1
    return None,-1




def get_assignment_map_from_checkpoint(tvars, init_checkpoint, num_of_group=0):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    logger.info("使用 keras transformers API 定义模型的训练参数：")
    for name, var in name_to_variable.items():
        print(f"{name},{var.shape}")

    init_vars = tf.train.list_variables(init_checkpoint)
    init_vars_name = [name for (name, _) in init_vars]
    logger.info(f"init_checkpoint 中的训练参数：\n")
    for init_var in init_vars:
        print(f"{init_var[0]} , {init_var[1]}")

    if num_of_group > 0:
        assignment_map = []
        for gid in range(num_of_group):
            assignment_map.append(collections.OrderedDict())
    else:
        assignment_map = collections.OrderedDict()

    for name in name_to_variable:
        if name in init_vars_name:
            tvar_name = name
        elif mapping_transformers_variable_to_tf_variable(name, init_vars_name):
            # 建立 模型参数的 初始化参数的对应关系
            tvar_name = mapping_transformers_variable_to_tf_variable(name, init_vars_name)

        # elif (re.sub(r"/group_\d+/", "/group_0/", six.ensure_str(name)) in init_vars_name and num_of_group > 1):
        #     tvar_name = re.sub(r"/group_\d+/", "/group_0/", six.ensure_str(name))
        # elif (re.sub(r"/ffn_\d+/", "/ffn_1/", six.ensure_str(name)) in init_vars_name and num_of_group > 1):
        #     tvar_name = re.sub(r"/ffn_\d+/", "/ffn_1/", six.ensure_str(name))
        # elif (re.sub(r"/attention_\d+/", "/attention_1/", six.ensure_str(name)) in init_vars_name and num_of_group > 1):
        #     tvar_name = re.sub(r"/attention_\d+/", "/attention_1/",
        #                        six.ensure_str(name))
        else:
            tf.logging.info("name %s does not get matched", name)
            continue

        tf.logging.info("name %s match to %s", name, tvar_name)

        # if num_of_group > 0:
        #     group_matched = False
        #     for gid in range(1, num_of_group):
        #         if (("/group_" + str(gid) + "/" in name) or
        #                 ("/ffn_" + str(gid) + "/" in name) or
        #                 ("/attention_" + str(gid) + "/" in name)):
        #             group_matched = True
        #             tf.logging.info("%s belongs to %dth", name, gid)
        #             assignment_map[gid][tvar_name] = name
        #     if not group_matched:
        #         assignment_map[0][tvar_name] = name
        # else:
        #     assignment_map[tvar_name] = name

        assignment_map[tvar_name] = name

        initialized_variable_names[name] = 1
        initialized_variable_names[six.ensure_str(name) + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def mapping_transformers_variable_to_tf_variable(name, init_vars_name):
    """
    @time:  2021/6/15 11:53
    @author:wangfc
    @version:
    @description:

    @params: name: 模型的变量名称
             init_vars_name： 初始化模型的名称
    @return: 与模型名称对应的初始化名称

    使用 keras transformers API 定义模型的训练参数：
    albert/embeddings/word_embeddings/weight,(21128, 128)
    albert/embeddings/token_type_embeddings/embeddings,(2, 128)
    albert/embeddings/position_embeddings/embeddings,(512, 128)
    albert/embeddings/LayerNorm/gamma,(128,)
    albert/embeddings/LayerNorm/beta,(128,)
    albert/encoder/embedding_hidden_mapping_in/kernel,(128, 312)
    albert/encoder/embedding_hidden_mapping_in/bias,(312,)
    albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/query/kernel,(312, 312)
    albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/query/bias,(312,)
    albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/key/kernel,(312, 312)
    albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/key/bias,(312,)
    albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/value/kernel,(312, 312)
    albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/value/bias,(312,)
    albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/dense/kernel,(312, 312)
    albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/dense/bias,(312,)
    albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/LayerNorm/gamma,(312,)
    albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/LayerNorm/beta,(312,)
    albert/encoder/albert_layer_groups_._0/albert_layers_._0/ffn/kernel,(312, 1248)
    albert/encoder/albert_layer_groups_._0/albert_layers_._0/ffn/bias,(1248,)
    albert/encoder/albert_layer_groups_._0/albert_layers_._0/ffn_output/kernel,(1248, 312)
    albert/encoder/albert_layer_groups_._0/albert_layers_._0/ffn_output/bias,(312,)
    albert/encoder/albert_layer_groups_._0/albert_layers_._0/full_layer_layer_norm/gamma,(312,)
    albert/en/albert_layer_groups_._0/albert_layers_._0/full_layer_layer_norm/beta,(312,)
    albert/pooler/kernel,(312, 312)
    albert/pooler/bias,(312,)

    classifier_dense/kernel,(312, 250)
    classifier_dense/bias,(250,)


    brightmart albert_tiny_google  init_checkpoint 中的训练参数：
    bert/embeddings/LayerNorm/beta , [128]
    bert/embeddings/LayerNorm/beta/adam_m , [128]
    bert/embeddings/LayerNorm/beta/adam_v , [128]
    bert/embeddings/LayerNorm/gamma , [128]
    bert/embeddings/LayerNorm/gamma/adam_m , [128]
    bert/embeddings/LayerNorm/gamma/adam_v , [128]
    bert/embeddings/position_embeddings , [512, 128]
    bert/embeddings/position_embeddings/adam_m , [512, 128]
    bert/embeddings/position_embeddings/adam_v , [512, 128]
    bert/embeddings/token_type_embeddings , [2, 128]
    bert/embeddings/token_type_embeddings/adam_m , [2, 128]
    bert/embeddings/token_type_embeddings/adam_v , [2, 128]
    bert/embeddings/word_embeddings , [21128, 128]
    bert/embeddings/word_embeddings/adam_m , [21128, 128]
    bert/embeddings/word_embeddings/adam_v , [21128, 128]
    bert/encoder/embedding_hidden_mapping_in/bias , [312]
    bert/encoder/embedding_hidden_mapping_in/bias/adam_m , [312]
    bert/encoder/embedding_hidden_mapping_in/bias/adam_v , [312]
    bert/encoder/embedding_hidden_mapping_in/kernel , [128, 312]
    bert/encoder/embedding_hidden_mapping_in/kernel/adam_m , [128, 312]
    bert/encoder/embedding_hidden_mapping_in/kernel/adam_v , [128, 312]

    bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta , [312]
    bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta/adam_m , [312]
    bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta/adam_v , [312]
    bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma , [312]
    bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma/adam_m , [312]
    bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma/adam_v , [312]

    bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta , [312]
    bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta/adam_m , [312]
    bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta/adam_v , [312]
    bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma , [312]
    bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma/adam_m , [312]
    bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma/adam_v , [312]

    bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias , [312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias/adam_m , [312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias/adam_v , [312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel , [312, 312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/adam_m , [312, 312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/adam_v , [312, 312]

    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias , [312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias/adam_m , [312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias/adam_v , [312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel , [312, 312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/adam_m , [312, 312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/adam_v , [312, 312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias , [312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias/adam_m , [312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias/adam_v , [312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel , [312, 312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/adam_m , [312, 312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/adam_v , [312, 312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias , [312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias/adam_m , [312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias/adam_v , [312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel , [312, 312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/adam_m , [312, 312]
    bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/adam_v , [312, 312]

    bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias , [1248]
    bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias/adam_m , [1248]
    bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias/adam_v , [1248]
    bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel , [312, 1248]
    bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/adam_m , [312, 1248]
    bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/adam_v , [312, 1248]
    bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias , [312]
    bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias/adam_m , [312]
    bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias/adam_v , [312]
    bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel , [1248, 312]
    bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/adam_m , [1248, 312]
    bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/adam_v , [1248, 312]

    bert/pooler/dense/bias , [312]
    bert/pooler/dense/bias/adam_m , [312]
    bert/pooler/dense/bias/adam_v , [312]
    bert/pooler/dense/kernel , [312, 312]
    bert/pooler/dense/kernel/adam_m , [312, 312]
    bert/pooler/dense/kernel/adam_v , [312, 312]
    cls/predictions/output_bias , [21128]
    cls/predictions/output_bias/adam_m , [21128]
    cls/predictions/output_bias/adam_v , [21128]
    cls/predictions/transform/LayerNorm/beta , [128]
    cls/predictions/transform/LayerNorm/beta/adam_m , [128]
    cls/predictions/transform/LayerNorm/beta/adam_v , [128]
    cls/predictions/transform/LayerNorm/gamma , [128]
    cls/predictions/transform/LayerNorm/gamma/adam_m , [128]
    cls/predictions/transform/LayerNorm/gamma/adam_v , [128]
    cls/predictions/transform/dense/bias , [128]
    cls/predictions/transform/dense/bias/adam_m , [128]
    cls/predictions/transform/dense/bias/adam_v , [128]
    cls/predictions/transform/dense/kernel , [312, 128]
    cls/predictions/transform/dense/kernel/adam_m , [312, 128]
    cls/predictions/transform/dense/kernel/adam_v , [312, 128]
    cls/seq_relationship/output_bias , [2]
    cls/seq_relationship/output_bias/adam_m , [2]
    cls/seq_relationship/output_bias/adam_v , [2]
    cls/seq_relationship/output_weights , [2, 312]
    cls/seq_relationship/output_weights/adam_m , [2, 312]
    cls/seq_relationship/output_weights/adam_v , [2, 312]
    global_step , []



    brightmart  albert_tiny  ckpt 训练参数：
    bert/embeddings/LayerNorm/beta , [312]
    bert/embeddings/LayerNorm/gamma , [312]
    bert/embeddings/position_embeddings , [512, 312]
    bert/embeddings/token_type_embeddings , [2, 312]
    bert/embeddings/word_embeddings , [21128, 128]
    bert/embeddings/word_embeddings_2 , [128, 312]
    bert/encoder/layer_shared/attention/output/LayerNorm/beta , [312]
    bert/encoder/layer_shared/attention/output/LayerNorm/gamma , [312]
    bert/encoder/layer_shared/attention/output/dense/bias , [312]
    bert/encoder/layer_shared/attention/output/dense/kernel , [312, 312]
    bert/encoder/layer_shared/attention/self/key/bias , [312]
    bert/encoder/layer_shared/attention/self/key/kernel , [312, 312]
    bert/encoder/layer_shared/attention/self/query/bias , [312]
    bert/encoder/layer_shared/attention/self/query/kernel , [312, 312]
    bert/encoder/layer_shared/attention/self/value/bias , [312]
    bert/encoder/layer_shared/attention/self/value/kernel , [312, 312]
    bert/encoder/layer_shared/intermediate/dense/bias , [1248]
    bert/encoder/layer_shared/intermediate/dense/kernel , [312, 1248]
    bert/encoder/layer_shared/output/LayerNorm/beta , [312]
    bert/encoder/layer_shared/output/LayerNorm/gamma , [312]
    bert/encoder/layer_shared/output/dense/bias , [312]
    bert/encoder/layer_shared/output/dense/kernel , [1248, 312]
    bert/pooler/dense/bias , [312]
    bert/pooler/dense/kernel , [312, 312]
    cls/predictions/output_bias , [21128]
    cls/predictions/transform/LayerNorm/beta , [312]
    cls/predictions/transform/LayerNorm/gamma , [312]
    cls/predictions/transform/dense/bias , [312]
    cls/predictions/transform/dense/kernel , [312, 312]
    cls/seq_relationship/output_bias , [2]
    cls/seq_relationship/output_weights , [2, 312]

    """
    name = 'tf_albert_model/albert/embeddings/word_embeddings/weight'
    shape = (21128, 128)
    # tf_albert_model/albert 对应 bert
    bert_matched = re.match(pattern=r'tf_albert_model/al(bert.*)', string=name)
    # if bert_matched is not None:
    #   bert_matched_str = bert_matched.group(1)
    #   if bert_matched_str in init_vars_name:


def load_tf_model(file_path: Union[Text, Path], from_savedmodel=True, from_ckpt=False, model=None, ) -> Model:
    if from_ckpt:
        model = load_tf_model_from_ckpt(model=model, ckpt_path=file_path)
        return model
    elif from_savedmodel:
        serving_model = load_serving_model_from_savedmodel(savedmodel_path=file_path)
        return serving_model


def load_tf_model_from_ckpt(model: Model, ckpt_path: Union[Text, Path]) -> Model:
    model.load_weights(filepath=ckpt_path)
    logger.info(f"加载ckpt参数到模型中 from {ckpt_path}")
    return model


def load_serving_model_from_savedmodel(savedmodel_path: Union[Text, Path],
                                       serving_signature_def_key=tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
    """
    loaded = tf.saved_model.load(mobilenet_save_path)
    loaded.signatures : Imported signatures always return dictionaries.
    infer = loaded.signatures["serving_default"]
    infer.structured_outputs: {'predictions': TensorSpec(shape=(None, 1000), dtype=tf.float32, name='predictions')}

    """
    savedmodel_loaded = load_savedmodel(filepath=savedmodel_path)
    serving_model = get_serving_model_from_savedmodel(savedmodel_loaded=savedmodel_loaded,
                                                      serving_signature_def_key=serving_signature_def_key)
    return serving_model


def load_savedmodel(filepath: Union[Text, Path]):
    """
    A SavedModel is a directory containing serialized signatures and the state needed to run them, including variable values and vocabularies.

    - assets         : contains files used by the TensorFlow graph, for example text files used to initialize vocabulary tables.
    - saved_model.pb : stores the actual TensorFlow program, or model, and a set of named signatures,
                       each identifying a function that accepts tensor inputs and produces tensor outputs.
    - variables      : contains a standard training checkpoint

    """

    savedmodel_loaded = tf.keras.models.load_model(filepath)
    return savedmodel_loaded


def get_serving_model_from_savedmodel(savedmodel_loaded,
                                      serving_signature_def_key=None):
    serving_model = savedmodel_loaded.signatures[serving_signature_def_key]
    return serving_model
