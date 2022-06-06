#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/14 16:03 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/14 16:03   wangfc      1.0         None
"""
import os
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    # 测试 tfv1.13  加载 tf2.3 pb 模型： NO
    from utils.tensorflow.environment import setup_tf_environment
    setup_tf_environment(gpu_memory_config=None)
    savedmodel_path ="output/juyuan_ner_classifier_output_20211228/models/HFAlbertClassifierModel_RANDOM_BATCH_STRATEGY_adam_1e-05/best_export/savedmodel_9/"
    # serving_model = load_serving_model_from_savedmodel(savedmodel_path=savedmodel_path)
    os.path.exists(savedmodel_path)
    serving_model = tf.contrib.predictor.from_saved_model(savedmodel_path)
    max_seq_lengh= 128
    input_keys = ["input_ids",'token_type_ids',"attention_mask"]
    model_inputs_dict = {key: np.ones(shape=(1,max_seq_lengh),dtype=np.int) for key in input_keys}

    output = serving_model(model_inputs_dict)


