#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/30 10:55 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/30 10:55   wangfc      1.0         None
"""
import os
from typing import Dict, Text
from utils.io import read_from_yaml
from utils.tensorflow.environment import get_gpu_memory_config



def update_environment_paras(task_configs:Dict[Text,Dict[Text,Text]]):
    environment_paras = task_configs.get('environment_paras')
    gpu_memory_config = get_gpu_memory_config(gpu_no=environment_paras.get('gpu_no'),
                                              gpu_memory_fraction=environment_paras.get('gpu_memory_fraction'))
    environment_paras.update({"gpu_memory_config":gpu_memory_config})
    return task_configs

def update_task_paras(task_configs:Dict[Text,Dict[Text,Text]]):
    task_paras = task_configs.get('task_paras')
    DATA_DIR = os.path.join(task_paras['corpus'], task_paras['dataset_name'], task_paras['data_subdir'])
    STOPWORDS_PATH = os.path.join(task_paras['corpus'], 'stop_words_zh.txt')
    task_paras.update({"data_dir":DATA_DIR,"stopwords_path":STOPWORDS_PATH })
    return task_configs

def update_model_paras(task_configs:Dict[Text,Dict[Text,Text]]):
    model_paras = task_configs.get('model_paras')
    MODEL_NAME = model_paras["model_name"]
    PRETRAINED_MODEL_DIR = model_paras["pretrained_model_dir"]
    BERT_BASE_MODEL = model_paras["bert_base_model"]
    VOCAB_FILENAME = model_paras["vocab_filename"]
    # bert_config.json是BERT的配置(超参数)，比如网络的层数
    BERT_CONFIG_FILE = os.path.join(PRETRAINED_MODEL_DIR, BERT_BASE_MODEL + '_config.json')
    VOCAB_FILE = os.path.join(PRETRAINED_MODEL_DIR, VOCAB_FILENAME)
    # bert_model.ckpt*，这是预训练好的模型的checkpoint，我们的Fine-Tuning模型的初始值就是来自于这些文件，然后根据不同的任务进行Fine-Tuning。
    INIT_CHECKPOINT = os.path.join(PRETRAINED_MODEL_DIR, BERT_BASE_MODEL+ '_model.ckpt')
    model_paras.update({"bert_config_file":BERT_CONFIG_FILE,'vocab_file':VOCAB_FILE,'init_checkpoint':INIT_CHECKPOINT})

    MAX_SEQ_LENGTH = model_paras['max_seq_length']

    train_paras = task_configs.get('train_paras')
    OPTIMIZER_NAME = train_paras['optimizer_name']
    EPOCH = train_paras['epoch']
    LEARNING_RATE = train_paras['learning_rate']

    OUTPUT_MODEL_SUBDIR = f"{MODEL_NAME}_{OPTIMIZER_NAME}_epoch{EPOCH}_lr{LEARNING_RATE}_{MAX_SEQ_LENGTH}"
    train_paras.update({"output_model_subdir":OUTPUT_MODEL_SUBDIR})
    return task_configs

def update_task_configs(task_configs:Dict[Text,Dict[Text,Text]]):
    task_configs = update_environment_paras(task_configs)
    task_configs = update_task_paras(task_configs)
    task_configs = update_model_paras(task_configs)
    all_paras = {}
    for k, paras_dict in task_configs.items():
        all_paras.update(paras_dict)
    task_configs.update({'all_paras':all_paras})
    return task_configs





# EPOCH = config_parser.getint('train', 'epoch')
# TRAIN_BATCH_SIZE= config_parser.getint('train', 'train_batch_size')
# MODEL_NAME = config_parser.get('train', 'model_name')
# WITH_POOL = config_parser.getboolean('train','with_pool')
# DROPOUT_RATE = config_parser.getfloat('train','dropout_rate')
# BERT_BASE_MODEL = config_parser.get('train', 'bert_base_model')
# MULTILABEL_CLASSIFIER = config_parser.getboolean('train', 'multilabel_classifier')
# MAX_SEQ_LENGTH= config_parser.getint('train', 'max_seq_length')
# OPTIMIZER_NAME = config_parser.get('train','optimizer_name')
# LEARNING_RATE = config_parser.getfloat('train', 'learning_rate')
# WEIGHT_DECAY_RATE = config_parser.getfloat('train','weight_decay_rate')
# EARLY_STOPPING_STEPS= config_parser.getint('train', 'early_stopping_steps')
# SAVE_CHECKPOINTS_STEPS= config_parser.getint('train', 'save_checkpoints_steps')
# NUM_WARMUP_EPOCH= config_parser.getint('train', 'num_warmup_epoch')
# EXPORTS_TO_KEEP = int(config_parser.get('train', 'exports_to_keep'))
# crf layer parameters
# IF_USE_CRF = config_parser.getboolean('train', 'if_use_crf')
# CRF_ONLY=config_parser.getboolean('train', 'crf_only')
# DROPOUT_RATE=config_parser.getfloat('train', 'dropout_rate')
# CELL=config_parser.get('train', 'cell')
# LSTM_SIZE=config_parser.getint('train', 'lstm_size')
# NUM_LAYERS=config_parser.getint('train', 'num_layers')
# PRETRAINED_MODEL_DIR = config_parser.get('train', 'PRETRAINED_MODEL_DIR')


if __name__ == '__main__':
    config_file_name = 'classifier_pipeline_config.yml'
    config_path = os.path.join(os.path.dirname(__file__), config_file_name)
    task_configs = read_from_yaml(config_path)
    TASK_CONFIGS = update_task_configs(task_configs)
    TASK_CONFIGS = TASK_CONFIGS

    # 模型配置参数文件
    albert_config_filename = 'albert_config.yml'
    model_config_path = os.path.join(os.path.dirname(__file__), albert_config_filename)
    model_config = read_from_yaml(model_config_path)['model_paras']


    # init_checkpoint = os.path.join(BERT_BASE_DIR,'model.ckpt-best')
    TRAIN_EPOCHS = model_config['train_epochs']
    TRAIN_BATCH_SIZE = model_config['train_batch_size']
    BATCH_STRATEGY = model_config['batch_strategy']

    MODEL_NAME = model_config['model_name']
    OPTIMIZER_NAME = model_config['optimizer_name']
    LEARNING_RATE = model_config['learning_rate']
    WARMUP_EPOCH = model_config.get('warmup_epoch')
    MAX_SEQ_LENGTH = model_config.get('max_seq_length')
    ADAM_EPSILON = model_config.get('adam_epsilon')
    PRETRAINED_MODEL_DIR = model_config['pretrained_model_dir']
    MODEL_CONFIG_FILENAME = model_config['model_config_filename']
    TRANSFORMER_BASE_MODEL = model_config['transformer_base_model']
    VOCAB_FILENAME = model_config['vocab_filename']

    FREEZE_PRETRAINED_WEIGHTS = model_config['freeze_pretrained_weights']
    FREEZE_EMBEDDING_WEIGHTS = model_config['freeze_embedding_weights']
    # bert_config.json是BERT的配置(超参数)，比如网络的层数
    MODEL_CONFIG_FILE = os.path.join(PRETRAINED_MODEL_DIR, MODEL_CONFIG_FILENAME)
    # # vocab.txt是模型的词典
    # VOCAB_FILENAME = 'vocab.txt'
    # MODEL_CONFIG_FILENAME= 'albert_config.json'
    VOCAB_FILE = os.path.join(PRETRAINED_MODEL_DIR, VOCAB_FILENAME)
    # bert_model.ckpt*，这是预训练好的模型的checkpoint，我们的Fine-Tuning模型的初始值就是来自于这些文件，然后根据不同的任务进行Fine-Tuning。
    INIT_CHECKPOINT = os.path.join(PRETRAINED_MODEL_DIR, TRANSFORMER_BASE_MODEL + '_model.ckpt')

    OUTPUT_MODEL_SUBDIR = f"{MODEL_NAME}_{OPTIMIZER_NAME}_epoch{TRAIN_EPOCHS}_batch_strategy{BATCH_STRATEGY}_freeze{FREEZE_PRETRAINED_WEIGHTS}_embedding_weights{FREEZE_EMBEDDING_WEIGHTS}_warmup_epoch{WARMUP_EPOCH}_lr{LEARNING_RATE}_adam_epsilon{ADAM_EPSILON}_{MAX_SEQ_LENGTH}"

    model_config.update({"model_config_file": MODEL_CONFIG_FILE,
                         "vocab_file": VOCAB_FILE,
                         "init_checkpoint": INIT_CHECKPOINT,
                         })



