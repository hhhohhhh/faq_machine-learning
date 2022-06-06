#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/4/15 15:18 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/4/15 15:18   wangfc      1.0         None

"""
import os
from configparser import ConfigParser
from pathlib import Path

from utils.io import read_from_yaml

# 模型配置参数文件
yml_config_filename = "polyencoder_config.yml"
model_config_path = os.path.join(os.path.dirname(__file__), yml_config_filename)
model_config = read_from_yaml(model_config_path)['model_paras']


CONFIG_FILENAME = 'parlai_configs.cfg'
config_file_path = os.path.join(os.path.dirname(__file__), CONFIG_FILENAME)
config_parser = ConfigParser()
config_parser.read(config_file_path,encoding='utf-8')

# 自定义的参数
TASK = config_parser.get('main', 'task')
PARLAI_MODE = config_parser.get('main','parlai_mode')
MODE = config_parser.get('main', 'mode')
ZOO_MODEL = config_parser.get('main', 'zoo_model')
MODEL_NAME = config_parser.get('main', 'model_name')
GPU_NO = config_parser.getint('main','gpu_no',fallback=None)


DATAPATH = config_parser.get('train','datapath')
OUTPUT_DIR = config_parser.get('train', 'output_dir')
OUTPUT_DIR = str(Path(OUTPUT_DIR).joinpath(MODEL_NAME))
# 输出模型的名称
# MODEL_FILENAME = config_parser.get('train', 'model_filename')
MODEL_FILE = os.path.join(OUTPUT_DIR, MODEL_NAME)
# 增加 biendocer 对于bert的设定
PRETRAINED_PATH = config_parser.get('train','pretrained_dir')
OUT_DIM = config_parser.get('train','out_dim')


NUM_EPOCHS = config_parser.getint('train', 'num_epochs')
BATCHSIZE = config_parser.getint('train', 'batchsize')
EVAL_BATCHSIZE = config_parser.getint('train', 'eval_batchsize')
LR = config_parser.getfloat('train', 'lr')



DATA_PARALLEL = config_parser.getboolean('train', 'data_parallel')
FP16 = config_parser.getboolean('train', 'fp16')

# 固定的 candidates
FIXED_CANDIDATES_FILENAME = config_parser.get('train','fixed_candidates_filename')
FIXED_CANDIDATES_PATH = os.path.join(DATAPATH,FIXED_CANDIDATES_FILENAME)
CANDIDATES = config_parser.get('train','candidates')
EVAL_CANDIDATES = config_parser.get('train','eval_candidates')


WARMUP_UPDATES = config_parser.getint('train', 'warmup_updates')
LR_SCHEDULER_PATIENCE = config_parser.getint('train', 'lr_scheduler_patience')
LR_SCHEDULER_DECAY = config_parser.getfloat('train', 'lr_scheduler_decay')

CLIP = config_parser.getfloat('train', 'clip',fallback=0.1)


HISTORY_SIZE = config_parser.getint('train', 'history_size')
LABEL_TRUNCATE = config_parser.getint('train', 'label_truncate')
TEXT_TRUNCATE = config_parser.getint('train', 'text_truncate')

MAX_TRAIN_TIME = config_parser.getint('train', 'max_train_time')
VEPS = config_parser.getfloat('train', 'validation-every-n-epochs')
VME = config_parser.getint('train', 'validation-max-exs')

VALIDATION_EVERY_N_SECS = config_parser.getint('train','validation_every_n_secs',fallback=-1)
VALIDATION_METRIC = config_parser.get('train', 'validation_metric')
VALIDATION_METRIC_MODE = config_parser.get('train', 'validation_metric_mode')
SAVE_AFTER_VALID = config_parser.getboolean('train', 'save_after_valid')

LOG_EVERY_N_SECS = config_parser.getint('train', 'log_every_n_secs')
LOG_FILENAME = f'{MODE}'



# 需要使用中文的字典: 默认是 english : bpe , 中文的时候 chinese: bert_tokenizer
# DICT_LANGUAGE = config_parser.get('train','dict_language')
# DICT_CLASS = config_parser.get('train', 'dict_class')
# DICT_TOKENIZER = config_parser.get('train', 'dict_tokenizer')
# DICT_LOWER = config_parser.getboolean('train', 'dict_lower')
# DICT_ENDTOKEN = config_parser.get('train', 'dict_endtoken')


# TransformerRankerAgent（PolyencoderAgent，Biencoder） 参数
# OPTIMIZER = config_parser.get('train', 'optimizer')
# OUTPUT_SCALING = config_parser.getfloat('train', 'output_scaling')
# VARIANT = config_parser.get('train', 'variant')
# REDUCTION_TYPE = config_parser.get('train', 'reduction_type')
# SHARE_ENCODERS = config_parser.getboolean('train', 'share_encoders')
# LEARN_POSITIONAL_EMBEDDINGS = config_parser.getboolean('train', 'learn_positional_embeddings')
# N_LAYERS = config_parser.getint('train', 'n_layers')
# N_HEADS = config_parser.getint('train', 'n_heads')
# FFN_SIZE = config_parser.getint('train', 'ffn_size')
# ATTENTION_DROPOUT = config_parser.getfloat('train', 'attention_dropout')
# RELU_DROPOUT = config_parser.getfloat('train', 'relu_dropout')
# DROPOUT = config_parser.getfloat('train', 'dropout')
# N_POSITIONS = config_parser.getint('train', 'n_positions')
# EMBEDDING_SIZE = config_parser.getint('train', 'embedding_size')
# ACTIVATION = config_parser.get('train', 'activation')
# EMBEDDINGS_SCALE = config_parser.getboolean('train', 'embeddings_scale')
# N_SEGMENTS = config_parser.getint('train', 'n_segments')
# LEARN_EMBEDDINGS = config_parser.getboolean('train', 'learn_embeddings')


# INIT_MODEL = config_parser.get('train', 'init_model',fallback=None)
# POLYENCODER_TYPE = config_parser.get('train', 'polyencoder_type',fallback=None)
# POLY_N_CODES = config_parser.getint('train', 'poly_n_codes',fallback=None)
# CODES_ATTENTION_TYPE = config_parser.get('train','codes_attention_type')
# CODES_ATTENTION_NUM_HEADS =config_parser.getint('train','codes_attention_num_heads')
# POLY_ATTENTION_TYPE = config_parser.get('train', 'poly_attention_type',fallback=None)
