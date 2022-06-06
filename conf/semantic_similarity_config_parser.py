# coding=utf-8
import os
import json
from configparser import ConfigParser
from .yaml_config_parser import read_from_yaml
import logging




TASK_NAME = 'semantic_similarity'
# "kbqa_diagnosis"
# "clue_classification"

# 训练参数配置文件
config_filename = f'{TASK_NAME.lower()}.cfg'
# config_file_name = 'sentiment_classifier.cfg'
config_path = os.path.join('conf', config_filename)
config_parser = ConfigParser()
config_parser.read(config_path,encoding='utf-8')

# [mode]
RUN_EAGERLY = config_parser.getboolean('mode','run_eagerly')
DEBUG = config_parser.getboolean('mode','debug')
MODE =config_parser.get('mode', 'mode')


# [main]
GPU_NO = config_parser.get('main', 'gpu_no')
GPU_MEMORY_GB = config_parser.getfloat('main', 'gpu_memory_gb')
GPU_MEMORY_MILLION_BYTES_PER_GB = 1024
GPU_MEMORY =  int(GPU_MEMORY_MILLION_BYTES_PER_GB*GPU_MEMORY_GB)
GPU_MEMORY_CONFIG = f"{GPU_NO}:{GPU_MEMORY}"
# [train]


CORPUS = config_parser.get('train', 'corpus')
DATASET_NAME = config_parser.get('train', 'dataset_name')
DATA_SUBDIR = config_parser.get('train','data_subdir')
DATA_OUTPUT_SUBDIR = config_parser.get('train','data_output_subdir')

# 获取训练数据的路径
DATASET_DIR = os.path.join(CORPUS,DATASET_NAME)
DATA_DIR = os.path.join(DATASET_DIR,DATA_SUBDIR)

LABEL_COLUMN = config_parser.get('train','label_column')

OUTPUT_DIR = config_parser.get('train', 'output_dir')


TEST_OUTPUT_SUBDIR = config_parser.get('train','test_output_subdir')
EXPORT_DIR = config_parser.get('train', 'export_dir')
STOPWORDS_PATH = os.path.join(CORPUS,'stop_words_zh.txt')

USEING_RASA = config_parser.getboolean('train','useing_rasa')
RASA_MODE = config_parser.get('train','rasa_mode')
RASA_CONFIG_FILENAME =  config_parser.get('train', 'rasa_config_filename')
RASA_MODELS_SUBDIRNAME = config_parser.get('train','rasa_models_subdirname')




#[data_preprocessing]
TRAIN_SIZE = config_parser.getfloat('data_preprocessing', 'train_size')
DEV_SIZE = config_parser.getfloat('data_preprocessing', 'dev_size')
RANDOM_STATE= config_parser.getint('data_preprocessing', 'random_state')
LABEL_COUNT_THRESHOLD = config_parser.getint('data_preprocessing', 'label_count_threshold')


# [predict]
NUM_OF_WORKERS = config_parser.getint('predict','num_of_workers')
TOP_K = config_parser.getint('predict', 'top_k')
PREDICT_TOP_K=config_parser.getint('predict', 'predict_top_k')
P_THRESHOLD= config_parser.getfloat('predict', 'p_threshold')



LOG_FILENAME = f'{MODE}_{OUTPUT_MODEL_SUBDIR}'
LOG_LEVEL = config_parser.get('logging', 'log_level')  #logging.DEBUG






