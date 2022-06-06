# coding=utf-8
import os
import json
from configparser import ConfigParser


# 聚源主体识别配置文件
TASK_NAME = "juyuan_ner_classifier"
config_file_name = f'{TASK_NAME}.cfg'
config_path = os.path.join('conf', config_file_name)
config_parser = ConfigParser()
config_parser.read(config_path,encoding='utf-8')


# [main]
GPU_NO = config_parser.get('main', 'gpu_no')
GPU_MEMORY_GB = config_parser.getfloat('main', 'gpu_memory_gb')
GPU_MEMORY_MILLION_BYTES_PER_GB = 1024
GPU_MEMORY =  int(GPU_MEMORY_MILLION_BYTES_PER_GB*GPU_MEMORY_GB)
GPU_MEMORY_CONFIG = f"{GPU_NO}:{GPU_MEMORY}"

# [train]
CORPUS = config_parser.get('train', 'corpus')
DATASET = config_parser.get('train', 'dataset')
DEFINITION_SUBDIR = config_parser.get('train', 'definition_subdir')
LAST_EVENT_DEFINITION_FILE = config_parser.get('train', 'last_event_definition_file')
LAST_OUTPUT_DATA_SUBDIR = config_parser.get('train', 'last_output_data_subdir')
LAST_OUTPUT_RAW_DATA_FILENAME = config_parser.get('train','last_output_raw_data_filename')
NEW_EVENT_DEFINITION_FILE = config_parser.get('train', 'new_event_definition_file')
NEW_RAW_DATA_SUBDIR = config_parser.get('train', 'new_raw_data_subdir')
NEW_RAW_DATA_FILENAME = config_parser.get('train','new_raw_data_filename')
DATA_OUTPUT_SUBDIR = config_parser.get('train','data_output_subdir')
OUTPUT_DIR = config_parser.get('train', 'output_dir')
EXPORT_DIR = config_parser.get('train', 'export_dir')

MODEL_CONFIG_FILEPATH = config_parser.get('train','model_config_filepath')


# BERR_BASE_MODEL = config_parser.get('train', 'bert_base_model')
# BERT_BASE_DIR = config_parser.get('train', 'bert_base_dir')
# # bert_config.json是BERT的配置(超参数)，比如网络的层数
# bert_config_file = os.path.join(BERT_BASE_DIR, BERR_BASE_MODEL + '_config.json')
# # vocab.txt是模型的词典
# vocab_file = os.path.join(BERT_BASE_DIR, 'vocab.txt')
# # bert_model.ckpt*，这是预训练好的模型的checkpoint，我们的Fine-Tuning模型的初始值就是来自于这些文件，然后根据不同的任务进行Fine-Tuning。
# init_checkpoint = os.path.join(BERT_BASE_DIR,BERR_BASE_MODEL+ '_model.ckpt')
# # init_checkpoint = os.path.join(BERT_BASE_DIR,'model.ckpt-best')
#
# # [data_preprocessing]
# TRAIN_SIZE = config_parser.getfloat('data_preprocessing', 'train_size')
# DEV_SIZE = config_parser.getfloat('data_preprocessing', 'dev_size')
# RANDOM_STATE= config_parser.getint('data_preprocessing', 'random_state')
# LABEL_COUNT_THRESHOLD = config_parser.getint('data_preprocessing', 'label_count_threshold')
#
# MULTILABEL_CLASSIFIER = config_parser.getboolean('train', 'multilabel_classifier')
# MAX_SEQ_LENGTH= config_parser.getint('train', 'max_seq_length')
# LEARNING_RATE = config_parser.getfloat('train', 'learning_rate')
#
# EPOCH = config_parser.getint('train', 'epoch')
# TRAIN_BATCH_SIZE= config_parser.getint('train', 'train_batch_size')
# EARLY_STOPPING_STEPS= config_parser.getint('train', 'early_stopping_steps')
# SAVE_CHECKPOINTS_STEPS= config_parser.getint('train', 'save_checkpoints_steps')
# NUM_WARMUP_EPOCH= config_parser.getint('train', 'num_warmup_epoch')
# EXPORTS_TO_KEEP = int(config_parser.get('train', 'exports_to_keep'))
#
# # crf layer parameters
# IF_USE_CRF = config_parser.getboolean('train', 'if_use_crf')
# # CRF_ONLY=config_parser.getboolean('train', 'crf_only')
# # DROPOUT_RATE=config_parser.getfloat('train', 'dropout_rate')
# # CELL=config_parser.get('train', 'cell')
# # LSTM_SIZE=config_parser.getint('train', 'lstm_size')
# # NUM_LAYERS=config_parser.getint('train', 'num_layers')
#
# # [predict]
# TOP_K = config_parser.getint('predict', 'top_k')
# PREDICT_TOP_K=config_parser.getint('predict', 'predict_top_k')
# P_THRESHOLD=config_parser.getfloat('predict', 'p_threshold')

