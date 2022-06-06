# coding=utf-8
import os
from configparser import ConfigParser


root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LOG_FILE = os.path.join(root_dir,'event_classifier_server','conf','logging.cfg')
config_path = os.path.join(root_dir,'event_classifier_server', 'conf', 'config.cfg')
config_parser = ConfigParser()
config_parser.read(config_path,encoding='utf-8')

# [main]
# service port
PORT = config_parser.getint('main', 'port')
P_THRESHOLD = config_parser.getfloat('main', 'p_threshold')

PROC_NAME = config_parser.get('main', 'proc_name')
TITLE = config_parser.get('main', 'title')
LOGIN_URL = config_parser.get('main', 'login_url')
COOKIE_SECRET = config_parser.get('main', 'cookie_secret')
PRED_TYPE = config_parser.get('main', 'pred_type')

# gpu_no
GPU_NO = config_parser.get('main', 'gpu_no')
GPU_MEMORY_FRACTION = config_parser.getfloat('main', 'gpu_memory_fraction')

CLASSIFIER =config_parser.get('main', 'classifier')
DATA_DIR = config_parser.get('main', 'data_dir')
OUTPUT_DIR = config_parser.get('main', 'output_dir')
BERT_MODEL_VERSION = config_parser.get('main', 'bert_model_version')
MODEL_VERSION = config_parser.get('main', 'bert_model_version')
# MAX_SEQ_LENGTH= config_parser.getint('main', 'max_seq_length')

DE_BUG= config_parser.getboolean('main', 'debug')

# [event_stock_classifier]
SERVICE_URL_DIR = config_parser.get('event_stock_classifier', 'service_url_dir')
VERSION_URL_DIR = config_parser.get('event_stock_classifier', 'version_url_dir')

MAX_SEQ_LENGTH=512

IF_PREDICT_WITH_TF_SERVING = False











