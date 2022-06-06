#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/4/15 14:07 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/4/15 14:07   wangfc      1.0         None

"""
from data_process.dataset.intent_classifier_dataset import CombinedRawIntentDataset, IVRRawDataset, \
    IVRStandardToExtendRawData
from data_process.dataset.rasa_dataset import IVRToRasaDataset
from utils.constants import SUB_INTENT_COLUMN


def preprocess_sentence_pair_to_dstc7_data():
    # from conf.polyencoder_config_parser import *
    from utils.common import init_logger
    # logger = init_logger(output_dir=OUTPUT_DIR, log_filename=LOG_FILENAME)
    from data_process.sentence_pair_data_reader import SentencePairDataReader
    # 蚂蚁金服的数据集
    data_dir = 'data'
    data_files = [{"dataset": 'atec', "data_filenames": ['atec_nlp_sim_train.csv', 'atec_nlp_sim_train_add.csv']},
                  {"dataset": 'security_similary_data', "data_filenames": ["六家评估集.xlsx"]}]

    output_dirname = 'sentence_pairdata_to_dstc7_003'
    sentence_pair_data_reader = SentencePairDataReader(data_dir=data_dir, data_files=data_files,
                                                       output_dirname=output_dirname,
                                                       if_overwrite=False, if_convert_to_dstc7=True,
                                                       train_size=0.8, dev_size=0.1, test_size=0.1)


def prepare_intent_data(dataset='ivr_data', dataset_dir=None,
                        output_data_subdir=None, output_dir=None,
                        support_labels=None, min_threshold=10,
                        sample_size_for_label=None,
                        raw_data_format='csv'):
    import os
    from utils.common import init_logger
    from conf.config_parser import VOCAB_FILE, STOPWORDS_PATH
    from data_process.dataset.intent_classifier_dataset import IntentClassifierProcessor
    logger = init_logger(output_dir=os.path.join('corpus', dataset, output_data_subdir, 'log'),
                         log_filename=f'{output_data_subdir}_data_process')

    if dataset == 'combine_dataset':
        support_labels = ['faq', 'advice', 'chat']
        # 融合多种意图的数据，最后存放在 output_data_subdir
        combined_raw_intent_dataset = CombinedRawIntentDataset(output_data_subdir=output_data_subdir,
                                                               vocab_file=VOCAB_FILE)
    elif dataset == 'ivr_data':
        use_retrieve_intent = False
        use_intent_column_as_label = False
        if use_intent_column_as_label == False:
            label_column = SUB_INTENT_COLUMN

        # 读取嘉琦提供的ivr 海通 数据
        ivr_raw_dataset = IVRRawDataset(dataset_dir='ivr_data_haitong_shaojiaqi_20210831',
                                        output_data_subdir="ivr_intent_classifier_data_20210831",
                                        use_intent_column_as_label=use_intent_column_as_label)
        # support_labels = ivr_raw_dataset.support_labels
        # 转换为 rasa 的数据格式
        # IVRToRasaDataset(raw_data=ivr_raw_dataset.filter_intent_data,
        #                  output_dir=output_dir)

        # 读取高峰提供的 ivr 海通数据
        ivr_standard_to_extend_raw_dataset = IVRStandardToExtendRawData(
            dataset_dir=dataset_dir,
            output_data_subdir=output_data_subdir)

    # 初始化 IntentClassifierProcessor 对象
    indent_classifier_processor = IntentClassifierProcessor(
        dataset_dir=dataset_dir,
        # 对过滤的数据进行分割处理
        intent_classifier_data_filename=ivr_standard_to_extend_raw_dataset.filter_intent_filename,
        output_data_subdir=output_data_subdir,
        output_dir=output_dir,
        label_column=label_column,  # 使用子意图作为 label
        min_threshold=min_threshold,
        raw_data_format=raw_data_format,
        stopwords_path=STOPWORDS_PATH,
        support_labels=None,
        sample_size_for_label=sample_size_for_label,
        vocab_file=VOCAB_FILE,
        # 转换 rasa 数据的时候使用
        use_retrieve_intent=use_retrieve_intent,
        use_intent_column_as_label=use_intent_column_as_label
    )

    raw_data = indent_classifier_processor.new_version_data_dict['train']

    vocabulary = ivr_raw_dataset.get_vocabulary(filter_intents=indent_classifier_processor.support_types_dict)

    ivr_to_rasa_dataset = IVRToRasaDataset(raw_data=raw_data,
                                           vocabulary= vocabulary,
                                           output_dir=output_dir)




def prepare_hundsun_faq_data():
    """
    @time:  2021/8/2 10:27
    @author:wangfc
    @version:
    @description: 使用 rasa response selector 做实验


    @params:
    @return:
    """
    import os
    from utils.common import init_logger
    output_dir = os.path.join('corpus', 'hsnlp_faq_data', 'log')
    log_filename = 'log'
    logger = init_logger(output_dir=output_dir, log_filename=log_filename)

    from data_process.dataset.hsnlp_faq_dataset import HSNLPFaqRawDataSet
    hsnlp_faq_raw_dataset = HSNLPFaqRawDataSet()
    hsnlp_faq_raw_dataset.transform_to_rasa_nlu_data()


def analysis_haitong_raw_data():
    """
    海通意图分类数据分析
    """
    import os
    from utils.common import init_logger
    data_dir = os.path.join('corpus', 'intent_classification')
    log_output_dir = os.path.join(data_dir, 'log')
    log_filename = 'haitong_intent_data_log'
    logger = init_logger(output_dir=log_output_dir, log_filename=log_filename)
    from data_process.dataset.intent_classifier_dataset import RawIntentClassifierDataProcessor
    raw_dataprocessor = RawIntentClassifierDataProcessor(
        dataset='haitong_intent_classification_data',
        output_data_subdir='haitong_intent_classification_data_01',
        raw_data_format='csv',
        intent_classifier_data_filename='kms_intention_record.csv'
    )
    raw_dataprocessor.read_haitong_raw_data()


def combine_raw_intent_data(output_data_subdir, logger=None):
    """
    融合各种 意图识别的数据
    """
    import os
    from utils.common import init_logger
    from conf.config_parser import VOCAB_FILE, STOPWORDS_PATH
    from data_process.dataset.intent_classifier_dataset import CombinedRawIntentDataset
    if logger is None:
        logger = init_logger(output_dir=os.path.join('corpus', 'intent_classification', 'log'),
                             log_filename=f'intent_data_process')
    #
    combined_raw_intent_dataset = CombinedRawIntentDataset(output_data_subdir=output_data_subdir,
                                                           vocab_file=VOCAB_FILE)


if __name__ == '__main__':
    from conf.config_parser import *

    # preprocess_sentence_pair_to_dstc7_data()
    # prepare_hundsun_faq_data()
    # analysis_haitong_raw_data()

    # output_data_subdir = 'gf_intent_classifier_data_20210729'
    # dataset = 'intent_classification'
    # output_data_subdir = 'intent_classifier_data_20210812'
    # output_dir = 'output/intent_classifier_02'

    # dataset = 'ivr_data'
    # output_data_subdir = 'ivr_intent_classifier_data_20210831'
    # output_dir = 'output/ivr_intent_classifier_01'

    prepare_intent_data(dataset=DATASET_NAME,
                        dataset_dir=DATASET_DIR,
                        output_data_subdir=DATA_SUBDIR,
                        output_dir=OUTPUT_DIR)
