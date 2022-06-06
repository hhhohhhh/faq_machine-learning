#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@time: 2021/3/1 16:37

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/1 16:37   wangfc      1.0         None

ivr: 意图识别模型
1） 增加扩展性： 使用大意图 + slot来进一步细分意图
2） 答非所问的情况： 正则匹配
3） 分类错误或者分类超出原有类型的范围的情况： 阈值控制 +正则匹配

"""

# import os
# os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda/lib64:/usr/local/cuda-10.1/lib64:/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH"
from conf.config_parser import RASA_TEST_DATA_SUBDIRNAME


def prepare_ivr_data(dataset='ivr_data',
                     dataset_dir=None,
                     subdataset_name=None,
                     output_data_subdir=None,
                     output_dir=None,
                     support_labels=None, min_threshold=5,
                     sample_size_for_label=None,
                     raw_data_format='csv',
                     label_column="intent",
                     use_retrieve_intent=False,
                     use_intent_column_as_label=False

                     ):
    import pandas as pd
    # from conf.config_parser import VOCAB_FILE, STOPWORDS_PATH
    from data_process.dataset.intent_classifier_dataset import IntentClassifierProcessor
    from data_process.dataset.intent_classifier_dataset import IVRRawDataReader, \
        IVRStandardToExtendDataReader, IVRRawDataReaderV2
    from data_process.dataset.rasa_dataset import IVRToRasaDataset

    train_data_path = os.path.join(dataset_dir, output_data_subdir, 'train.csv')
    if not os.path.exists(train_data_path):
        # # 读取嘉琦提供的ivr 海通 数据
        # ivr_raw_dataset = IVRRawDataset(dataset_dir='ivr_data_haitong_shaojiaqi_20210831',
        #                                 output_data_subdir="ivr_intent_classifier_data_20210831",
        #                                 use_intent_column_as_label=use_intent_column_as_label)

        # 读取金曙提供的 60+ ivr 标准问的数据
        ivr_new_raw_dataset = IVRRawDataReaderV2(dataset_dir=dataset_dir,
                                                 subdataset_name=subdataset_name,
                                                 intent_structure_sheetname='intent&entity-20211111',
                                                 output_data_subdir=output_data_subdir,
                                                 use_intent_column_as_label=use_intent_column_as_label,
                                                 if_get_raw_intent_data=True,
                                                 if_get_function_name_information_dict=True)

        # 读取高峰提供的 ivr 海通数据
        ivr_standard_to_extend_raw_dataset = IVRStandardToExtendDataReader(
            dataset_dir=dataset_dir,
            subdataset_name='ivr_data_haitong_gaofeng_20210915',
            output_data_subdir=output_data_subdir,
            filter_intents=ivr_new_raw_dataset.get_support_function_name_or_intents(if_get_intent=False),
            function_name_information_dict=ivr_new_raw_dataset.function_name_information_dict
        )

        # 对数据进行合并
        raw_intent_data = pd.concat(
            [ivr_new_raw_dataset.raw_intent_data, ivr_standard_to_extend_raw_dataset.raw_intent_data])
        function_name_information_dict = ivr_new_raw_dataset.function_name_information_dict
        ivr_combined_raw_dataset = IVRRawDataReaderV2(dataset_dir=dataset_dir,
                                                      output_data_subdir=output_data_subdir)
        ivr_combined_raw_dataset.raw_intent_data = raw_intent_data
        ivr_combined_raw_dataset.function_name_information_dict = function_name_information_dict

        # 保存数据
        ivr_combined_raw_dataset._dump_raw_intent_data(raw_intent_data)
        ivr_combined_raw_dataset._dump_function_name_information_dict(function_name_information_dict)

    else:
        ivr_combined_raw_dataset = IVRRawDataReaderV2(dataset_dir=dataset_dir,
                                                      output_data_subdir=output_data_subdir,
                                                      if_get_raw_intent_data=True,
                                                      if_build_raw_intent_data=False,
                                                      if_get_function_name_information_dict=True)

    ivr_combined_raw_dataset.check_intersection_data()
    intent_count = ivr_combined_raw_dataset.raw_intent_data.intent.value_counts()

    # 初始化 IntentClassifierProcessor 对象
    indent_classifier_processor = IntentClassifierProcessor(
        dataset_dir=dataset_dir,
        # 对过滤的数据进行分割处理
        intent_classifier_data_filename=ivr_combined_raw_dataset.intent_classifier_data_filename,
        output_data_subdir=output_data_subdir,
        output_dir=output_dir,
        label_column=label_column,  # 使用子意图作为 label
        min_threshold=min_threshold,
        raw_data_format=raw_data_format,
        stopwords_path=STOPWORDS_PATH,
        support_labels=None,
        sample_size_for_label=sample_size_for_label,
        vocab_file=None,
        # 转换 rasa 数据的时候使用
        use_retrieve_intent=use_retrieve_intent,
        use_intent_column_as_label=use_intent_column_as_label
    )

    raw_data = indent_classifier_processor.new_version_data_dict['train']

    vocabulary = ivr_combined_raw_dataset.get_vocabulary(filter_intents=indent_classifier_processor.support_types_dict)

    function_name_information_dict = ivr_combined_raw_dataset.function_name_information_dict

    ivr_to_rasa_dataset = IVRToRasaDataset(raw_data=raw_data,
                                           vocabulary=vocabulary,
                                           function_name_information_dict=function_name_information_dict,
                                           output_dir=output_dir,
                                           rasa_config_filename=RASA_CONFIG_FILENAME,
                                           if_get_core_data=True,
                                           if_get_vocabulary_data=True,
                                           if_get_synonym_data=True,
                                           )
    return ivr_to_rasa_dataset


def prepare_faq_test_data(data_dir, test_data_subdir='faq_test_data_20211108',
                          test_data_filename='业务返回定义模糊数据.xlsx',
                          new_columns=['question', 'first_intent']
                          ):
    from pathlib import Path
    import pandas as pd
    from utils.io import dataframe_to_file

    test_data_subdir = '海通faq-20211110'
    test_data_filename = "海通faq--大意图--标准问.xlsx"
    question_column = 'question'
    standard_question_column = 'standard_question'
    first_intent_column = 'first_intent'
    sheet_name = '评估问--标准问--大意图'

    data_dir = Path(data_dir)
    test_path = data_dir / test_data_filename
    test_data = dataframe_to_file(path=test_path, mode='r', sheet_name=sheet_name)

    if test_data_filename == '业务返回定义模糊数据.xlsx':
        columns = test_data.columns.tolist()
        column_df_ls = []
        for column in columns:
            column_to_questions = test_data.loc[:, column].dropna().values.tolist()
            print(f"column={column}共 {column_to_questions.__len__()} 条数据")
            column_df = pd.DataFrame({question_column: column_to_questions, first_intent_column: column})
            column_df_ls.append(column_df)
        test_data = pd.concat(column_df_ls)
    elif test_data_filename == "海通faq--大意图--标准问.xlsx":
        columns = [question_column, standard_question_column, first_intent_column]
        test_data.columns = columns

    print(f"共读取测试数据 {test_data.shape}:\n{test_data.loc[:, first_intent_column].value_counts()}")
    return test_data


def train_and_evaluate(mode='train',
         rasa_mode='train_nlu',
         taskbot_name='rasa_bot',
         dataset=None,
         data_dir=None,
         output_dir=None,
         test_output_subdir='test',
         config_filename='config.yml',
         models_subdir='models',
         test_filename='test.csv'
         ):
    import sys
    import os
    from utils.common import init_logger
    logger = init_logger(output_dir=os.path.join(output_dir, 'log'), log_filename=mode)
    from conf.config_parser import GPU_MEMORY_CONFIG
    from utils.tensorflow.environment import setup_tf_environment
    from utils.io import dataframe_to_file
    from utils.constants import INTENT_COLUMN, SUB_INTENT_COLUMN
    import rasa
    from rasa.__main__ import main as rasa_main

    if mode == 'train' or mode == 'train_and_evaluate':
        setup_tf_environment(gpu_memory_config=GPU_MEMORY_CONFIG)
        prepare_ivr_data(dataset_dir=DATASET_DIR,
                         subdataset_name=SUBDATASET_NAME,
                         output_data_subdir=DATA_OUTPUT_SUBDIR,
                         label_column=INTENT_COLUMN,
                         output_dir=OUTPUT_DIR)
        os.chdir(output_dir)
        argv = rasa_mode.split('_')
        if argv[0] in ['train', 'data']:
            # cmd : train /train nlu /train core
            # cmd : run / run nlu /run core / run actions
            argv.extend(['--config', config_filename, '--out', models_subdir])
        elif argv[0] == 'shell':  # shell /shell nlu / shell core
            argv.extend(['-m', models_subdir])

        sys.argv.extend(argv)
        rasa_main()

    # elif mode == 'run_server':
    #     from servers.rasa_servers.rasa_application import run_rasa_application
    #     # from conf.config_parser import *
    #     run_rasa_application(mode=MODE, output_dir=OUTPUT_DIR,
    #                          config_filename=RASA_CONFIG_FILENAME,
    #                          models_subdir=RASA_MODELS_SUBDIRNAME,
    #                          num_of_workers=NUM_OF_WORKERS,
    #                          gpu_memory_per_worker=GPU_MEMORY_CONFIG,
    #                          if_build_agent_application=True,
    #                          if_enable_api=False
    #                          )

    if mode == 'evaluate' or mode == 'train_and_evaluate':
        from apps.rasa_apps.rasa_application import RasaApplication
        setup_tf_environment(gpu_memory_config=None)

        # 加载测试数据
        # test_path = os.path.join(data_dir, test_filename)
        # test_data = dataframe_to_file(path=test_path, mode='r')

        test_data_subdir = '海通faq-20211110'
        test_data_dir = os.path.join('corpus', dataset, test_data_subdir)
        test_data = prepare_faq_test_data(test_data_dir)

        # 新建  server
        # rasa_sever = RasaServer(taskbot_dir=taskbot_dir,trained_nlu_model=None)

        # 新建 interpreter
        rasa_application = RasaApplication(taskbot_dir=output_dir,
                                           config_filename=config_filename,
                                           models_subdir=models_subdir)

        # 对测试数据进行预测
        rasa_application.evaluate(test_data=test_data,
                                  test_data_dir=test_data_dir,
                                  # test_output_subdir=test_output_subdir,
                                  label_column='first_intent',
                                  agent_application=True,
                                  nlu_application=False,
                                  debug=False)

def main(mode):
    import os
    from conf.config_parser import MODE,DATASET_NAME,DATA_DIR,OUTPUT_DIR,RASA_CONFIG_FILENAME,RASA_MODELS_SUBDIRNAME,\
    RASA_TRAIN_DATA_SUBDIRNAME,GPU_MEMORY_CONFIG,\
        RAW_FAQ_KNOWLEDGE_DEFINITION_FILENAME,FAQ_KNOWLEDGE_DEFINITION_SHEET_NAME,STANDARD_QUESTION_KNOWLEDGE_SHEET_NAME,\
        RAW_STANDARD_TO_EXTEND_DATA_FILENAME,STANDARD_TO_EXTEND_DATA_SHEET_NAME,ENTITY_DATA_FILENAME

    from utils.common import init_logger

    logger = init_logger(output_dir=os.path.join(OUTPUT_DIR, 'log'), log_filename=mode)
    from utils.tensorflow.environment import setup_tf_environment
    setup_tf_environment(gpu_memory_config=GPU_MEMORY_CONFIG)

    from tasks.rasa_task import RasaTask
    import rasa

    rasa_task = RasaTask(mode=mode,
                         dataset=DATASET_NAME,
                         data_dir=DATA_DIR,
                         raw_faq_knowledge_definition_filename=RAW_FAQ_KNOWLEDGE_DEFINITION_FILENAME,
                         faq_knowledge_definition_sheet_name=FAQ_KNOWLEDGE_DEFINITION_SHEET_NAME,
                         standard_question_knowledge_sheet_name= STANDARD_QUESTION_KNOWLEDGE_SHEET_NAME,
                         raw_standard_to_extend_data_filename=RAW_STANDARD_TO_EXTEND_DATA_FILENAME,
                         standard_to_extend_data_sheet_name=STANDARD_TO_EXTEND_DATA_SHEET_NAME,
                         output_dir=OUTPUT_DIR,
                         rasa_config_filename=RASA_CONFIG_FILENAME,
                         rasa_train_data_subdirname = RASA_TRAIN_DATA_SUBDIRNAME,
                         rasa_test_data_subdirname=RASA_TEST_DATA_SUBDIRNAME,
                         rasa_models_subdirname=RASA_MODELS_SUBDIRNAME)

    # 准备标注的数据
    # rasa_task._prepare_data_to_label()

    if mode == 'train':
        # 进行训练
        rasa_task.train()
    # 进行评估
    if mode == 'test':
        filter_test_filename = "ivr短句和扩展问无限制_评估集_20211216.xlsx" #"评估集4-16字符.xlsx"
        rasa_task._prepare_data(if_get_train_data=False,
                                if_get_test_data=True,
                                filter_test_filename=filter_test_filename)
        # rasa_task.test_nlu()
        # 测试问题是否返回单一的标准问
        rasa.utils.common.run_in_loop(rasa_task.evaluate(filter_test_filename=filter_test_filename))


if __name__ == '__main__':

    main(mode='test')






