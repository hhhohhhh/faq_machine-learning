#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/26 9:52 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/26 9:52   wangfc      1.0         None
"""
import os
import time

from utils.io import get_file_stem
import sys
import subprocess


def main(mode='train', debug=False, port=8015,gpu_memory_config=None,num_processes=None,
         with_preprocess_component=True, use_intent_attribute_regex_classifier=True,
         run_app_when_test=False,
         test_data_filename=None,output_test_data_filename=None):
    from cli.arguments.task_argument_parser import create_argument_parser
    from conf.config_parser import MODE, DATASET_NAME, DATA_DIR, TEST_DATA_SUBDIR, OUTPUT_DIR, RASA_CONFIG_FILENAME, \
        RASA_MODELS_SUBDIRNAME, \
        RASA_TRAIN_DATA_SUBDIRNAME, GPU_MEMORY_CONFIG, \
        RAW_FAQ_KNOWLEDGE_DEFINITION_FILENAME, FAQ_INTENT_ATTRIBUTE_DEFINITION_SHEET_NAME, \
        FAQ_ENTITY_DEFINITION_SHEET_NAME, \
        STANDARD_QUESTION_KNOWLEDGE_SHEET_NAME, \
        RAW_STANDARD_TO_EXTEND_DATA_FILENAME, STANDARD_TO_EXTEND_DATA_SHEET_NAME, ENTITY_DATA_FILENAME, \
        RASA_TEST_DATA_SUBDIRNAME, MODEL_CONFIG_FILEPATH

    # 设置默认参数：从参数中直接获取或者 config 文件中读取
    mode = mode if mode else MODE
    gpu_memory_config = gpu_memory_config if gpu_memory_config else GPU_MEMORY_CONFIG
    num_processes = num_processes if num_processes else NUM_OF_WORKERS
    # 设置 arg_parser: 可以从 cmd 命令中读取参数
    arg_parser = create_argument_parser(mode=mode, port=port,gpu_memory_config=gpu_memory_config,
                                        num_processes=num_processes)
    cmdline_arguments = arg_parser.parse_args()

    # log_level = (
    #     cmdline_arguments.loglevel if hasattr(cmdline_arguments, "loglevel") else None
    # )
    # logger = init_logger(output_dir=os.path.join(OUTPUT_DIR, 'log'), log_filename=cmdline_arguments.mode,
    #                      log_level=log_level)

    from tasks.faq_intent_attribute_classifier_task import FaqIntentAttributeClassifierTaskTfV2
    faq_intent_attribute_classifier_task = FaqIntentAttributeClassifierTaskTfV2(
        debug=debug,
        run_eagerly=True,
        mode=cmdline_arguments.mode,
        dataset=DATASET_NAME,
        data_dir=DATA_DIR,
        test_data_subdir=TEST_DATA_SUBDIR,
        test_data_filename=test_data_filename,
        output_test_data_filename= output_test_data_filename,
        raw_faq_knowledge_definition_filename=RAW_FAQ_KNOWLEDGE_DEFINITION_FILENAME,
        faq_intent_attribute_definition_sheet_name=FAQ_INTENT_ATTRIBUTE_DEFINITION_SHEET_NAME,
        faq_entity_definition_sheet_name=FAQ_ENTITY_DEFINITION_SHEET_NAME,
        standard_question_knowledge_sheet_name=STANDARD_QUESTION_KNOWLEDGE_SHEET_NAME,
        raw_standard_to_extend_data_filename=RAW_STANDARD_TO_EXTEND_DATA_FILENAME,
        standard_to_extend_data_sheet_name=STANDARD_TO_EXTEND_DATA_SHEET_NAME,
        intent_attribute_mapping_filename="haitong_ivr_intent_attribute_mapping.json",
        output_dir=OUTPUT_DIR,
        model_config_filepath=MODEL_CONFIG_FILEPATH,
        rasa_config_filename=RASA_CONFIG_FILENAME,
        rasa_train_data_subdirname=RASA_TRAIN_DATA_SUBDIRNAME,
        rasa_test_data_subdirname=RASA_TEST_DATA_SUBDIRNAME,
        rasa_models_subdirname=RASA_MODELS_SUBDIRNAME)

    if cmdline_arguments.mode == 'train':
        from utils.tensorflow.environment import setup_tf_environment
        setup_tf_environment(gpu_memory_config=cmdline_arguments.gpu_memory_config)
        faq_intent_attribute_classifier_task.train()

    elif cmdline_arguments.mode == 'run':
        faq_intent_attribute_classifier_task.run(run_nlu_app=True,
                                                 port=cmdline_arguments.port,
                                                 gpu_memory_config=cmdline_arguments.gpu_memory_config,
                                                 num_processes= cmdline_arguments.num_processes,
                                                 run_shell_nlu=False,
                                                 with_preprocess_component=with_preprocess_component,
                                                 use_intent_attribute_regex_classifier=use_intent_attribute_regex_classifier
                                                 )
    elif cmdline_arguments.mode == 'test':
        # 1. 加载 app
        app = faq_intent_attribute_classifier_task._build_app(
            port=cmdline_arguments.port,
            gpu_memory_config=cmdline_arguments.gpu_memory_config,
            num_processes=cmdline_arguments.num_processes,
            with_preprocess_component=with_preprocess_component,
            use_intent_attribute_regex_classifier=use_intent_attribute_regex_classifier)

        # 2. 后台启动 app.run()
        if run_app_when_test:
            env = os.environ.copy()
            python = "/home/wangfc/anaconda3/envs/python3710/bin/python"
            child = subprocess.Popen([python,sys.argv[0],'-m','run'],env=env)
            # child.wait()
            logger.info(f"已经运行 app")
            time.sleep(10)

        # 获取 url
        url = app._get_service_url()

        # 获取测试结果的后缀名
        test_result_file_suffix = get_test_result_file_suffix(with_preprocess_component=with_preprocess_component,
                                                              run_app_when_test=run_app_when_test,output_dir=OUTPUT_DIR)
        # 4. 进行评估
        faq_intent_attribute_classifier_task.evaluate(url=url, test_result_file_suffix=test_result_file_suffix)

        # 5. 终止app
        if run_app_when_test:
            child.kill()
            logger.info(f"终止 app")


def get_test_result_file_suffix(with_preprocess_component=True,
                                run_app_when_test=True,
                                output_dir=None,
                                version='v1',
                                test_model_num=None,
                                with_entity=True):
    test_result_file_suffix= 'result'
    if run_app_when_test:
        version = 'v2'
        test_model_num = get_file_stem(output_dir).split("_")[-1]

    if with_preprocess_component:
        test_result_file_suffix = f'{test_result_file_suffix}_preprocessed'

    if test_model_num:
        model = f"model{test_model_num}"
        if with_entity:
            model = f"{model}withentity"
        test_result_file_suffix = f"{version}_{model}_{test_result_file_suffix}"
    else:
        test_result_file_suffix = f"{version}_{test_result_file_suffix}"
    return test_result_file_suffix



if __name__ == '__main__':
    from utils.common import init_logger
    from conf.config_parser import OUTPUT_DIR, NUM_OF_WORKERS
    mode = 'run'
    log_level = 'info'
    logger = init_logger(output_dir=os.path.join(OUTPUT_DIR, 'log'), log_filename=mode,
                         log_level=log_level)

    try:
        gpu_memory_config = '-1:128'
        debug = False
        port = 8014
        num_processes = 1
        test_data_filename = "第一轮测试结果（增加扩展问）.xlsx" #  "评估集4-16字符-new_02.xlsx"
        main(mode=mode, debug=debug, port=port,
             num_processes=num_processes,
             gpu_memory_config=gpu_memory_config,
             with_preprocess_component=False,
             use_intent_attribute_regex_classifier=False,
             test_data_filename =test_data_filename,
             run_app_when_test=True)
    except Exception as e:
        logger.error(e,exc_info=True)