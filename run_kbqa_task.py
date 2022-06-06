#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/10/22 11:03 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/22 11:03   wangfc      1.0         None
"""


def main():
    import os
    import sys
    from conf.config_parser import MODE,RASA_MODE,CORPUS,DATASET_DIR, OUTPUT_DIR,\
        RASA_CONFIG_FILENAME,RASA_MODELS_SUBDIRNAME,GPU_MEMORY_CONFIG
    from utils.common import init_logger
    logger = init_logger(output_dir=os.path.join(OUTPUT_DIR, 'log'), log_filename=MODE, log_level='debug')

    neo4j_database_config = dict(host="127.0.0.1", port=7687, user="neo4j", password="123456")

    # 训练模式
    if MODE == 'train' or MODE == 'train_and_evaluate':
        from data_process.dataset.kg_dataset import MedicalDiagnosisDataProcessor
        from utils.tensorflow.environment import setup_tf_environment
        from rasa.__main__ import main as rasa_main

        # 创建 rasa格式的标注数据
        medical_extractor = MedicalDiagnosisDataProcessor(corpus=CORPUS, dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR,
                                                          neo4j_database_config=neo4j_database_config)
        # 创建 知识图谱数据库
        medical_extractor.get_kg()

        # 训练 模型
        setup_tf_environment(gpu_memory_config=GPU_MEMORY_CONFIG)
        os.chdir(OUTPUT_DIR)
        argv = RASA_MODE.split('_')
        if argv[0] in ['train', 'data']:
            # cmd : train /train nlu /train core
            # cmd : run / run nlu /run core / run actions
            argv.extend(['--config', RASA_CONFIG_FILENAME, '--out', RASA_MODELS_SUBDIRNAME])
        elif argv[0] == 'shell':  # shell /shell nlu / shell core
            argv.extend(['-m', RASA_MODELS_SUBDIRNAME])
        sys.argv.extend(argv)
        rasa_main()

    # 启动服务
    if MODE=='run_server':
        from servers.rasa_servers.rasa_application import RasaApplication
        from conf.config_parser import NUM_OF_WORKERS
        rasa_application = RasaApplication(taskbot_dir=OUTPUT_DIR,
                                           config_filename=RASA_CONFIG_FILENAME,
                                           models_subdir= RASA_MODELS_SUBDIRNAME,
                                           num_of_workers=NUM_OF_WORKERS,
                                           gpu_memory_per_worker=GPU_MEMORY_CONFIG,
                                           if_build_nlu_interpreter=False,
                                           if_build_nlu_application=False,
                                           if_build_agent_application=False,
                                           if_enable_api=False,
                                           if_build_medical_diagnosis_application=True,
                                           neo4j_database_config=neo4j_database_config
                                           )

        # result = await rasa_application.nlu_interpreter.parse("脑瘤传染吗")
        # print(result)

    # 评估



if __name__ == '__main__':

    main()
