#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/8 10:32 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/8 10:32   wangfc      1.0         None
"""


def main(debug=False,mode='train',gpu_memory_config = None,log_level='debug'):
    from utils.tensorflow.environment import setup_tf_environment
    if gpu_memory_config is None:
        gpu_memory_config = GPU_MEMORY_CONFIG
    setup_tf_environment(gpu_memory_config=gpu_memory_config)
    from tasks.juyuan_ner_classifier_task import JuyuanNerClassifierTask

    juyuan_ner_classifier_task = JuyuanNerClassifierTask(
        debug=debug,
        mode=mode,
        log_level= log_level,
        data_subdir=DATA_OUTPUT_SUBDIR,
        output_dir=OUTPUT_DIR,
        model_config_filepath=MODEL_CONFIG_FILEPATH,
        last_event_definition_file=LAST_EVENT_DEFINITION_FILE,
        last_output_data_subdir=LAST_OUTPUT_DATA_SUBDIR,
        last_output_raw_data_filename =LAST_OUTPUT_RAW_DATA_FILENAME,
        new_event_definition_file=NEW_EVENT_DEFINITION_FILE,
        new_raw_data_subdir=NEW_RAW_DATA_SUBDIR,
        new_raw_data_filename=NEW_RAW_DATA_FILENAME,
    )

    juyuan_ner_classifier_task._prepare_data()

    if mode == 'train':
        juyuan_ner_classifier_task._prepare_model()
        juyuan_ner_classifier_task.train()
    elif mode == 'evaluate':
        juyuan_ner_classifier_task.evaluate()


if __name__ == '__main__':
    from conf.juyuan_ner_classifier_config_parser import *

    debug = False
    mode = 'evaluate'
    main(debug=debug,mode =mode)
