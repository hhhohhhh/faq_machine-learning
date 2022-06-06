#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@time: 2020/9/2 21:45

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/9/2 21:45   wangfc      1.0         None

"""


import sys
import os



class RasaExampleTask():
    def __init__(self,example_data_dir,output_dir,
                 rasa_config_filename='config.yml',
                 rasa_train_data_subdirname='data',
                 rasa_domain_filename = 'domain.yml',
                 rasa_models_subdirname='models' ):
        self.data_dir = example_data_dir
        self.output_dir =  output_dir
        self.rasa_config_filename = rasa_config_filename
        self.rasa_train_data_subdirname = rasa_train_data_subdirname
        self.rasa_domain_filename = rasa_domain_filename
        self.rasa_models_subdirname = rasa_models_subdirname


        self.rasa_config_filepath = os.path.join(self.data_dir,self.rasa_config_filename)
        self.rasa_train_data_dir = os.path.join(self.data_dir,rasa_train_data_subdirname)
        self.rasa_domain_filepath = os.path.join(self.data_dir,rasa_domain_filename)
        self.rasa_model_output_dir = os.path.join(output_dir,rasa_models_subdirname)

    def run_rasa_subtask(self,mode, argv, gpu_memory_config='-1:0',):
        from utils.tensorflow.environment import setup_tf_environment
        setup_tf_environment(gpu_memory_config=gpu_memory_config)
        from rasa.__main__ import main as rasa_main
        # os.chdir(self.output_dir)

        sys.argv.extend(argv)
        rasa_main()

    def train(self,train_nlu=False,train_core=False,shell_nlu=False,
              train_agent=False,gpu_memory_config='-1:0'):
        if train_agent:
            argv = ['train',
                    '--config', self.rasa_config_filepath,
                    '--data', self.rasa_train_data_dir,
                    '--domain', self.rasa_domain_filepath,
                    '--out', self.rasa_model_output_dir]
        elif train_nlu:
            argv = ['train','nlu',
                    '--config', self.rasa_config_filepath,
                    '--nlu', self.rasa_train_data_dir,
                    '--domain', self.rasa_domain_filepath,
                    '--out', self.rasa_model_output_dir]
        elif shell_nlu:
            argv = ['shell','nlu',
                    '--config', self.rasa_config_filepath,
                    '--nlu', self.rasa_train_data_dir,
                    '--domain', self.rasa_domain_filepath,
                    '--out', self.rasa_model_output_dir]

        self.run_rasa_subtask(mode='train',argv=argv,gpu_memory_config=gpu_memory_config)

    def run(self,run_agent=False,shell_nlu=False, gpu_memory_config='-1:0'):
        if run_agent:
            argv = ['run',
                    '--model', self.rasa_model_output_dir]
        elif shell_nlu:
            argv = ['shell','nlu',
                    '--model', self.rasa_model_output_dir]
        self.run_rasa_subtask(mode='run', argv=argv, gpu_memory_config=gpu_memory_config)




def main(mode='train'):
    example= 'responseselectorbot'
    example_data_dir = os.path.join('examples','rasa_examples',example)
    output_dir = os.path.join('output',example)
    GPU_MEMORY_CONFIG = '1:5120'
    from utils.common import init_logger
    logger = init_logger(output_dir=os.path.join(output_dir, 'log'), log_filename=mode)
    # from utils.tensorflow.environment import setup_tf_environment
    # setup_tf_environment(gpu_memory_config=GPU_MEMORY_CONFIG)
    rasa_example_task = RasaExampleTask(example_data_dir=example_data_dir,output_dir=output_dir)
    if mode == 'train':
        rasa_example_task.train(train_nlu=True,gpu_memory_config=GPU_MEMORY_CONFIG)
    elif mode =='run':
        rasa_example_task.run(shell_nlu=True)


if __name__ == '__main__':
    mode = 'run'
    main(mode=mode)


