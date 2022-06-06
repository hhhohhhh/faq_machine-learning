#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/11 14:53 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/11 14:53   wangfc      1.0         None
"""

from tasks.ner_task import BilstmCrfEntityExtrator
from utils.tensorflow.environment import setup_tf_environment


def main(mode='train',debug=True):
    extractor = BilstmCrfEntityExtrator(debug=debug,run_eagerly=True,vocab_size=1000,
                                        )
    if mode:
        setup_tf_environment(gpu_memory_config="0:5120")
        extractor.train(train_epochs=3)



if __name__ == '__main__':
    main()
