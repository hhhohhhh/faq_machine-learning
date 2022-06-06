#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:
https://github.com/howl-anderson/addons/blob/add_crf_tutorial/docs/tutorials/layers_crf.ipynb

@time: 2022/3/9 13:54 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/9 13:54   wangfc      1.0         None
"""


from tasks.ner_task import BilstmCrfEntityExtrator




def main():
    extractor = BilstmCrfEntityExtrator()
    extractor.train()







if __name__ == '__main__':
    main()
