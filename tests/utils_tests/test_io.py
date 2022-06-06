#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/8/12 9:13 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/12 9:13   wangfc      1.0         None
"""
import os
import pytest

from utils.io import get_matched_filenames

@pytest.fixture(scope='session')
def test_dir_data():
    cwd = os.getcwd()

    data_dir = os.path.join('/home/wangfc/faq/','corpus','haitong_intent_classification_data','haitong_preprocessed_intent_data')

    print(data_dir)
    filename_pattern = 'select_faq_data_\d{3}.yml'
    return {'data_dir': data_dir, 'filename_pattern': filename_pattern}

def test_get_matched_filenames(test_dir_data):
    filenames = get_matched_filenames(**test_dir_data)
    assert filenames== ['select_faq_data_000.yml','select_faq_data_001.yml']

# if __name__ == '__main__':
#     test_get_matched_filenames(**test_dir_nata)