#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/10 10:30 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/10 10:30   wangfc      1.0         None
"""
import pytest
from data_process.data_generator import KerasTextClassifierDataGenerator
from data_process.data_generator import SEQUENCE_BATCH_STRATEGY, RANDOM_BATCH_STRATEGY, BALANCED_BATCH_STRATEGY

data_size = 5
string = 'abcdefgh'
data01 = [{'guid': str(i), 'id': str(i).zfill(3), 'text_a': string[i], 'label_id': i} for i in range(data_size)]

data_size = 8
mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
data02 = [{'guid': str(i), 'id': str(i).zfill(3), 'text_a': string[i], 'label_id': i % 3} for i in range(data_size)]


@pytest.mark.parametrize(
    "data, batch_size, batch_strategy, batch_expected",
    [
        # (data01, 2, SEQUENCE_BATCH_STRATEGY, [(['a', 'b'], [0, 1]),
        #                                       (['c', 'd'], [2, 3]),
        #                                       (['e'], [4]),
        #                                       ]),
        (data02, 3, BALANCED_BATCH_STRATEGY, [(['d', 'h', 'f'], [0, 1, 2]),
                                              (['g', 'c', 'b'], [0, 2, 1]),
                                              (['e', 'a'], [1, 0])],

         )
    ])


def test_keras_data_generator(data, batch_size, batch_strategy, batch_expected):
    data_generator = KerasTextClassifierDataGenerator(data=data, batch_size=batch_size, batch_strategy=batch_strategy)
    batches = []
    steps = data_generator.__len__()
    for step in range(steps):
        batch = data_generator.__getitem__(index=step)
        batches.append(batch)

    assert batches == batch_expected
