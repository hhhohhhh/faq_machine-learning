#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/14 23:19 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/14 23:19   wangfc      1.0         None
"""

import numpy as np
import tensorflow as tf
import spektral

adj = spektral.datasets.citation.load_data(dataset_name='cora')