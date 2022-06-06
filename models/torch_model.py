#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.***
@time: 2021/3/1 17:27 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/1 17:27   wangfc      1.0         None

"""

from abc import ABCMeta, abstractmethod
import torch
import logging
logger = logging.getLogger(__name__)


class BasicTorchModel(torch.nn.Module,metaclass=ABCMeta):
    def __init__(self,device=None):
        super(BasicTorchModel, self).__init__()
        self.fp16_enabled = False
        # 这个参数是做什么用的？
        self.best_score = float('inf')


    def forward(self):
        raise NotImplementedError


