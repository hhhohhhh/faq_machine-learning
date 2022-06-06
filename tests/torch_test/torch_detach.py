#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/4/6 16:49 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/4/6 16:49   wangfc      1.0         None
"""

import torch

def test_basic_grad():
    a = torch.tensor([1, 2, 3.], requires_grad=True)
    print(a.grad)
    out = a.sigmoid()
    out.sum().backward()
    print(a.grad)
'''返回：
None
tensor([0.1966, 0.1050, 0.0452])
'''



def test_basic_detach():
    a = torch.tensor([1, 2, 3.], requires_grad=True)
    print(a.grad)
    out = a.sigmoid()

    # 添加detach(),c的requires_grad为False
    c = out.detach()
    print(c)

    # 这时候没有对c进行更改，所以并不会影响backward()
    out.sum().backward()
    print(a.grad)