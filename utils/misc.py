#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   misc.py
@Version:  
@Desc:
@Time: 2022/6/5 16:03
@Contact :   wang_feicheng@163.com
@License :   Create by wangfc27441 on 2022/6/5, Copyright 2022 wangfc27441. All rights reserved.

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/5 16:03   wangfeicheng      1.0         None
'''

import progressbar


bar_widgets = [
    'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA()
]