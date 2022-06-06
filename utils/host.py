#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/10/11 17:02 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/11 17:02   wangfc      1.0         None
"""
import socket
from typing import Text


def get_host_ip()-> [Text,Text]:
    try:
        # 获取本机计算机名称
        host_name = socket.gethostname()
        # 获取本机ip
        host_ip = socket.gethostbyname(host_name)
        return host_name, host_ip
    except:
        raise EnvironmentError("Unable to get Hostname and IP")
