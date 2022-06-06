#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.***
@time: 2021/1/7 10:17 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/7 10:17   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：***股份有限公司 2019
 * 注意：本内容仅限于***股份有限公司内部传阅，禁止外泄以及用于其他的商业目的

"""
# import os
# import logging
# import time
# from pathlib import Path
#
# from utils.constants import LOG_FORMAT
# from utils.time import TIME_FORMAT, LOG_DATE_FORMAT
#
#
#
#
#
# def init_logger(output_dir,log_filename=None, log_level=logging.INFO):
#     '''
#     Example:
#         >>> init_logger(log_file)
#         >>> logger.info("abc'")
#     '''
#     if isinstance(log_filename,Path):
#         log_filename = str(log_filename)
#     if isinstance(log_level,str):
#         if log_level.lower()=='debug':
#             log_level = logging.DEBUG
#         elif log_level.lower() =='info':
#             log_level = logging.INFO
#         elif log_level.lower()=='warn':
#             log_level = logging.WARNING
#
#     os.makedirs(output_dir,exist_ok=True)
#     timestamp = time.strftime(TIME_FORMAT, time.localtime())
#     log_filename = f'{log_filename}_{timestamp}.log'
#     log_file_path = os.path.join(output_dir,log_filename)
#
#     log_format = logging.Formatter(fmt=LOG_FORMAT,
#                                    datefmt=LOG_DATE_FORMAT)
#     # 当没有参数时, 默认访问`root logger`
#     logger = logging.getLogger()
#     logger.setLevel(log_level)
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(log_format)
#     logger.handlers = [console_handler]
#     if log_file_path and log_file_path != '':
#         # os.makedirs(os.path.dirname(log_file),exist_ok=True)
#         file_handler = logging.FileHandler(log_file_path)
#         file_handler.setLevel(log_level)
#         file_handler.setFormatter(log_format)
#         logger.addHandler(file_handler)
#     return logger

