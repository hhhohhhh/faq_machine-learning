#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: train
@version: 02

@author: wangfc
@site: http://www.***
@time: 2019/7/16 15:50

 * 密级：秘密
 * 版权所有：***股份有限公司 2019
 * 注意：本内容仅限于***股份有限公司内部传阅，禁止外泄以及用于其他的商业目的
 * logger日志参数解析和配置模块
"""
import os
import configparser
import logging

# def get_logger(log_file):
#     """
#     @author:wangfc27441
#     @time:  2019/10/22  15:39
#     @desc: 自定义的logger
#     """
#
#     # 读取LOG配置文件
#     logger_cfg = configparser.ConfigParser()
#     logger_cfg.read(log_file, encoding='utf-8')
#     # 读取LOG配置参数
#     logger_level= logger_cfg.get('tornado','level')
#     logfilename= logger_cfg.get('tornado', 'log_file_prefix')
#
#     # 设置logger
#     # 定义log文件名称
#     logfile_path = os.path.join(os.getcwd(),'logs',logfilename)
#
#     # 初始化记录器
#     logger= logging.getLogger()
#     # 定义logger的输出级别
#     logger.setLevel(logger_level)
#     # logger.setLevel(logger.INFO)
#
#     # 定义格式
#     format = logging.Formatter('%(asctime)s %(module)s %(funcName)s [line:%(lineno)d] [%(levelname)s] - %(message)s')
#     # 处理程序对象Handler将（记录器产生的）日志记录发送至合适的目的地。
#     # 文件输出
#     os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
#     file_handler = logging.FileHandler(logfile_path, 'w', 'utf-8')
#     file_handler.setFormatter(format) # or whatever
#     # 从记录器对象中添加处理程序对象。
#     logger.addHandler(file_handler)
#
#     # 屏幕打印输出
#     stream_handler = logging.StreamHandler()
#     stream_handler.setFormatter(format)
#     logger.addHandler(stream_handler)
#     return logger
#
# # 日志配置文件 global LOG_FILE
# LOG_FILE = os.path.join('event_classifier_server','conf','logging.cfg')
#
# logger = get_logger(LOG_FILE)
