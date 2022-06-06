#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@file: event_classifier_server.py
#@version: 02
#@author: $ {USER}
#@contact: wangfc27441@***
#@time: 2019/10/22 17:03 
#@desc:

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "event_classifier_server"))

import socket
import tornado
from tornado.options import define,options
from tornado.escape import json_decode
import tensorflow as tf
from .event_classifier_server.conf.config_parser import GPU_NO,GPU_MEMORY_FRACTION, DE_BUG, PORT,SERVICE_URL_DIR,VERSION_URL_DIR



if GPU_NO != '-1':
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NO
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION)
    tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from .event_classifier_server.handlers.classify import classifier_handlers
from .event_classifier_server.handlers.base import NotFoundHandler
import logging
logger = logging.getLogger(__name__)


def init_handlers(classifier_handlers, NotFoundHandler):
    """
    @author:wangfc27441
    @time:  2019/10/25  11:22
    @desc:  路由信息映射元组的列表，用于Application对象初始化，
            每个handler 处理对应特定HTTP请求的响应 :

    :param classifier_handlers:
    :param NotFoundHandler:
    :return:
    """
    h = []
    h.extend(classifier_handlers)
    h.extend([(r'(.*)', NotFoundHandler)])
    return h


class Application(tornado.web.Application):
    """
    @author:wangfc27441
    @time:  2019/10/25  11:25
    @desc: 创建 Tornado 的 Application 类
    """
    # 初始化 Application 对象
    def __init__(self,handlers, settings):
        super(Application,self).__init__(handlers,**settings)


def main():
    # 1. 初始化 路由信息映射元组的列表
    handlers = init_handlers(classifier_handlers, NotFoundHandler)

    # 初始化 web.Application 的配置
    setting = dict(
        title = 'HSNLP',
        autoreload=True, # 自动加载主要用于开发和测试阶段，要不每次修改，都重启tornado服务 ,
        debug= DE_BUG, # 通过 option 参数 debug 来选择是否autoreload
        login_url = '/login'
    )

    # 2. 构建一个 Application 对象的实例
    # 初始化接收的第一个参数就是一个路由信息映射元组的列表
    application = Application(handlers,setting)

    # 定义 HTTPServer 监听HTTP请求的端口
    define('port',default = PORT, help = 'run on the given port',type=int)

    # 使用 tornado 的 options 模块解析命令行：转换命令行参数，并将转换后的值对应的设置到全局options对象相关属性上
    tornado.options.parse_command_line()

    # 3. 创建HTTPServer 对象：将 Application实例传递给 Tornado 的  HTTPServer 对象
    http_server = tornado.httpserver.HTTPServer(application)

    # 获取本机计算机名称
    hostname = socket.gethostname()
    # 获取本机ip
    ip = socket.gethostbyname(hostname)

    # 启动单进程服务
    if GPU_NO=='-1':
        mode = 'cpu'
    else:
        mode = 'gpu:'+ GPU_NO

    num_processes = 1

    # 4. 启动IOLoop

    logger.info('启动 bert 单进程服务 num_processes={} with mode={} at url={}:{}{}'.format(num_processes, mode, ip,options.port,SERVICE_URL_DIR))
    # 服务器绑定到指定端口
    http_server.bind(options.port)   # http_server.listen()这个方法只能在单进程模式中使用
    # 开启num_processes进程（默认值为1）
    http_server.start(num_processes=num_processes)
    # 启动当前线程的IOLoop
    tornado.ioloop.IOLoop.current().start()
    # tornado.ioloop.IOLoop.instance().start()


    # 启动高级多进程 HTTPServer 服务
    # logger.info('启动 bert 高级多进程服务 with mode={} at url={}:{}{}'.format(mode, ip, options.port, SERVICE_URL_DIR))
    # sockets = tornado.netutil.bind_sockets( options.port)
    # # 启动多进程
    # tornado.process.fork_processes(num_processes = num_processes)
    # http_server.add_sockets(sockets)
    # # IOLoop.current() : 返回当前线程的IOLoop实例
    # # IOLoop.start():  启动IOLoop实例的I / O循环, 同时服务器监听被打开。
    # tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(e,exc_info=True)