#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/10/20 9:50 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/20 9:50   wangfc      1.0         None
"""
import os
from typing import Optional, List, Type, Any

import tornado
from tornado.options import define, options
from tornado.routing import (
    AnyMatches,
    DefaultHostMatches,
    HostMatches,
    ReversibleRouter,
    Rule,
    ReversibleRuleRouter,
    URLSpec,
    _RuleList,
)

from rasa.core.interpreter import RasaNLUInterpreter
from .tornado_handlers import NLUParseHandler
from utils.host import get_host_ip
import logging
logger = logging.getLogger(__name__)

def create_tornado_server(port,route=r'/nlu/parse/tornado',num_processes=10,interpreter:RasaNLUInterpreter=None):
    """
    @time:  2021/10/20 9:50
    @author:wangfc
    @version:
    @description:

    @params:
    @return:
    """

    # 定义 HTTPServer 监听HTTP请求的端口
    define('port',default = port, help = 'run on the given port',type=int)

    # 使用 tornado 的 options 模块解析命令行：转换命令行参数，并将转换后的值对应的设置到全局options对象相关属性上
    tornado.options.parse_command_line()

    # app = tornado.web.Application([(route,NLUParseHandler)])
    # 试图讲 interpreter 对象传递给 handler对象，作为其 interpreter 属性，但是多进程的时候会卡死
    handlers = [(route,NLUParseHandler,dict(interpreter=interpreter))]
    app = NLPApplication(handlers=handlers)


    # 创建HTTPServer 对象：将 Application实例传递给 Tornado 的  HTTPServer 对象
    http_server = tornado.httpserver.HTTPServer(app)

    # 服务器绑定到指定端口
    # app.listern(port)
    http_server.bind(options.port)   # http_server.listen()这个方法只能在单进程模式中使用
    # 开启num_processes进程（默认值为1）
    http_server.start(num_processes=num_processes)

    host = get_host_ip()
    logger.info(f"开启tornado模型：route = {host}:{port}{route}, pid={os.getpid()}")
    tornado.ioloop.IOLoop.current().start()


class NLPApplication(tornado.web.Application):
    """
    @time:  2021/10/20 15:50
    @author:wangfc
    @version:
    @description:

    @params:
    @return:
    """
    # 初始化 Application 对象
    def __init__(self,handlers: Optional[_RuleList] = None,
        default_host: Optional[str] = None,
        transforms: Optional[List[Type["OutputTransform"]]] = None,
        **settings: Any):
        super(NLPApplication,self).__init__(handlers,default_host,transforms,**settings)
        logger.info(f"初始化 Application 对象, pid={os.getpid()}")
