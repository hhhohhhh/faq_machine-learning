#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/10/20 9:23 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/20 9:23   wangfc      1.0         None
"""
import os
from typing import Text, Any
import asyncio
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.httpclient

import urllib
import json
import datetime
import time

from tornado import httputil

from rasa.core.interpreter import RasaNLUInterpreter
import  logging
logger = logging.getLogger(__name__)


class NLUParseHandler(tornado.web.RequestHandler):

    """
    当你使用@tornado.web.asynchonous装饰器时，Tornado永远不会自己关闭连接。
    你必须在你的RequestHandler对象中调用finish方法来显式地告诉Tornado关闭连接。
    （否则，请求将可能挂起，浏览器可能不会显示我们已经发送给客户端的数据。）

    """
    # def __init__(self,application: "Application",
    #     request: httputil.HTTPServerRequest,
    #     **kwargs: Any):
    #     super.__init__(application,request,**kwargs)
    #     model_path = kwargs.get('model_path',None)
    #     if model_path is not None:
    #         logger.info(f"初始化模型")

    def initialize(self, interpreter:RasaNLUInterpreter=None):
        self.interpreter = interpreter
        logger.info(f"初始化对象 {self.__class__.__name__}的 interpreter 属性 ,pid={os.getpid()}")


    async def post(self,url="https://www.baidu.com/s?wd=pants"):
        data = tornado.escape.json_decode(self.request.body)
        message = data.get("message")
        if self.interpreter is not None:
            result = await self.interpreter.parse(message)
        else:
            logger.info(f"self.interpreter 为空")
            result = message
        # response = await self.asynchronous_fetch(url)
        # body = tornado.escape.xhtml_unescape(response.body)
        # result_count = len(body)
        # now = datetime.datetime.utcnow()
        # self.write("""
        # <div style="text-align: center">
        #     <div style="font-size: 72px">%s</div>
        #     <div style="font-size: 144px">%.02f</div>
        #     <div style="font-size: 24px">tweets per second</div>
        # </div>""" % (message,result_count))
        self.write(json.dumps(result))
        self.finish()


    # @tornado.web.asynchronous
    # def post(self):
    #     data = tornado.escape.json_decode(self.request.body)
    #     # json_string = self.request.body.decode('utf-8')
    #     # data = json.loads(json_string)


    async def asynchronous_fetch(self,url):
        http_client = tornado.httpclient.AsyncHTTPClient()
        response = await http_client.fetch(url)
        return response


async def load_interpreter_on_start(model_path: Text, gpu_memory_per_worker: int) -> RasaNLUInterpreter:
    """
    参考：  rase.core.run.load_agent_on_start() 方法
           rasa.core.agent.Agent.load_local_model()
    如果使用 interpreter 赋值给 app的时候，当worker >1 的时候无法运行
    因此，使用 model_path 后在各个的进程当中解压模型，可能会成功

    """
    from utils.tensorflow.environment import setup_tf_environment
    setup_tf_environment(gpu_memory_config=None)
    from servers.rasa_servers.rasa_application import RasaApplication
    import rasa

    tmp_model_dir, core_model, nlu_model_dir = RasaApplication.get_tmp_nlu_and_core_model_dir(models_dir=model_path)
    interpreter = rasa.core.interpreter.create_interpreter(nlu_model_dir)
    interpreter.tmp_model_dir = tmp_model_dir
    app.interpreter = interpreter
    return app.interpreter
