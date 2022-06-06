#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/10/19 9:14 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/19 9:14   wangfc      1.0         None
"""
import time
from typing import Text, Callable, Awaitable, Any, Optional
from sanic import Sanic, Blueprint
from sanic.request import Request
from sanic.response import HTTPResponse,json
from sanic import response
from rasa.core.channels import RestInput


import logging
logger = logging.getLogger(__name__)


def register_interpreter(app:Sanic):
    """
    # 参考 : rasa.core.channels.channel.register()

    1) 生成一个 blueprint 作为 input channel:
    根据 route 生成一个 blueprint
    将 消息处理逻辑 agent.handle_message()  传递给 blueprint.receive handler 来处理接受的信息 request
    将 blueprint 注册到 app

    2) 注册一个 listener：
    生成 agent
    将 agent 附加到 app context 中 作为 app.agent 属性

    3) 开启服务
    request -> receive handler -> agent.handle_message() -> response

    """
    async def handler(message=None) -> Text:
        """
        具体 agent 处理消息的逻辑
        """
        result = await app.interpreter.parse(message)
        return result

    interpreter_input_channel = InterpreterInputChannel()
    bp = interpreter_input_channel.blueprint(handle_message=handler)
    app.blueprint(bp)






class InterpreterInputChannel(RestInput):
    """
    参考 : rasa.core.channels.rest.RestInput.blurprint()
    """
    def __init__(self,url_prefix='/nlu',blueprint_uri='/parse'):
        super(InterpreterInputChannel,self).__init__()
        self.url_prefix= url_prefix
        self.blueprint_uri = blueprint_uri

    def blueprint(self, handle_message: Callable[[Text], Awaitable[Any]]):
        """
        生成一个 blueprint
        handler_message: 是处理 message 的逻辑
        """
        interpreter_blueprint = Blueprint(name='interpreter_blueprint', url_prefix=self.url_prefix)

        # 对 blueprint 增加 一个 route + handler 来处理信息
        @interpreter_blueprint.route(uri=self.blueprint_uri, methods=['POST'])
        async def parse(request: Request) -> HTTPResponse:
            """
            处理 request 请求 ：  json = {“sender”:,"message":}
            """
            # sender = await self._extract_sender(req=request)

            start_time = time.time()
            text = await self._extract_message(req=request)
            # 使用 消息处理逻辑 处理消息
            result = await handle_message(text)
            logger.debug(f"解析数据text = {text},cost={time.time()- start_time}")
            return response.text(result)

        return interpreter_blueprint


    # 参考 rasa.core.channels.rest.RestInput._extract_message() 方法
    async def _extract_message(self, req: Request) -> Optional[Text]:
        return req.json.get("message", None)
