#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/14 9:17 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/14 9:17   wangfc      1.0         None
"""
from typing import Text,Optional
from functools import partial
from asyncio import AbstractEventLoop
from sanic import Sanic
from apps.sanic_apps.channel import UserMessage
from apps.sanic_apps.rest import HsnlpInputChannel
from utils.sanic_utils import list_routes



import logging
logger = logging.getLogger(__name__)



class BasicAgent():
    def __init__(self, model_path):
        pass

    def handle_message(self):
        raise NotImplementedError



class SanicApplication():
    def __init__(self,
                 app_name='snaic_app',
                 url_prefix='/hsnlp/',
                 blueprint_route_name="classifier",
                 host="0.0.0.0", port=8015,
                 num_of_workers=1,
                 auto_reload=True,
                 model_path = None,
                 gpu_memory_per_worker=None):
        self.app_name = app_name
        self.url_prefix = url_prefix
        self.blueprint_route_name = blueprint_route_name
        self.host = host
        self.port = port
        self.num_of_workers = num_of_workers
        self.auto_reload = auto_reload
        self.model_path = model_path
        self.gpu_memory_per_worker = gpu_memory_per_worker

    def run(self):
        """
        参考: rasa.core.run.serve_application() 方法 设置 app
        Agent.handle_message() 方法
        """
        app = Sanic(name=self.app_name)

        # 注册一个 classifier blueprint
        self._register_blueprint(app)

        # 注册一个 listener  before_server_start
        app.register_listener(
            partial(self._load_agent_on_start, self.model_path, self.gpu_memory_per_worker),
            event='before_server_start')

        # noinspection PyUnresolvedReferences
        # async def clear_model_files(_app: Sanic, _loop: Text) -> None:
        #     if app.interpreter.tmp_model_dir:
        #         logger.info(f"删除路径下的文件{_app.interpreter.tmp_model_dir}")
        #         shutil.rmtree(_app.interpreter.tmp_model_dir)
        #
        # app.register_listener(clear_model_files, "after_server_stop")

        # 打印所有 route
        listed_routes = list_routes(app=app)

        url_table = "\n".join(listed_routes[url] for url in sorted(listed_routes))
        logger.info(url_table)
        logger.info(f"启动 Sanic 应用 {self.host} :{self.port}{self.url_prefix} with workers={self.num_of_workers}")

        app.run(host=self.host, port=self.port, workers=self.num_of_workers)



    def _register_blueprint(self, app: Sanic):
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

        async def handler(message:UserMessage) -> Text:
            """
            具体 agent.handle_message() 方法处理消息的逻辑
            """
            # 使用 agent.async_handle_message 异步协程来处理数据
            result = await app.agent.async_handle_message(message)
            return result

        # 生成 InputChannel 对象: 在 InputChannel  增加注册多个 api
        input_channel = HsnlpInputChannel(url_prefix=self.url_prefix,
                                          blueprint_route_name=self.blueprint_route_name)
        # 使用 InputChannel 生成一个blueprint
        bp = input_channel.blueprint(on_new_message=handler)
        # 注册blueprint到app
        app.blueprint(bp)


    async def _load_agent_on_start(self,model_path: Optional[Text] = None, gpu_memory_per_worker: int = None,
                                  app: Sanic = None, loop: AbstractEventLoop = None) -> BasicAgent:
        """
        参考：  rase.core.run.load_agent_on_start() 方法
               rasa.core.agent.Agent.load_local_model()
        如果使用 interpreter 赋值给 app的时候，当worker >1 的时候无法运行
        因此，使用 model_path 后在各个的进程当中解压模型，可能会成功

        """
        logger.debug("设置 tensorflow 环境变量")
        from utils.tensorflow.environment import setup_tf_environment
        setup_tf_environment(gpu_memory_config=gpu_memory_per_worker)
        logger.debug("加载 agent")
        agent = self._load_agent(model_path)
        app.agent = agent
        return app.agent

    # @classmethod 改成 instance method
    def _load_agent(self,model_path)->BasicAgent:
        raise NotImplementedError



if __name__ == '__main__':
    pass