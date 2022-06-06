#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/14 9:20 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/14 9:20   wangfc      1.0         None
"""

from typing import Text
from py2neo import Graph
from functools import partial
from asyncio import AbstractEventLoop
from sanic import Sanic
from sanic.response import HTTPResponse, json, text

import rasa
from apps.rasa_apps.blueprints import InterpreterInputChannel
from models.kbqa_models.medical_diagnosis_agent import MedicalDiagnosisAgent
from rasa.core.utils import list_routes
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from apps.rasa_apps.rasa_application import NLUApplication, RasaApplication

import logging
logger = logging.getLogger(__name__)


class MedicalDiagnosisApplication(NLUApplication):
    def __init__(self, graph: Graph,
                 model_path: Text = None,
                 interpreter: NaturalLanguageInterpreter = None,
                 name='nlu_app', route='/nlu/parse/',
                 host="0.0.0.0", port=9001,
                 num_of_workers=1,
                 gpu_memory_per_worker=None
                 ):
        super(MedicalDiagnosisApplication, self).__init__(model_path, interpreter, name,
                                                          host=host, port=port, num_of_workers=num_of_workers,
                                                          gpu_memory_per_worker=gpu_memory_per_worker)
        self.graph = graph

    def create_app(self):
        """
        参考:rasa.core.run.serve_application() 方法 设置 app
        Agent.handle_message() 方法
        """
        app = Sanic(name=self.name)
        # 注册一个 blueprint
        register_diagnosis_agent(app)
        # 注册一个 listener  before_server_start
        app.register_listener(
            partial(load_diagnosis_agent_on_start, self.graph, self.model_path, self.gpu_memory_per_worker),
            event='before_server_start')

        # noinspection PyUnresolvedReferences
        async def clear_model_files(_app: Sanic, _loop: Text) -> None:
            if app.diagnosis_agent.tmp_model_dir:
                logger.info(f"删除路径下的文件{_app.diagnosis_agent.tmp_model_dir}")
                shutil.rmtree(_app.diagnosis_agent.tmp_model_dir)

        app.register_listener(clear_model_files, "after_server_stop")
        # 打印所有 route
        listed_routes = list_routes(app=app)
        url_table = "\n".join(listed_routes[url] for url in sorted(listed_routes))
        logger.info(url_table)
        logger.info(
            f"启动 medical diagnosis agent as a sanic application on {self.host}:{self.port} with workers={self.num_of_workers}")
        app.run(host=self.host, port=self.port, workers=self.num_of_workers)



def register_diagnosis_agent(app:Sanic,
                             url_prefix='/diagnosis_agent',
                             blueprint_uri='/handle_message'):
    """

    """
    async def handler(message=None) -> Text:
        """
        具体 agent 处理消息的逻辑
        """
        response = await app.diagnosis_agent.handle_message(message)
        return response

    interpreter_input_channel = InterpreterInputChannel(url_prefix=url_prefix,blueprint_uri=blueprint_uri)
    bp = interpreter_input_channel.blueprint(handle_message=handler)
    app.blueprint(bp)


async def load_diagnosis_agent_on_start(graph, model_path: Text, gpu_memory_per_worker: int,
                                        app: Sanic, loop: AbstractEventLoop) -> MedicalDiagnosisAgent:
    from utils.tensorflow.environment import setup_tf_environment
    setup_tf_environment(gpu_memory_config=gpu_memory_per_worker)
    tmp_model_dir, core_model, nlu_model_dir = RasaApplication.get_tmp_nlu_and_core_model_dir(models_dir=model_path)
    interpreter = rasa.core.interpreter.create_interpreter(nlu_model_dir)
    interpreter.tmp_model_dir = tmp_model_dir
    diagnosis_agent = MedicalDiagnosisAgent(graph=graph, tmp_model_dir=tmp_model_dir, nlu_interpreter=interpreter, )
    app.diagnosis_agent = diagnosis_agent
    return app.diagnosis_agent
