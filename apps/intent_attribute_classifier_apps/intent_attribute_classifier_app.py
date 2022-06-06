#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/13 13:52 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/13 13:52   wangfc      1.0         None
"""
import os
from functools import partial
from typing import Tuple, Text, Optional, Any, Dict, Union, List

from sanic import Sanic
from sanic.request import Request
from sanic import response
from urllib.parse import urljoin

from apps.responses import HsnlpResponse
from apps.sanic_apps.channel import UserMessage
from apps.sanic_apps.custom_app import CustomSanic
from apps.sanic_apps.rest import HsnlpFAQIntentAttributeClassifierInputChannel
from apps.sanic_apps.sanic_application import SanicApplication
from apps.tornado_apps.tornado_server import TornadoServer
from tornado.web import RequestHandler

from models.classifiers.regex_classifiers import PatternData, PATTERN, add_pattern_info
from utils.host import get_host_ip
from utils.sanic_utils import list_routes
from utils.io import dump_obj_as_json_to_file, read_json_file
import logging


logger = logging.getLogger(__name__)



class IntentAttributeClassifierSanicApp(SanicApplication):
    """
    意图属性的 SanicApp
    参考 rasa/core/run/serve_application()
    """

    def __init__(self, gpu_memory_per_worker="1:1024",
                 url_prefix='/hsnlp/faq/intent_attribute_service/',
                 classifier_route_name='classifier',
                 new_standard_question_classifier_route_name="new_standard_question_classifier",
                 update_pattern_data_route_name="update_pattern_data",
                 port=8016,
                 num_of_workers=1,
                 auto_reload=False,
                 pattern_data_dir=None,
                 new_pattern_data_subdir ="new_data",
                 with_preprocess_component=True,
                 hsnlp_word_segment_url=None,
                 from_tfv1=True,
                 model_path=None,
                 use_intent_attribute_regex_classifier=True,

                 ):
        super(IntentAttributeClassifierSanicApp, self).__init__(url_prefix=url_prefix,
                                                                blueprint_route_name=None,
                                                                port=port,
                                                                gpu_memory_per_worker=gpu_memory_per_worker,
                                                                num_of_workers=num_of_workers,
                                                                auto_reload=auto_reload,
                                                                model_path=model_path
                                                                )
        self.classifier_route_name = classifier_route_name
        # 增加  new_standard_question_classifier api
        self.new_standard_question_classifier_route_name = new_standard_question_classifier_route_name
        # 增加  update_regex_file api
        self.update_pattern_data_route_name = update_pattern_data_route_name
        self.update_pattern_data_uri = urljoin(self.url_prefix, self.update_pattern_data_route_name)

        self.with_preprocess_component = with_preprocess_component
        self.hsnlp_word_segment_url = hsnlp_word_segment_url

        self.from_tfv1 = from_tfv1
        self.pattern_data_dir = pattern_data_dir
        self.new_pattern_data_subdir = new_pattern_data_subdir
        if os.path.exists(self.pattern_data_dir):
            self.new_pattern_data_dir = os.path.join(self.pattern_data_dir,self.new_pattern_data_subdir)

        self.use_intent_attribute_regex_classifier = use_intent_attribute_regex_classifier

    def run(self):
        """
        启动服务
        参考: rasa.core.run.serve_application() 方法 设置 app
        Agent.handle_message() 方法
        """
        # 加载规则数据
        pattern_data = self._load_pattern_data()
        # 自动以的 Sanic 21 对象，添加 ctx 属性用于记录和更新 pattern_data
        app = CustomSanic(name=self.app_name, ctx=pattern_data)

        # 注册一个 classifier blueprint
        self._register_blueprint(app)

        # 注册一个 listener  before_server_start
        app.register_listener(
            partial(self._load_agent_on_start, self.model_path, self.gpu_memory_per_worker),
            event='before_server_start')

        # 增加注册 update_pattern_data 路由
        self._register_update_pattern_data_route(app=app)

        # 增加 注册 中间件
        self._register_middleware(app=app)

        # 打印所有 route
        listed_routes = list_routes(app=app)

        url_table = "\n".join(listed_routes[url] for url in sorted(listed_routes))

        logger.info(f"启动 Sanic 应用 {self.host}:{self.port}{self.url_prefix} with workers={self.num_of_workers}"
                    f"\n{url_table}")
        # 启动服务
        app.run(host=self.host, port=self.port, workers=self.num_of_workers, auto_reload=self.auto_reload)

    def _register_blueprint(self, app: Sanic):
        """
        @time:  2022/3/14 11:15
        @author:wangfc
        @version:
        @description: 注册 blueprint
            1) 生成一个 blueprint 作为 input channel:
            根据 route 生成一个 blueprint
            将 消息处理逻辑 agent.handle_message()  传递给 blueprint.receive handler 来处理接受的信息 request
            将 blueprint 注册到 app

            2) 注册一个 listener：
            生成 agent
            将 agent 附加到 app context 中 作为 app.agent 属性

            3) 开启服务
            request -> receive handler -> agent.handle_message() -> response
            参考 : rasa.core.channels.channel.register()

        @params:  app 为 Sanic 实例
        @return:
        """
        """

        """
        logger.debug("开始注册 blueprint")
        # 定义数据处理的 handler
        async def handler(message: UserMessage, pattern_data: Optional[PatternData] = None) -> Text:
            """
            具体 agent.handle_message() 方法处理消息的逻辑
            新增可选参数 pattern_data:
            """
            # 使用 agent.async_handle_message 异步协程来处理数据
            result = await app.agent.async_handle_message(message, pattern_data)
            return result

        # 生成 InputChannel 对象: 在 InputChannel  增加注册多个 api
        input_channel = HsnlpFAQIntentAttributeClassifierInputChannel(
            url_prefix=self.url_prefix,
            classifier_route_name=self.classifier_route_name,
            new_standard_question_classifier_route_name=self.new_standard_question_classifier_route_name,
            # update_pattern_data_route_name= self.update_pattern_data_route_name
        )

        # 使用 InputChannel 生成一个blueprint
        bp = input_channel.blueprint(on_new_message=handler)
        # 注册blueprint到app
        logger.debug("注册 blueprint到app")
        app.blueprint(bp)

    def _load_agent(self, model_path) -> "IntentAttributeInterpreter":
        """
        @time:  2022/3/14 11:15
        @author:wangfc
        @version:
        @description: 加载 agent

        @params:
         model_path 模型的路径
        @return:
        """

        logger.debug("开始加载 agent")
        from apps.intent_attribute_classifier_apps.intent_attribute_interpreter import IntentAttributeInterpreter
        from hsnlp_faq_utils.preprocess.word_segement import HSNLPWordSegmentApi
        if self.with_preprocess_component:
            # 是否使用 预处理部分
            hsnlp_word_segment_api = HSNLPWordSegmentApi(url=self.hsnlp_word_segment_url)
        else:
            hsnlp_word_segment_api = None

        intent_attribute_interpreter = IntentAttributeInterpreter(use_intent_attribute_regex_classifier=
                                                                  self.use_intent_attribute_regex_classifier,
                                                                  hsnlp_word_segment_api=hsnlp_word_segment_api,
                                                                  model_dir=model_path,
                                                                  from_tfv1=self.from_tfv1)
        return intent_attribute_interpreter

    def _get_service_url(self):
        """
        @time:  2022/3/14 11:16
        @author:wangfc
        @version:
        @description: 获取主要服务的 url 地址

        @params:
        @return:
        """
        host_name, host_ip = get_host_ip()
        service_url = f"http://{host_ip}:{self.port}{self.url_prefix}{self.classifier_route_name}"
        return service_url

    def _load_pattern_data(self) -> PatternData:
        """
        初始化加载 规则数据： 系统数据和 新增数据
        """
        # 加载系统的 规则数据
        if os.path.exists(self.pattern_data_dir):
            pattern_data = PatternData(data_dir=self.pattern_data_dir)
        # 加载新增的数据
        if os.path.exists(self.new_pattern_data_dir):
            new_pattern_data = PatternData(data_dir=self.new_pattern_data_dir)
            patterns = pattern_data.patterns + new_pattern_data.patterns
            # 优先级按照 index 索引来确定
            patterns = sorted(patterns,key=lambda x: x.index,reverse=True)
            pattern_data.patterns = patterns
        return pattern_data

    def _register_update_pattern_data_route(self, app: CustomSanic):
        """
        注册 更新规则数据的路由
        """

        @app.post(uri=self.update_pattern_data_uri)
        async def update_pattern_data(request: Request):
            """
            更新 app.ctx.pattern_data 中的数据
            """
            try:
                new_pattern_data = request.json

                pattern_data = app.ctx
                # 获取已有 pattern_data 数量
                pattern_data_num = pattern_data.__len__()
                # 增加pattern 信息
                new_pattern_data = self._add_new_patten_data_info(new_pattern_data, pattern_data_num=pattern_data_num)
                # 更新 已有存在的 pattern_data
                pattern_data.update_patterns(new_pattern_data)
                # 保存到本地
                self._save_new_pattern_data(new_pattern_data)

                hsnlp_response = HsnlpResponse()
            except Exception as e:
                logger.error(e)
                hsnlp_response = HsnlpResponse(status='failure', exception=e)
            # app.ctx = pattern_data
            output_info = hsnlp_response.output_info
            return response.json(body=output_info)

    def _add_new_patten_data_info(self, new_pattern_data: List[Dict[Text, Text]], pattern_data_num=0):
        """给更新的数据增加信息:
        index, pattern_id, time"""
        for index, new_pattern in enumerate(new_pattern_data):
            new_pattern = add_pattern_info(pattern=new_pattern, index=pattern_data_num + index + 1)
            # new_pattern_data_info.append(new_pattern)
        return new_pattern_data

    def _save_new_pattern_data(self, new_pattern_data: List[Dict],
                               new_data_dirname='new_data', filename="pattern_data.json"):
        """
        @time:  2022/3/14 11:25
        @author:wangfc
        @version:
        @description: 保存新的规则数据

        @params:
        @return:
        """
        new_pattern_data_path = os.path.join(self.pattern_data_dir, new_data_dirname, filename)
        original_new_data = []
        if os.path.exists(new_pattern_data_path):
            original_new_data = read_json_file(json_path=new_pattern_data_path)
        original_new_data.extend(new_pattern_data)
        dump_obj_as_json_to_file(filename=new_pattern_data_path, obj=original_new_data)
        return original_new_data

    def _register_middleware(self, app: CustomSanic):
        """
        @time:  2022/3/14 11:25
        @author:wangfc
        @version:
        @description: 注册 middleware 到 app

        @params:
        @return:
        """
        # 生产 request 的处理函数
        logger.debug("开始注册 middleware")

        async def get_pattern_data_before_handler(request: Request):
            """
            在中间件中使用ctx记录一些数据，这样，此中间件之后的所有流程，都能使用此数据
            """
            request.ctx.pattern_data = app.ctx

        logger.debug("注册为中间件并挂载到 request")
        app.register_middleware(get_pattern_data_before_handler, "request")





class IntentAttributeClassifierServer(TornadoServer):
    def __init__(self,
                 gpu_memory_config="1:5120",
                 url='/hsnlp/faq_intent_attribute_classifier', port=8015,
                 *args, **kwargs):
        super(IntentAttributeClassifierServer, self).__init__(gpu_memory_config=gpu_memory_config,
                                                              url=url, port=port, *args, **kwargs)
        # 初始化模型
        from apps.intent_attribute_classifier_apps.intent_attribute_classifier import IntentAttributeClassifierTfv1
        self.intent_attribute_classifier = IntentAttributeClassifierTfv1()

    def _build_url_handler_tuple(self) -> Tuple[Text, RequestHandler, Optional[Dict[Text, Any]]]:
        from apps.intent_attribute_classifier_apps.tornado_handlers import IntentAttributeClassifierHandler
        kwargs = dict(classifier=self.intent_attribute_classifier)
        return [(self.url, IntentAttributeClassifierHandler, kwargs)]




def run_intent_attribute_classifier_app(run_tornado_app=False, gpu_memory_config="1:1024", num_processes=1,
                                        port=8015,
                                        use_intent_attribute_regex_classifier=True,
                                        hsnlp_word_segment_url=None):
    # 设置环境变量
    if run_tornado_app:
        server = IntentAttributeClassifierServer(gpu_memory_config=gpu_memory_config,
                                                 num_processes=num_processes,
                                                 port=port)
        server.run()
    else:
        app = IntentAttributeClassifierSanicApp(gpu_memory_per_worker=gpu_memory_config,
                                                num_of_workers=num_processes,
                                                port=port,
                                                use_intent_attribute_regex_classifier=use_intent_attribute_regex_classifier,
                                                hsnlp_word_segment_url=hsnlp_word_segment_url
                                                )
        app.run()

if __name__ == '__main__':
    run_intent_attribute_classifier_app(run_tornado_app=False, gpu_memory_config="1:512", num_processes=1)
