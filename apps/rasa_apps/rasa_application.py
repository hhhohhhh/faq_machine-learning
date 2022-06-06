#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/10/18 11:13 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/18 11:13   wangfc      1.0         None
"""
import os
import time
from collections import OrderedDict
from pathlib import Path
import argparse

import tqdm
import pandas as pd

from data_process.dataset.hsnlp_faq_knowledge_dataset import ATTRIBUTE, FIRST_KNOWLEDGE
from rasa.shared.nlu.constants import INTENT, OUTPUT_INTENT, PREDICTED_CONFIDENCE_KEY, TEXT
from sklearn.metrics import classification_report
from functools import partial
from typing import Text
import shutil

import asyncio
from asyncio import AbstractEventLoop
from sanic import Sanic

import rasa
from rasa.core.interpreter import RasaNLUInterpreter
from rasa.model import get_local_model, get_model, get_model_subdirectories
from rasa.core.agent import Agent
from rasa.core.utils import list_routes
from rasa.nlu.model import Interpreter
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter

from apps.server_requests import post_data, post_rasa_data, parse_rasa_intent_result, parse_rasa_nlu_result, \
    parse_rasa_attribute, parse_rasa_entity_result
from data_process.nlu_constants import STANDARD_QUESTION, INTENT_COLUMN, QUESTION_COLUMN
from utils.host import get_host_ip
from utils.io import dump_obj_as_json_to_file, dataframe_to_file, read_json_file
from py2neo import Graph
from .blueprints import register_interpreter
import logging

# from rasax.community.constants import *
from ..medical_diagnosis_apps.medical_diagnosis_app import MedicalDiagnosisApplication

logger = logging.getLogger(__name__)


# class RasaApplication():
#     def __init__(self, trained_nlu_model: Text= None,model_dir:Text=None,
#                  port:int=5005):
#         self.model_dir = model_dir
#         self.trained_nlu_model = trained_nlu_model
#         self.port = port
#
#
#     async def build_nlu_app(self):
#         # 创建 agent
#         nlu_agent = await load_agent(model_path=self.trained_nlu_model)
#
#         # 创建 nlu_server
#         app = server.create_app(agent=nlu_agent)
#         channel.register([RestInput()], app, "/webhooks/")
#
#         # 创建 nlu_app
#         # nlu_app = rasa_app_nlu(rasa_nlu_server=nlu_server)
#         nlu_app = app.asgi_client
#         return nlu_app
#
#     def run_application(self):
#         """
#         @time:  2021/10/8 9:46
#         @author:wangfc
#         @version:
#         @description:
#         参考 ： rasa.__main__.create_argument_parser()
#
#         @params:
#         @return:
#         """
#         from rasa.cli import run
#         from rasa.cli.arguments import run as arguments
#
#         run_parser = argparse.ArgumentParser(description="创建 rasa server")
#         arguments.set_run_arguments(parser=run_parser)
#         args = run_parser.parse_args(['--model', self.model_dir,'--port',str(self.port)])
#         rasa.cli.run.run(args)


class RasaApplication():
    def __init__(self, taskbot_dir: Path = None,
                 config_filename='config.yml',
                 models_subdir='models',
                 model_extract_directory_name='model_tmp',
                 trained_model_path: Path = None,
                 if_build_nlu_interpreter=False,
                 if_build_nlu_application=False,
                 if_build_agent_application=False,
                 if_build_medical_diagnosis_application=False,
                 if_enable_api=True,
                 output_with_nlu_parse_data=False,
                 host=None,
                 agent_port=5005,
                 num_of_workers=1,
                 gpu_memory_per_worker=None,
                 **kwargs
                 ):
        self.taskbot_dir = taskbot_dir
        self.config_filename = config_filename
        self.models_subdir = models_subdir
        self.model_extract_directory_name = model_extract_directory_name
        self.nlu_interpreter: Interpreter = None
        self.agent: Agent = None
        self.host = host if host is not None else get_host_ip()
        self.num_of_workers = num_of_workers
        self.gpu_memory_per_worker = gpu_memory_per_worker
        self.agent_route = "/webhooks/rest/webhook/"
        self.agent_port = agent_port
        self.nlu_interpreter_route = "/nlu/parse/"
        self.nlu_interpreter_parsing_port = 9001
        self.rasa_rest_server_url = f"http://{self.host}:{self.agent_port}{self.agent_route}"
        self.rasa_nlu_interpreter_url = f"http://{self.host}:{self.nlu_interpreter_parsing_port}{self.nlu_interpreter_route}"
        # 'http://10.20.33.3:5005/webhooks/rest/webhook'

        if self.taskbot_dir is not None:
            self.config_path = os.path.join(self.taskbot_dir, self.config_filename)
            self.models_dir = os.path.join(self.taskbot_dir, self.models_subdir)
            self.working_directory = os.path.join(self.taskbot_dir, self.model_extract_directory_name)
            self.model_local_path = get_local_model(self.models_dir)

        self.output_with_nlu_parse_data = output_with_nlu_parse_data
        self.if_build_nlu_interpreter = if_build_nlu_interpreter
        self.if_build_nlu_application = if_build_nlu_application
        self.if_build_agent_application = if_build_agent_application
        self.if_build_medical_diagnosis_application = if_build_medical_diagnosis_application

        self.kwargs = kwargs

        if self.if_build_nlu_interpreter:
            self.nlu_interpreter = self.create_nlu_interpreter()

        elif if_build_nlu_application:
            self.build_nlu_application()
        elif if_build_agent_application:
            self.build_agent_application(if_enable_api, port=self.agent_port)

        elif if_build_medical_diagnosis_application:
            self.build_medical_diagnosis_application()

    def create_interpreter(self) -> Interpreter:
        """
        参考 ：
        rasa.nlu.model.Interpreter 对象 生成方式
        1) rasa.cli.shell.shell_nlu


        {'text': '帮我开通创业版',
         'intent': {'id': -7479419365565818984, 'name': '开户咨询', 'confidence': 1.0},
         'entities': [{'entity': 'block',
           'start': 4,
           'end': 7,
           'confidence_entity': 0.9917227625846863,
           'value': '创业版',
           'extractor': 'DIETClassifier'}],
         'intent_ranking': [{'id': -7479419365565818984,
           'name': '开户咨询',
           'confidence': 1.0},
          {'id': -1140346640274628313, 'name': '软件下载', 'confidence': 0.0},
          {'id': -1479861286211517141, 'name': '转人工', 'confidence': 0.0},
          {'id': -4071793577210808117, 'name': '账号和密码都忘记了', 'confidence': 0.0},
          {'id': 4569455476194073243, 'name': '查账户状态', 'confidence': 0.0},
          {'id': -8935007165621339853, 'name': '查行情', 'confidence': 0.0},
          {'id': 8763397788308059026, 'name': '查手续费', 'confidence': 0.0},
          {'id': 8440622777159744395, 'name': '查开户营业部', 'confidence': 0.0},
          {'id': -4137195239777861876, 'name': '查开户日期', 'confidence': 0.0},
          {'id': -3140996479061972612, 'name': '查就近营业部', 'confidence': 0.0}],
         'response_selector': {'all_retrieval_intents': ['软件下载'],
          'default': {'response': {'id': 2423077775077337445,
            'responses': [{'text': 'e海通财下载方式：苹果手机打开“App Store”搜索“e海通财”、安卓手机打开应用市场搜索“e海通财”下载安装。如需短信发送到手机请说短信发送'}],
            'response_templates': [{'text': 'e海通财下载方式：苹果手机打开“App Store”搜索“e海通财”、安卓手机打开应用市场搜索“e海通财”下载安装。如需短信发送到手机请说短信发送'}],
            'confidence': 1.0,
            'intent_response_key': '软件下载/手机软件下载',
            'utter_action': 'utter_软件下载/手机软件下载',
            'template_name': 'utter_软件下载/手机软件下载'},
           'ranking': [{'id': 2423077775077337445,
             'confidence': 1.0,
             'intent_response_key': '软件下载/手机软件下载'},
            {'id': 7843697584334410241,
             'confidence': 0.0,
             'intent_response_key': '软件下载/电脑软件下载'}]}}}

        """
        # 获取 nlu 模型的dir
        # core_model , nlu_model_dir = get_model_subdirectories(tmp_working_dir)
        models_dir, core_model, nlu_model_dir = self._get_tmp_nlu_and_core_model_dir()

        # 创建 interpreter
        interpreter = Interpreter.load(model_dir=nlu_model_dir)
        result = interpreter.parse(u"帮我开通创业版")

        return interpreter

    def create_nlu_interpreter(self) -> RasaNLUInterpreter:
        """
        参考： rasa.core.run.load_agent_on_start() 方法 创建 RasaNLUInterpreter 对象
        """
        models_dir, core_model, nlu_model_dir = self.get_tmp_nlu_and_core_model_dir(self.models_dir)
        _interpreter = rasa.core.interpreter.create_interpreter(obj=nlu_model_dir)
        return _interpreter

    def build_nlu_application(self, app_name='nlu_interpreter'):
        # _interpreter = self.create_nlu_interpreter()
        nlu_application = NLUApplication(model_path=self.models_dir, interpreter=None,
                                         num_of_workers=self.num_of_workers,
                                         gpu_memory_per_worker=self.gpu_memory_per_worker)
        nlu_application.create_app()
        # nlu_application.create_tornado_app()

    def build_medical_diagnosis_application(self):

        # 创建 graph
        neo4j_database_config = self.kwargs.get('neo4j_database_config')
        graph = Graph(**neo4j_database_config)

        medical_diagnosis_application = MedicalDiagnosisApplication(
            graph=graph,
            model_path=self.models_dir, interpreter=None,
            num_of_workers=self.num_of_workers,
            gpu_memory_per_worker=self.gpu_memory_per_worker)
        medical_diagnosis_application.create_app()

    def build_agent_application(self, if_enable_api, port=5005):
        """
        参考： rasa run / rasa shell command
        rasa.cli.run.run() : 运行一个 sanic server
        rasa.cli.shell.shell():  运行一个 本地的 sanic server
        rasa.api.run()
        rasa.core.run.serve_application()
        rasa.core.run.load_agent_on_start()
        rasa.core.agent.Agent.handle_text()

        """
        # from rasa.core.agent import Agent
        # from rasa.core.interpreter import RasaNLUInterpreter
        from rasa.cli import run
        from rasa.cli.arguments import run as arguments

        # tmp_model_dir, core_model, nlu_model_dir = self.get_tmp_nlu_and_core_model_dir(models_dir=self.models_dir)
        # 通过 model_dir 加载 agent
        # agent = Agent.load(tmp_model_dir)
        # 解析 text
        # response = agent.handle_text("hello")
        # result = agent.handle_text(u"帮我开通创业版")
        # self.agent = agent

        # 通过 model_dir 加载 sever
        run_parser = argparse.ArgumentParser(description="创建 rasa server")
        arguments.set_run_arguments(parser=run_parser)

        args_list = ['--model', self.models_dir, '--port', str(port)]
        if if_enable_api:
            args_list.extend(['--enable-api'])

        args = run_parser.parse_args(args_list)

        # 使用  output_with_nlu_parse_data 控制 agent 输出 nlu parse 数据
        # agent.handle_message
        args.__setattr__('output_with_nlu_parse_data', self.output_with_nlu_parse_data)
        rasa.cli.run.run(args)

    def predict(self, index, question,
                agent_application=True, nlu_application=False):
        if agent_application:
            """
            使用 rasa 的rest channel 进行服务
            """
            data = {
                "sender": f"test_{index}",
                "message": question
            }
            results = post_data(url=self.rasa_rest_server_url, json_data=data)
        elif nlu_application:
            data = {"message": question}
            results = post_data(url=self.rasa_nlu_interpreter_url, json_data=data)
        elif self.agent:
            task = self.agent.handle_text(question)
            # 最简单的形式，产生一个  coroutine 对象，运行 run得到结果,但是 这样运行的速度很慢，可能是所有的结果都在一个 session
            results = asyncio.run(task)
            # 或者 gather task 之后一起运行,
            # 但是这样做是的 成为一个同一个 conversation_id 对应的很长的对话
            # tasks.append(task)

            # async def get_results():
            #     return await asyncio.gather(*tasks)
            # results_ls = asyncio.run(get_results())

        elif self.nlu_interpreter:
            results = self.agent.parse(question)
        return results

    def evaluate(self, test_data: pd.DataFrame,
                 test_data_dir,
                 text_column=QUESTION_COLUMN,
                 standard_question_column=STANDARD_QUESTION,

                 predict_intent_column='predict_intent',
                 predict_intent_confidence_column=PREDICTED_CONFIDENCE_KEY,
                 output_intent_column=OUTPUT_INTENT,
                 output_intent_confidence_column=f"{OUTPUT_INTENT}_{PREDICTED_CONFIDENCE_KEY}",
                 predict_attribute_column="predict_attribute",
                 intent_column=INTENT_COLUMN,
                 response_text_key=TEXT,
                 tag_column='tag',
                 agent_application=False,
                 nlu_application=True,
                 debug=False
                 ):
        test_data = test_data.copy()
        if debug:
            test_data = test_data.iloc[:10]

        tested_data_path = os.path.join(test_data_dir, 'tested_data.csv')
        results_ls_path = os.path.join(test_data_dir, 'results_ls.json')

        if agent_application:
            url = self.rasa_rest_server_url
        elif nlu_application:
            url = self.rasa_nlu_interpreter_url

        data_result_ls = []
        results_ls = []

        start_time = time.time()

        load_results_success = False
        if os.path.exists(results_ls_path):
            results_ls = read_json_file(results_ls_path)
            assert results_ls.__len__() == test_data.__len__()
            load_results_success = True

        for index in tqdm.trange(test_data.__len__()):
            single_test_data = test_data.iloc[index]
            question = single_test_data.loc[text_column]
            if load_results_success:
                assert results_ls[index]["question"] == question
                # 获取存储的 results
                results = results_ls[index]["results"]
                question_loaded = results_ls[index]["question"]
                assert question_loaded == question
            else:
                results = post_rasa_data(index, question, url=url)
                results_ls.append({'results': results, 'question': question})

            # 获取 text 对应的标准问
            if results:
                try:
                    text = results[0][response_text_key]
                except:
                    logger.error(f"index={index},question={question},text 获取失败,results={results}")
                    text = None

                # 获取大意图
                output_intent, output_intent_confidence = parse_rasa_intent_result(results,
                                                                                   output_intent_key=OUTPUT_INTENT)
                # 获取模型意图
                pred_intent, intent_confidence = parse_rasa_intent_result(results=results, output_intent_key=INTENT)

                # 获取 attribute
                pred_attribute = parse_rasa_attribute(pred_intent)

                # 获取 entities
                entity_results_dict = parse_rasa_entity_result(question, results[0])
            else:
                logger.error(f"index={index},question={question},results获取失败,results={results}")
                text, output_intent, output_intent_confidence, pred_attribute, pred_intent, intent_confidence = [
                                                                                                                    None] * 6
                entity_results_dict = {}

            # 获取原始数据
            data_dict = OrderedDict(single_test_data.to_dict())

            parsed_result_dict = {
                predict_intent_column: pred_intent,
                output_intent_column: output_intent,
                output_intent_confidence_column: output_intent_confidence,
                predict_attribute_column: pred_attribute,
                predict_intent_confidence_column: intent_confidence,
                'text': text,
            }

            data_dict.update(parsed_result_dict)
            data_dict.update(entity_results_dict)

            data_result_ls.append(data_dict)

        duration = time.time() - start_time
        fps = test_data.__len__() / duration
        mspf = duration * 1000 / test_data.__len__()

        dump_obj_as_json_to_file(obj=results_ls, filename =results_ls_path)

        data_result_df = pd.DataFrame(data_result_ls)
        # 增加 意图识别是否准确的 tag
        data_result_df[f"{intent_column}_{tag_column}"] = data_result_df.loc[:, intent_column] \
                                                          == data_result_df.loc[:, predict_intent_column]
        data_result_df[f"{output_intent_column}_{tag_column}"] = data_result_df.loc[:, FIRST_KNOWLEDGE] \
                                                                 == data_result_df.loc[:, output_intent_column]
        data_result_df[f"{ATTRIBUTE}_{tag_column}"] = data_result_df.loc[:, ATTRIBUTE] \
                                                                 == data_result_df.loc[:, predict_attribute_column]

        dataframe_to_file(path=tested_data_path, data=data_result_df)

        # if output_intent_column in data_result_df.columns:
        #     # 计算大意图的准确率
        #     self._build_classification_report(data=data_result_df, label_column=first_intent_column,
        #                                       predict_column=output_intent_column,
        #                                       fps=fps, mspf=mspf,
        #                                       test_data_dir=test_data_dir)
        # if predict_intent_column in data_result_df.columns:
        #     # 计算模型的准确率
        #     self._build_classification_report(data=data_result_df, label_column=intent_column,
        #                                       predict_column=predict_intent_column,
        #                                       fps=fps, mspf=mspf,
        #                                       test_data_dir=test_data_dir
        #                                       )

        # 计算覆盖率
        if standard_question_column:
            count = 0
            for data_result_dict in data_result_ls:
                response_text = data_result_dict[response_text_key]
                standard_question = data_result_dict[standard_question_column]
                if response_text == standard_question:
                    count += 1
            cover_and_accurate_rate = count / data_result_ls.__len__()
            print(f"扩展问共有{data_result_ls.__len__()}个，回复的标准问唯一且正确的格式{count}，"
                  f"覆盖率={cover_and_accurate_rate}")
            cover_and_accurate_result = {"test_question_num": data_result_ls.__len__(),
                                         "cover_and_accurate_num": count,
                                         "cover_and_accurate_rate": cover_and_accurate_rate
                                         }
            cover_and_accurate_result_path = os.path.join(test_data_dir, "cover_and_accurate_result.json")
            dump_obj_as_json_to_file(filename=cover_and_accurate_result_path, obj=cover_and_accurate_result)
        return test_data

    @staticmethod
    def get_nlu_parse_result(index, question, url):

        # post
        results = post_rasa_data(index=index, question=question, url=url)
        # 获取模型意图
        nlu_result_dict = parse_rasa_nlu_result(results=results)
        return nlu_result_dict

    @staticmethod
    def _build_classification_report(data, label_column, predict_column,
                                     test_data_dir,
                                     fps, mspf,
                                     tag_column='tag'):
        tag_column = f"{label_column}_{tag_column}"
        data.loc[:, tag_column] = data.loc[:, predict_column] == \
                                  data.loc[:, label_column]

        true_labels = data.loc[:, label_column]
        predict_labels = data.loc[:, predict_column]

        report_dict = classification_report(y_true=true_labels, y_pred=predict_labels,
                                            output_dict=True, labels=list(set(true_labels)))
        report_df = pd.DataFrame(report_dict).T
        report_df.loc['fps', 'precision'] = fps
        report_df.loc['mspf', 'precision'] = mspf

        report_path = os.path.join(test_data_dir, f'{label_column}_classification_report.xlsx')
        dataframe_to_file(path=report_path, data=report_df)

    @staticmethod
    def get_tmp_nlu_and_core_model_dir(models_dir):
        from rasa.cli.run import _validate_model_path
        from rasa.shared.constants import DEFAULT_MODELS_PATH
        # 验证 models_dir 是否存在
        models_dir = _validate_model_path(models_dir, "model", DEFAULT_MODELS_PATH)
        # 获取最小的模型路径
        model_path = get_local_model(model_path=models_dir)

        # 生产临时的model_dir
        tmp_model_dir = get_model(models_dir)

        # 获取 core 和 nlu 模型的dir
        core_model, nlu_model_dir = get_model_subdirectories(tmp_model_dir)
        return tmp_model_dir, core_model, nlu_model_dir


class NLUApplication():
    def __init__(self,
                 model_path: Text = None,
                 interpreter: NaturalLanguageInterpreter = None,
                 name='nlu_app', route='/nlu/parse/',
                 host="0.0.0.0", port=9001,
                 num_of_workers=1,
                 gpu_memory_per_worker=None):
        self.model_path = model_path
        self.interpreter = interpreter
        self.name = name
        self.route = route
        self.host = host
        self.port = port
        self.num_of_workers = num_of_workers
        self.gpu_memory_per_worker = gpu_memory_per_worker

    def create_app(self):
        """
        参考:rasa.core.run.serve_application() 方法 设置 app
        Agent.handle_message() 方法
        """
        app = Sanic(name=self.name)
        # 注册一个 blueprint
        register_interpreter(app)
        # 注册一个 listener  before_server_start
        app.register_listener(
            partial(load_interpreter_on_start, self.model_path, self.gpu_memory_per_worker),
            event='before_server_start')

        # noinspection PyUnresolvedReferences
        async def clear_model_files(_app: Sanic, _loop: Text) -> None:
            if app.interpreter.tmp_model_dir:
                logger.info(f"删除路径下的文件{_app.interpreter.tmp_model_dir}")
                shutil.rmtree(_app.interpreter.tmp_model_dir)

        app.register_listener(clear_model_files, "after_server_stop")
        # 打印所有 route
        listed_routes = list_routes(app=app)
        url_table = "\n".join(listed_routes[url] for url in sorted(listed_routes))
        logger.info(url_table)
        logger.info(f"启动 sanic application on {self.host}:{self.port}{self.route} with workers={self.num_of_workers}")
        app.run(host=self.host, port=self.port, workers=self.num_of_workers)

    def create_tornado_app(self):
        """
        @time:  2021/10/20 9:12
        @author:wangfc
        @version:
        @description:
        TODO: 尝试使用 tornado 异步方法来启动 app

        @params:
        @return:
        """
        from .torando_severs import create_tornado_server
        interpreter = load_interpreter(model_path=self.model_path)
        create_tornado_server(port=self.port, interpreter=interpreter)



async def load_interpreter_on_start(model_path: Text, gpu_memory_per_worker: int,
                                    app: Sanic, loop: AbstractEventLoop) -> RasaNLUInterpreter:
    """
    参考：  rase.core.run.load_agent_on_start() 方法
           rasa.core.agent.Agent.load_local_model()
    如果使用 interpreter 赋值给 app的时候，当worker >1 的时候无法运行
    因此，使用 model_path 后在各个的进程当中解压模型，可能会成功

    """
    from utils.tensorflow.environment import setup_tf_environment
    setup_tf_environment(gpu_memory_config=gpu_memory_per_worker)
    tmp_model_dir, core_model, nlu_model_dir = RasaApplication.get_tmp_nlu_and_core_model_dir(models_dir=model_path)
    interpreter = rasa.core.interpreter.create_interpreter(nlu_model_dir)
    interpreter.tmp_model_dir = tmp_model_dir
    app.interpreter = interpreter
    return app.interpreter





def load_interpreter(model_path: Text, gpu_memory_per_worker: int = None) -> RasaNLUInterpreter:
    """
    参考：  rase.core.run.load_agent_on_start() 方法
           rasa.core.agent.Agent.load_local_model()
    如果使用 interpreter 赋值给 app的时候，当worker >1 的时候无法运行
    因此，使用 model_path 后在各个的进程当中解压模型，可能会成功

    """
    from utils.tensorflow.environment import setup_tf_environment
    setup_tf_environment(gpu_memory_config=gpu_memory_per_worker)
    tmp_model_dir, core_model, nlu_model_dir = RasaApplication.get_tmp_nlu_and_core_model_dir(models_dir=model_path)
    interpreter = rasa.core.interpreter.create_interpreter(nlu_model_dir)

    return interpreter


def run_rasa_application(mode, output_dir, config_filename, models_subdir, num_of_workers,
                         gpu_memory_per_worker=None,
                         if_build_nlu_application=False,
                         if_build_agent_application=False,
                         if_enable_api=False,
                         output_with_nlu_parse_data=False
                         ):
    # 新建 interpreter
    rasa_application = RasaApplication(taskbot_dir=output_dir,
                                       config_filename=config_filename,
                                       models_subdir=models_subdir,
                                       num_of_workers=num_of_workers,
                                       gpu_memory_per_worker=gpu_memory_per_worker,
                                       if_build_nlu_application=if_build_nlu_application,
                                       if_build_agent_application=if_build_agent_application,
                                       if_enable_api=if_enable_api,
                                       output_with_nlu_parse_data=output_with_nlu_parse_data)

# if __name__ == '__main__':
#     from conf.config_parser import *
#     from conf.logger_config import init_logger
#
#     logger = init_logger(output_dir=os.path.join(OUTPUT_DIR, 'log'), log_filename='server')
#     run_rasa_application(mode=MODE, output_dir=OUTPUT_DIR,
#                          config_filename=RASA_CONFIG_FILENAME,
#                          models_subdir=RASA_MODELS_SUBDIRNAME,
#                          num_of_workers=NUM_OF_WORKERS
#                          )
