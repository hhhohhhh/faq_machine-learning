#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/14 13:54 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/14 13:54   wangfc      1.0         None
"""
import io
import asyncio
import inspect
import json
import logging
import time
import pandas as pd
from asyncio import Queue, CancelledError

import tqdm
from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse
from typing import Text, Dict, Any, Optional, Callable, Awaitable, NoReturn

from data_process.dataframe_utils import load_excel_file_to_dataframe, export_dataframe_to_http_response
from data_process.dataset.hsnlp_faq_knowledge_dataset import ATTRIBUTE
from models.classifiers.regex_classifiers import PATTERN
from utils.constants import INTENT, STANDARD_QUESTION, EXTEND_QUESTION
from utils.exceptions import InvalidParameterException
from utils.io import get_file_stem
from utils.time import get_current_time
from ..app_constants import FAILURE
from ..app_utils.data_validate import validate_json_payload
from ..intent_attribute_classifier_apps.intent_attribute_classifier_app_constants import PREDICT_INTENT, \
    PREDICT_ATTRIBUTE
from ..responses import HsnlpResponse
from openpyxl import load_workbook

logger = logging.getLogger(__name__)

from .channel import InputChannel, CollectingOutputChannel, UserMessage


class RestInput(InputChannel):
    """A custom http input channel.

    This implementation is the basis for a custom implementation of a chat
    frontend. You can customize this to send messages to Rasa and
    retrieve responses from the assistant."""

    @classmethod
    def name(cls) -> Text:
        return "rest"

    # @staticmethod
    # async def on_message_wrapper(
    #     on_new_message: Callable[[UserMessage], Awaitable[Any]],
    #     text: Text,
    #     queue: Queue,
    #     sender_id: Text,
    #     input_channel: Text,
    #     metadata: Optional[Dict[Text, Any]],
    # ) -> None:
    #     collector = QueueOutputChannel(queue)
    #
    #     message = UserMessage(
    #         text, collector, sender_id, input_channel=input_channel, metadata=metadata
    #     )
    #     await on_new_message(message)
    #
    #     await queue.put("DONE")

    async def _extract_sender(self, req: Request) -> Optional[Text]:
        return req.json.get("sender", None)

    # noinspection PyMethodMayBeStatic
    def _extract_message(self, req: Request) -> Optional[Text]:
        return req.json.get("message", None)

    def _extract_input_channel(self, req: Request) -> Text:
        return req.json.get("input_channel") or self.name()

    def stream_response(
            self,
            on_new_message: Callable[[UserMessage], Awaitable[None]],
            text: Text,
            sender_id: Text,
            input_channel: Text,
            metadata: Optional[Dict[Text, Any]],
    ) -> Callable[[Any], Awaitable[None]]:
        async def stream(resp: Any) -> None:
            q = Queue()
            task = asyncio.ensure_future(
                self.on_message_wrapper(
                    on_new_message, text, q, sender_id, input_channel, metadata
                )
            )
            while True:
                result = await q.get()
                if result == "DONE":
                    break
                else:
                    await resp.write(json.dumps(result) + "\n")
            await task

        return stream

    def blueprint(
            self, on_new_message: Callable[[UserMessage], Awaitable[None]]
    ) -> Blueprint:
        custom_webhook = Blueprint(
            "custom_webhook_{}".format(type(self).__name__),
            inspect.getmodule(self).__name__,
        )

        # noinspection PyUnusedLocal
        @custom_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @custom_webhook.route("/webhook", methods=["POST"])
        async def receive(request: Request) -> HTTPResponse:
            sender_id = await self._extract_sender(request)
            text = self._extract_message(request)
            # from rasa.utils import endpoints
            from apps.app_utils import endpoints
            should_use_stream = endpoints.bool_arg(
                request, "stream", default=False
            )
            input_channel = self._extract_input_channel(request)
            metadata = self.get_metadata(request)

            if should_use_stream:
                return response.stream(
                    self.stream_response(
                        on_new_message, text, sender_id, input_channel, metadata
                    ),
                    content_type="text/event-stream",
                )
            else:
                collector = CollectingOutputChannel()
                # noinspection PyBroadException
                try:
                    # 将转入的 request 请求的参数包装为 UserMessage 对象,交个 on_new_message方法 处理
                    await on_new_message(
                        UserMessage(
                            text,
                            collector,
                            sender_id,
                            input_channel=input_channel,
                            metadata=metadata,
                        )
                    )
                except CancelledError:
                    logger.error(
                        f"Message handling timed out for " f"user message '{text}'."
                    )
                except Exception:
                    logger.exception(
                        f"An exception occured while handling "
                        f"user message '{text}'."
                    )
                return response.json(collector.messages)

        return custom_webhook


class HsnlpInputChannel(RestInput):
    """
    参考 : rasa.core.channels.rest.RestInput.blurprint()
    """

    def __init__(self, url_prefix='hsnlp', blueprint_route_name="classifier"):
        super(HsnlpInputChannel, self).__init__()
        self.url_prefix = url_prefix
        self.blueprint_route_name = blueprint_route_name

    def blueprint(self, on_new_message: Callable[[UserMessage], Awaitable[Any]]):
        """
        生成一个 blueprint
        handler_message: 是处理 message 的逻辑
        """
        custom_blueprint = Blueprint(name=self.name(), url_prefix=self.url_prefix)

        # 对 blueprint 增加 一个 route + handler 来处理信息,对其分类
        @custom_blueprint.route(uri=self.classifier_route_name, methods=['POST'])
        async def receive(request: Request) -> HTTPResponse:
            """
            处理 request 请求 ：  json = {“sender”:,"message":}
            """
            sender_id = await self._extract_sender(req=request)
            start_time = time.time()
            text = await self._extract_message(req=request)
            # 生成 UserMessage 对象
            user_message = UserMessage(text=text, sender_id=sender_id)
            # 使用 消息处理逻辑 处理消息
            result = await on_new_message(user_message)

            logger.debug(f"解析数据text = {text},cost={time.time() - start_time}")
            # 返回 text 内容:response.text(result)
            if isinstance(result, dict):
                output_response = {"code": "00", "msg": "Success",
                                   "text": text,
                                   "pred": result}
            elif isinstance(result, tuple):
                preprocessed_text = result[0]
                pred = result[1]
                output_response = {"code": "00", "msg": "Success",
                                   "text": text,
                                   'preprocessed_text': preprocessed_text,
                                   "pred": pred}
            return response.json(body=output_response)

        return custom_blueprint

    # 参考 rasa.core.channels.rest.RestInput._extract_message() 方法
    async def _extract_message(self, req: Request) -> Optional[Text]:
        return req.json.get("message", None)


class HsnlpFAQIntentAttributeClassifierInputChannel(HsnlpInputChannel):
    """
    参考 : rasa.core.channels.rest.RestInput.blurprint()
    """

    def __init__(self, url_prefix: Text = 'hsnlp/faq/intent_attribute_classifier/',
                 classifier_route_name: Text = "classifier",
                 new_standard_question_classifier_route_name: Text = "new_standard_question_classifier",
                 update_pattern_data_route_name: Text = "update_pattern_data",
                 standard_question_key: Text = STANDARD_QUESTION,  # '标准问',
                 extend_question_key: Text = EXTEND_QUESTION,  # '扩展问',
                 extend_question_least_num: int = 10,
                 patten_key: Text = PATTERN,
                 intent_key: Text = INTENT,
                 attribute_key: Text = ATTRIBUTE
                 ):
        super(HsnlpFAQIntentAttributeClassifierInputChannel, self).__init__(url_prefix=url_prefix)
        self.classifier_route_name = classifier_route_name
        self.new_standard_question_classifier_route_name = new_standard_question_classifier_route_name
        self.update_pattern_data_route_name = update_pattern_data_route_name

        self.standard_question_key = standard_question_key
        self.extend_question_key = extend_question_key
        self.new_standard_question_data_keys = [self.standard_question_key, self.extend_question_key]
        self.extend_question_least_num = extend_question_least_num

        self.patten_key = patten_key
        self.intent_key = intent_key
        self.attribute_key = attribute_key
        self.update_regex_data_keys = [self.standard_question_key, self.intent_key, self.attribute_key]

    def _validate_extend_questions(self, data: Dict):
        extend_questions = data.get(self.extend_question_key)
        if extend_questions is None:
            raise ValueError("扩展问数据为空")
        elif not isinstance(extend_questions, list):
            raise TypeError(f"扩展问数据格式错误，数据类型为 {type(extend_questions)},要求是一个列表类型")
        elif extend_questions.__len__() < self.extend_question_least_num:
            raise ValueError(f"扩展问数据共有{extend_questions.__len__()}个，不满足要求的至少{self.extend_question_least_num}个")
        return extend_questions

    def blueprint(self, on_new_message: Callable[[UserMessage], Awaitable[Any]]):
        """
        生成一个 blueprint
        on_new_message : 是处理 message 的逻辑
        """
        custom_blueprint = Blueprint(name=self.name(), url_prefix=self.url_prefix)

        # 对 blueprint 增加 一个 route + handler 来处理信息
        @custom_blueprint.route(uri=self.classifier_route_name, methods=['POST'])
        async def classifier(request: Request) -> HTTPResponse:
            """
            处理 request 请求 ：  json = {“sender”:,
                                        "message":}
            """

            sender_id = await self._extract_sender(req=request)
            start_time_str,start_time = get_current_time(with_time_stamp=True)
            # message 字段默认传入原始文本
            text = await self._extract_message(req=request)
            preprocessed_info = await  self._extract_preprocessed_info(req=request)

            # 生成 UserMessage 对象
            user_message = UserMessage(text=text, sender_id=sender_id,metadata=preprocessed_info)

            # 提取中间件加入的 ctx，其实就是  pattern_data,将其传递给 on_new_message
            pattern_data = request.ctx.pattern_data

            # 使用 消息处理逻辑 处理消息
            pred_result = await on_new_message(user_message, pattern_data)
            # pred_result = {'pred':result}
            end_time_str,end_time = get_current_time(with_time_stamp=True)

            logger.info(f"start_time={start_time_str},end_time={end_time_str},cost_time={end_time-start_time},"
                        f"text = {text}, pred_result={pred_result}")
            # 返回 text 内容:response.text(result)
            hsnlp_response = HsnlpResponse(**pred_result)
            return response.json(body=hsnlp_response.output_info)

        @custom_blueprint.route(uri="batch_" + self.classifier_route_name,methods=['POST'])
        async def batch_classifier(request:Request,eval_question_key="评估问")-> HTTPResponse:
            """
            新增批处理的意图识别的 API
            """
            try:
                # 获取前端传的文件对象，注意是File对象，并不是我们平时使用的file对象
                file = request.files.get('file')
                filname = file.name
                data = load_excel_file_to_dataframe(file=file)
                data = data.iloc[:10]

                if eval_question_key not in data.columns:
                    raise InvalidParameterException(f"{eval_question_key}不存在，请检查数据")

                # 提取中间件加入的 ctx，其实就是  pattern_data,将其传递给 on_new_message
                pattern_data = request.ctx.pattern_data
                pred_result_ls = []
                for index in tqdm.tqdm(data.index,desc='意图批处理'):
                    text = data.loc[index,eval_question_key]
                    # 生成 UserMessage 对象
                    user_message = UserMessage(text=text, sender_id=None, metadata=None)
                    # 使用 消息处理逻辑 处理消息
                    pred_result = await on_new_message(user_message, pattern_data)
                    preprocessed_text = pred_result.get('preprocessed_text')
                    pred_dict = pred_result.get('pred')
                    pred_dict.update({"preprocessed_text":preprocessed_text})
                    pred_result_ls.append(pred_dict)

                pred_result_df = pd.DataFrame(pred_result_ls)
                result_df = pd.concat([data,pred_result_df],axis=1)

                filename_stem = get_file_stem(file=filname)
                new_filename = f"{filename_stem}_result.xlsx"
                hsnlp_response = export_dataframe_to_http_response(data=result_df,filename=new_filename)
                return hsnlp_response


            except Exception as e:
                logger.error(e)
                hsnlp_response = HsnlpResponse(status=FAILURE, exception=e)
                return response.json(body=hsnlp_response.output_info)


        @custom_blueprint.route(uri=self.new_standard_question_classifier_route_name, methods=['POST'])
        async def new_standard_question_classifier(request: Request) -> HTTPResponse:
            """
            新增标准问的 接口 API
            """
            try:
                sender_id = await self._extract_sender(req=request)
                start_time = time.time()
                data = request.json
                validate_json_payload(request_payload=data, request_keys=self.new_standard_question_data_keys)

                extend_questions = self._validate_extend_questions(data=data)
                standard_question = data.get(self.standard_question_key)
                # 识别意图和属性
                questions = [standard_question] + extend_questions[:self.extend_question_least_num]
                results = []
                for question in questions:
                    # 生成 UserMessage 对象
                    user_message = UserMessage(text=question, sender_id=sender_id)
                    # 使用 消息处理逻辑 处理消息
                    result = await on_new_message(user_message)
                    results.append(result)

                predict_intents = [result.get(PREDICT_INTENT) for result in results]
                predict_attributes = [result.get(PREDICT_ATTRIBUTE) for result in results]
                predict_intent_set = set(predict_intents)
                predict_attribute_set = set(predict_attributes)
                if predict_intent_set.__len__() == 1:
                    predict_intent = predict_intents[0]
                    predict_intent_msg = f"前{self.extend_question_least_num}条数据预测的意图一致为:{predict_intent}"
                else:
                    predict_intent = None
                    predict_intent_msg = f"前{self.extend_question_least_num}条数据预测的意图不一致为:{predict_intents}，无法获取统一的意图"
                if predict_attribute_set.__len__() == 1:
                    predict_attribute = predict_attributes[0]
                    predict_attribute_msg = f"前{self.extend_question_least_num}条数据预测的属性一致为:{predict_attribute}"
                else:
                    predict_attribute = None
                    predict_attribute_msg = f"前{self.extend_question_least_num}条数据预测的属性不一致为:{predict_attributes}，无法获取统一的属性"
                pred_result = {PREDICT_INTENT: predict_intent, PREDICT_ATTRIBUTE: predict_attribute,
                               "predict_intent_reason": f"{predict_intent_msg}",
                               "predict_attribute_reason": f"{predict_attribute_msg}"}
                hsnlp_response = HsnlpResponse(**{"pred": pred_result})
            except Exception as e:
                logger.error(e)
                hsnlp_response = HsnlpResponse(status=FAILURE, exception=e)

            return response.json(body=hsnlp_response.output_info)


        @custom_blueprint.route(uri='test',methods=['GET'])
        async def test(request:Request)->HTTPResponse:
            return response.json({'a':[1,2,3]})

        return custom_blueprint
        # @custom_blueprint.route(uri=self.update_pattern_data_route_name, methods=['POST'])
        # async def update_pattern_data(request: Request) -> HTTPResponse:
        #     """
        #     新增标准问的 接口 API
        #     思路1: postman: updated_pattern_data -> save and update pattern data -> update IntentRegexClassifier -> update IntentAttributeInterpreter
        #     TODO: 多进程的情况是否 update 到每个 进程的 IntentAttributeInterpreter 对象
        #     思路 2： 中间件
        #     思路 3： 升级 21版本 ，使用 context 对象
        #     """
        #     try:
        #         sender_id = await self._extract_sender(req=request)
        #         start_time = time.time()
        #         # 获取和验证数据
        #         data = await self._extract_data(req=request)
        #         validate_json_payload(request_payload=data, request_keys=self.update_regex_data_keys)
        #         self._validate_pattern_data()
        #         self._save_pattern_data()
        #         self._update_pattern_data()
        #         self._update_intent_regex_classifier()
        #         self._update_intent_attribute_classififer()
        #
        #     except Exception as e:
        #         logger.error(e)
        #         hsnlp_response = HsnlpResponse(status=FAILURE, exception=e)



    async def _extract_new_standard_question_data(self, req: Request) -> Dict:
        """
        具体 agent.handle_message() 方法处理消息的逻辑
        new_standard_question_data :{"标准问": , "扩展问": []}
        """
        data = req.json
        validate_json_payload(request_payload=data, request_keys=self.new_standard_question_data_keys)
        return data

    async def _extract_preprocessed_info(self,req:Request,key='baseInfo')-> Optional[Dict]:
        """
        提取预处理的信息
        """
        data=  req.json
        preprocessed_info = data.get(key,None)
        if preprocessed_info and isinstance(preprocessed_info,list) and preprocessed_info.__len__()==1:
            preprocessed_info_dict = preprocessed_info[0]
            if isinstance(preprocessed_info_dict,dict):
                return preprocessed_info_dict

