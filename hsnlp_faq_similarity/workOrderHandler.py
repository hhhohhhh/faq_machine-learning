#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: gaofeng
@Contact: gaofeng27692@***
@Site: http://www.***
@File: workOrderHandler.py
@Time: 2020-06-01 09:39:46
@Version:V201901.12.000

 * 密级：秘密
 * 版权所有：***
 * 注意：本内容仅限于***内部传阅，禁止外泄以及用于其他的商业目的

1.获取对话数据并做解析，返回检测结果
2. todo 检查配置的模型是否存在

"""
import os
import json
import time
import numpy as np

from utils.utils import logger
from handlers.base import BaseHandler
from handlers.exceptionHandlers import SUCCESS_RESPONSE, UNKOWN_EXCEPTION, \
    FAILED_CODE_01, FAILED_CODE_03, FAILED_CODE_04

from .workorderService import get_result, push_product, get_query_process, get_sentence_process, get_extend_similar
# from service_core.workorderService import get_result, push_product

ROOT_PATH = os.getcwd()
_CONF_PATH = ROOT_PATH + "/service_core/config_file"
_SERVICE_MODEL_PATH = ROOT_PATH + "/models/service_model/"
_DIM_MODEL_PATH = ROOT_PATH + "/models/dim_model/"


def check_file_exist(file_path):
    """检测文件或文件夹是否存在"""
    if os.path.exists(file_path):
        return True
    else:
        return False


def check_dialogue_data(data):
    try:
        sents = []
        ids = []
        for dt in data:
            # print(dt)
            # 检测所有字段是否都在
            for field in ["sender", "id", "sentence_id", "sentence"]:
                if field not in dt.keys():
                    logger.error("msg: 字段{}缺失".format(field))
                    return {"code": "03", "msg": "字段{}缺失".format(field)}
                    # return FAILED_CODE_03

            #  检查字段的类型。要先检查类型！！！！
            # if (isinstance(dt["sender"], str) and isinstance(dt["id"],
            #                                                 str) and \
            #         isinstance(dt["sentence_id"], str) and isinstance(
            #     dt["sentence"], str)) is False:
            #     logger.error("字段类型错误")
            #     return {"code": "03", "msg": "字段类型错误"}
            # return FAILED_CODE_03
            if isinstance(dt["sender"], str) is False:
                # print(dt["sender"])
                # print(type(dt["sender"]))
                logger.error("字段sender类型错误")
                return {"code": "03", "msg": "字段sender类型错误"}
            if isinstance(dt["id"], str) is False:
                logger.error("字段id类型错误")
                return {"code": "03", "msg": "字段id类型错误"}
            if isinstance(dt["sentence_id"], str) is False:
                logger.error("字段sentence_id类型错误")
                return {"code": "03", "msg": "字段sentence_id类型错误"}
            if isinstance(dt["sentence"], str) is False:
                logger.error("字段sentence类型错误")
                return {"code": "03", "msg": "字段sentence类型错误"}

            # 检查字段是否为空
            if str(dt["sender"]).strip() == "":
                logger.error("msg: 字段sender不能为空")
                return {"code": "01", "msg": "字段sender不能为空"}
                # return FAILED_CODE_01
            if str(dt["id"]).strip() == "":
                logger.error("msg: 字段id不能为空")
                return {"code": "01", "msg": "字段id不能为空"}

                # return FAILED_CODE_01
            if str(dt["sentence_id"]).strip() == "":
                logger.error("msg: 字段sentence_id不能为空")
                return {"code": "01", "msg": "字段sentence_id不能为空"}

                # return FAILED_CODE_01
            # 检测参数长度
            if len(dt["sender"]) >= 65:
                logger.error("msg: 参数sender长度过长")
                return {"code": "04", "msg": "参数sender长度过长"}
                # return FAILED_CODE_04
            if len(dt["id"]) >= 65:
                logger.error("msg: 参数id长度过长")
                return {"code": "04", "msg": "参数id长度过长"}
                # return FAILED_CODE_04
            if len(dt["sentence_id"]) >= 65:
                logger.error("msg: 参数sentence_id长度过长")
                return {"code": "04", "msg": "参数sentence_id长度过长"}
                # return FAILED_CODE_04

            # 检查传入的id是否都相同
            ids.append(dt["id"])
            # print(dt["id"])

            sents.append(dt["sentence"])
        sent_check = [s for s in sents if s.strip() != ""]
        if len(sent_check) == 0:
            logger.error("msg: 不能所有的sentence都为空")
            return {"code": "01", "msg": "不能所有的sentence都为空"}
            # return FAILED_CODE_01
        # 检查传入的id是否都相同
        ids = list(set(ids))
        # print(ids)
        if len(ids) != 1:
            logger.error("msg: 会话id不一致")
            return {"code": "09", "msg": "传入的会话id不一致"}
            # return FAILED_CODE_03

        return SUCCESS_RESPONSE
    except Exception as ue:
        return {"code": "99", "msg": "未知异常"}


def is_json(json_data):
    try:
        json.loads(json_data)
    except Exception as e:
        # print("解析错误")
        logger.error("JSON数据解析失败")
        return False
    return True


def check_empty_data(data):
    """检测是否是字符串为空或者List为空"""
    if isinstance(data, list):
        return True
    if isinstance(data, str):
        if len(data.strip()) == 0:
            return True
        else:
            return False


def check_push_data(data):
    try:
        for dt in data:
            # 检测所有字段是否都在
            # print(dt)
            for field in ["product_code", "name", "short_name", "type"]:
                if field not in dt.keys():
                    logger.error("msg: 字段{}缺失".format(field))
                    return {"code": "03", "msg": "字段{}缺失".format(field)}
                    # return FAILED_CODE_03

            # 检查字段类型是否有误 要先检查类型 不然后面strip()函数会出错
            # if (isinstance(dt["product_code"], str) and isinstance(
            #         dt["name"],
            #         str) \
            #         and isinstance(dt["short_name"], list) and (
            #         isinstance(dt["type"], str) or isinstance(dt["type"],
            #                                                   list))) is False:
            #     logger.error("字段类型错误")
            #     return {"code": "03", "msg": "字段类型错误"}

            if isinstance(dt["product_code"], str) is False:
                logger.error("字段product_code类型错误")
                return {"code": "03", "msg": "字段product_code类型错误"}
            if isinstance(dt["name"], str) is False:
                logger.error("字段name类型错误")
                return {"code": "03", "msg": "字段name类型错误"}
            if isinstance(dt["short_name"], list) is False:
                logger.error("字段short_name类型错误")
                return {"code": "03", "msg": "字段short_name类型错误"}
            if (isinstance(dt["type"], str) or isinstance(dt["type"],
                                                          list)) is False:
                logger.error("字段type类型错误")
                return {"code": "03", "msg": "字段type类型错误"}

                # return FAILED_CODE_03

            # 检查字段是否为空
            if dt["product_code"].strip() == "":
                logger.error("msg: 字段product_code不能为空")
                return {"code": "01", "msg": "字段product_code不能为空"}
                # return FAILED_CODE_01
            if dt["name"].strip() == "":
                logger.error("msg: 字段name不能为空")
                return {"code": "01", "msg": "字段name不能为空"}

                # return FAILED_CODE_01

            # 检测参数长度
            if len(dt["product_code"].strip()) >= 65:
                logger.error("msg: 参数product_code长度过长")
                return {"code": "04", "msg": "参数product_code长度过长"}
                # return FAILED_CODE_04
            if len(dt["name"]) >= 65:
                logger.error("msg: 参数name长度过长")
                return {"code": "04", "msg": "参数name长度过长"}
                # return FAILED_CODE_04
            if dt["type"] != []:
                if len(dt["type"]) >= 65:
                    logger.error("msg: 参数type长度过长")
                    return {"code": "04", "msg": "参数type长度过长"}
                    # return FAILED_CODE_04
        return SUCCESS_RESPONSE
    except Exception as e:
        return {"code": "99", "msg": "未知异常"}


class QueryClassifyHandler(BaseHandler):
    """
    获取请求的对话数据，对数据字段做检查并返回检查结果
    """

    def get(self):
        self.post()

    def post(self):
        """
        获取post 请求
        :return:
        """
        content = self.get_argument("query",
                                    "")  # 得到list字符串，需要用json.load解析成list：'[{"id":"1234", "sentence":"测试"}]'
        threshold = self.get_argument("threshold", default=0.0)
        threshold = float(threshold)
        if content.strip() == "":
            self.write({"code": "01", "msg": "请求不能为空", "result": ""})
            return
        else:
            t1 = time.time()
            recall_question, recall_std_question, recall_std_id, recall_score, recall_bm25_yy, recall_bm25_ss, recall_bm25_yy_std, recall_logits = get_result(
                query_input=content, threshold=threshold)
            t2 = time.time()
            print("请求耗时{}".format(t2-t1))
            result = dict()
            result["recall_question"] = recall_question
            result["recall_std_question"] = recall_std_question
            result["recall_score"] = [np.float(rs) for rs in recall_score]
            result["recall_std_id"] = recall_std_id
            result["bm25_recall"] = recall_bm25_yy
            result["bm25_recall_standard"] = recall_bm25_yy_std
            result["bm25_score"] = recall_bm25_ss
            result["logits"] = recall_logits
            self.write({"code": "00", "msg": "请求成功", "result": result})
            return


def check_duplicate(push_data):
    """检查标准问的重复性"""
    question_count = dict()
    for dt in push_data:
        is_standard = dt["is_standard"]
        if is_standard != "2":
            question = dt["question"]
            question = str(question).upper()
            if question not in question_count:
                question_count[question] = "1"
            else:
                return False
    return True


class PushProduct(BaseHandler):
    """产品数据推送接口"""

    def get(self):
        self.post()

    def post(self):

        data = self.get_argument("data", "")
        # print(data)
        if data.strip() == "":
            self.write({"code": "01", "msg": "参数不能为空"})
            return
        if not is_json(data):
            logger.error("参数非json格式")
            self.write({"code": "03", "msg": "非json格式数据"})
            return

        push_data = json.loads(
            data)  # shortName有可能为空， type有可能为空 。如果type为空 那么忽略掉这条数据。

        check_res = check_duplicate(push_data)
        if check_res is False:
            self.write({"code": "01", "msg": "推送的标准问有重复"})
            return

        try:
            logger.info("开始推送数据")
            res = push_product(push_data)
            if res:
                self.write({"code": "00", "msg": "推送成功"})
                return
            else:
                self.write({"code": "01", "msg": "推送失败"})
                return
        except Exception as e:
            self.write({"code": "01", "msg": "推送失败"})
            logger.error("读取推送数据失败")
            return

class QueryProcessHandler(BaseHandler):
    """
    返回query预处理后的拼接结果
    """

    def get(self):
        self.post()

    def post(self):
        """
        获取post 请求
        :return:
        """
        content = self.get_argument("sentence",
                                    "")  # 得到list字符串，需要用json.load解析成list：'[{"id":"1234", "sentence":"测试"}]'
        if content.strip() == "":
            self.write({"code": "01", "msg": "请求不能为空", "result": ""})
            return
        else:
            sent_join = get_query_process(content)
            result = dict()
            result["process_res"] = sent_join
            self.write({"code": "00", "msg": "请求成功", "result": result})
            return


class QuestionFilter(BaseHandler):
    def get(self):
        self.post()

    def post(self):
        """
        获取post 请求
        :return:
        """
        content = self.get_argument("query",
                                    "")  # 得到list字符串，需要用json.load解析成list：'[{"id":"1234", "sentence":"测试"}]'
        threshold = self.get_argument("threshold", default=0.85)
        threshold = float(threshold)
        std_question = self.get_argument("standard_question", "")
        if content.strip() == "":
            self.write({"code": "01", "msg": "请求不能为空", "result": ""})
            return
        else:
            t1 = time.time()
            print("请求相似度")
            recall_question, recall_std_question, recall_score = get_extend_similar(content, std_question, threshold)
            t2 = time.time()
            print("请求耗时{}".format(t2-t1))
            result = dict()
            if len(recall_std_question) == 0:
                result["msg"] = "扩展问可添加"
            else:
                result["msg"] = "扩展问不可添加"
            result["recall_question"] = recall_question
            result["recall_std_question"] = recall_std_question
            result["recall_score"] = [np.float(rs) for rs in recall_score]
            self.write({"code": "00", "msg": "请求成功", "result": result})
            return

class SentProcessHandler(BaseHandler):
    """
    返回query预处理后的拼接结果
    """

    def get(self):
        self.post()

    def post(self):
        """
        获取post 请求
        :return:
        """
        content = self.get_argument("sentence",
                                    "")  # 得到list字符串，需要用json.load解析成list：'[{"id":"1234", "sentence":"测试"}]'
        if content.strip() == "":
            self.write({"code": "01", "msg": "请求不能为空", "result": ""})
            return
        else:
            sent_join, sent_word, sent_word_ext = get_sentence_process(content)
            result = dict()
            result["sent_join"] = sent_join
            result["sent_word"] = sent_word
            result["sent_word_ext"] = sent_word_ext
            self.write({"code": "00", "msg": "请求成功", "result": result})
            return