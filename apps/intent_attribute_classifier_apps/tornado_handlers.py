#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/13 10:24 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/13 10:24   wangfc      1.0         None
"""
import json

import tornado
from tornado.web import RequestHandler
from apps.intent_attribute_classifier_apps.intent_attribute_classifier import IntentAttributeClassifier


class IntentAttributeClassifierHandler(RequestHandler):
    def initialize(self, classifier: IntentAttributeClassifier):
        """
        函数的参数是 Application 配置中的关键字 参数定义
        """
        self.classifier = classifier

    # 定义 post 方法
    @tornado.gen.coroutine
    def post(self,data_key = 'message'):
        """
        @author:wangfc27441
        @time:  2019/10/25  11:35
        @desc: 对应http的post请求方式
        """
        # 解析tornado查询参数：
        # question = self.get_argument('message', '')
        data = json.loads(self.request.body.decode('utf-8'))
        message = data.get(data_key)
        results = self.classifier.handle_message(message=message)
        # results = json.dumps(eval(str(results),))
        response = {'code': '00', 'msg': 'Success', 'pred': results}
        self.write(response)