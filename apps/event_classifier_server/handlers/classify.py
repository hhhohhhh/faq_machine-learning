#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@file: classify.py
#@version: 02
#@author: $ {USER}
#@contact: wangfc27441@***
#@time: 2019/10/22 17:24 
#@desc:

import os
import json
import ast
import tornado.gen
import tensorflow as tf
import tokenization

from event_classifier.raw_data_prepare import read_support_types_dict_json
from event_classifier.predict import predict, tf_serving_predict,RuleEventPrediction
from run_classifier_multilabel import FLAGS
from classifier_utils import TextClassifierProcessor

from ..conf.config_parser import *
from ..handlers.base import BaseHandler
from build_event_rule import event_type2regular_expression_dict,sub_other_type2same_level_labels_dict,\
    CORRESPONDING_EVENT_LABEL_TUPLE, NEGATIVE_NEGLECT_EVENT_LABELS,LOW_RULE_PROBABILITY_EVENT_LABELS,INCOMPATIBLE_EVENT_LABEL_TUPLE
import logging
logger = logging.getLogger(__name__)

class ClassifierInitializer(object):
    def __init__(self,classifier):
        self.classifier = classifier

    def get_initialization(self):
        # if self.classifier=='event':
        port = PORT
        service_url = SERVICE_URL_DIR
        support_types_dict_json_path = os.path.join(DATA_DIR, 'support_types_dict.json')
        model_version = MODEL_VERSION
        bert_model_dir = os.path.join(OUTPUT_DIR, BERT_MODEL_VERSION)
        pred_type = PRED_TYPE

        # tf_serving_model_url = r'http://localhost:8501/v3/models/concept_classifier:predictions'
        # container_port = kwargs.get('container_port')
        # tf_servering_model_version = kwargs.get('tf_servering_model_version')
        # tf_servering_model_target_path = kwargs.get('tf_servering_model_target_path')
        # tf_serving_model_url = 'http://localhost:' + container_port + 'v' + tf_servering_model_version + '/' + tf_servering_model_target_path + ':predict'

        support_types_dict = read_support_types_dict_json(support_types_dict_json_path, mode='r')
        # 提前加载 processor
        # logger.info('提前加载 {} processor '.format(self.classifier))
        processor = TextClassifierProcessor(support_types_dict=support_types_dict, use_text_b=True,classifier=self.classifier)
        # 提前加载 模型
        logger.info('提前加载 bert_model from {}'.format(bert_model_dir))
        # model_predict_fn=None
        model_predict_fn = tf.contrib.predictor.from_saved_model(bert_model_dir)
        label_list = processor.get_labels()
        # 创建 tokenizer 对象：标记化 输入文本为 单个单词
        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

        # 加载 规则的列表
        rules_dict = {label: rule_info for label,rule_info in event_type2regular_expression_dict.items() if rule_info['classifier']!='model'}

        default_other_type = '其他事件类型'
        # 创建 规则识别的实例
        rule_event_prediction = RuleEventPrediction(rules_dict, default_other_type,
                                                    sub_other_type2same_level_labels_dict,
                                                    CORRESPONDING_EVENT_LABEL_TUPLE,
                                                    NEGATIVE_NEGLECT_EVENT_LABELS,
                                                    LOW_RULE_PROBABILITY_EVENT_LABELS,
                                                    INCOMPATIBLE_EVENT_LABEL_TUPLE)

        return dict(service_url = service_url,bert_model_dir=bert_model_dir,support_types_dict=support_types_dict,
                    processor=processor,model_predict_fn=model_predict_fn,label_list=label_list,tokenizer=tokenizer,
                    pred_type=pred_type,
                    tf_serving_model_url= None,
                    default_other_type=default_other_type,
                    rule_event_prediction=rule_event_prediction)
                    # classifier=self.classifier)


classifier_initialization = ClassifierInitializer(classifier=CLASSIFIER).get_initialization()

def check_test_type(test_type,support_types):
    if test_type is not None and test_type!='':
        try:
            assert test_type in support_types
        except :
            raise(AssertionError('type={} is not in support_types={}'.format(test_type,support_types)))



def check_batch_data_format(data,support_types):
    if type(data) != list:
        raise (TypeError("data type should be list,but get type(data)={}\ndata={}".format(type(data), data)))
    batch_data = []
    test_types = []
    for index,event in enumerate(data):
        assert type(data[index]) == dict
        keys = list(data[index].keys())
        assert  keys == ['title'] or keys == ['title','content'] or keys == ['title','content','test_type']
        batch_data.append(dict(title =event['title'], content=event.get('content',None)))
        test_type = event.get('test_type',None)
        check_test_type(test_type,support_types=support_types)
        test_types.append(test_type)
    return batch_data, test_types



# 定义 ClassifyHandler
class ClassifyHandler(BaseHandler):
    # def __init__(self):
    #     """
    #     函数的参数是 Application 配置中的关键字 参数定义
    #     """
    #     super(ClassifyHandler).__init__()
    #     self.classifier_initialization = ClassifierInitializer(classifier=CLASSIFIER).get_initialization()


    def prepare(self):
        """
        @author:wangfc27441
        @time:  2019/11/7  15:27
        @desc:  我们只需要在请求到来之后，body取值之前，判断一下此请求是否为Json形式
        （选择在RequestHanlder.prepare中进行处理也是因为其会在get、post这些方法之前运行）。
        如果是，将收到的HTTP body进行一下解码以备后用就可以了：
        :return:
        """
        content_type = self.request.headers.get('Content-Type')
        self.if_read_json = False
        self.dict_batch_data = None
        self.read_json_error_response = None

        # 判断是否是json 格式
        content_type = content_type.lower()
        # logger.info("From requests.post, content_type={}".format(content_type))
        if (content_type in ('application/json', 'application/json;charset=utf-8')):
            try:
                self.if_read_json =True
                # 当使用postman时,使用 request.body.decode('utf-8')  解析 json 数据为 str：
                self.json_batch_data = self.request.body.decode('utf-8')
                # 因为单引号的原因不符合 json 格式的要求，会报错
                # self.args = json_decode(self.request.body.decode('utf-8'))
                # 使用 ast.literal_eval() 解析 使用单引号的不标准的json格式
                self.dict_batch_data = ast.literal_eval(self.json_batch_data)
                # self.dict_batch_data = json_decode(self.json_batch_data)
                # logger.info("type(self.dict_batch_data)={}, self.dict_batch_data={}"
                #             .format(type(self.dict_batch_data), self.dict_batch_data))
            except Exception as e:
                # self.set_status(422, 'Unprocessable Entity,{}'.format(e))
                # self.finish()
                self.read_json_error_response = {'code': '00', 'msg': 'Unprocessable Json Entity', 'pred': ''}

    # 定义 get 方法
    def get(self):
        pass

    # 定义 post 方法
    @tornado.gen.coroutine
    def post(self):
        """
        @author:wangfc27441
        @time:  2019/10/25  11:35
        @desc: 对应http的post请求方式
        """
        # 解析tornado查询参数：
        # 当数据是web页面提供的单条数据时候
        try:
            if self.if_read_json ==False:
                # 当使用 request.post(url， data = ) 方法传入data参数作为POST请求的数据：
                # 解析post 参数,获取每个 request.post()方法 的data 内容，返回的都是 unicode字符串
                title = self.get_argument('title', '')
                content = self.get_argument('content', '')
                test_type = self.get_argument('type', None)
                check_test_type(test_type,classifier_initialization.get('label_list'))
                logger.info("title={},\ncontent[0:50]={},\ntest_type={}".format(title,content[0:50],test_type))
                if content==title:
                    content=''
                batch_data = [dict(title = title,text = content)]
                test_types = [test_type]
            else:
                if self.read_json_error_response:
                    self.write(self.read_json_error_response)
                else:
                    # 当使用 request.post(url， json = ) 方法传入 json 作为POST请求的数据：
                    # dict_batch_data格式：
                    batch_data, test_types = check_batch_data_format(self.dict_batch_data, classifier_initialization.get('label_list'))

            ### 在这中间是否可以写入批处理的响应函数，然后将结果分发给每个 request
            # 使用预测函数对数据进行预测

            if IF_PREDICT_WITH_TF_SERVING:
                results = tf_serving_predict(batch_data, test_types, **classifier_initialization)
            else:
                results = predict(batch_data, test_types, **classifier_initialization)
            # logger.info("results={}".format(results))

            if not isinstance(results, list):
                pass
            else:
                # 当返回的是单条数据，默认为原来的web接口，故输出 results[0]
                if  self.if_read_json :
                    ### 在这中间是否可以写入批处理的响应函数，然后将结果分发给每个 request
                    # 使用预测函数对数据进行预测
                    response = {'code': '00', 'msg': 'Success', 'pred': results}
                else:
                    response = {'code': '00', 'msg': 'Success', 'pred': results[0]}
                self.write(response)
        except Exception as e:
            logger.error(e, exc_info=True)
            response = {'code': '99', 'msg': e, 'pred': []}
            self.write(response)


# 定义多个不同分类器的handler:
# 定义事件分类服务的路由地址
# event_service_url= '/hsnlp/event_stock/classifier'
# industry_service_url = '/hsnlp-tools-industry-classifier/industry/classifier'
# codetest_service_url =  '/hsnlp/codetest/classifier'

classifier_handlers = [(SERVICE_URL_DIR, ClassifyHandler)]
# classifier_handlers = [(industry_service_url,ClassifyHandler)]
