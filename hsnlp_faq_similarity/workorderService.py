#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: gaofeng
@Contact: gaofeng27692@***
@Site: http://www.***
@File: workorderService.py
@Time: 2021-10-28
@Version:V2020.01.000

 * 密级：秘密
 * 版权所有：***
 * 注意：本内容仅限于***内部传阅，禁止外泄以及用于其他的商业目的


1.预处理阶段不做分割,同一个标准问的相同bm25分数也放在topN中。

"""
import pdb
import os
import re
import json
import time
import copy
import string
import random
import datetime
import pandas as pd
import tensorflow as tf
from math import log
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from pyhanlp import *

KBeamArcEagerDependencyParser = JClass('com.hankcs.hanlp.dependency.perceptron.parser.KBeamArcEagerDependencyParser')

parser = KBeamArcEagerDependencyParser()


from collections import defaultdict

from service_core.trie import Trie
from utils.utils import logger

from service_core.config import SEQ_LENGTH, RULE_FLAG, PROCESS, INTENT_FLAG

# from service_core.predict_concate import InputExample, convert_single_example, \
#     tokenizer, label_list
from service_core.predict import InputExample, convert_single_example, \
    tokenizer, label_list

import service_core.intent_predict as ip

import requests
from gensim import corpora
from gensim.summarization import bm25
import numpy as np

# import xgboost as xgb

#生产随机数
def num_unique():
    nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 生成当前时间
    randomNum = random.randint(0, 100)  # 生成的随机整数n，其中0<=n<=100
    if randomNum <= 10:
        randomNum = str(0) + str(randomNum)
    uniqueNum = str(nowTime) + str(randomNum)
    return uniqueNum

# 意图结果请求
def get_intent(query):
    url = "http://10.20.33.11:5005/webhooks/rest/webhook/"
    uid = num_unique();
    post_data = {"sender": uid, "message": query}
    result = requests.post(url, json.dumps(post_data)).content
    return json.loads(result)[0]["text"]

def _get_module_path(path):
    return os.path.normpath(os.path.join(os.getcwd(),
                                         os.path.dirname(__file__), path))

intent_text = ["查资金账号", "账号和密码都忘记了", "密码重置", "禁止银行转账支取", "转账失败", "科创板开通", "查科创板权限", "创业板开通", "查创业板权限"]


_CONF_PATH = _get_module_path("./config_file")

# _MODEL = _get_module_path("./models/1613668763")
# _MODEL = _get_module_path("./models/0323")
# _MODEL = _get_module_path("./models/04081")
# _MODEL = _get_module_path("./models/ivr_concat")
_MODEL = _get_module_path("./models/ivr_sim")
# _MODEL = _get_module_path("./models/0531")
print("ivr sim 模型")

_THRE_MODEL = _get_module_path("./models/thre_model/model_6_features_depth_3.bin")
# _MODEL = _get_module_path("./models/1615194190")
_INTENT_MODEL = _get_module_path("./models/intention_model/1615441100")  #意图分类模型

with open(_CONF_PATH+'/stop_words', 'r', encoding='utf8') as f:
        words = f.readlines()
        words = [w.strip() for w in words]

with open(_CONF_PATH + '/tone_word.txt', 'r', encoding='utf8') as f:
    tone_words = f.readlines()
    tone_words = [w.strip() for w in tone_words]
    tone_words = list(set(tone_words))
    tone_words = list(sorted(tone_words, key=lambda x: len(x), reverse=True))

# punc = [":", "→", "℃", "&", ""]
#
# text = "fdafa"
# s = re.sub('[:→℃&*一~’.『/\-』＝″【—?、。“”《》！，：；？．,\'·<>（）〔〕\[\]()+～×／①②③④⑤⑥⑦⑧⑨⑩ⅢВ";#@γμφΔ■▲＃％＆＇〉｡＂＄＊＋－＜＞＠［＼］＾｀｛｜｝｟｠｢｣､〃「」】〖〗〘〙〚〛〜〝〞〟〰〾〿‘‛„‟…‧﹏]+', "", text)

punc = ':→℃&*一~’.『/\-』＝″【—?、。“”《》！，：；？．,\'·<>（）〔〕\[\]()+～×／①②③④⑤⑥⑦⑧⑨⑩ⅢВ";#@γμφΔ■▲＃％＆＇〉｡＂＄＊＋－＜＞＠［＼］＾｀｛｜｝｟｠｢｣､〃「」】〖〗〘〙〚〛〜〝〞〟〰〾〿‘‛„‟…‧﹏'


def softmax(x):
    xx = np.exp(x)
    s = sum(xx)
    p = xx/s
    return p


def remove_stop_words(x, stop_words=None):
    xx = [w for w in x if w not in stop_words]
    return xx

def score_trans(x):
    return ((max(min(x, 20), -50) + 50) / 70) ** log(0.8, 42 / 70)

#http请求
session = requests.session()
#重连次数
requests.adapters.DEFAULT_RETRIES = 100

def wordseg(x, stop_words=None):
    # url = "http://192.168.73.51:18080/hsnlp-tools-server/nlp/word_segment"
    url = "http://10.20.33.3:8180/hsnlp-tools-server/nlp/word_segment"
    data = {"text": x}
    head = {"Connection": "close"}
    r = session.post(url, data=data, headers=head)
    session.keep_alive = False
    rows = r.json()["rows"]
    words = [w["word"] for w in rows]
    words = remove_stop_words(words, stop_words=stop_words)
    return words

def model_exist(file):
    """检测配置文件中配置的模型的模型文件是否存在"""
    if os.path.exists(file):
        return True
    else:
        return False



class SmartWorkorder():
    """对传入的对话语料做服务类型和维度类型检测"""

    def init(self):
        self.q_2_1_2_2_reg = ["自助开户时，选错营业部应如何处理甭",
                              "A股自助开户时转账失败，如何处理哎",
                              "自助开户时哪些银行需要输入卡密码别",
                              "通过港股通持有的港股，可以提取纸面股票吗啐",
                              "开户绑定银行卡时，输入的密码是什么吧哒",
                              "营业部如何开通港股通别",
                              "港股通换汇过程中，中国结算收取费用或赚取汇差吗一样",
                              "网上开通港股通，报沪A账户未指定交易和",
                              "未开通创业板，能申购创业板新股吗呜呼",
                              "当日申请转户或销户，当日可以办理完成吗呜呼",
                              "开市前时段，港股账户的竞价盘是优于限价盘成交吗罢了",
                              "香港市场的半日市，可以正常交收吗啊呀",
                              "沪深港通股票为A+H股的，A股停牌与H股停牌是否同步也好",
                              "申购基金银行卡已扣款，但交易失败吧",
                              "基金申购失败，资金如何处理着哩",
                              "周五申购的货币基金，周末享有收益吗会",
                              "申购基金银行卡已扣款，但交易失败‟",
                              "当日申请转户或销户，当日可以办理完成吗〝",
                              "基金申购失败，资金如何处理〘",
                              "未开通创业板，能申购创业板新股吗<",
                              "周五申购的货币基金，周末享有收益吗&",
                              "港股通换汇过程中，中国结算收取费用或赚取汇差吗」",
                              "开市前时段，港股账户的竞价盘是优于限价盘成交吗】",
                              "A股自助开户时转账失败，如何处理＄",
                              "自助开户时，选错营业部应如何处理＞",
                              "网上开通港股通，报沪A账户未指定交易＋",
                              "通过港股通持有的港股，可以提取纸面股票吗｝",
                              "沪深港通股票为A+H股的，A股停牌与H股停牌是否同步〚",
                              "香港市场的半日市，可以正常交收吗\"",
                              "开户绑定银行卡时，输入的密码是什么「"

                              ]
        for i in range(len(self.q_2_1_2_2_reg)):
            self.q_2_1_2_2_reg[i] = self.q_2_1_2_2_reg[i].upper()

        self.stdq_id = dict()  # {id:std_question}
        self.stdq_raw_id = dict() # {id:std_question_raw} # 存放未转换大小写的标准问，即推送进来的原始的标准问
        self.id_stdq = dict()  # {std_question:id}
        self.extq_id = dict()  # {id:ext_question}
        self.extq_raw_id = dict() # {id:ext_question_raw} # 存放原始的扩展问
        # self.extinstd = defaultdict(list) # {stdid:[extid,extid]}
        self.extqstdid = dict() #{extq:stdid} 扩展问属于哪个标准问id
        # self.questions = []  # 标准问和扩展问的合并
        self.stop_words = words
        self.sentences = []    # 标准问和扩展问的合并
        self.texts = None
        self.sent_join = None  # 将预处理后的词join为一个长文本
        self.dictionary = None
        self.bm25_obj = None

        self.neg_question = [] # 不应命中的原始文本
        self.neg_question_process = [] # 不应命中的原始文本预处理后的结果。


        self.question2reg = defaultdict(dict) # 存放pos和neg正则表达式
        # self.topk = None
        print(_MODEL)
        self.predict_fn = tf.contrib.predictor.from_saved_model(_MODEL)
        logger.info("相似度模型加载完成")
        if INTENT_FLAG == "1":  # intent_flag为1时，使用意图分类。
            self.predict_fn_intent = tf.contrib.predictor.from_saved_model(_INTENT_MODEL)
            logger.info("意图分类模型加载完成")
        else:
            self.predict_fn_intent = None

        # self.fn_threshold_predict = xgb.Booster({'nthread': 8})
        # self.fn_threshold_predict.load_model(_THRE_MODEL)
        # logger.info("xgboost模型加载完成")
        self.synword_map = dict()
        with open(_CONF_PATH + '/synonym_word.json', 'r', encoding='utf8') as f:
            synonym_word = json.load(f)
            for std_word, syn_words in synonym_word.items():
                for sw in syn_words:
                    self.synword_map[sw] = std_word  # 同义词到标准词的映射
        logger.info("同义词加载完成")

        self.std2coreW = dict()
        # with open(_CONF_PATH + '/std2coreW.json', 'r', encoding='utf8') as f:
        #     self.std2coreW = json.load(f)

        # self.extend_word = defaultdict(list)  # 对每一个词建立一个list映射，加快查找速度
        # with open(_CONF_PATH + "/extend_word.json", "r", encoding="utf-8") as fr:
        #     for i, line in enumerate(fr):
        #         linex = line.strip()
        #         linex = json.loads(linex)
        #         self.extend_word[list(linex.keys())[0]] = list(linex.values())[0]
        # logger.info("扩展词加载完成")


        # 应对测试规则干预
        self.neg_rule = []
        self.pos_rule = []
        with open(_CONF_PATH + "/neg_rule.txt", "r", encoding="utf-8") as fr:
            for i, line in enumerate(fr):
                linex = line.strip()
                if linex == "":
                    continue
                self.neg_rule.append(str(linex).upper())
        with open(_CONF_PATH+"/pos_rule.txt", "r", encoding="utf-8") as fr:
            for i, line in enumerate(fr):
                linex = line.strip()
                if linex == "":
                    continue
                self.pos_rule.append(str(linex).upper())

        df_drop_word = pd.read_excel(_CONF_PATH+"/语气词_20210413.xlsx", dtype=str)
        df_drop_word = df_drop_word[df_drop_word["lrtail"]=='1']
        self.drop_word_right = df_drop_word["word"].tolist()
        # print(self.drop_word_right)

        # 反向场景过滤
        self.df_opp = pd.read_excel(_CONF_PATH+"/反向关键词.xlsx",
                                    sheet_name="反向关键词", converters={"关键词": str}, header=0)

        # 根据词汇来降低问题的分数。
        self.neg_words = []
        with open(_CONF_PATH+"/neg_words.txt", "r", encoding="utf-8") as fr:
            for i, line in enumerate(fr):
                linex = line.strip()
                if linex == "":
                    continue
                self.neg_words.append(str(linex).upper())

    # def batch_input(self,text_as, text_bs):
    #     input_ids_as = []
    #     input_mask_as = []
    #     segment_ids_as = []
    #     label_idss = []
    #     for i in range(len(text_as)):
    #         predict_example = InputExample("id", text_as[i], text_bs[i], label_list[0])
    #         feature = convert_single_example(3, predict_example, label_list,
    #                                          30, tokenizer)
    #         input_ids_as.append(feature.input_ids)
    #         input_mask_as.append(feature.input_mask)
    #         segment_ids_as.append(feature.segment_ids)
    #         label_idss.append(feature.label_id)
    #     return input_ids_as, input_mask_as, segment_ids_as, label_idss

    def batch_input(self,text_as, text_bs):
        input_ids_as = []
        input_mask_as = []
        segment_ids_as = []
        label_idss = []
        input_ids_bs = []
        input_mask_bs = []
        segment_ids_bs = []
        for i in range(len(text_as)):
            predict_example = InputExample("id", text_as[i], text_bs[i], label_list[0])
            feature = convert_single_example(5, predict_example, label_list,
                                             30, tokenizer)
            input_ids_as.append(feature.input_ids_a)
            input_mask_as.append(feature.input_mask_a)
            segment_ids_as.append(feature.segment_ids_a)
            label_idss.append(feature.label_id)
            input_ids_bs.append(feature.input_ids_b)
            input_mask_bs.append(feature.input_mask_b)
            segment_ids_bs.append(feature.segment_ids_b)
        return input_ids_as, input_mask_as, segment_ids_as, label_idss, input_ids_bs, \
               input_mask_bs, segment_ids_bs

    # def predict_siamese(self, text_as, text_bs, label_list=label_list):
    #     input_ids_as, input_mask_as, segment_ids_as, label_idss = self.batch_input(text_as, text_bs)
    #
    #     feature = {
    #         "input_ids": input_ids_as,
    #         "input_mask": input_mask_as,
    #         "segment_ids": segment_ids_as,
    #         "label_ids": label_idss,
    #     }
    #     prediction = self.predict_fn(feature)
    #     # return prediction["logits"], prediction["sims"]  # 模型相似度输出
    #     # return prediction["logits_rank"], prediction["sims"]
    def predict_siamese(self, text_as, text_bs, label_list=label_list):
        input_ids_as, input_mask_as, segment_ids_as, label_idss, input_ids_bs, \
        input_mask_bs, segment_ids_bs = self.batch_input(text_as, text_bs)
        prediction = self.predict_fn({
            "input_ids_a": input_ids_as,
            "input_mask_a": input_mask_as,
            "segment_ids_a": segment_ids_as,
            "label_ids": label_idss,
            "input_ids_b": input_ids_bs,
            "input_mask_b": input_mask_bs,
            "segment_ids_b": segment_ids_bs,
        })
        return prediction["sims"]


    def query_split(self, query):
        """为应对多主题评估问的召回不够的问题，增加分割逻辑"""
        is_join = False
        for q in self.q_2_1_2_2_reg:
            if query == q:
                is_join = True
        if "，" in query:
            eval_q_list = query.split("，")
        else:
            eval_q_list = query.split(",")
        eval_q_list = [el for el in eval_q_list if el != ""]
        eval_len = [len(el) for el in eval_q_list]
        if len(eval_q_list) == 3:
            ind = eval_len.index(max(eval_len))
            if ind == 0:
                query_1 = eval_q_list[0]
                query_2 = "".join(eval_q_list[1:])
            elif ind == 2:
                query_1 = "".join(eval_q_list[:2])
                query_2 = eval_q_list[2]
            else:
                query_1 = eval_q_list[0]
                query_2 = "".join(eval_q_list[1:])
        elif len(eval_q_list) == 2:
            if [l for l in eval_len if l < 4]:
                query_1 = "".join(eval_q_list)
                query_2 = None
            else:
                query_1 = eval_q_list[0]
                query_2 = eval_q_list[1]
        else:
            query_1 = "".join(eval_q_list)
            query_2 = None

        sent_join_raw, sent_word_raw, _, _ = self.preprocess(query)  #用于直接匹配。

        if query_2 is None:
            logger.info("分割后只有一个句子")
            logger.info("分割后的句子query_1:{}".format(query_1))
            sent_join_1, sent_word_1, sent_word_ext_1, _ = self.preprocess(query_1)
            return sent_join_1, sent_word_1, sent_word_ext_1, "", [], [], sent_join_1, sent_join_raw, sent_word_raw
        else:
            logger.info("分割后的句子query_1:{}".format(query_1))
            logger.info("分割后的句子query_2:{}".format(query_2))
            sent_join_1, sent_word_1, sent_word_ext_1, _ = self.preprocess(query_1)
            sent_join_2, sent_word_2, sent_word_ext_2, _ = self.preprocess(query_2)
            # sent_join_nosplit, sent_word_nosplit, sent_word_ext_nosplit = self.preprocess(query)
            if is_join:
                sent_join_nosplit = "".join([sent_join_1, sent_join_2])
            else:
                sent_join_nosplit = ",".join([sent_join_1, sent_join_2])
            return sent_join_1, sent_word_1, sent_word_ext_1, sent_join_2, sent_word_2, sent_word_ext_2, sent_join_nosplit,sent_join_raw, sent_word_raw


    def similar_calculate_process(self, query, recall_flag = True,use_reg=True, use_intent=False,topk=10, threshold=0.0):
        """带有预处理的相似度计算模块 recall_flag=True 启用多次召回"""
        # print(topk)
        # 先预召回
        # t1 = time.time()
        logger.info("请求问题：{}".format(query))
        logger.info("请求阈值:{}".format(threshold))
        """
        意图判断，如果有精细意图能对应到标准问则直接输出，如果确定不了标准问则先经过核心词过滤，再走相似度模型。
        """

        query = str(query).upper()
        question = copy.copy(query)

        yy = []  # 存放召回的前200个问题
        ss = []  # 存放召回得前200个问题的分数
        yy_sent_join = []  # 存放召回的前200个问题预处理后的文本

        if recall_flag:
            # 下面一行的sent_join没用到。
            # sent_join_1, sent_word_1, sent_word_ext_1, sent_join_2, sent_word_2, sent_word_ext_2, sent_join,\
            # sent_join_raw, sent_word_raw = self.query_split(question)
            sent_join_raw, sent_word_raw, _, _ = self.preprocess(question)


            # 判断是否直接命中
            if sent_join_raw in self.sent_join:
                recall_question_dir, recall_std_question_dir, recall_std_id_dir, recall_score_dir, \
                recall_bm25_yy_dir, recall_bm25_ss_dir, recall_bm25_yy_std_dir, recall_logits_dir = [], [], [], [], [], [], [], []
                sentence_hit = self.sentences[self.sent_join.index(sent_join_raw)]
                if sentence_hit in self.extqstdid:
                    stdid = self.extqstdid[sentence_hit]
                    stdq = self.stdq_id[stdid]
                    recall_std_question_dir.append(stdq)
                    recall_std_id_dir.append(stdid)
                    recall_question_dir.append(sentence_hit)
                    recall_score_dir.append(1.0)
                    recall_bm25_yy_std_dir.append(stdq)
                    recall_bm25_yy_dir.append(sentence_hit)
                    recall_bm25_ss_dir.append(1.0)
                    recall_logits_dir.append(1.0)
                else:
                    recall_std_question_dir.append(sentence_hit)
                    recall_std_id_dir.append(self.id_stdq[sentence_hit])
                    recall_question_dir.append(sentence_hit)
                    recall_score_dir.append(1.0)
                    recall_bm25_yy_std_dir.append(sentence_hit)
                    recall_bm25_yy_dir.append(sentence_hit)
                    recall_bm25_ss_dir.append(1.0)
                    recall_logits_dir.append(1.0)
                return recall_question_dir, recall_std_question_dir, recall_std_id_dir, recall_score_dir, \
                recall_bm25_yy_dir, recall_bm25_ss_dir, recall_bm25_yy_std_dir, recall_logits_dir

            if len(sent_word_raw) == 0:  # 预处理后为空 直接返回空值
                return [], [], [], [], [], [], [], []

            # 增加一个不分割的召回
            yy_3 = []  # 存放不分割query召回的前200个问题
            ss_3 = []  # 存放不分割query召回得前200个问题的分数
            yy_sent_join_3 = []  # 存放不分割query召回的前200个问题预处理后的文本

            sent_join_3, sent_word_3, sent_word_ext_3, _ = self.preprocess(question)
            if len(sent_word_3) != 0:
                sent_word_3 = list(set(sent_word_3))  # todo 召回
                scores_3 = self.bm25_obj.get_scores(sent_word_3)
                rank_ids_3 = sorted(range(len(scores_3)), key=lambda i: scores_3[i], reverse=True)[:200]

                for i in range(len(rank_ids_3)):
                    if len(yy_3) == 200:
                        break
                    if scores_3[rank_ids_3[i]] > 0:  # 过滤掉bm25为0的情况
                        yy_3.append(self.sentences[rank_ids_3[i]])
                        ss_3.append(scores_3[rank_ids_3[i]])
                        yy_sent_join_3.append(self.sent_join[rank_ids_3[i]])
                        # answer.add(self.sent_join[rank_ids[i]])
                        logger.info(f"不分割的query:{sent_join_3} ,bm25召回的结果：{self.sentences[rank_ids_3[i]]}--{scores_3[rank_ids_3[i]]}")
            if len(yy_3) == 0:
                logger.error("query的bm25没有召回任何数据")
                return [], [], [], [], [], [], [], []

            rank_ids = sorted(range(len(yy_3)), key=lambda i: ss_3[i], reverse=True)[:200]
            for i in range(len(rank_ids)):
                yy.append(yy_3[rank_ids[i]])
                ss.append(ss_3[rank_ids[i]])
                yy_sent_join.append(yy_sent_join_3[rank_ids[i]])


        yy_std = []  # 存放召回的问题所对应的标准问
        yy_filter = []  # 相同标准问下只存放Bm25分数最大的那个，取top10~N,相同分数则都召回。
        ss_filter = []  # 相同标准问下只存放Bm25分数最大的那个，取top10~N,相同分数则都召回。
        yy_sent_join_filter = []  # 相同标准问下只存放Bm25分数最大的那个，取top10~N,相同分数则都召回。

        std_q_in_flag = []  # 判断是否已经有属于该标准问的扩展问或者标准问在里面
        std_q_in_score_flag = [] # 判断是否已经有属于该标准问的扩展问或者标准问在里面切分数是否一致
        # pdb.set_trace()
        for i, yq in enumerate(yy):
            if len(set(std_q_in_flag)) == topk:  # 存放topk个就中止。
                break
            if yq in self.extqstdid:  # 判断这个问题是标准问还是扩展问
                stdid = self.extqstdid[yq]
                stdq = self.stdq_id[stdid]
                if stdq not in std_q_in_flag:
                    #pass
                    std_q_in_flag.append(stdq)
                    std_q_in_score_flag.append(ss[i])
                    yy_filter.append(yy[i])
                    ss_filter.append(ss[i])
                    yy_sent_join_filter.append(yy_sent_join[i])
                    yy_std.append(stdq)
                else:
                    ind = std_q_in_flag.index(stdq)
                    if ss[i] == std_q_in_score_flag[ind]:
                        std_q_in_flag.append(stdq)
                        std_q_in_score_flag.append(ss[i])
                        yy_filter.append(yy[i])
                        ss_filter.append(ss[i])
                        yy_sent_join_filter.append(yy_sent_join[i])
                        yy_std.append(stdq)
                    else:
                        continue
            else:
                stdq = yq
                if stdq not in std_q_in_flag:
                    # pass
                    std_q_in_flag.append(stdq)
                    std_q_in_score_flag.append(ss[i])
                    yy_filter.append(yy[i])
                    ss_filter.append(ss[i])
                    yy_sent_join_filter.append(yy_sent_join[i])
                    yy_std.append(stdq)
                else:
                    ind = std_q_in_flag.index(stdq)
                    if ss[i] == std_q_in_score_flag[ind]:
                        std_q_in_flag.append(stdq)
                        std_q_in_score_flag.append(ss[i])
                        yy_filter.append(yy[i])
                        ss_filter.append(ss[i])
                        yy_sent_join_filter.append(yy_sent_join[i])
                        yy_std.append(stdq)
                    else:
                        continue

        logger.info("bm25召回完成")  # todo 如果推送扩展问的话bm25的topk里面要保证问题属于不同的标准问

        # 使用核心词缩小范围
        # ind_stdcore = []  #核心词出现在query中的index
        # for i in range(len(yy_std)):
        #     std_q = yy_std[i]
        #     coreW = self.std2coreW[std_q]
        #     for w in coreW:
        #         if w in query:
        #             ind_stdcore.append(i)
        #             break
        # yy_std_tmp = []
        # yy_filter_tmp = []
        # ss_filter_tmp = []
        # yy_sent_join_filter_tmp = []
        # for d in ind_stdcore:
        #     yy_std_tmp.append(yy_std[d])
        #     yy_filter_tmp.append(yy_filter[d])
        #     ss_filter_tmp.append(ss_filter[d])
        #     yy_sent_join_filter_tmp.append(yy_sent_join_filter[d])
        # yy_std, yy_filter, ss_filter, yy_sent_join_filter = yy_std_tmp, yy_filter_tmp, ss_filter_tmp, yy_sent_join_filter_tmp
        # if len(yy_std) == 0:
        #     return [], [], [], [0.0001], [], [], [], []


        sent_join, _, _, _ = self.preprocess(question)
        query_batch = [sent_join] * len(yy_filter)
        # query_batch = [question] * len(yy_filter) # 不使用预处理后的结果

        logger.info("输入模型的句子：{}".format(query_batch[0]))
        # pdb.set_trace()
        model_input = yy_sent_join_filter
        # model_input = yy_filter  #不使用预处理后的结果
        for i in range(len(model_input)):
            logger.info("输入模型的召回的句子为:{}".format(model_input[i]))

        # logits, sim_score = self.predict_siamese(query_batch, model_input, label_list=label_list)
        sim_score = self.predict_siamese(query_batch, model_input, label_list=label_list)
        # logger.info(str(logits) + " " + str(sim_score))
        # logits, sim_score = self.predict_siamese(query_batch, yy_filter, label_list=label_list) # 不使用预处理后的结果
        # logits = [np.float(bs[0]) for bs in logits]
        sim_score = sim_score.tolist()
        logits = sim_score.copy()
        rerank_ids = sorted(range(len(sim_score)), key=lambda i: sim_score[i],
                          reverse=True)
        # bert_score = [score_trans(bs) for bs in logits]  # logits转换分数 用于排序和拦截
        # bert_score = softmax(bert_score)  # 转换分数
        rerank_ids = rerank_ids[:len(model_input)]
        rerank_yy = []  # 重排序后的问题
        rerank_ss = []  # 重排序后的分数
        rerank_logits = []  # 重排序后的logits
        rerank_bm25_ss = []  # 重排序后的bm25分数的排序
        recall_bm25_yy = []  # 重排序后把bm25召回也重排序
        recall_bm25_ss = []  # 重排序后把bm25召回的问题分数也排序好
        recall_bm25_yy_std = []  # 重排序后把bm25召回的问题对应的标准问放如其中
        recall_logits = []
        rerank_sent_join = []  # 存放召回的预处理后的文本
        for i in range(len(rerank_ids)):
            # if bert_score[rerank_ids[i]] >= threshold:  # 阈值拦截
            if sim_score[rerank_ids[i]] >= threshold:  # 阈值拦截
                rerank_yy.append(yy_filter[rerank_ids[i]])
                # rerank_ss.append(bert_score[rerank_ids[i]])
                rerank_ss.append(sim_score[rerank_ids[i]])
                rerank_sent_join.append(yy_sent_join_filter[rerank_ids[i]])
                rerank_logits.append(logits[rerank_ids[i]])
                rerank_bm25_ss.append(ss_filter[rerank_ids[i]])
            logger.info(f"bert召回结果：{yy_filter[rerank_ids[i]]}--{sim_score[rerank_ids[i]]}")
        # pdb.set_trace()
        # 检查问句是否小于5
        if len(sent_join) < 4:
            if sent_join in rerank_sent_join:
                ind = rerank_sent_join.index(sent_join)
                recall_question, recall_std_question, recall_std_id, recall_score = [rerank_yy[ind]], [], [], [np.float(1.0)]
                recall_logits.append(rerank_logits[ind])
                recall_bm25_yy.append(rerank_yy[ind])
                ind_bm25 = yy_filter.index(rerank_yy[ind])
                recall_bm25_ss.append(ss_filter[ind_bm25])
                if rerank_yy[ind] in self.extqstdid:
                    stdid = self.extqstdid[rerank_yy[ind]]
                    stdq = self.stdq_id[stdid]
                    recall_std_question.append(stdq)
                    recall_std_id.append(stdid)
                    recall_bm25_yy_std.append(stdq)
                else:
                    stdq = rerank_yy[ind]
                    recall_std_question.append(stdq)
                    recall_std_id.append(self.id_stdq[stdq])
                    recall_bm25_yy_std.append(stdq)
                logger.info("短句子直接命中")

                # 把转换的std_question 替换为推送的原始的raw。
                recall_std_question = []
                for sid in recall_std_id:
                    recall_std_question.append(self.stdq_raw_id[sid])

                #规则模块
                t1 = time.time()
                if use_reg:
                    recall_question, recall_std_question, recall_std_id, recall_score, recall_bm25_yy, \
                    recall_bm25_ss, recall_bm25_yy_std, recall_logits = self.reg_filter(query, recall_question,
                                                                                        recall_std_question,
                                                                                        recall_std_id, recall_score, \
                                                                                        recall_bm25_yy, recall_bm25_ss,
                                                                                        recall_bm25_yy_std,
                                                                                        recall_logits)

                t2 = time.time()
                print("规则模块耗时{}".format(t2-t1))
                return recall_question, recall_std_question, recall_std_id, recall_score, recall_bm25_yy, \
                       recall_bm25_ss, recall_bm25_yy_std, recall_logits
            else:
                rerank_ss = [np.float(rs*0.8) for rs in rerank_ss]
                recall_question, recall_std_question, recall_std_id, recall_score = rerank_yy, [], [], rerank_ss
                recall_bm25_yy = recall_question
                recall_logits = rerank_logits
                recall_bm25_ss = rerank_bm25_ss
                for i in range(len(rerank_yy)):
                    if rerank_yy[i] in self.extqstdid:
                        stdid = self.extqstdid[rerank_yy[i]]
                        stdq = self.stdq_id[stdid]
                        recall_std_question.append(stdq)
                        recall_std_id.append(stdid)
                        recall_bm25_yy_std.append(stdq)
                    else:
                        recall_std_question.append(rerank_yy[i])
                        recall_std_id.append(self.id_stdq[rerank_yy[i]])
                        recall_bm25_yy_std.append(rerank_yy[i])
        else:
            # #判断是否直接命中，直接命中则赋值1
            # if sent_join_raw in rerank_sent_join:
            #     ind = rerank_sent_join.index(sent_join_raw)
            #     rerank_ss[ind] = 1.0
            # pdb.set_trace()
            recall_question, recall_std_question, recall_std_id, recall_score = rerank_yy, [], [], rerank_ss
            recall_bm25_yy = recall_question
            recall_logits = rerank_logits
            recall_bm25_ss = rerank_bm25_ss
            for i in range(len(rerank_yy)):
                if rerank_yy[i] in self.extqstdid:
                    stdid = self.extqstdid[rerank_yy[i]]
                    stdq = self.stdq_id[stdid]
                    recall_std_question.append(stdq)
                    recall_std_id.append(stdid)
                    recall_bm25_yy_std.append(stdq)
                else:
                    recall_std_question.append(rerank_yy[i])
                    recall_std_id.append(self.id_stdq[rerank_yy[i]])
                    recall_bm25_yy_std.append(rerank_yy[i])
        if len(recall_question) == 0:  # 被阈值拦截导致召回为空
            logger.info("返回为空")
            return [], [], [], [], [], [], [], []

        # 把转换的std_question 替换为推送的原始的raw。
        recall_std_question = []
        for sid in recall_std_id:
            recall_std_question.append(self.stdq_raw_id[sid])

        # 把bm25召回的标准问转换为原始标准问
        yy_std_raw = []
        for ys in yy_std:
            sid = self.id_stdq[ys]
            std_q_raw = self.stdq_raw_id[sid]
            yy_std_raw.append(std_q_raw)

        t3 = time.time()
        # 意图模块
        # if use_intent:
        #     if recall_std_question[0] in intent_text and recall_score[0] > 0.8:
        #     # if recall_std_question[0] in intent_text:
        #         recall_question, recall_std_question, recall_std_id, recall_score, \
        #         yy_filter, ss_filter, yy_std_raw, recall_logits = self.intent_filter(query, recall_question, recall_std_question,
        #                                                                              recall_std_id, recall_score, yy_filter, ss_filter,
        #                                                                              yy_std_raw, recall_logits)


        # 正则模块
        if use_reg:
            recall_question, recall_std_question, recall_std_id, recall_score, \
            yy_filter, ss_filter, yy_std_raw, recall_logits = self.reg_filter(query, recall_question, recall_std_question,
                                                                              recall_std_id, recall_score,
                                                                              yy_filter, ss_filter, yy_std_raw,
                                                                              recall_logits)
        t4 = time.time()
        print("规则过滤时间{}".format(t4-t3))

        return recall_question, recall_std_question, recall_std_id, recall_score, \
               yy_filter, ss_filter, yy_std_raw, recall_logits


    def reg_filter(self, query, recall_question, recall_std_question, recall_std_id, recall_score,
                   yy_filter, ss_filter, yy_std_raw, recall_logits):

        # 0. 对标准问做标点符号处理
        recall_std_question_process = []
        t1 = time.time()
        if recall_std_question:
            # for sq in recall_std_question:
            #     # _, _, _, pro_stdq = self.preprocess(sq)
            # pro_stdq = re.sub(
            #     '[:→℃&*一~’.『/\-』＝″【—?、。“”《》！，：；？．,\'·<>（）〔〕\[\]()+～×／①②③④⑤⑥⑦⑧⑨⑩ⅢВ";#@γμφΔ■▲＃％＆＇〉｡＂＄＊＋－＜＞＠［＼］＾｀｛｜｝｟｠｢｣､〃「」】〖〗〘〙〚〛〜〝〞〟〰〾〿‘‛„‟…‧﹏]+',
            #     "", sq)
            #     recall_std_question_process.append(pro_stdq)
            query_process = re.sub(
                '[:→℃&*一~’.『/\-』＝″【—?、。“”《》！，：；？．,\'·<>（）〔〕\[\]()+～×／①②③④⑤⑥⑦⑧⑨⑩ⅢВ";#@γμφΔ■▲＃％＆＇〉｡＂＄＊＋－＜＞＠［＼］＾｀｛｜｝｟｠｢｣､〃「」】〖〗〘〙〚〛〜〝〞〟〰〾〿‘‛„‟…‧﹏]+',
                "", query)
        else:
            return [], [], [], [], [], [], [], []
        t2 = time.time()
        logger.info("预处理耗时{}".format(t2 - t1))
        # 1.正规则的匹配
        pos_match_flag = False
        print(self.question2reg)
        for i, sq in enumerate(recall_std_question):
            print(i)
            print(sq)
            if sq not in self.question2reg:
                continue
            else:
                pos_regs = self.question2reg[sq]["pos"]
                print(pos_regs)
                if pos_regs:
                    for pn in pos_regs:
                        print(pn)
                        # print(recall_std_question_process[i])
                        if re.search(pn, query_process):
                            logger.info("标准问{}的正向正则{}命中".format(sq, pn))
                            recall_score[i] = 1.0
                            pos_match_flag = True
                            break
        inds_rerank = sorted(range(len(recall_score)), key=lambda i: recall_score[i],
                             reverse=True)
        recall_question_new, recall_std_question_new, recall_std_id_new, recall_score_new, \
        yy_filter_new, ss_filter_new, yy_std_raw_new, recall_logits_new = [], [], [], [], [], [], [], []
        if pos_match_flag:
            for ind in inds_rerank:
                recall_question_new.append(recall_question[ind])
                recall_std_question_new.append(recall_std_question[ind])
                recall_std_id_new.append(recall_std_id[ind])
                recall_score_new.append(recall_score[ind])
                yy_filter_new.append(yy_filter[ind])
                ss_filter_new.append(ss_filter[ind])
                yy_std_raw_new.append(yy_std_raw[ind])
                recall_logits_new.append(recall_logits[ind])
        t3 = time.time()
        logger.info("正向正则耗时{}".format(t3 - t2))
        if pos_match_flag:
            return recall_question_new, recall_std_question_new, recall_std_id_new, recall_score_new, \
                   yy_filter_new, ss_filter_new, yy_std_raw_new, recall_logits_new

        # 2.如果正规则没有匹配到，就匹配负规则，负规则就只看top1。

        if pos_match_flag is False:
            sq_top1 = recall_std_question[0]
            if sq_top1 not in self.question2reg:
                return recall_question, recall_std_question, recall_std_id, recall_score, \
                       yy_filter, ss_filter, yy_std_raw, recall_logits
            else:
                neg_regs = self.question2reg[sq_top1]["neg"]
                for nr in neg_regs:
                    if re.search(nr, query_process):
                        recall_score = [0.5 * rs for rs in recall_score]
                        logger.info("标准问{}的负向正则{}命中".format(sq_top1, nr))
        t4 = time.time()
        print("反向正则耗时{}".format(t4 - t3))
        return recall_question, recall_std_question, recall_std_id, recall_score, \
               yy_filter, ss_filter, yy_std_raw, recall_logits

    def intent_filter(self, query, recall_question, recall_std_question, recall_std_id, recall_score,
                   yy_filter, ss_filter, yy_std_raw, recall_logits):
        intent_res = get_intent(query)
        recall_question[0] = intent_res
        recall_std_question[0] = intent_res
        recall_std_id[0] = self.id_stdq[intent_res]
        recall_score[0] = 1.0
        return recall_question, recall_std_question, recall_std_id, recall_score, yy_filter, ss_filter, yy_std_raw, recall_logits



    def preprocess(self, sent):
        """

        :param sent:
        :return: sent_join 不去停用词的预处理结果(去标点和做同义词替换)
                 sent_word 去同义词和标点、同义词替换后的词表，用于BM25召回
                 sent_word_ext 增加扩展词后（废弃不用）
                 sent_droppunc_nostopword 不去停用词的预处理结果(去标点和不做同义词替换)

        """
        """去标点符号，去停用词，同义词替换"""
        logger.info("原句子：{}".format(sent))
        # 过滤掉右边的语气词 应对孙慧玲测试
        # for w in self.drop_word_right:
        #     if w in sent:
        #         sent = sent.rstrip(w)
        #         logger.info("去掉句子右边的语气词后的结果:{}".format(sent))
        # for tw in tone_words:  # 即句子中间的词
        #     if tw in sent:
        #         sent = re.sub(tw, "", sent)
        # logger.info("去掉语气词后的结果：{}".format(sent))
        # if sent == "":
        #     return "", [], []

        sent_cut = wordseg(sent, self.stop_words) # 完成去停用词和分词
        logger.info("分词后结果：{}".format(sent_cut))
        sent_droppunc = [sc for sc in sent_cut if sc not in punc]
        logger.info("去掉标点符号后的结果：{}".format(sent_droppunc))
        sent_word = [self.synword_map[sc] if sc in self.synword_map else sc for sc in sent_droppunc]
        logger.info("同义词替换后的结果：{}".format(sent_word))
        # sent_word = [sw for sw in sent_word if sw not in tone_words]
        # logger.info("去掉语气词后的结果：{}".format(sent_word))
        sent_cut_nostopword = wordseg(sent, [])
        logger.info("不去停用词分词后结果：{}".format(sent_cut_nostopword))
        sent_droppunc_nostopword = [sc for sc in sent_cut_nostopword if sc not in punc]  #可用于做正则的匹配
        logger.info("不去停用词去掉标点符号后的结果：{}".format(sent_droppunc_nostopword))
        sent_word_nostopword = [self.synword_map[sc] if sc in self.synword_map else sc for sc in sent_droppunc_nostopword]
        logger.info("不去停用词同义词替换后的结果：{}".format(sent_word_nostopword))



        sent_join = "".join(sent_word_nostopword)
        # 扩展词扩展
        sent_word_ext = sent_word.copy()  # 扩展后的分词结果
        # for sw in sent_word:
        #     if sw in self.extend_word:
        #         sent_word_ext.extend(self.extend_word[str(sw)])
        # logger.info("扩展词扩展后的结果：{}".format(sent_word_ext))
        return sent_join, sent_word, sent_word_ext, ''.join(sent_droppunc_nostopword)


    def init_ext_std(self, data_init):
        try:
            self.stdq_id = dict()  # {id:std_question}
            self.id_stdq = dict()  # {std_question:id}
            self.extq_id = dict()  # {id:ext_question}
            self.stdq_raw_id = dict()  # {id:std_question_raw} # 存放未转换大小写的标准问，即推送进来的原始的标准问
            self.extq_raw_id = dict()  # {id:ext_question_raw} # 存放原始的扩展问
            # self.extinstd = defaultdict(list) # {stdid:[extid,extid]}
            self.extqstdid = dict()  # {extq:stdid} 扩展问属于哪个标准问id
            # self.questions = []  # 标准问和扩展问的合并
            self.sentences = []  # 标准问和扩展问的合并
            self.texts = []
            self.sent_join = []
            self.dictionary = None
            self.bm25_obj = None
            self.question2reg = defaultdict(dict)

            for dt in data_init:
                is_standard = dt["is_standard"]
                if is_standard=="0":
                    question_id = dt["question_id"]
                    question = dt["question"]
                    self.extq_raw_id[question_id] = question # 保存原始的question 和id对应关系
                    # print(question)  # todo 删掉
                    standard_id = str(dt["standard_id"])
                    logger.info("扩展问:{},所属的标准问id:{}".format(question, standard_id))
                    question = str(question).upper()
                    if PROCESS == "1":
                        #预处理：a.去除特殊字符与标点符号 b.去除停用词 c.同义词替换
                        sent_join, sent_word, sent_word_ext, _ = self.preprocess(question)
                        if len(sent_word) == 0:
                            continue
                        self.sent_join.append(sent_join)  # 预处理后拼接起来的句子用过来计算模型相似度
                        sent_word_ext = list(set(sent_word_ext))
                        self.texts.append(sent_word_ext)

                        # sent_word = list(set(sent_word))  # 去重
                        # self.texts.append(sent_word)

                    self.extq_id[question_id] = question
                    self.extqstdid[question] = standard_id
                    self.sentences.append(question)
                elif is_standard=="1":
                    question_id = str(dt["question_id"])
                    question = dt["question"]
                    logger.info("标准问:{},标准问id:{}".format(question, question_id))
                    self.stdq_raw_id[question_id] = question # 保存原始的id与question对应关系
                    question = str(question).upper()
                    # print("标准问1111111111")  # todo 删掉
                    if PROCESS == "1":
                        # print("22222222222")
                        sent_join, sent_word, sent_word_ext, _ = self.preprocess(question)
                        if len(sent_word) == 0:
                            continue
                        self.sent_join.append(sent_join)  # 预处理后拼接起来的句子用过来计算模型相似度
                        sent_word_ext = list(set(sent_word_ext)) # todo 测试是否是扩展的问题
                        # sent_word_ext = list(set(sent_word)) # 词条里不使用扩展，query中扩展
                        self.texts.append(sent_word_ext)

                        # sent_word = list(set(sent_word))  # 去重
                        # self.texts.append(sent_word)

                    self.stdq_id[question_id] = question
                    self.id_stdq[question] = question_id
                    self.sentences.append(question)
                elif is_standard == "2":   #  正则模块。
                    print("正则")
                    question = dt["question"]
                    pos_reg, neg_reg = [], []
                    if "pos_reg" in dt:
                        pos_reg = dt["pos_reg"].split("#")
                    if "neg_reg" in dt:
                        neg_reg = dt["neg_reg"].split("#")
                    question_with_reg = defaultdict(list)
                    question_with_reg["pos"] = pos_reg
                    question_with_reg["neg"] = neg_reg
                    self.question2reg[question] = question_with_reg
            logger.info("数据接收完成")
            if PROCESS != "1":
                self.texts = [wordseg(w, stop_words=self.stop_words) for w in
                              self.sentences]
            # self.dictionary = corpora.Dictionary(self.texts) # todo 召回
            # corpus = [self.dictionary.doc2bow(text) for text in self.texts]
            # self.bm25_obj = bm25.BM25(corpus)
            self.bm25_obj = bm25.BM25(self.texts)
            logger.info("数据初始化完成")
            return True
        except:
            logger.error("数据初始化失败")
            return False

    def intention_filter(self, query):
        """意图检查"""
        logger.info("请求的问题:{}".format(query))
        label_list_intent = ["0", "1"]
        predict_example = ip.InputExample("id", query, None, label_list_intent[0])
        feature = ip.convert_single_example(5, predict_example, label_list,
                                         30, tokenizer)
        # print(feature.input_ids)
        prediction = self.predict_fn_intent({
            "input_ids": [feature.input_ids],
            "input_mask": [feature.input_mask],
            "segment_ids": [feature.segment_ids],
            "label_ids": [feature.label_id],
        })
        prob = prediction["probabilities"].tolist()
        intent_label = label_list_intent[prob[0].index(max(prob[0]))]
        if intent_label == "1":
            logger.info("意图命中faq")
            return True
        else:
            logger.info("意图未命中faq")
            return False


workorder = SmartWorkorder()

def push_product(data_init):
    res = workorder.init_ext_std(data_init)
    return res


def get_result(query_input, threshold=0.1):
    """请求结果"""
    # if PROCESS == "1":
    if INTENT_FLAG == "1":  #启用意图分类
        logger.info("使用意图分类模型")
        intent_res = workorder.intention_filter(query_input)
        if intent_res is True:
            recall_question, recall_std_question, recall_std_id, recall_score, recall_bm25_yy, recall_bm25_ss, recall_bm25_yy_std, recall_logits = workorder.similar_calculate_process(query=query_input, recall_flag=True, threshold=threshold)
        else:
            recall_question, recall_std_question, recall_std_id, recall_score, recall_bm25_yy, recall_bm25_ss, \
            recall_bm25_yy_std, recall_logits = [], [], [], [], [], [], [], []
    else:
        logger.info("不使用意图分类模型")
        recall_question, recall_std_question, recall_std_id, recall_score, recall_bm25_yy, recall_bm25_ss, recall_bm25_yy_std, recall_logits = workorder.similar_calculate_process(
            query=query_input, recall_flag=True, use_reg=True, use_intent=False, threshold=threshold)
    # else:
    #     recall_question, recall_std_question, recall_std_id, recall_score, yy, ss, yy_standard = workorder.similar_calculate(query=query_input, threshold=threshold)
    return recall_question, recall_std_question, recall_std_id, recall_score, recall_bm25_yy, recall_bm25_ss, recall_bm25_yy_std, recall_logits

def get_extend_similar(ext_add, standard_question, threshold):
    print("222")
    logger.info("请求的问题{}".format(ext_add))
    recall_question, recall_std_question, recall_std_id, recall_score, recall_bm25_yy, recall_bm25_ss, recall_bm25_yy_std, recall_logits = workorder.similar_calculate_process(
        query=ext_add, recall_flag=True,use_reg=True)
    print("请求的结果")
    # pdb.set_trace()
    print(recall_std_question)
    recall_question_new, recall_std_question_new, recall_score_new = [], [], []
    for i, rsn in enumerate(recall_score):
        if rsn >= threshold:
            recall_question_new.append(recall_question[i])
            recall_std_question_new.append(recall_std_question[i])
            recall_score_new.append(recall_score[i])
    print(recall_std_question_new)
    if standard_question in recall_std_question_new:
        ind = recall_std_question_new.index(standard_question)
        recall_question_new.pop(ind)
        recall_std_question_new.pop(ind)
        recall_score_new.pop(ind)
        return recall_question_new, recall_std_question_new, recall_score_new
    else:
        return recall_question_new, recall_std_question_new, recall_score_new


def get_query_process(query):
    """返回对query预处理后的结果"""
    sent_join_1, sent_word_1, sent_word_ext_1, sent_join_2, sent_word_2, sent_word_ext_2, sent_join,sent_join_raw, sent_word_raw = workorder.query_split(query)
    return sent_join


def get_sentence_process(sent):
    """返回对标准问和扩展问预处理后的结果"""
    sent_join, sent_word, sent_word_ext, _ = workorder.preprocess(sent)
    return sent_join, sent_word, sent_word_ext


def caculate_score():
    """计算文本之间的相似度"""
    token_replace = '创业板'
    xkt = pd.read_excel("./tmp/实体+句式.xlsx", sheet_name="合并句式1")
    ktqx = pd.read_excel("./tmp/实体+句式.xlsx", sheet_name="合并句式2")
    xkt_text = xkt["创业板开通和科创板开通合并"].tolist()
    xkt_text_rep = [xt.replace("[token]", token_replace) for xt in xkt_text]

    ktqx_text = ktqx["查创业板权限和查科创板合并"].tolist()
    ktqx_text_rep = [ktx.replace("[token]", token_replace) for ktx in ktqx_text]

    xkt_text_pre, ktqx_text_pre = [], []
    for xtr in xkt_text_rep:
        pre_result, _, _, _ = workorder.preprocess(xtr)
        xkt_text_pre.append(pre_result)
    for ktr in ktqx_text_rep:
        pre_result, _, _, _ = workorder.preprocess(ktr)
        ktqx_text_pre.append(pre_result)
    xkt_s, simest, simscore = [], [], []

    for i, tt in enumerate(xkt_text):
        print(tt)
        texta = [xkt_text_pre[i]] * len(ktqx_text)
        textb = ktqx_text_pre
        logits, scores = workorder.predict_siamese(texta, textb)
        # inds = [i for i in range(len(ktqx_text_pre))]
        inds_rerank = sorted(range(len(ktqx_text_pre)), key=lambda i: scores[i],
                             reverse=True)
        xkt_tmp, simest_tmp, score_tmp = [], [], []
        for ind in inds_rerank:
            if scores[ind] < 0.8:
                break
            else:
                xkt_tmp.append(tt)
                simest_tmp.append(ktqx_text[ind])
                score_tmp.append(scores[ind][0])
        if len(simest_tmp) != 0:
            xkt_s.extend(xkt_tmp)
            simest.extend(simest_tmp)
            simscore.extend(score_tmp)
    # print(len())
    df_writer = pd.DataFrame({"科创板或创业板开通": xkt_s, "查科创板或创业板权限": simest, "相似度": simscore})
    with pd.ExcelWriter("/home/gaof/tmp/new_faq_python_word_ivr/service_core/tmp/相似度比较.xlsx") as writer:
        df_writer.to_excel(excel_writer=writer, index=False)



    # sent_join_raw, sent_word_raw, _, _ = workorder.preprocess(text)

if __name__ == '__main__':
    # workorder.init()
    # data = [
    #     {"sender": "1", "id": "123", "sentence_id": "323",
    #      "sentence": "哦四，五。广发多元新兴股票基金。啊。这个广发多元新兴股票基金，他是在场内是不难。嗯。买卖也不能申购赎回的，他只能在场外申购。嗯。"},
    #     {"sender": "1", "id": "123", "sentence_id": "323",
    #      "sentence": "核实一下您的身份信息"}]
    workorder.init()
    # data = [{"sender": "1", "id": "2172600", "sentence_id": "6475", "sentence": "请您提供身份证号，我马上为您查询。"}, {"sender": "1", "id": "2172600", "sentence_id": "6476", "sentence": "您好，请问有什么可以帮到您？"}, {"sender": "2", "id": "2172600", "sentence_id": "6477", "sentence": "喂，你好，我先问一下我。那个。基金啊。一直。就是。"}, {"sender": "1", "id": "2172600", "sentence_id": "6478", "sentence": "嗯。"}, {"sender": "2", "id": "2172600", "sentence_id": "6479", "sentence": "在在我那个。"}, {"sender": "2", "id": "2172600", "sentence_id": "6480", "sentence": "呃，那个。证券的基金要怎么不同意呀？"}, {"sender": "1", "id": "2172600", "sentence_id": "6481", "sentence": "呃。请问先生，您具体购买的基金名称是什么？还记得吗？"}, {"sender": "2", "id": "2172600", "sentence_id": "6482", "sentence": "我。没有。根本。购买，我已经卖掉我接着做他手续费啊，什么？"}, {"sender": "2", "id": "2172600", "sentence_id": "6483", "sentence": "完全不一样的一个什么？不同的。"}, {"sender": "1", "id": "2172600", "sentence_id": "6484", "sentence": "手续费是吧？"}, {"sender": "1", "id": "2172600", "sentence_id": "6485", "sentence": "啊。"}, {"sender": "2", "id": "2172600", "sentence_id": "6486", "sentence": "呃，就是说。一一个是当天可以。交易的。"}, {"sender": "2", "id": "2172600", "sentence_id": "6487", "sentence": "嗯。"}, {"sender": "2", "id": "2172600", "sentence_id": "6488", "sentence": "嗯家，就是那个证券。率的基金是当天可以交易的。那个。要是要七。加三才，可以那个大。的二三把区别的。"}, {"sender": "1", "id": "2172600", "sentence_id": "6489", "sentence": "呃。那先生您有没有具体的基金？因为不同基金的话，他的交易规则可能有些不一样，有些基金的话呢，他可以在证券公司就是二级市场买卖的，但是有一些基金，它是普通的场外基金。只能。申购赎回呀，不支持在场内进行买卖的。呃。不同基金的话呢？交易规则有可能不一样，要看具体的基金来具体分析了。"}, {"sender": "2", "id": "2172600", "sentence_id": "6490", "sentence": "那我给你呀。同样是。广发的啊，怎么怎么样的一个市场，那？是不是？"}, {"sender": "1", "id": "2172600", "sentence_id": "6491", "sentence": "呃。就是有一些基金，它是可以支持在场内交易的，有些他可能不支持，要看具体的基金的。"}, {"sender": "2", "id": "2172600", "sentence_id": "6492", "sentence": "啊。那买涨。不一样的。那个对那个。我当天可以交易，可以很少的。那个。吧。要是场外的。"}, {"sender": "1", "id": "2172600", "sentence_id": "6493", "sentence": "设。嗯。"}, {"sender": "1", "id": "2172600", "sentence_id": "6494", "sentence": "就是您是指在场内交易的手续费跟场外交易的手续费。"}, {"sender": "2", "id": "2172600", "sentence_id": "6495", "sentence": "他那个不是那个证券交易了。"}, {"sender": "2", "id": "2172600", "sentence_id": "6496", "sentence": "啊。"}, {"sender": "1", "id": "2172600", "sentence_id": "6497", "sentence": "场内交易的话，就是在二级市场那嘛，就是。呃。证券公司的那个。"}, {"sender": "2", "id": "2172600", "sentence_id": "6498", "sentence": "要是。股票。这个。持仓的股票，那个市场。"}, {"sender": "1", "id": "2172600", "sentence_id": "6499", "sentence": "对。"}, {"sender": "1", "id": "2172600", "sentence_id": "6500", "sentence": "对对，对对。嗯。"}, {"sender": "2", "id": "2172600", "sentence_id": "6501", "sentence": "哎。"}, {"sender": "2", "id": "2172600", "sentence_id": "6502", "sentence": "那。"}, {"sender": "2", "id": "2172600", "sentence_id": "6503", "sentence": "交易。席位。不同。"}, {"sender": "2", "id": "2172600", "sentence_id": "6504", "sentence": "差别很大。嘞。"}, {"sender": "1", "id": "2172600", "sentence_id": "6505", "sentence": "嗯，因为有些基金的话，的确场外交易跟场内的话，他的确是会有手续费不一样，这种情况的要看具体的基金的。"}, {"sender": "2", "id": "2172600", "sentence_id": "6506", "sentence": "嗯。那么是什么？区别，那。"}, {"sender": "1", "id": "2172600", "sentence_id": "6507", "sentence": "这个要看具体的基金具体分析了您这边有没有，比如说具体基金的一个名称的呢。"}, {"sender": "2", "id": "2172600", "sentence_id": "6508", "sentence": "现在。"}, {"sender": "2", "id": "2172600", "sentence_id": "6509", "sentence": "呃，就是那个。"}, {"sender": "2", "id": "2172600", "sentence_id": "6510", "sentence": "嗯，就是广发下班啊。"}, {"sender": "1", "id": "2172600", "sentence_id": "6511", "sentence": "哦。。四，五。广发多元新兴股票基金。啊。这个广发多元新兴股票基金，他是在场内是不难。嗯。买卖也不能申购赎回的，他只能在场外申购。嗯。"}, {"sender": "2", "id": "2172600", "sentence_id": "6512", "sentence": "。四，五。"}, {"sender": "2", "id": "2172600", "sentence_id": "6513", "sentence": "哎，我知道。他有一个那个。广发医药。那么。差。啊。股票。交易市场g。加一的。"}, {"sender": "1", "id": "2172600", "sentence_id": "6514", "sentence": "医药。是吗？"}, {"sender": "2", "id": "2172600", "sentence_id": "6515", "sentence": "广发医药，广发医药。"}, {"sender": "1", "id": "2172600", "sentence_id": "6516", "sentence": "广发医药。"}, {"sender": "1", "id": "2172600", "sentence_id": "6517", "sentence": "呃。"}, {"sender": "1", "id": "2172600", "sentence_id": "6518", "sentence": "广发医药。您是。呃指数分级。指数基金吗？还是说是股票型的基金的呢？"}, {"sender": "2", "id": "2172600", "sentence_id": "6519", "sentence": "股票型的。"}, {"sender": "1", "id": "2172600", "sentence_id": "6520", "sentence": "哦。是医疗保健股票基金嘛，是这个吗？"}, {"sender": "2", "id": "2172600", "sentence_id": "6521", "sentence": "嗯。呃。"}, {"sender": "2", "id": "2172600", "sentence_id": "6522", "sentence": "我给你还有一个问题，就是说。连连续交易的。都是。公布的。还是私募的呀。那个。场内的。"}, {"sender": "1", "id": "2172600", "sentence_id": "6523", "sentence": "这些都是公募的基金。"}, {"sender": "2", "id": "2172600", "sentence_id": "6524", "sentence": "哦。直播在哪里？在？"}, {"sender": "1", "id": "2172600", "sentence_id": "6525", "sentence": "嗯。私募基金的话呢？就是。嗯，一般要看不同的基金。私募基金的话呢，它有一些客户，他应该是不支持在场内交易的。呃。您指的私募基金是指我们的高端理财的那种产品嘛，就是一般，它的那个起点金额可能是申购的起点金额可能是相对来说比较高，有些可能是？呃。万七，有些的话可能是的。"}, {"sender": "2", "id": "2172600", "sentence_id": "6526", "sentence": "哦。这样的。没有的。没有。万的是吧？"}, {"sender": "1", "id": "2172600", "sentence_id": "6527", "sentence": "哦。暂时高端理财的那种产品暂时。就没有十万或者是五万起的。一般都是。比较多，可能。40万或者是一百万起。的。"}, {"sender": "2", "id": "2172600", "sentence_id": "6528", "sentence": "哦。"}, {"sender": "2", "id": "2172600", "sentence_id": "6529", "sentence": "哦，这样的这样，我就。那个。"}, {"sender": "1", "id": "2172600", "sentence_id": "6530", "sentence": "嗯。嗯。"}, {"sender": "2", "id": "2172600", "sentence_id": "6531", "sentence": "是在。交易的也是公布的是吧？"}, {"sender": "1", "id": "2172600", "sentence_id": "6532", "sentence": "对一般就是断普通的，就是我们在网站上公布这种啊。呃。就普通的这种股票型啊。混合型指数型，债券型，货币型，这种一般都是普通的那种公募基金，然后有些基金的话呢，他可能就是可以支持在二级市场的。比如说有异动，乐福的。它可以支持在场内申购，赎回或者是买卖，如果是普通的那种。股票型基金的话，他不支持在场内申购，赎回或者是买卖，只能普通的场外申购赎回的。"}, {"sender": "2", "id": "2172600", "sentence_id": "6533", "sentence": "哦，这样子。那。呃。一般在二级市场上。就是公布的。"}, {"sender": "2", "id": "2172600", "sentence_id": "6534", "sentence": "没有什么的？"}, {"sender": "1", "id": "2172600", "sentence_id": "6535", "sentence": "大部分的话应该是公布，但是具体的基金要具体分析了。"}, {"sender": "2", "id": "2172600", "sentence_id": "6536", "sentence": "啊，这我。"}, {"sender": "2", "id": "2172600", "sentence_id": "6537", "sentence": "啊。银光。那个光木的。"}, {"sender": "1", "id": "2172600", "sentence_id": "6538", "sentence": "大部分来说的话，应该都是。这种。大部分都是公募的基金。"}, {"sender": "2", "id": "2172600", "sentence_id": "6539", "sentence": "啊，这样的。"}, {"sender": "1", "id": "2172600", "sentence_id": "6540", "sentence": "对，但是。呃。嗯。嗯。"}, {"sender": "2", "id": "2172600", "sentence_id": "6541", "sentence": "那个。"}, {"sender": "2", "id": "2172600", "sentence_id": "6542", "sentence": "啊。一半。场内的都是，就是公布的是吧？"}, {"sender": "1", "id": "2172600", "sentence_id": "6543", "sentence": "呃。二，八是吗？"}, {"sender": "1", "id": "2172600", "sentence_id": "6544", "sentence": "交割。是消费etf，然后。是吧？啊。要不就就三，九。"}, {"sender": "2", "id": "2172600", "sentence_id": "6545", "sentence": "诶。"}, {"sender": "2", "id": "2172600", "sentence_id": "6546", "sentence": "三，九。"}, {"sender": "2", "id": "2172600", "sentence_id": "6547", "sentence": "九。"}, {"sender": "1", "id": "2172600", "sentence_id": "6548", "sentence": "这个是信息技术。哦。对。"}, {"sender": "2", "id": "2172600", "sentence_id": "6549", "sentence": "对信息技术。"}, {"sender": "2", "id": "2172600", "sentence_id": "6550", "sentence": "就是那个。"}, {"sender": "2", "id": "2172600", "sentence_id": "6551", "sentence": "还有南方家务接近。"}, {"sender": "1", "id": "2172600", "sentence_id": "6552", "sentence": "哦。"}, {"sender": "2", "id": "2172600", "sentence_id": "6553", "sentence": "都是观摩的是吧？"}, {"sender": "1", "id": "2172600", "sentence_id": "6554", "sentence": "对这种是etf的基金。对是公募的指数基金。指数基金。"}, {"sender": "2", "id": "2172600", "sentence_id": "6555", "sentence": "啊，这我放心。如果。什么转换？没有做保险是吧？"}, {"sender": "1", "id": "2172600", "sentence_id": "6556", "sentence": "额。这个也不一定要具体看具体的基金来分析了。"}, {"sender": "2", "id": "2172600", "sentence_id": "6557", "sentence": "嗯。好的好的。谢谢你啊。啊。"}, {"sender": "1", "id": "2172600", "sentence_id": "6558", "sentence": "对。嗯。"}, {"sender": "1", "id": "2172600", "sentence_id": "6559", "sentence": "不客气，不客气，那先生这边再提醒一下，因为像这种。比如说etf。的他是指数型基金，大部分风险等级是属于中风险，还有一些股票型基金的话，风险等级是属于中风险的，像这个多元，新兴。股票。一般建议您在购买之前呢，您可以先了解一下基金的合同，嗯，账目说明书，再结合您的风险承受能力进行投资了。"}, {"sender": "2", "id": "2172600", "sentence_id": "6560", "sentence": "哦，这样的。那要不结结。巴列应该是公布的。"}, {"sender": "1", "id": "2172600", "sentence_id": "6561", "sentence": "啊。对。"}, {"sender": "1", "id": "2172600", "sentence_id": "6562", "sentence": "是吗？"}, {"sender": "2", "id": "2172600", "sentence_id": "6563", "sentence": "。哎。"}, {"sender": "1", "id": "2172600", "sentence_id": "6564", "sentence": "幺五。"}, {"sender": "1", "id": "2172600", "sentence_id": "6565", "sentence": "。"}, {"sender": "1", "id": "2172600", "sentence_id": "6566", "sentence": "这个是消费，这个好像不是我们广发基金的产品的。"}, {"sender": "2", "id": "2172600", "sentence_id": "6567", "sentence": "这个。也不是这个板块一样买。"}, {"sender": "1", "id": "2172600", "sentence_id": "6568", "sentence": "幺五。是吗？"}, {"sender": "2", "id": "2172600", "sentence_id": "6569", "sentence": "。"}, {"sender": "1", "id": "2172600", "sentence_id": "6570", "sentence": "啊。。"}, {"sender": "1", "id": "2172600", "sentence_id": "6571", "sentence": "对，这个是我们。广发。医药卫生etf，这个也是etf的是公募的基金，对。"}, {"sender": "2", "id": "2172600", "sentence_id": "6572", "sentence": "啊。好的好的啊，行啊。那个我在。这个费用。很少买。这个。就像股票差不多的费用。"}, {"sender": "1", "id": "2172600", "sentence_id": "6573", "sentence": "好。您看有没有其他可以。"}, {"sender": "1", "id": "2172600", "sentence_id": "6574", "sentence": "嗯，这种etf基金的话呢？它的那个交易费用的话就是一般按照证券公司那边收取对应的佣金为准的，具体要看证券公司那边的标准了。"}, {"sender": "2", "id": "2172600", "sentence_id": "6575", "sentence": "哎呦。反正。这边显示做的很小。很小。我做。选股票。"}, {"sender": "1", "id": "2172600", "sentence_id": "6576", "sentence": "对他。"}, {"sender": "1", "id": "2172600", "sentence_id": "6577", "sentence": "对他这种etf是在场内交易的，所以。具体。的费用要看证券公司那边的规定为准了。"}, {"sender": "2", "id": "2172600", "sentence_id": "6578", "sentence": "一天卖掉就是。指数型的。"}, {"sender": "1", "id": "2172600", "sentence_id": "6579", "sentence": "对他是指数型的。"}, {"sender": "2", "id": "2172600", "sentence_id": "6580", "sentence": "最新风险大那个股票新。上线。的。"}, {"sender": "1", "id": "2172600", "sentence_id": "6581", "sentence": "呃。其实指数型的话更。呃。这种。股票行的话。呢和风险都是差不多的，就是它中指数型的话呢？他其实也是类似。股票型的。所以它的风险的话都是比那种混合型啊。债券型，货币型的。风险会高。这样子说行的话呢，比如说像etf，它都是完全跟踪标的指数的表现的嘛。他标的指数的话，里面也是投资一些。成份股备选成份股。其实他的风险也是类似于股票型的，他就是说。整个整个。股票的那个持股比例的话，都是比较高的，像这种etf。的话。它采用这种完全复制。法。然后它一般的。成份股啊，还有。备选成份股的比例不低于基金资产的百分之。95。相对来说的话，风险也是比较高的。"}, {"sender": "2", "id": "2172600", "sentence_id": "6582", "sentence": "他不在的。"}, {"sender": "2", "id": "2172600", "sentence_id": "6583", "sentence": "啊。还。"}, {"sender": "2", "id": "2172600", "sentence_id": "6584", "sentence": "好的。好的。啊，谢谢你啊。"}, {"sender": "1", "id": "2172600", "sentence_id": "6585", "sentence": "不客气的，您看有没有其他可以帮到您的呢？"}, {"sender": "2", "id": "2172600", "sentence_id": "6586", "sentence": "没有了，没有了先拿。啊。"}, {"sender": "1", "id": "2172600", "sentence_id": "6587", "sentence": "嗯。好。不客气，那请您稍后不要挂机，对我的服务做个评价，感谢来电再见，嗯。"}, {"sender": "2", "id": "2172600", "sentence_id": "6588", "sentence": "啊哈哈哈。"}]
    # service_label, dim_label, dialogue_id = get_debug_result(data)
    # print(service_label)
    # print(dim_label)
    # print(dialogue_id)
    caculate_score()
