#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/30 14:07 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/30 14:07   wangfc      1.0         None
"""
import os
import copy
import json
import time
from collections import defaultdict
from pathlib import Path
import re

import requests
# gensim 3.8.3  使用 github的源码
from gensim.summarization import bm25

import numpy as np
import pandas as pd
import tensorflow as tf


from .predict import InputExample, convert_single_example #, tokenizer, label_list
from  .hs_tokenization_0715 import FullTokenizer
import logging

from hsnlp_faq_similarity.utils import wordseg, PUNCTUATIONS, get_intent

logger = logging.getLogger(__name__)



class HsnlpFaqSimilarityTask():
    """
    authors: liaozhilin & gaofeng
    调用该功能

    1) 初始化 字典数据： 停用词等
    2） 推送 标准问和扩展问到服务器
    3）
    """
    def __init__(self):

        self.config_files_dir = Path(__file__).parent / "config_file"
        self.data_dir =  Path(__file__).parent / "data"

        self.standard_question_filename = 'hsnlp_standard_question.xlsx'
        self.extend_question_filename = "hsnlp_extend_question.xlsx"
        self.standard_question_path = self.data_dir / self.standard_question_filename
        self.extend_question_path = self.data_dir / self.extend_question_filename
        self.model_dir = str(Path(__file__).parent / "models/ivr_sim/1123")
        self.vocab_file = str(self.config_files_dir / "hs_bert_vocab.txt")
        self.wordseg_file_path = str(self.config_files_dir/ "标准库_0723.xlsx")
        self.label_list = ["0", "1"]


        # 加载停用词 和 tone_words
        self.stop_words = self._get_stop_words(str(self.config_files_dir))
        # self.tone_words = self._get_tone_words(str(self.config_files_dir))
        self.synword_map = self._get_synword_map(str(self.config_files_dir))

        # 加载标点符号
        self.punc = PUNCTUATIONS


        #  初始化标准问和扩展问
        self.data_init = self._load_data(path_std_q=self.standard_question_path,
                                         path_ext_q=self.extend_question_path)

        # 初始化内部对象, 加入到 bm25
        self._init_ext_std(data_init=self.data_init)

        # 初始化tokenizer 和 模型
        self.tokenizer = FullTokenizer(
            vocab_file= self.vocab_file, do_lower_case=True,wordseg_file_path = self.wordseg_file_path)
        # self.predict_fn = tf.contrib.predictor.from_saved_model(self.model_dir)
        # 使用 tf2.0:
        self._predict_fn = tf.saved_model.load(self.model_dir)




    def _upload(self,path_std_q, path_ext_q=None, path_reg=None, address="192.168.73.51", port=20028):
        """
        将数据上传到 PushProduct 对象
        """
        # def upload(path_std_q, path_ext_q=None, path_reg=None, address="10.20.33.3", port=20028):
        url = "http://" + address + ":" + str(port) + "/hsnlp/qa/sentence_similarity/push_data"

        std_df = pd.read_excel(path_std_q, sheet_name="hsnlp_standard_question")
        question_id_std = std_df["question_id"].tolist()
        question_std = std_df["question"].tolist()
        # 推送标准问
        print("开始组装标准问")
        data = []
        for i in range(len(std_df)):
            d = dict()
            d["question_id"] = question_id_std[i]
            d["question"] = "".join(str(question_std[i]).strip().split())
            d["is_standard"] = "1"
            d["standard_id"] = question_id_std[i]
            data.append(d)
        # cont = requests.post(url, data={"data": json.dumps(data)}).content
        # print(cont)

        if path_ext_q is not None:
            ext_df = pd.read_excel(path_ext_q, sheet_name="hsnlp_extend_question")
            question_id_ext = ext_df["question_id"].tolist()
            question_ext = ext_df["question"].tolist()
            standard_id = ext_df["standard_id"].tolist()
            print("开始组装扩展问")
            for i in range(len(ext_df)):
                d = dict()
                d["question_id"] = question_id_ext[i]
                d["question"] = question_ext[i]
                d["is_standard"] = "0"
                d["standard_id"] = standard_id[i]
                data.append(d)
        print("开始组装正则表达式")
        print(len(data))
        if path_reg is not None:
            reg_df = pd.read_excel(path_reg, dtype=str)
            question = reg_df["标准问"].tolist()
            pos_regs = reg_df["pos正则"].tolist()
            neg_regs = reg_df["neg正则"].tolist()
            assert len(question) == len(pos_regs) == len(neg_regs)
            for i in range(len(question)):
                if pos_regs[i].strip() == "nan" and neg_regs[i].strip() == "nan":
                    continue
                d = dict()
                d["question"] = question[i]
                d["is_standard"] = "2"
                if pos_regs[i].strip() != "nan" and pos_regs[i].strip() != "":
                    d["pos_reg"] = pos_regs[i]
                if neg_regs[i].strip() != "nan" and neg_regs[i].strip() != "":
                    d["neg_reg"] = neg_regs[i]
                data.append(d)
        print(len(data))

        cont = requests.post(url, data={"data": json.dumps(data)}).content
        return json.loads(cont)

    def _load_data(self,path_std_q, path_ext_q=None, path_reg=None):
        """
        @time:  2021/12/1 8:42
        @author:wangfc
        @version:
        @description:
        修改 原来 gaovfeng提供的 upload()  方法加载数据，对 HsnlpFaqSimilarityTask 对象进行初始化，用以bm25检索

        @params:
        @return:
        """
        std_df = pd.read_excel(path_std_q, sheet_name="hsnlp_standard_question")
        question_id_std = std_df["question_id"].tolist()
        question_std = std_df["question"].tolist()
        # 推送标准问
        print("开始组装标准问")
        data = []
        for i in range(len(std_df)):
            d = dict()
            d["question_id"] = question_id_std[i]
            d["question"] = "".join(str(question_std[i]).strip().split())
            d["is_standard"] = "1"
            d["standard_id"] = question_id_std[i]
            data.append(d)
        # cont = requests.post(url, data={"data": json.dumps(data)}).content
        # print(cont)

        if path_ext_q is not None:
            ext_df = pd.read_excel(path_ext_q, sheet_name="hsnlp_extend_question")
            question_id_ext = ext_df["question_id"].tolist()
            question_ext = ext_df["question"].tolist()
            standard_id = ext_df["standard_id"].tolist()
            print("开始组装扩展问")
            for i in range(len(ext_df)):
                d = dict()
                d["question_id"] = question_id_ext[i]
                d["question"] = question_ext[i]
                d["is_standard"] = "0"
                d["standard_id"] = standard_id[i]
                data.append(d)
        print("开始组装正则表达式")
        print(len(data))
        if path_reg is not None:
            reg_df = pd.read_excel(path_reg, dtype=str)
            question = reg_df["标准问"].tolist()
            pos_regs = reg_df["pos正则"].tolist()
            neg_regs = reg_df["neg正则"].tolist()
            assert len(question) == len(pos_regs) == len(neg_regs)
            for i in range(len(question)):
                if pos_regs[i].strip() == "nan" and neg_regs[i].strip() == "nan":
                    continue
                d = dict()
                d["question"] = question[i]
                d["is_standard"] = "2"
                if pos_regs[i].strip() != "nan" and pos_regs[i].strip() != "":
                    d["pos_reg"] = pos_regs[i]
                if neg_regs[i].strip() != "nan" and neg_regs[i].strip() != "":
                    d["neg_reg"] = neg_regs[i]
                data.append(d)
        print(len(data))
        return data



    def _get_stop_words(self,_CONF_PATH):
        with open(_CONF_PATH + '/stop_words', 'r', encoding='utf8') as f:
            words = f.readlines()
            words = [w.strip() for w in words]
        return words


    def _get_tone_words(self,_CONF_PATH):
        with open(_CONF_PATH + '/tone_word.txt', 'r', encoding='utf8') as f:
            tone_words = f.readlines()
            tone_words = [w.strip() for w in tone_words]
            tone_words = list(set(tone_words))
            tone_words = list(sorted(tone_words, key=lambda x: len(x), reverse=True))
        return tone_words

    def _get_synword_map(self,_CONF_PATH):
        synword_map = dict()
        with open(_CONF_PATH + '/synonym_word.json', 'r', encoding='utf8') as f:
            synonym_word = json.load(f)
            for std_word, syn_words in synonym_word.items():
                for sw in syn_words:
                    synword_map[sw] = std_word  # 同义词到标准词的映射
        logger.info("同义词加载完成")
        return synword_map



    def _push_product(self, data_init):
        res = self._init_ext_std(data_init)
        return res


    def _init_ext_std(self, data_init,PROCESS='1'):
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
                        sent_join, sent_word, sent_word_ext, _ = self._preprocess(question)
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
                        sent_join, sent_word, sent_word_ext, _ = self._preprocess(question)
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
        except Exception as e:
            logger.error(f"数据初始化失败:{e}")
            return False

    def _preprocess(self, sent):
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
        sent_droppunc = [sc for sc in sent_cut if sc not in self.punc]
        logger.info("去掉标点符号后的结果：{}".format(sent_droppunc))
        sent_word = [self.synword_map[sc] if sc in self.synword_map else sc for sc in sent_droppunc]
        logger.info("同义词替换后的结果：{}".format(sent_word))
        # sent_word = [sw for sw in sent_word if sw not in tone_words]
        # logger.info("去掉语气词后的结果：{}".format(sent_word))
        sent_cut_nostopword = wordseg(sent, [])
        logger.info("不去停用词分词后结果：{}".format(sent_cut_nostopword))
        sent_droppunc_nostopword = [sc for sc in sent_cut_nostopword if sc not in self.punc]  #可用于做正则的匹配
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



    def similar_calculate_process(self, query, recall_flag = True,
                                  use_reg=False, use_intent=False,
                                  topk=10, threshold=0.0):
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
            sent_join_raw, sent_word_raw, _, _ = self._preprocess(question)


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

            sent_join_3, sent_word_3, sent_word_ext_3, _ = self._preprocess(question)
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


        sent_join, _, _, _ = self._preprocess(question)
        query_batch = [sent_join] * len(yy_filter)
        # query_batch = [question] * len(yy_filter) # 不使用预处理后的结果

        logger.info("输入模型的句子：{}".format(query_batch[0]))
        # pdb.set_trace()
        model_input = yy_sent_join_filter
        # model_input = yy_filter  #不使用预处理后的结果
        for i in range(len(model_input)):
            logger.info("输入模型的召回的句子为:{}".format(model_input[i]))

        # logits, sim_score = self.predict_siamese(query_batch, model_input, label_list=label_list)
        sim_score = self._predict_siamese(query_batch, model_input)
        # logger.info(str(logits) + " " + str(sim_score))
        # logits, sim_score = self.predict_siamese(query_batch, yy_filter, label_list=label_list) # 不使用预处理后的结果
        # logits = [np.float(bs[0]) for bs in logits]
        # sim_score = sim_score.tolist()


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
                    recall_bm25_ss, recall_bm25_yy_std, recall_logits = self._reg_filter(query, recall_question,
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
            yy_filter, ss_filter, yy_std_raw, recall_logits = self._reg_filter(query, recall_question, recall_std_question,
                                                                              recall_std_id, recall_score,
                                                                              yy_filter, ss_filter, yy_std_raw,
                                                                              recall_logits)
        t4 = time.time()
        print("规则过滤时间{}".format(t4-t3))

        return recall_question, recall_std_question, recall_std_id, recall_score, \
               yy_filter, ss_filter, yy_std_raw, recall_logits


    def _predict_siamese(self, text_as, text_bs):
        input_ids_as, input_mask_as, segment_ids_as, label_idss, input_ids_bs, \
        input_mask_bs, segment_ids_bs = self._batch_input(text_as, text_bs,
                                                          tokenizer=self.tokenizer,label_list=self.label_list)
        # prediction = self._predict_fn({
        #     "input_ids_a": input_ids_as,
        #     "input_mask_a": input_mask_as,
        #     "segment_ids_a": segment_ids_as,
        #     "label_ids": label_idss,
        #     "input_ids_b": input_ids_bs,
        #     "input_mask_b": input_mask_bs,
        #     "segment_ids_b": segment_ids_bs,
        # })

        # tf2.0 预测的时候 注意 tensor 和 model 是否在同一个device
        def _to_tensor(inputs):
            return tf.convert_to_tensor(inputs,dtype=tf.int32)

        prediction = self._predict_fn.signatures["serving_default"](
            input_ids_a=_to_tensor(input_ids_as),
            input_mask_a= _to_tensor(input_mask_as),
            segment_ids_a = _to_tensor(segment_ids_as),
            label_ids =_to_tensor(label_idss),
            input_ids_b=_to_tensor(input_ids_bs),
            input_mask_b= _to_tensor(input_mask_bs),
            segment_ids_b=  _to_tensor(segment_ids_bs))

        sim_scores = prediction["sims"].numpy().tolist()
        return sim_scores


    def _batch_input(self, text_as, text_bs,tokenizer,label_list):
        input_ids_as = []
        input_mask_as = []
        segment_ids_as = []
        label_idss = []
        input_ids_bs = []
        input_mask_bs = []
        segment_ids_bs = []
        for i in range(len(text_as)):
            predict_example = InputExample("id", text_as[i], text_bs[i], label_list[0])
            feature = convert_single_example(5, predict_example, label_list,30, tokenizer)
            input_ids_as.append(feature.input_ids_a)
            input_mask_as.append(feature.input_mask_a)
            segment_ids_as.append(feature.segment_ids_a)
            label_idss.append(feature.label_id)
            input_ids_bs.append(feature.input_ids_b)
            input_mask_bs.append(feature.input_mask_b)
            segment_ids_bs.append(feature.segment_ids_b)
        return input_ids_as, input_mask_as, segment_ids_as, label_idss, input_ids_bs, \
               input_mask_bs, segment_ids_bs



    def _reg_filter(self, query, recall_question, recall_std_question, recall_std_id, recall_score,
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



    def _intent_filter(self, query, recall_question, recall_std_question, recall_std_id, recall_score,
                   yy_filter, ss_filter, yy_std_raw, recall_logits):
        intent_res = get_intent(query)
        recall_question[0] = intent_res
        recall_std_question[0] = intent_res
        recall_std_id[0] = self.id_stdq[intent_res]
        recall_score[0] = 1.0
        return recall_question, recall_std_question, recall_std_id, recall_score, yy_filter, ss_filter, yy_std_raw, recall_logits

