#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/30 15:32 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/30 15:32   wangfc      1.0         None
"""
import datetime
import random

import requests
import json

PUNCTUATIONS = ':→℃&*一~’.『/\-』＝″【—?、。“”《》！，：；？．,\'·<>（）〔〕\[\]()+～×／①②③④⑤⑥⑦⑧⑨⑩ⅢВ";#@γμφΔ■▲＃％＆＇〉｡＂＄＊＋－＜＞＠［＼］＾｀｛｜｝｟｠｢｣､〃「」】〖〗〘〙〚〛〜〝〞〟〰〾〿‘‛„‟…‧﹏'


def wordseg(x, stop_words=None):
    # http请求
    session = requests.session()
    # 重连次数
    requests.adapters.DEFAULT_RETRIES = 100

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


def remove_stop_words(x, stop_words=None):
    xx = [w for w in x if w not in stop_words]
    return xx


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