#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@time: 2020/10/19 21:40

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/10/19 21:40   wangfc      1.0         None


"""
import logging
import os
from typing import Dict,Text,Any
import rasa
from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer,List
from rasa.shared.nlu.training_data.message import Message
from tokenizations.whitespace_tokenizer import WhitespaceTokenizer
from transformers.models.bert import BasicTokenizer,BertTokenizer
logger = logging.getLogger(__name__)


class HfTransfromersZhTokizer(Tokenizer):
    """
    @author:wangfc27441
    @desc:
    @version：
    @time:2020/10/19 21:38

    """
    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "+",
        # Regular expression to detect tokens
        "token_pattern": None,
    }

    def __init__(self, component_config: Dict[Text, Any] = None,tokenizer:Any=None) -> None:
        """Construct a new tokenizer using the WhitespaceTokenizer framework."""

        super().__init__(component_config)
        if "case_sensitive" in self.component_config:
            rasa.shared.utils.io.raise_warning(
                "The option 'case_sensitive' was moved from the tokenizers to the "
                "featurizers.",
                docs=DOCS_URL_COMPONENTS,
            )
        # 增加初始化参数 tokenizer
        self.tokenizer = tokenizer

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """
        @author:wangfc27441
        @desc:  中文使用
        @version：
        @time:2020/10/19 21:17

        """
        text = message.get(attribute)
        # 如果使用 BertTokenizer 会产生 [UNK] token，因此我们只做最简单的按字进行分割
        words = self.tokenizer.tokenize(text)
        # 转换 为 rasa Token 对象
        tokens = self._convert_words_to_tokens(words, text)

        return self._apply_token_pattern(tokens)


class HFTransformersTokenizer(HfTransfromersZhTokizer):
    """
    继承  HfTransfromersZhTokizer component
    父类初始化的时候需要 传入 tokenizer 进行初始化，不是很合理，而是应该自己根据词典进行初始化


    1. 支持中文 + 英文
    2. 创建 初始化 tokenizer 方法：
       self._init_tokenizer() ->  BertTokenizer

    """

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "+",
        # Regular expression to detect tokens
        "token_pattern": None,
        # vocab_file 默认路径
        "vocab_file": os.path.join('pretrained_model','albert-chinese-tiny','vocab.txt')
    }

    def __init__(self,component_config:Dict[Text,Any]=None)-> None:
        super().__init__(component_config)

        self.tokenizer = self._init_tokenizer()

    def _init_tokenizer(self) -> BertTokenizer:
        """
        TODO
        创建 初始化 tokenizer 方法
        1. 加载 预训练的 vocabulary
        2. 初始化 tokenizer 对象

        BertTokenizer 是否可以使用：
        tokens = self._convert_words_to_tokens(words, text)
        word_offset = text.index(word, running_offset)
        ValueError: substring not found

        BasicTokenizer 进行初始化:
        tokenize的时候 将会去除空格

        """
        # tokenizer = BertTokenizer(vocab_file = self.component_config.get('vocab_file'))

        tokenizer = BasicTokenizer(do_lower_case=False)
        return tokenizer



