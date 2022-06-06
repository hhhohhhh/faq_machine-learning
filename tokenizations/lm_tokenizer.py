#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/5/24 10:01 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/24 10:01   wangfc      1.0         None
"""
import os
from typing import Text, Optional, Dict, Any, List

import numpy as np
from transformers.file_utils import PaddingStrategy

from tokenizations.tokenizer import Tokenizer, Token
from transformers import BertTokenizer,BatchEncoding


import logging

logger =  logging.getLogger(__name__)


class LanguageModelTokenizer(Tokenizer):
    """This tokenizer is deprecated and will be removed in the future.

    Use the LanguageModelFeaturizer with any other Tokenizer instead.
    """
    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
        # Regular expression to detect tokens
        "token_pattern": None,
    }


    def __init__(self, component_config: Dict[Text, Any] = {}) -> None:
        """Initializes LanguageModelTokenizer for tokenization.

        Args:
            component_config: Configuration for the component.
        """
        super().__init__(component_config)
        # rasa.shared.utils.io.raise_warning(
        #     f"'{self.__class__.__name__}' is deprecated and "
        #     f"will be removed in the future. "
        #     f"It is recommended to use the '{WhitespaceTokenizer.__name__}' or "
        #     f"another {Tokenizer.__name__} instead.",
        #     category=DeprecationWarning,
        # )


    # # 定义 tokenize
    # def tokenize(self, message: Message, attribute: Text) -> List[Token]:
    #     # 在 HFTransformersNLP 训练的时候已经生成的 doc，通过 tokenize 方法 从 message 中提取 attribute 对应 doc:
    #     # attribute = 'text' ---> text_language_model_doc = ['token_ids', 'tokens', 'sequence_features', 'sentence_features']
    #     doc = self.get_doc(message, attribute)
    #     TOKENS = "tokens"
    #     return doc[TOKENS]
    #
    # def get_doc(self, message: Message, attribute: Text) -> Dict[Text, Any]:
    #     return message.get(LANGUAGE_MODEL_DOCS[attribute])


class HfTransfromersTokizer(LanguageModelTokenizer):
    """
    @author:wangfc27441
    @desc:
    @version：
    @time:2020/10/19 21:38

    """

    def __init__(self, component_config: Dict[Text, Any] = {},
                 tokenizer:BertTokenizer=None,
                 max_length=128,pretrained_path=None,do_lower_case =True,
                 padding_strategy= PaddingStrategy.LONGEST,only_get_input_ids=False) -> None:
        """Construct a new tokenizer using the WhitespaceTokenizer framework."""

        super().__init__( component_config=component_config)
        # if "case_sensitive" in self.component_config:
        #     rasa.shared.utils.io.raise_warning(
        #         "The option 'case_sensitive' was moved from the tokenizers to the "
        #         "featurizers.",
        #         docs=DOCS_URL_COMPONENTS,
        #     )
        self.max_length=max_length
        self.pretrained_path = pretrained_path
        self.do_lower_case = do_lower_case
        self.padding_strategy =padding_strategy
        self.only_get_input_ids = only_get_input_ids

        if tokenizer is None:
            # 增加初始化参数 tokenizer, 从预训练模型中初始化
            self.tokenizer = self.load_from_pretrained(self.pretrained_path)
        else:
            self.tokenizer = tokenizer


    def load_from_pretrained(self, pretrainded_path)-> BertTokenizer:
        tokenizer = BertTokenizer.from_pretrained(pretrainded_path)
        logger.info(f"加载 tokenizer={tokenizer.__class__.__name__} from {pretrainded_path}")
        return tokenizer



    def tokenize(self,text) -> List[Token]:
        """
        @author:wangfc27441
        @desc:  中文使用
        @version：
        @time:2020/10/19 21:17

        """
        # text = message.get(attribute)
        if self.do_lower_case:
            text = text.lower()
        words = self.tokenizer.tokenize(text)

        tokens = self._convert_words_to_tokens(words, text)

        return self._apply_token_pattern(tokens)

    @staticmethod
    def _convert_words_to_tokens(words: List[Text], text: Text) -> List[Token]:
        running_offset = 0
        tokens = []

        for word in words:
            # 英文 word 中可能含有 ## ，需要替换
            word = word.replace('#','')
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))

        return tokens

    def convert_example_to_feature(self,text,max_length:int=None,padding_strategy=None)-> BatchEncoding:
        """
        @time:  2021/5/24 10:46
        @author:wangfc
        @version: 返回 bert 的输入
        @description:

        @params:
        @return:
        """
        if max_length is None:
            max_length = self.max_length
        if padding_strategy is None:
            padding_strategy = self.padding_strategy

        # self.tokenizer.encode(text=text,padding=True,max_length=10)
        # self.tokenizer(text=text)
        # self.tokenizer.tokenize(text=text)
        batch_encoding = self.tokenizer.encode_plus(text,add_special_tokens=True,padding=padding_strategy,truncation=True,
                                          max_length=max_length)
        return batch_encoding



    # def tokenize_with_padding(self,texts,max_length=None,padding_strategy=PaddingStrategy.MAX_LENGTH):
    #     """
    #     @time:  2021/5/24 23:02
    #     @author:wangfc
    #     @version: 对 texts 进行处理
    #     @description:
    #
    #     @params:
    #     @return:
    #     """
    #     self.tokenize_to_bert_input()


    def encode_examples(self,texts:List[Text],max_length=None,padding_strategy=None,only_get_input_ids=None):
        if max_length is None:
            max_length = self.max_length
        if padding_strategy is None:
            padding_strategy = self.padding_strategy
        if only_get_input_ids is None:
            only_get_input_ids = self.only_get_input_ids

        # prepare list, so that we can build up final TensorFlow dataset from slices.
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []

        for index, text in enumerate(texts):
            bert_input = self.convert_example_to_feature(text=text,max_length=max_length,padding_strategy=padding_strategy)

            input_ids_list.append(bert_input['input_ids'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            attention_mask_list.append(bert_input['attention_mask'])

        input_ids = np.array(input_ids_list)
        attention_masks = np.array(attention_mask_list)
        token_type_ids = np.array(token_type_ids_list)
        features_dict = self.map_example_to_dict(input_ids,attention_masks,token_type_ids)
        if only_get_input_ids:
            return input_ids
        else:
            return features_dict

        # return tf.data.Dataset.from_tensor_slices(
        #     (input_ids_list, attention_mask_list, token_type_ids_list).map(self.map_example_to_dict)

        # map to the expected input to TFBertForSequenceClassification, see here

    def map_example_to_dict(self,input_ids, attention_masks, token_type_ids):
        return {
                   "input_ids": input_ids,
                   "token_type_ids": token_type_ids,
                   "attention_mask": attention_masks,
               }





if __name__ == '__main__':
    chinese_pretrained_dir = "/home/wangfc/event_extract/pretrained_model/bert-base-chinese"
    english_pretrained_dir = "/home/wangfc/event_extract/pretrained_model/bert-base-uncased"
    vocab_filename = "vocab.txt"
    pretrained_path = os.path.join(english_pretrained_dir,vocab_filename)
    component_config = dict(pretrained_path=pretrained_path,max_length=10)
    transformer_tokenizer = HfTransfromersTokizer(component_config= component_config)

    # sentence = "返回 bert 的输入"
    sentence = "I use bert to tokenization"
    tokens = transformer_tokenizer.tokenize(text=sentence)
    batch_encoding = transformer_tokenizer.tokenize_to_bert_input(text=sentence,max_length=16,padding_strategy=PaddingStrategy.MAX_LENGTH)
    batch_encoding.get('input_ids')