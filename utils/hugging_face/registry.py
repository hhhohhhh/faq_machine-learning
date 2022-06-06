#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/2/9 16:18 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/9 16:18   wangfc      1.0         None
"""

import logging

from transformers import (  # noqa: F401, E402
    TFAlbertModel, # 增加 TFAlbertModel，使用 tiny 模型试图
    TFBertModel,
    TFOpenAIGPTModel,
    TFGPT2Model,
    TFXLNetModel,
    # TFXLMModel,
    TFDistilBertModel,
    TFRobertaModel,
    BertTokenizer,
    OpenAIGPTTokenizer,
    GPT2Tokenizer,
    XLNetTokenizer,
    # XLMTokenizer,
    DistilBertTokenizer,
    RobertaTokenizer,
)

from .transformers_pre_post_processors import *
model_class_dict = {
    "albert":TFAlbertModel,
    "bert": TFBertModel,
    "gpt": TFOpenAIGPTModel,
    "gpt2": TFGPT2Model,
    "xlnet": TFXLNetModel,
    # "xlm": TFXLMModel, # Currently doesn't work because of a bug in transformers
    # library https://github.com/huggingface/transformers/issues/2729
    "distilbert": TFDistilBertModel,
    "roberta": TFRobertaModel,
}


MAX_SEQUENCE_LENGTHS = {
    "albert": 512,
    "bert": 512,
    "gpt": 512,
    "gpt2": 512,
    # "xlnet": NO_LENGTH_RESTRICTION,
    "distilbert": 512,
    "roberta": 512,
}


model_weights_defaults = {
    "bert": "rasa/LaBSE",
    "gpt": "openai-gpt",
    "gpt2": "gpt2",
    "xlnet": "xlnet-base-cased",
    # "xlm": "xlm-mlm-enfr-1024",
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
}


model_tokenizer_dict = {
    "albert": BertTokenizer, # albert 中文 使用 BertTokenizer
    "bert": BertTokenizer,
    "gpt": OpenAIGPTTokenizer,
    "gpt2": GPT2Tokenizer,
    "xlnet": XLNetTokenizer,
    # "xlm": XLMTokenizer,
    "distilbert": DistilBertTokenizer,
    "roberta": RobertaTokenizer,
}


model_special_tokens_pre_processors = {
    "albert": bert_tokens_pre_processor, # 增加 albert
    "bert": bert_tokens_pre_processor,
    "gpt": gpt_tokens_pre_processor,
    "gpt2": gpt_tokens_pre_processor,
    "xlnet": xlnet_tokens_pre_processor,
    # "xlm": xlm_tokens_pre_processor,
    "distilbert": bert_tokens_pre_processor,
    "roberta": roberta_tokens_pre_processor,
}

model_tokens_cleaners = {
    "albert": bert_tokens_cleaner,  # 增加 albert
    "bert": bert_tokens_cleaner,
    "gpt": openaigpt_tokens_cleaner,
    "gpt2": gpt2_tokens_cleaner,
    "xlnet": xlnet_tokens_cleaner,
    # "xlm": xlm_tokens_pre_processor,
    "distilbert": bert_tokens_cleaner,  # uses the same as BERT
    "roberta": gpt2_tokens_cleaner,  # Uses the same as GPT2
}

model_embeddings_post_processors = {
    "albert": bert_embeddings_post_processor, # 增加 albert
    "bert": bert_embeddings_post_processor,
    "gpt": gpt_embeddings_post_processor,
    "gpt2": gpt_embeddings_post_processor,
    "xlnet": xlnet_embeddings_post_processor,
    # "xlm": xlm_embeddings_post_processor,
    "distilbert": bert_embeddings_post_processor,
    "roberta": roberta_embeddings_post_processor,
}