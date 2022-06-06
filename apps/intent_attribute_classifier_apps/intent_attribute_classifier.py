#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/12 17:07 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/12 17:07   wangfc      1.0         None
"""
import os
from typing import Text, Dict

from apps.intent_attribute_classifier_apps.intent_attribute_classifier_app_constants import PREDICT_INTENT, \
    PREDICT_INTENT_CONFIDENCE, PREDICT_ATTRIBUTE, PREDICT_ATTRIBUTE_CONFIDENCE
from data_process.training_data.message import Message
from models.components import Component
from utils.constants import TEXT, INTENT_AND_ATTRIBUTE
from run_script_tf_v1.intent_attribute_predict import AttributeClass
from run_script_tf_v1.intent_predict import IntentPredict
import run_script_tf_v1.intent_tokenization as tokenization



class IntentAttributeClassifierTfv1(Component):
    def __init__(self,
                 config_dir="corpus/hsnlp_kbqa_faq_data/haitong_intent_attribute_classification_data/intent_config",
                 vocab_filename="hs_bert_vocab.txt",
                 model_dir="output/intent_model_saved/",
                 max_seq_length=30,
                 ):
        self.config_dir = config_dir
        self.vocab_file = os.path.join(config_dir, vocab_filename)
        self.tokenizer = tokenization.FullTokenizer(
            config_dir=config_dir,
            vocab_file=self.vocab_file,
            do_lower_case=True)

        self.max_seq_length = max_seq_length

        self.intent_function = IntentPredict(config_dir, model_dir=model_dir, tokenizer=self.tokenizer,
                                             max_seq_length=max_seq_length,
                                             )
        self.att_function = AttributeClass(model_dir=model_dir, tokenizer=self.tokenizer, max_seq_length=max_seq_length)

    def parse(self, message: Message, output_as_predict_label_to_confidence_dict=True) \
            -> Dict[Text, Dict[Text, float]]:
        text = message.get(TEXT)
        intent_label, intent_confidence = self.intent_function.get_intent(text=text)

        att_label, att_confidence = self.att_function.get_attribute(text=text, intent=intent_label)
        # if output_as_predict_label_to_confidence_dict:
        #     result = {PREDICT_INTENT: intent_label, PREDICT_INTENT_CONFIDENCE: intent_confidence,
        #               PREDICT_ATTRIBUTE: att_label, PREDICT_ATTRIBUTE_CONFIDENCE: att_confidence
        #               }
        # else:
        result = {'intent': {'name': intent_label, 'confidence': intent_confidence},
                  'attribute': {'name': att_label, 'confidence': att_confidence}}

        return result


    # async def async_handle_message(self, message: Message, output_as_predict_label_to_confidence_dict=True) \
    #         -> Dict[Text, Dict[Text, float]]:
    #     result = self.parse(message,output_as_predict_label_to_confidence_dict)
    #     return result


if __name__ == '__main__':
    text = "账号查询"
    intent_attribute_evaluator = IntentAttributeEvaluator()
    intent_attribute_evaluator.get_intent_attribute(text=text)
