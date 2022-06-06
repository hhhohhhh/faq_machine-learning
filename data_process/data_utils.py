#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/12/2 13:40 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/12/2 13:40   wangfc      1.0         None
"""
from typing import Text, List, Dict, Any

from apps.app_constants import UTTER_PREFIX
from utils.constants import RESPONSE_IDENTIFIER_DELIMITER
from utils.io import raise_warning


def is_rasa_retrieve_intent( intent):
    return intent.split(RESPONSE_IDENTIFIER_DELIMITER).__len__() == 2


class Action():
    def __init__(self, action_name: Text, response_text: List[Text]):
        self.action_name = action_name
        self.response_text = response_text

    def __repr__(self):
        return f"action: action_name={self.action_name}, response_text={self.response_text}"


def make_rasa_action(action_name:Text,response:Text=None , use_custom_actions=False) -> [Action, bool]:
    """
    rasa retrieve_intent: 我们构建 response = f"utter_{intent}/{sub_intent}"
    参考 https://rasa.com/docs/rasa/chitchat-faqs
    对于 retrieval_intent ,我们只需要对 retrival intent 生成 rule
    因为  self.intents 包括 retrieval_intents + retrieval_intent/response_key
    所以省略对 intent = retrieval_intent/response_key 这种意图生成 rule

    rules:
      - rule: respond to FAQs
        steps:
        - intent: faq
        - action: utter_faq
      - rule: respond to chitchat
        steps:
        - intent: chitchat
        - action: utter_chitchat
    """

    action_name = f"utter_{action_name}"
    # 直接根据 response 做出 action
    action = Action(action_name=action_name, response_text=response)
    return action


def check_duplicate_synonym(
    entity_synonyms: Dict[Text, Any], text: Text, syn: Text, context_str: Text = ""
) -> None:
    if text in entity_synonyms and entity_synonyms[text] != syn:
        raise_warning(
            f"Found inconsistent entity synonyms while {context_str}, "
            f"overwriting {text}->{entity_synonyms[text]} "
            f"with {text}->{syn} during merge."
        )


def intent_response_key_to_template_key(intent_response_key: Text) -> Text:
    """Resolve the response template key for a given intent response key.

    Args:
        intent_response_key: retrieval intent with the response key suffix attached.

    Returns: The corresponding response template.

    """
    return f"{UTTER_PREFIX}{intent_response_key}"




