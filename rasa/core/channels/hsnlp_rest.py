#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/8 10:01 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/8 10:01   wangfc      1.0         None
"""
from typing import Text
from rasa.core.channels.channel import InputChannel
from rasa.core.channels.rest import RestInput


class HsnlpRest(RestInput):
    """
    The webhook you give to the custom channel to call would be
    http://<host>:<port>/webhooks/myio/webhook

    """
    @classmethod
    def name(self) -> Text:
        """Name of your custom channel."""
        return "hsnlp"