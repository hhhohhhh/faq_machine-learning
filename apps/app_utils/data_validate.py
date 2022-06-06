#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/2/23 11:46 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/23 11:46   wangfc      1.0         None
"""
from http import HTTPStatus
from typing import Text, Optional, List, Dict
from sanic.request import Request

from apps.app_utils.server_exceptions import ErrorResponse, InvalidKeyException


# from rasa.server import validate_request_body,_validate_json_training_payload
def validate_request_body(request: Request, error_message: Text) -> None:
    """Check if `request` has a body."""
    if not request.body:
        raise ErrorResponse(HTTPStatus.BAD_REQUEST, "BadRequest", error_message)


def validate_json_payload(request_payload:Dict,request_keys:Optional[List[Text]]=None) -> None:
    if request_keys:
        if set(request_payload.keys()) !=  set(request_keys):
            message = f"request请求数据字段和要求的不一致，request请求字段为:{set(request_payload.keys())},而要求的字段为:{set(request_keys)}"
            raise InvalidKeyException(message=message)





