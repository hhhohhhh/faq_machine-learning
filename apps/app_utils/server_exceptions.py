#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/2/23 11:53 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/23 11:53   wangfc      1.0         None
"""

from http import HTTPStatus
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Text,
    Union,
    Dict,
    TYPE_CHECKING,
    NoReturn,
    Coroutine,
)

import logging

from utils.exceptions import HsnlpException

logger = logging.getLogger(__name__)


class EmptyParameterException(HsnlpException):
    """
    参数为空
    """

class InvalidParameterCharaterException(HsnlpException):
    """
    参数包含非法字符
    """


class ParameterErrorFormatException(HsnlpException):
    """
    参数格式错误
    """

class ParameterLengthException(HsnlpException):
    """
    参数长度非法
    """



class InvalidKeyException(HsnlpException):
    """
    输入数据含有不一致的 key的时候发出异常
    """
    pass




# from rasa.server import ErrorResponse
class ErrorResponse(Exception):
    """Common exception to handle failing API requests."""

    def __init__(
        self,
        status: Union[int, HTTPStatus],
        reason: Text,
        message: Text,
        details: Any = None,
        help_url: Optional[Text] = None,
    ) -> None:
        """Creates error.

        Args:
            status: The HTTP status code to return.
            reason: Short summary of the error.
            message: Detailed explanation of the error.
            details: Additional details which describe the error. Must be serializable.
            help_url: URL where users can get further help (e.g. docs).
        """
        self.error_info = {
            # "version": rasa.__version__,
            "status": "failure",
            "message": message,
            "reason": reason,
            "details": details or {},
            "help": help_url,
            "code": status,
        }
        self.status = status
        logger.error(message)
        super(ErrorResponse, self).__init__()