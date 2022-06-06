#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/2/23 16:09 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/23 16:09   wangfc      1.0         None
"""
import traceback
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

from apps.app_constants import SUCCESS, FAILURE
from utils.exceptions import HsnlpException
from apps.app_utils.server_exceptions import EmptyParameterException,InvalidParameterCharaterException,\
    ParameterErrorFormatException,ParameterLengthException

logger = logging.getLogger(__name__)




class HsnlpResponse():
    """Common response to handle API requests."""

    def __init__(
            self,
            status:Text= "success",
            exception:Exception=None,
            *args, **kwargs,
    ) -> None:
        """Creates error.

        Args:
            status: The HTTP status code to return.
            reason: Short summary of the error.
            message: Detailed explanation of the error.
            details: Additional details which describe the error. Must be serializable.
            help_url: URL where users can get further help (e.g. docs).
        """
        # self.code = code
        self.status = status
        self.exception = exception
        self.args = args
        self.kwargs = kwargs

        traceback_str = self._get_traceback_str()

        output_info = {
            "code": self._get_code(),
            "msg": self._get_message(),
        }

        if self.kwargs:
            output_info.update(self.kwargs)

        if traceback_str:
            output_info.update({'exception':traceback_str})
        self.output_info = output_info

    def _get_code(self,):
        if self.status == SUCCESS:
            return '00'
        else:
            if isinstance(self.exception,EmptyParameterException):
                return '01'
            elif isinstance(self.exception,InvalidParameterCharaterException):
                return '02'
            elif isinstance(self.exception,ParameterErrorFormatException):
                return '03'
            elif isinstance(self.exception, ParameterLengthException):
                return '04'
            elif isinstance(self.exception, HsnlpException):
                return '09'
            elif isinstance(self.exception, Exception):
                return '99'


    def _get_message(self):
        if self.status == SUCCESS:
            message = SUCCESS
        else:
            message = f"{FAILURE}: {self.exception.__str__()}"
        return message

    def _get_traceback_str(self):
        if self.status==FAILURE and self.exception:
           traceback_str = ''.join(traceback.format_tb(self.exception.__traceback__))
           return traceback_str


