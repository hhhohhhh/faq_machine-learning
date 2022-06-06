#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:   rasa\shared\nlu\training_data\formats\readerwriter.py
@time: 2022/2/8 16:27 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/8 16:27   wangfc      1.0         None
"""
import abc
import json
from collections import OrderedDict
import operator
from pathlib import Path
import typing
from typing import Text, Dict, Any, Union

from utils.io import read_file


class TrainingDataReader(abc.ABC):
    """Reader for NLU training data."""

    def __init__(self) -> None:
        """Creates reader instance."""
        self.filename: Text = ""

    def read(self, filename: Union[Text, Path], **kwargs: Any) -> "TrainingData":
        """Reads TrainingData from a file."""
        self.filename = filename
        return self.reads(read_file(filename), **kwargs)

    @abc.abstractmethod
    def reads(self, s: Text, **kwargs: Any) -> "TrainingData":
        """Reads TrainingData from a string."""
        raise NotImplementedError