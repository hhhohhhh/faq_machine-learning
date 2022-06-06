#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/2/17 10:55 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/17 10:55   wangfc      1.0         None
"""
import typing
from typing import Optional, Text

if typing.TYPE_CHECKING:
    from .persistor import Persistor


def create_persistor(persistor: Optional[Text]) -> Optional["Persistor"]:
    """Create a remote persistor to store the model if configured."""

    if persistor is not None:
        # from rasa.nlu.persistor import get_persistor

        from models.nlu.persistor import get_persistor
        return get_persistor(persistor)
    else:
        return None
