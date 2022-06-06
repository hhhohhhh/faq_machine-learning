#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/2/8 17:32 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/8 17:32   wangfc      1.0         None
"""
import json
import logging
import os
import re
from typing import Any, Dict, Optional, Text, Match, List

from utils.constants import ENTITY_ATTRIBUTE_TYPE, ENTITY_ATTRIBUTE_VALUE, ENTITY_ATTRIBUTE_GROUP, \
    ENTITY_ATTRIBUTE_ROLE, ENTITY_ATTRIBUTE_START, ENTITY_ATTRIBUTE_END



def build_entity(
    start: int,
    end: int,
    value: Text,
    entity_type: Text,
    role: Optional[Text] = None,
    group: Optional[Text] = None,
    **kwargs: Any,
) -> Dict[Text, Any]:
    """Builds a standard entity dictionary.

    Adds additional keyword parameters.

    Args:
        start: start position of entity
        end: end position of entity
        value: text value of the entity
        entity_type: name of the entity type
        role: role of the entity
        group: group of the entity
        **kwargs: additional parameters

    Returns:
        an entity dictionary
    """

    entity = {
        ENTITY_ATTRIBUTE_START: start,
        ENTITY_ATTRIBUTE_END: end,
        ENTITY_ATTRIBUTE_VALUE: value,
        ENTITY_ATTRIBUTE_TYPE: entity_type,
    }

    if role:
        entity[ENTITY_ATTRIBUTE_ROLE] = role
    if group:
        entity[ENTITY_ATTRIBUTE_GROUP] = group

    entity.update(kwargs)
    return entity
