#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/19 14:21 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/19 14:21   wangfc      1.0         None
"""
import json
from hashlib import md5
from typing import Text, Optional, Any, Type, Dict, Union, List

from utils.io import DEFAULT_ENCODING


def deep_container_fingerprint(
    obj: Union[List[Any], Dict[Any, Any]], encoding: Text = DEFAULT_ENCODING
) -> Text:
    """Calculate a hash which is stable, independent of a containers key order.

    Works for lists and dictionaries. For keys and values, we recursively call
    `hash(...)` on them. Keep in mind that a list with keys in a different order
    will create the same hash!

    Args:
        obj: dictionary or list to be hashed.
        encoding: encoding used for dumping objects as strings

    Returns:
        hash of the container.
    """
    if isinstance(obj, dict):
        return get_dictionary_fingerprint(obj, encoding)
    if isinstance(obj, list):
        return get_list_fingerprint(obj, encoding)
    else:
        return get_text_hash(str(obj), encoding)


def get_dictionary_fingerprint(
    dictionary: Dict[Any, Any], encoding: Text = DEFAULT_ENCODING
) -> Text:
    """Calculate the fingerprint for a dictionary.

    The dictionary can contain any keys and values which are either a dict,
    a list or a elements which can be dumped as a string.

    Args:
        dictionary: dictionary to be hashed
        encoding: encoding used for dumping objects as strings

    Returns:
        The hash of the dictionary
    """
    stringified = json.dumps(
        {
            deep_container_fingerprint(k, encoding): deep_container_fingerprint(
                v, encoding
            )
            for k, v in dictionary.items()
        },
        sort_keys=True,
    )
    return get_text_hash(stringified, encoding)


def get_list_fingerprint(
    elements: List[Any], encoding: Text = DEFAULT_ENCODING
) -> Text:
    """Calculate a fingerprint for an unordered list.

    Args:
        elements: unordered list
        encoding: encoding used for dumping objects as strings

    Returns:
        the fingerprint of the list
    """
    stringified = json.dumps(
        [deep_container_fingerprint(element, encoding) for element in elements]
    )
    return get_text_hash(stringified, encoding)


def get_text_hash(text: Text, encoding: Text = DEFAULT_ENCODING) -> Text:
    """Calculate the md5 hash for a text."""
    return md5(text.encode(encoding)).hexdigest()  # nosec
