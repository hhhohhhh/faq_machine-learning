#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/6/8 17:38 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/8 17:38   wangfc      1.0         None
"""
from utils.constants import TF_INPUT_EXAMPLE_LABEL_COLUMN

GUID_KEY = 'guid'
INPUT_EXAMPLE_LABEL_KEY = 'label'


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a,id=None,text_b=None, label=None,label_id=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.id = id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.label_id = label_id

    @property
    def label_column(self):
        return TF_INPUT_EXAMPLE_LABEL_COLUMN


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

