#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: gaofeng
@Contact: gaofeng27692@***
@Site: http://www.***
@File: predict.py
@Time: 2020-06-01 16:58:27
@Version:V201901.12.000

 * 密级：秘密
 * 版权所有：***
 * 注意：本内容仅限于***内部传阅，禁止外泄以及用于其他的商业目的

albert模型预测
"""
import os
import random

import tensorflow as tf
# from . import tokenization
from . import hs_tokenization_0715 as tokenization
# from . import hs_tokenization_0629 as tokenization
# from . import hs_tokenization as tokenization
# from service_core.config import GPU_NO, GPU_MEMORY_FRACTION, SEQ_LENGTH

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# if GPU_NO != "-1":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
#     tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# else:
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# label_list = ["0", "1"]

def _get_module_path(path):
    return os.path.normpath(os.path.join(os.getcwd(),
                                         os.path.dirname(__file__), path))
# _CONF_PATH = _get_module_path("./config_file")
#
# vocab_file = _CONF_PATH + "/hs_bert_vocab_0719.txt"
# # vocab_file = _CONF_PATH + "/vocab.txt"
# print(vocab_file)
#
# tokenizer = tokenization.FullTokenizer(
#       vocab_file=vocab_file, do_lower_case=True)


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
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
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids_a,
               input_mask_a,
               segment_ids_a,
               input_ids_b,
               input_mask_b,
               segment_ids_b,
               label_id,
               is_real_example=True):
    self.input_ids_a = input_ids_a
    self.input_mask_a = input_mask_a
    self.segment_ids_a = segment_ids_a
    self.input_ids_b = input_ids_b
    self.input_mask_b = input_mask_b
    self.segment_ids_b = segment_ids_b
    self.label_id = label_id
    self.is_real_example = is_real_example


def get_input_mask_segment(text,
                           max_seq_length, tokenizer, random_mask=0):
    tokens = tokenizer.tokenize(text)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]

    if random_mask:
        tokens[random.randint(0, len(tokens) - 1)] = "[MASK]"

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0 for _ in tokens]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return (input_ids, input_mask, segment_ids, tokens)


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, random_mask=0):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    input_ids_a, input_mask_a, segment_ids_a, tokens_a = \
        get_input_mask_segment(example.text_a, max_seq_length, tokenizer,
                               random_mask)
    input_ids_b, input_mask_b, segment_ids_b, tokens_b = \
        get_input_mask_segment(example.text_b, max_seq_length, tokenizer,
                               random_mask)

    if len(label_list) > 1:
        label_id = label_map[example.label]
    else:
        label_id = example.label

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens_a: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens_a]))
        tf.logging.info("tokens_b: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens_b]))
        tf.logging.info(
            "input_ids_a: %s" % " ".join([str(x) for x in input_ids_a]))
        tf.logging.info(
            "input_mask_a: %s" % " ".join([str(x) for x in input_mask_a]))
        tf.logging.info(
            "segment_ids_a: %s" % " ".join([str(x) for x in segment_ids_a]))
        tf.logging.info(
            "input_ids_b: %s" % " ".join([str(x) for x in input_ids_b]))
        tf.logging.info(
            "input_mask_b: %s" % " ".join([str(x) for x in input_mask_b]))
        tf.logging.info(
            "segment_ids_b: %s" % " ".join([str(x) for x in segment_ids_b]))
        tf.logging.info("label: %s (id = %s)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids_a=input_ids_a,
        input_mask_a=input_mask_a,
        segment_ids_a=segment_ids_a,
        input_ids_b=input_ids_b,
        input_mask_b=input_mask_b,
        segment_ids_b=segment_ids_b,
        label_id=label_id,
        is_real_example=True)
    return feature



def batch_input(text_as, text_bs):

    input_ids_as = []
    input_mask_as = []
    segment_ids_as = []
    label_idss = []
    input_ids_bs = []
    input_mask_bs = []
    segment_ids_bs = []
    for i in range(len(text_as)):
        predict_example = InputExample("id", text_as[i], text_bs[i], label_list[0])
        feature = convert_single_example(5, predict_example, label_list,
                                         30, tokenizer)
        input_ids_as.append(feature.input_ids_a)
        input_mask_as.append(feature.input_mask_a)
        segment_ids_as.append(feature.segment_ids_a)
        label_idss.append(feature.label_id)
        input_ids_bs.append(feature.input_ids_b)
        input_mask_bs.append(feature.input_mask_b)
        segment_ids_bs.append(feature.segment_ids_b)
    return input_ids_as, input_mask_as, segment_ids_as, label_idss, input_ids_bs, \
           input_mask_bs,segment_ids_bs



def predict_siamese(text_as,text_bs, label_list=None):
    input_ids_as, input_mask_as, segment_ids_as, label_idss, input_ids_bs, \
    input_mask_bs, segment_ids_bs = batch_input(text_as, text_bs)
    prediction = predict_fn({
        "input_ids_a": input_ids_as,
        "input_mask_a": input_mask_as,
        "segment_ids_a": segment_ids_as,
        "label_ids": label_idss,
        "input_ids_b": input_ids_bs,
        "input_mask_b": input_mask_bs,
        "segment_ids_b": segment_ids_bs,
    })
    return prediction["logits"]

if __name__ == '__main__':
    pass

