#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/22 9:00 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/22 9:00   wangfc      1.0         None
"""
import collections
import os
from typing import List, Text, Dict

from data_process.data_example import InputExample
import tensorflow as tf
import logging

from tokenizations import tokenization

logger = logging.getLogger(__name__)



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 guid=None,
                 example_id=None,
                 is_real_example=True):
        self.guid = guid
        self.example_id = example_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example






def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


def create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return f


def file_based_convert_examples_to_features(
        examples, label_to_id_mapping, max_seq_length, tokenizer, output_file=None, task_name=None,
        if_overwrite=False, if_print=True):
    """Convert a set of `InputExample`s to a TFRecord file."""

    # 判读是否已经存在 TF_RECORD 格式的数据
    if output_file and os.path.exists(output_file) and if_overwrite == False:
        tf.logging.info("output_file already exists in {}".format(output_file))
    elif output_file:
        writer = tf.python_io.TFRecordWriter(output_file)
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                tf.logging.info("Writing example %d of %d to %s" % (ex_index, len(examples), output_file))

            feature = convert_single_example(ex_index, example, label_to_id_mapping,
                                             max_seq_length, tokenizer, task_name)

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            # features["label_ids"] = create_float_feature([feature.label_id])\
            #     if task_name == "sts-b" else create_int_feature([feature.label_id])

            # features["label_ids"] = create_float_feature([feature.label_id]) \
            #     if task_name == "sts-b" else create_int_feature(feature.label_id)
            features["label_ids"] = create_int_feature(feature.label_id)

            features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()
    else:
        # 直接返回为 features
        features = []
        # 将 example 转换为 feature
        for index, example in enumerate(examples):
            feature = convert_single_example(index, example,label_to_id_mapping,
                                             max_seq_length, tokenizer)
            feature.is_real_example = [int(feature.is_real_example)]
            # tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            features.append(feature)
        return features


def file_based_input_fn_builder(input_file, seq_length, multilabel_length, is_training,
                                drop_remainder, task_name, use_tpu, bsz,
                                multiple=1, if_input_file=True, examples=None):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    labeltype = tf.float32 if task_name == "sts-b" else tf.int64

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length * multiple], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
        # "label_ids": tf.FixedLenFeature([], labeltype),
        # 设置为 multilabel 的情况
        'label_ids': tf.FixedLenFeature([multilabel_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        if use_tpu:
            batch_size = params["batch_size"]
        else:
            batch_size = bsz

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            # contrib_data.map_and_batch(
            tf.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn




def _truncate_seq_pair(tokens_a: object, tokens_b: object, max_length: object) -> object:
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(ex_index: int, example: InputExample,
                           label_to_id_mapping:Dict[Text,int],
                           max_seq_length: int,
                           tokenizer, if_print=True) -> InputFeatures:
    """Converts a single `InputExample` into a single `InputFeatures`."""
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in ALBERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

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

    if hasattr(example,"label_id"):
        label_id = example.label_id
    else:
        label_id = label_to_id_mapping[example.label]

    if ex_index < 5 and if_print:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        if isinstance(label_id,str):
            logger.info("label: %s (label_id = %d)" % (example.label, label_id))
        elif isinstance(label_id,list):
            logger.info("label: %s " % (" ".join(example.label), ))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_id]))

    feature = InputFeatures(
        guid=example.guid,
        # example_id=example.id,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature

