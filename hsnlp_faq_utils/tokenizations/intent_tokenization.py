#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
@version:
@author:***
@concat:***@***
@software: 
@file: intent_tokenization.py
@time: 2022/1/5 10:43

 * 密级：秘密
 * 版权所有：*** 2019
 * 注意：本内容仅限于***内部传阅，禁止外泄以及用于其他的商业目的
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import re
import unicodedata
import six
import tensorflow as tf

import pandas as pd
import requests
import json


class WordSegment(object):
    def __init__(self, intent_config_dir='hsnlp_faq_utils/intent_config',
                 entity_name_filename="entity_name.json",
                 entity_value_filename="entityValue.json",
                 sentence_process_url="http://10.20.33.3:8017/hsnlp/faq/sentenceProcess"
                 ):
        self.intent_config_dir = intent_config_dir
        self.entity_name_filename = entity_name_filename
        self.entity_value_filename = entity_value_filename
        self.sentence_process_url = sentence_process_url
        self.entity_name_filepath =os.path.join(self.intent_config_dir,self.entity_name_filename)
        self.entity_value_filepath =  os.path.join(self.intent_config_dir,self.entity_value_filename)

        self._get_entity_data()

    def _get_entity_data(self):
        with open(self.entity_name_filepath, "r", encoding='utf8') as f:
            self.entity_name = json.load(f)
        with open(self.entity_value_filename, "r", encoding='utf8') as f:
            entity_value = json.load(f)
            self.value2entity = dict()
            for k, v in entity_value.items():
                for w in v.split(","):
                    self.value2entity[w] = k
            # print(self.value2entity)

    def string_entity_replace(self, x):
        """x is a word list"""
        words = []
        for word in x:
            if word in self.value2entity:
                tag = self.value2entity[word]
                words.append(tag)
                words.append(word)
                words.append(tag)
            else:
                words.append(word)
        return words

    def wordseg(self, x):
        # url = "http://10.20.33.3:8017/hsnlp/faq/sentenceProcess"
        data = {"sentence": x, "domainId": 1}
        head = {"Connection": "close"}
        r = requests.post(self.sentence_process_url, data=data, headers=head)
        rows = r.json()["baseInfo"][0]
        words = rows["replaceSynonym"]
        # print("word1", words)
        words = self.string_entity_replace(words)
        return words

    def whitespace_tokenize(self, text):
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        text = text.strip()
        if not text:
            return []
        tokens = self.wordseg(text)
        return tokens


# HS_WORDSEG = WORDSEG()
# HS_WORDSEG.whitespace_tokenize("买卖沪A股票如何收费的")
def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
    """Checks whether the casing config is consistent with the checkpoint name."""

    # The casing has to be passed in by the user and there is no explicit check
    # as to whether it matches the checkpoint. The casing information probably
    # should have been stored in the bert_config.json file, but it's not, so
    # we have to heuristically detect it to validate.

    if not init_checkpoint:
        return

    m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
    if m is None:
        return

    model_name = m.group(1)

    lower_models = [
        "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
        "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
    ]

    cased_models = [
        "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
        "multi_cased_L-12_H-768_A-12"
    ]

    is_bad_config = False
    if model_name in lower_models and not do_lower_case:
        is_bad_config = True
        actual_flag = "False"
        case_name = "lowercased"
        opposite_flag = "True"

    if model_name in cased_models and do_lower_case:
        is_bad_config = True
        actual_flag = "True"
        case_name = "cased"
        opposite_flag = "False"

    if is_bad_config:
        raise ValueError(
            "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
            "However, `%s` seems to be a %s model, so you "
            "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
            "how the model was pre-training. If this error is wrong, please "
            "just comment out this check." % (actual_flag, init_checkpoint,
                                              model_name, case_name, opposite_flag))


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    # print("items:",items) #['[CLS]', '日', '##期', '，', '但', '被', '##告', '金', '##东', '##福', '载', '##明', '[MASK]', 'U', '##N', '##K', ']', '保', '##证', '本', '##月', '1', '##4', '[MASK]', '到', '##位', '，', '2', '##0', '##1', '##5', '年', '6', '[MASK]', '1', '##1', '日', '[', 'U', '##N', '##K', ']', '，', '原', '##告', '[MASK]', '认', '##可', '于', '2', '##0', '##1', '##5', '[MASK]', '6', '月', '[MASK]', '[MASK]', '日', '##向', '被', '##告', '主', '##张', '权', '##利', '。', '而', '[MASK]', '[MASK]', '自', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '年', '6', '月', '1', '##1', '日', '[SEP]', '原', '##告', '于', '2', '##0', '##1', '##6', '[MASK]', '6', '[MASK]', '2', '##4', '日', '起', '##诉', '，', '主', '##张', '保', '##证', '责', '##任', '，', '已', '超', '##过', '保', '##证', '期', '##限', '[MASK]', '保', '##证', '人', '依', '##法', '不', '##再', '承', '##担', '保', '##证', '[MASK]', '[MASK]', '[MASK]', '[SEP]']
    for i, item in enumerate(items):
        # print(i,"item:",item) #  ##期
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


# def whitespace_tokenize(text):
#   """Runs basic whitespace cleaning and splitting on a piece of text."""
#   text = text.strip()
#   if not text:
#     return []
#   tokens = text.split()
#   # tokens = HS_WORDSEG(text)
#   return tokens


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        # print(self.basic_tokenizer.tokenize(text))
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.HS_WORDSEG = WordSegment()

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        # text = self._tokenize_chinese_chars(text)

        orig_tokens = self.HS_WORDSEG.whitespace_tokenize(text)
        # print("\nbasic orig_token\n", orig_tokens)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
            split_tokens.append(token)
            # token = self._run_strip_accents(token)
            # split_tokens.extend(self._run_split_on_punc(token))

        # output_tokens = self.HS_WORDSEG.whitespace_tokenize(" ".join(split_tokens))
        output_tokens = split_tokens
        # print("output_tokens", output_tokens)
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.HS_WORDSEG = WORDSEG()

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        # for token in self.HS_WORDSEG.whitespace_tokenize(text):
        for token in [text]:
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


if __name__ == "__main__":
    hs_wordseg = WORDSEG()
    x = hs_wordseg.wordseg("手机如何开通A股账户")
    print(x)
