# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python2, python3
# coding=utf-8
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import six
from six.moves import range
import tensorflow as tf

# import tensorflow.compat.v1 as tf
# import tensorflow_hub as hub
# import sentencepiece as spm

SPIECE_UNDERLINE = u"▁".encode("utf-8")


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
    """Checks whether the casing config is consistent with the checkpoint name."""

    # The casing has to be passed in by the user and there is no explicit check
    # as to whether it matches the checkpoint. The casing information probably
    # should have been stored in the bert_config.json file, but it's not, so
    # we have to heuristically detect it to validate.

    if not init_checkpoint:
        return

    m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt",
                 six.ensure_str(init_checkpoint))
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


def preprocess_text(inputs, remove_space=True, lower=False,language='zh'):
    """preprocess data by removing extra space and normalize data."""
    if language=='zh':
        word_separator = ''
    elif language=='en':
        word_separator = ' '

    outputs = inputs
    if remove_space:
        outputs = f"{word_separator}".join(inputs.strip().split())

    if six.PY2 and isinstance(outputs, str):
        try:
            outputs = six.ensure_text(outputs, "utf-8")
        except UnicodeDecodeError:
            outputs = six.ensure_text(outputs, "latin-1")

    outputs = unicodedata.normalize("NFKD", outputs)
    outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
    """turn sentences into word pieces."""

    if six.PY2 and isinstance(text, six.text_type):
        text = six.ensure_binary(text, "utf-8")

    if not sample:
        # 使用 sentence piece model 来分词
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    for piece in pieces:
        piece = printable_text(piece)
        if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                six.ensure_binary(piece[:-1]).replace(SPIECE_UNDERLINE, b""))
            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    # note(zhiliny): convert back to unicode for py2
    if six.PY2 and return_unicode:
        ret_pieces = []
        for piece in new_pieces:
            if isinstance(piece, str):
                piece = six.ensure_text(piece, "utf-8")
            ret_pieces.append(piece)
        new_pieces = ret_pieces

    return new_pieces


def encode_ids(sp_model, text, sample=False):
    pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return ids


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return six.ensure_text(text, "utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return six.ensure_text(text, "utf-8", "ignore")
        elif isinstance(text, six.text_type):
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
            return six.ensure_text(text, "utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, six.text_type):
            return six.ensure_binary(text, "utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.io.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            # print('index= {}, token={},len(token)={}'.format(index,token,len(token)))
            if not token:
                break
            # token = token.strip().split()[0] # vocab 为英文的时候使用
            token = token.strip()  # vocab 为中文的 时候使用
            if token not in vocab:
                vocab[token] = len(vocab)
                index += 1
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True, spm_model_file=None):
        self.vocab = None
        self.sp_model = None
        if spm_model_file:
            # SentencePieceProcessor 不同于bert中
            self.sp_model = spm.SentencePieceProcessor()
            tf.logging.info("loading sentence piece model")
            self.sp_model.Load(spm_model_file)
            # Note(mingdachen): For the purpose of consisent API, we are
            # generating a vocabulary for the sentence piece tokenizer.
            self.vocab = {self.sp_model.IdToPiece(i): i for i
                          in range(self.sp_model.GetPieceSize())}
        else:
            self.vocab = load_vocab(vocab_file)
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
            self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_scratch(cls, vocab_file, do_lower_case, spm_model_file):
        return FullTokenizer(vocab_file, do_lower_case, spm_model_file)

    @classmethod
    def from_hub_module(cls, hub_module, spm_model_file):
        """Get the vocab file and casing info from the Hub module."""
        with tf.Graph().as_default():
            albert_module = hub.Module(hub_module)
            tokenization_info = albert_module(signature="tokenization_info",
                                              as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run(
                    [tokenization_info["vocab_file"],
                     tokenization_info["do_lower_case"]])
        return FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case,
            spm_model_file=spm_model_file)

    def tokenize(self, text):
        if self.sp_model:
            split_tokens = encode_pieces(self.sp_model, text, return_unicode=False)
        else:
            split_tokens = []
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        if self.sp_model:
            tf.logging.info("using sentence piece tokenzier.")
            return [self.sp_model.PieceToId(
                printable_text(token)) for token in tokens]
        else:
            return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        if self.sp_model:
            tf.logging.info("using sentence piece tokenzier.")
            return [self.sp_model.IdToPiece(id_) for id_ in ids]
        else:
            return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

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
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
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


class ChineseBasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=False):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, ori_text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(ori_text)
        # 不需要去除空格
        # text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        orig_tokens = self._tokenize_chinese_chars(text)

        # orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        # output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return split_tokens

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
        """Splits punctuation on a piece of text.
        根据 punctuation 标点符号分割文本"""
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
        output_tokens = []
        prev_is_chinese = False
        for char in text:
            # 返回对应的 ASCII 数值，或者 Unicode 数值
            cp = ord(char)
            # 我们此处为了保留中文中的空格信息而增加是否是 is_whitespace 的 这个判断
            if self._is_chinese_char(cp) or is_whitespace(char):  #
                output_tokens.append(char)
                prev_is_chinese = True
            elif prev_is_chinese == True:
                output_tokens.append(char)
                prev_is_chinese = False
            else:
                if output_tokens != []:
                    output_tokens[-1] = output_tokens[-1] + char
                else:
                    # 最开始第一个字符，即不是中文和空格
                    output_tokens.append(char)
                prev_is_chinese = False
        return output_tokens

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


class ChineseBaiscTokenizerV2(BasicTokenizer):
    """上个版本问题：
    中文字符串  ---> 分割为字 ---> token
    \r\n 等特殊字符可能直接被省略，而因此丢失原有的位置信息
    """

    def segment(self, sequence,absolute=False):
        """
        @author:wangfc27441
        @desc:
        1. 先将 sequence 不缺省 地 进行分割为 "中文字"
        2. 将 "中文字" 做 word_piece token

        @version：
        @time:2020/12/15 11:22
        Parameters
        ----------
        Returns:
            tokens:
            char_to_token_offset:  每个 char的位置 对应的 token_index
            token_to_char_index:
        -------
        """
        char_to_segment_offset =[]
        segments = []
        pre_token = ""
        if not absolute:
            for char_index,char in enumerate(sequence):
                cp = ord(char)
                if self._is_chinese_char(cp):
                    if pre_token != "":
                        # 在 pre_token 加入 segments之后，初始化为 ""
                        segments.append(pre_token)
                        pre_token =""
                        # 每次迭代一个char的时候，更新每个 char的位置对应的 token的 token_index
                        char_to_segment_offset.append(len(segments))
                        segments.append(char)
                    else:
                        # 每次迭代一个char的时候，更新每个 char的位置对应的 token的 token_index
                        char_to_segment_offset.append(len(segments))
                        segments.append(char)
                elif _is_whitespace(char) or _is_punctuation(char) or _is_control(char):
                    char_to_segment_offset.append(len(segments))
                    segments.append(char)
                else:
                    # 每次迭代一个char的时候，更新每个 char的位置对应的 token的 token_index
                    char_to_segment_offset.append(len(segments))
                    pre_token+=char
        else:
            for char in sequence:
                char_to_segment_offset.append(len(segments))
                segments.append(char)
        assert char_to_segment_offset.__len__() == sequence.__len__()
        return segments,char_to_segment_offset








class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

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
        for token in whitespace_tokenize(text):
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
                        substr = "##" + six.ensure_str(substr)
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
    # \t, \n, and \r are technically control characters but we treat them
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


# 检验是否全是中文字符
def is_chinese(char):
    if '\u4e00' <= char <= '\u9fa5':
        return True
    else:
        return False


def is_all_chenese(text):
    for char in text:
        if not is_chinese(char):
            return False
    return True


def is_chinese_text_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

# 中文的标点符号
CHINESE_PUNCTUATION = {
    '–', '—', '‘', '’', '“', '”',
    '…', '、', '。', '〈', '〉', '《',
    '》', '「', '」', '『', '』', '【',
    '】', '〔', '〕', '！', '（', '）',
    '，', '．', '：', '；', '？'
}
# 检验是否全是中文字符
def is_chinese_or_punctuation(char):
    if '\u4e00' <= char <= '\u9fa5' or char in CHINESE_PUNCTUATION or _is_punctuation(char) :
        return True
    else:
        return False


# 检验是否全是标签符号
def is_punctuation(char):
    if char in CHINESE_PUNCTUATION or _is_punctuation(char):
        return True
    else:
        return False

# 检验不是标签符号
def is_not_punctuation_or_whitespace(char):
    if char not in CHINESE_PUNCTUATION and not _is_punctuation(char) and not _is_whitespace(char):
        return True
    else:
        return False


def chinese_tokenize(ch_text):
    """
    @author:wangfc27441
    @desc:  将中文的字符串 转换为 汉子+ 标点符号+ 英文单词 + 数字的 token 列表
    @version：
    @time:2020/7/14 22:11
    return:
    char_to_word_offset: 每个字符位置 char_index 所对应的 token_index
    """

    # 英文：每个example的 以 white_space分割的list,所以有些字符是 'good,'等
    # 中文：这种分割是否合适？
    doc_tokens = []
    # 英文里表示char所在的第几个词,用于定位 answer的位置
    # 中文里：如果没有空格就会全部为 0，似乎有些不合适
    char_to_word_offset = []
    # 首个char默认
    prev_is_whitespace = True
    prev_is_chinese = True
    for c in ch_text:
        if is_chinese_text_whitespace(c):
            # 将空格等符号加入
            doc_tokens.append(c)
            # 如果是空格等符号
            prev_is_whitespace = True
            prev_is_chinese = False
            prev_is_digit = False
            prev_is_english =False
        else:
            # 判断当前字符的默认属性
            current_is_chinese = False
            current_is_digit = False
            current_is_english = False
            # 如果当前字符不是空格，需要判断前一个字符是否为中文
            if is_chinese_or_punctuation(c):
                current_is_chinese = True
            elif re.match(pattern=r'[0-9]',string=c):
                current_is_digit =True
            elif re.match(pattern= r'[a-zA-Z]',string=c):
                current_is_english =True

            if prev_is_whitespace or current_is_chinese or prev_is_chinese:
                # 当前一个字符为空格，而现在的不是空格的情况，新增一个element
                # 或者之前的字符为中文的时候，我们也应该开始一个新的token
                # 或者当前的字符是中文
                doc_tokens.append(c)
            elif (current_is_english and prev_is_digit) or (current_is_digit and prev_is_english):
                # 当前面是英文 + 数字的时候应该分割
                # 当前面是数字 + 英文的时候应该分割
                doc_tokens.append(c)
            else:
                # 如果是连续的 英文或者数字
                # 在最后一个元素上加长字符串
                doc_tokens[-1] += c

            # 为下一步设置
            prev_is_whitespace = False
            prev_is_chinese = current_is_chinese
            prev_is_digit = current_is_digit
            prev_is_english =current_is_english


        # 英文里表示char所在的第几个词，但是中文里不需要？？？
        char_to_word_offset.append(len(doc_tokens) - 1)
    return doc_tokens, char_to_word_offset




def get_subword_tokens(tokenizer,segments):
    tok_to_orig_index = []  # 每个word_pieces.token对应的原始index
    orig_to_tok_index = []  # 每个原始 doc_token 对应的 word_pieces.token 后 index
    all_doc_tokens = []  # 原始的 doc_tokens 经过  word_pieces.tokenize 后的list
    for (i, segment) in enumerate(segments):
        # 记录 doc_tokens 中 token 对应的位置
        orig_to_tok_index.append(len(all_doc_tokens))
        # doc_tokens 中的 token 对应于 all_doc_tokens的位置
        sub_tokens = tokenizer.tokenize(segment)
        for sub_token in sub_tokens:
            # 记录 sub_token 对应于 doc_token 的index
            tok_to_orig_index.append(i)
            # 增加 sub_token 到 all_doc_tokens 中
            all_doc_tokens.append(sub_token)
    return all_doc_tokens,tok_to_orig_index,orig_to_tok_index