#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version:  修改自 rasa.nlu.tokenizers.tokenizer
@desc:  
@time: 2021/5/24 10:00 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/24 10:00   wangfc      1.0         None
"""
import re
from typing import Text, Optional, Dict, Any, List
import tqdm
import logging

# from rasa.nlu.config import RasaNLUModelConfig
# from rasa.nlu.constants import MESSAGE_ATTRIBUTES
# from  rasa.nlu.tokenizers.tokenizer import Tokenizer
from data_process.training_data.message import Message
from data_process.training_data.training_data import TrainingData
from models.components import Component
from utils.constants import MESSAGE_ATTRIBUTES, INTENT, ACTION_NAME, INTENT_RESPONSE_KEY, TOKENS_NAMES, \
    RESPONSE_IDENTIFIER_DELIMITER

logger =  logging.getLogger(__name__)


class Token:
    def __init__(
        self,
        text: Text,
        start: int,
        end: Optional[int] = None,
        data: Optional[Dict[Text, Any]] = None,
        lemma: Optional[Text] = None,
    ) -> None:
        self.text = text
        self.start = start
        self.end = end if end else start + len(text)

        self.data = data if data else {}
        self.lemma = lemma or text



    def set(self, prop: Text, info: Any) -> None:
        self.data[prop] = info

    def get(self, prop: Text, default: Optional[Any] = None) -> Any:
        """Returns token value."""
        return self.data.get(prop, default)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Token):
            return NotImplemented
        return (self.start, self.end, self.text, self.lemma) == (
            other.start,
            other.end,
            other.text,
            other.lemma,
        )

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Token):
            return NotImplemented
        return (self.start, self.end, self.text, self.lemma) < (
            other.start,
            other.end,
            other.text,
            other.lemma,
        )

    def __repr__(self):
        return f"token:text={self.text},start={self.start},end={self.end}"




class Tokenizer(Component):
    def __init__(self,
                 component_config: Dict[Text, Any] = {}) -> None:
        """Construct a new tokenizer using the WhitespaceTokenizer framework."""

        super().__init__(component_config)

        # flag to check whether to split intents
        self.intent_tokenization_flag = self.component_config.get(
            "intent_tokenization_flag", False
        )
        # split symbol for intents
        self.intent_split_symbol = self.component_config.get("intent_split_symbol", "_")
        # token pattern to further split tokens
        token_pattern = self.component_config.get("token_pattern", None)

        self.token_pattern_regex = None
        if token_pattern:
            self.token_pattern_regex = re.compile(token_pattern)

    def tokenize(self, text) -> List[Token]:
        """Tokenizes the text of the provided attribute of the incoming message."""

        raise NotImplementedError


    def train(
        self,
        training_data: TrainingData,
        config = None, # : Optional[RasaNLUModelConfig]
        **kwargs: Any,
    ) -> None:
        """Tokenize all training data."""
        logger.info("使用 component = {} 对 MESSAGE_ATTRIBUTES ={} 进行 tokenizing."
                    .format(self.__class__.__name__, MESSAGE_ATTRIBUTES))
        for example in tqdm.tqdm(training_data.training_examples,desc='tokenizing'):
            for attribute in MESSAGE_ATTRIBUTES:
                if (
                    example.get(attribute) is not None
                    and not example.get(attribute) == ""
                ):
                    if attribute in [INTENT, ACTION_NAME, INTENT_RESPONSE_KEY]:
                        tokens = self._split_name(example, attribute)
                    else:
                        # 在 HFTransformersNLP 训练的时候已经生成的 doc，通过 tokenize 方法 从 message 中提取 attribute 对应 doc:
                        # attribute = 'text' ---> text_language_model_doc = ['token_ids', 'tokens', 'sequence_features', 'sentence_features']
                        tokens = self.tokenize(example, attribute)
                    example.set(TOKENS_NAMES[attribute], tokens)


    def process(self, message: Message, **kwargs: Any) -> None:
        """Tokenize the incoming message."""
        for attribute in MESSAGE_ATTRIBUTES:
            if isinstance(message.get(attribute), str):
                if attribute in [INTENT, ACTION_NAME, RESPONSE_IDENTIFIER_DELIMITER]:
                    tokens = self._split_name(message, attribute)
                else:
                    tokens = self.tokenize(message, attribute)

                message.set(TOKENS_NAMES[attribute], tokens)

    def _tokenize_on_split_symbol(self, text: Text) -> List[Text]:

        words = (
            text.split(self.intent_split_symbol)
            if self.intent_tokenization_flag
            else [text]
        )

        return words

    def _split_name(self, message: Message, attribute: Text = INTENT) -> List[Token]:
        text = message.get(attribute)

        # for INTENT_RESPONSE_KEY attribute,
        # first split by RESPONSE_IDENTIFIER_DELIMITER
        if attribute == INTENT_RESPONSE_KEY:
            intent, response_key = text.split(RESPONSE_IDENTIFIER_DELIMITER)
            words = self._tokenize_on_split_symbol(
                intent
            ) + self._tokenize_on_split_symbol(response_key)

        else:
            words = self._tokenize_on_split_symbol(text)

        return self._convert_words_to_tokens(words, text)



    def _apply_token_pattern(self, tokens: List[Token]) -> List[Token]:
        """Apply the token pattern to the given tokens.

        Args:
            tokens: list of tokens to split

        Returns:
            List of tokens.
        """
        if not self.token_pattern_regex:
            return tokens

        final_tokens = []
        for token in tokens:
            new_tokens = self.token_pattern_regex.findall(token.text)
            new_tokens = [t for t in new_tokens if t]

            if not new_tokens:
                final_tokens.append(token)

            running_offset = 0
            for new_token in new_tokens:
                word_offset = token.text.index(new_token, running_offset)
                word_len = len(new_token)
                running_offset = word_offset + word_len
                final_tokens.append(
                    Token(
                        new_token,
                        token.start + word_offset,
                        data=token.data,
                        lemma=token.lemma,
                    )
                )

        return final_tokens

    @staticmethod
    def _convert_words_to_tokens(words: List[Text], text: Text) -> List[Token]:
        running_offset = 0
        tokens = []

        for word in words:
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))

        return tokens



class TfTokenizer(Tokenizer):
    def __init__(self,vocab_size = 10000,oov_tok = "UNK",max_length=32 ,padding_type= 'post',trunc_type='post',
                 component_config={}):
        super().__init__(component_config=component_config)
        self.vocab_size = vocab_size
        self.oov_tok = oov_tok
        self.max_length =max_length
        self.padding_type =padding_type
        self.trunc_type = trunc_type

    def train(self,training_data):
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        #
        tf_tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_tok)
        # 训练 tf 的 tokenizer
        tf_tokenizer.fit_on_texts(training_data)
        self.tf_tokenizer = tf_tokenizer
        logger.info(f"训练完成 tokenizer={tf_tokenizer.__class__.__name__}, vocab_size={self.vocab_size} ")


    def encode_examples(self,texts,max_length=None):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        sequences = self.tf_tokenizer.texts_to_sequences(texts = texts)
        if max_length is None:
            max_length = self.max_length
        padded = pad_sequences(sequences=sequences,maxlen=max_length,padding=self.padding_type,truncating=self.trunc_type)
        return padded
