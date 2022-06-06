#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/5/24 10:00 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/24 10:00   wangfc      1.0         None
"""
import glob
import os
import shutil
from typing import Text, Optional, Dict, Any, List

from .tokenizer import Tokenizer,Token
import logging
logger = logging.getLogger(__name__)

class JiebaTokenizer(Tokenizer):
    """This tokenizer is a wrapper for Jieba (https://github.com/fxsjy/jieba)."""

    supported_language_list = ["zh"]

    defaults = {
        "dictionary_path": None,
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
        # Regular expression to detect tokens
        "token_pattern": None,
    }  # default don't load custom dictionary

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new intent classifier using the MITIE framework."""

        super().__init__(component_config)

        # path to dictionary file or None
        self.dictionary_path = self.component_config.get("dictionary_path")

        # load dictionary
        if self.dictionary_path is not None:
            self.load_custom_dictionary(self.dictionary_path)

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["jieba"]

    @staticmethod
    def load_custom_dictionary(path: Text) -> None:
        """Load all the custom dictionaries stored in the path.

        More information about the dictionaries file format can
        be found in the documentation of jieba.
        https://github.com/fxsjy/jieba#load-dictionary
        """
        import jieba

        jieba_userdicts = glob.glob(f"{path}/*")
        for jieba_userdict in jieba_userdicts:
            logger.info(f"Loading Jieba User Dictionary at {jieba_userdict}")
            jieba.load_userdict(jieba_userdict)

    def tokenize(self, text) -> List[Token]:
        import jieba

        tokenized = jieba.tokenize(text)
        tokens = [Token(word, start) for (word, start, end) in tokenized]

        return self._apply_token_pattern(tokens)


    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text,
        model_metadata: Optional["Metadata"] = None,
        # cached_component: Optional[Component] = None,
        **kwargs: Any,
    ) -> "JiebaTokenizer":
        """Loads trained component (see parent class for full docstring)."""
        relative_dictionary_path = meta.get("dictionary_path")

        # get real path of dictionary path, if any
        if relative_dictionary_path is not None:
            dictionary_path = os.path.join(model_dir, relative_dictionary_path)

            meta["dictionary_path"] = dictionary_path

        return cls(meta)

    @staticmethod
    def copy_files_dir_to_dir(input_dir: Text, output_dir: Text) -> None:
        # make sure target path exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        target_file_list = glob.glob(f"{input_dir}/*")
        for target_file in target_file_list:
            shutil.copy2(target_file, output_dir)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory."""

        # copy custom dictionaries to model dir, if any
        if self.dictionary_path is not None:
            target_dictionary_path = os.path.join(model_dir, file_name)
            self.copy_files_dir_to_dir(self.dictionary_path, target_dictionary_path)

            return {"dictionary_path": file_name}
        else:
            return {"dictionary_path": None}
