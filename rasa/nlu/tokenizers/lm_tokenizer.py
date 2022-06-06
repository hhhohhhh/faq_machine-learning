from typing import Dict, Text, Any, List

import rasa.shared.utils.io
from rasa.nlu.constants import LANGUAGE_MODEL_DOCS
from rasa.nlu.tokenizers.tokenizer import Tokenizer, Token
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer


# 改成基础 Tokenizer 类
# class LanguageModelTokenizer(WhitespaceTokenizer):
from rasa.shared.nlu.training_data.message import Message


class LanguageModelTokenizer(Tokenizer):
    """This tokenizer is deprecated and will be removed in the future.

    Use the LanguageModelFeaturizer with any other Tokenizer instead.
    """
    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
        # Regular expression to detect tokens
        "token_pattern": None,
    }


    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Initializes LanguageModelTokenizer for tokenization.

        Args:
            component_config: Configuration for the component.
        """
        super().__init__(component_config)
        rasa.shared.utils.io.raise_warning(
            f"'{self.__class__.__name__}' is deprecated and "
            f"will be removed in the future. "
            f"It is recommended to use the '{WhitespaceTokenizer.__name__}' or "
            f"another {Tokenizer.__name__} instead.",
            category=DeprecationWarning,
        )


    # 定义 tokenize
    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        # 在 HFTransformersNLP 训练的时候已经生成的 doc，通过 tokenize 方法 从 message 中提取 attribute 对应 doc:
        # attribute = 'text' ---> text_language_model_doc = ['token_ids', 'tokens', 'sequence_features', 'sentence_features']
        doc = self.get_doc(message, attribute)
        TOKENS = "tokens"
        return doc[TOKENS]

    def get_doc(self, message: Message, attribute: Text) -> Dict[Text, Any]:
        return message.get(LANGUAGE_MODEL_DOCS[attribute])
