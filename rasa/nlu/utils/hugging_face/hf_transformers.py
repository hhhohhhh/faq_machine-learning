import logging
import os
from typing import Any, Dict, List, Text, Tuple, Optional

import rasa.core.utils
from rasa.nlu.model import Metadata
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
# 对中文的 tokizer
from rasa.nlu.tokenizers.hf_transfromers_zh_tokizer import HfTransfromersZhTokizer
# from transformers.tokenization_bert import BasicTokenizer
from transformers.models.bert import BasicTokenizer

from rasa.nlu.featurizers.dense_featurizer.lm_featurizer import LanguageModelFeaturizer
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
import rasa.shared.utils.io
from rasa.nlu.tokenizers.tokenizer import Token
import rasa.utils.train_utils as train_utils
import numpy as np

from rasa.nlu.constants import (
    LANGUAGE_MODEL_DOCS,
    DENSE_FEATURIZABLE_ATTRIBUTES,
    SENTENCE_FEATURES,
    SEQUENCE_FEATURES,
    NUMBER_OF_SUB_TOKENS,
    NO_LENGTH_RESTRICTION,
)
from rasa.shared.nlu.constants import TEXT, ACTION_TEXT

MAX_SEQUENCE_LENGTHS = {
    "bert": 512,
    "gpt": 512,
    "gpt2": 512,
    "xlnet": NO_LENGTH_RESTRICTION,
    "distilbert": 512,
    "roberta": 512,
}

logger = logging.getLogger(__name__)


class HFTransformersNLP(Component):
    """This component is deprecated and will be removed in the future.

    Use the LanguageModelFeaturizer instead.
    """

    defaults = {
        # 增加 language  的component配置参数，用来控制是否用中文进行分词
        "language": 'en',
        # name of the language model to load.
        "model_name": "bert",
        # Pre-Trained weights to be loaded(string)
        "model_weights": None,
        # an optional path to a specific directory to download
        # and cache the pre-trained model weights.
        "cache_dir": None,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        skip_model_load: bool = False,
    ) -> None:
        """Initializes HFTransformsNLP with the models specified."""
        super(HFTransformersNLP, self).__init__(component_config)

        self._load_model_metadata()
        self._load_model_instance(skip_model_load)

        if self.component_config['language'] == "zh":
            # 针对中文的 tokizer
            logger.info("当中文的时候,使用 HfTransfromersZhTokizer,替换原来的 WhitespaceTokenizer，作为 hf_transfromers_tokizer,"
                        "在 tran() 的时候，获取每个 example 中的 'text_language_model_doc' 属性的 'token_ids', 'tokens'")
            # 如果 使用 self._load_model_instance(skip_model_load) 中加载的 Berttokenizer 进行 对 HfTransfromersZhTokizer进行初始化
            # _tokenize_example  当BertTokenizer 会产生 [UNK] token，因此我们只做最简单的按字进行分割
            self.hf_transfromers_tokizer = HfTransfromersZhTokizer(component_config=component_config,
                                                                   tokenizer=BasicTokenizer(do_lower_case=False))
        else:
            # self.whitespace_tokenizer = WhitespaceTokenizer(component_config)
            self.hf_transfromers_tokizer = WhitespaceTokenizer(component_config)

        rasa.shared.utils.io.raise_warning(
            f"'{self.__class__.__name__}' is deprecated and "
            f"will be removed in the future. "
            f"It is recommended to use the '{LanguageModelFeaturizer.__name__}' "
            f"instead.",
            category=DeprecationWarning,
        )

    def _load_model_metadata(self) -> None:

        from rasa.nlu.utils.hugging_face.registry import (
            model_class_dict,
            model_weights_defaults,
        )

        self.model_name = self.component_config["model_name"]

        if self.model_name not in model_class_dict:
            raise KeyError(
                f"'{self.model_name}' not a valid model name. Choose from "
                f"{str(list(model_class_dict.keys()))} or create "
                f"a new class inheriting from this class to support your model."
            )

        self.model_weights = self.component_config["model_weights"]
        self.cache_dir = self.component_config["cache_dir"]

        if not self.model_weights:
            logger.info(
                f"Model weights not specified. Will choose default model weights: "
                f"{model_weights_defaults[self.model_name]}"
            )
            self.model_weights = model_weights_defaults[self.model_name]

        self.max_model_sequence_length = MAX_SEQUENCE_LENGTHS[self.model_name]

    def _load_model_instance(self, skip_model_load: bool) -> None:
        """Try loading the model instance.

        Args:
            skip_model_load: Skip loading the model instances to save time.
            This should be True only for pytests
        """
        if skip_model_load:
            # This should be True only during pytests
            return

        from rasa.nlu.utils.hugging_face.registry import (
            model_class_dict,
            model_tokenizer_dict,
        )

        logger.debug(f"Loading Tokenizer and Model for {self.model_name}")
        #  使用 transformers 的 BertTokenizer
        model_weights = os.path.normpath(os.path.join(os.path.expanduser('~'),'faq',self.model_weights))
        self.tokenizer = model_tokenizer_dict[self.model_name].from_pretrained(
            model_weights, cache_dir=self.cache_dir
        )
        self.model = model_class_dict[self.model_name].from_pretrained(
            model_weights, cache_dir=self.cache_dir
        )
        logger.info(f"初始化 component = {self.__class__.__name__} Loading tokenizer={self.tokenizer.__class__.__name__}"
                    f"and model={self.model.__class__.__name__} ")
        # Use a universal pad token since all transformer architectures do not have a
        # consistent token. Instead of pad_token_id we use unk_token_id because
        # pad_token_id is not set for all architectures. We can't add a new token as
        # well since vocabulary resizing is not yet supported for TF classes.
        # Also, this does not hurt the model predictions since we use an attention mask
        # while feeding input.
        self.pad_token_id = self.tokenizer.unk_token_id

    @classmethod
    def cache_key(
        cls, component_meta: Dict[Text, Any], model_metadata: Metadata
    ) -> Optional[Text]:
        """Cache the component for future use. 'HFTransformersNLP-bert-7e993dc979f3e1bd2b7f83b87fcf6f06'

        Args:
            component_meta: configuration for the component.
            model_metadata: configuration for the whole pipeline.

        Returns: key of the cache for future retrievals.
        """
        weights = component_meta.get("model_weights") or {}

        return (
            f"{cls.name}-{component_meta.get('model_name')}-"
            f"{rasa.shared.utils.io.deep_container_fingerprint(weights)}"
        )

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["transformers"]

    def _lm_tokenize(self, text: Text) -> Tuple[List[int], List[Text]]:
        """Pass the text through the tokenizer of the language model.

        Args:
            text: Text to be tokenized.

        Returns:
            List of token ids and token strings.

        """
        split_token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        split_token_strings = self.tokenizer.convert_ids_to_tokens(split_token_ids)

        return split_token_ids, split_token_strings

    def _add_lm_specific_special_tokens(
        self, token_ids: List[List[int]]
    ) -> List[List[int]]:
        """Adds language model specific special tokens.

         These tokens were used during their training.

        Args:
            token_ids: List of token ids for each example in the batch.

        Returns:
            Augmented list of token ids for each example in the batch.
        """
        from rasa.nlu.utils.hugging_face.registry import (
            model_special_tokens_pre_processors,
        )
        # 对每个 exampel的 token_ids 增加 LM的 special_tokens:如果是 bert 模型，在首尾增加 CLS + SEP (101, 102)
        augmented_tokens = [
            model_special_tokens_pre_processors[self.model_name](example_token_ids)
            for example_token_ids in token_ids
        ]
        return augmented_tokens

    def _lm_specific_token_cleanup(
        self, split_token_ids: List[int], token_strings: List[Text]
    ) -> Tuple[List[int], List[Text]]:
        """Clean up special chars added by tokenizers of language models.

        Many language models add a special char in front/back of (some) words. We clean
        up those chars as they are not
        needed once the features are already computed.

        Args:
            split_token_ids: List of token ids received as output from the language
            model specific tokenizer.
            token_strings: List of token strings received as output from the language
            model specific tokenizer.

        Returns:
            Cleaned up token ids and token strings.
        """
        from rasa.nlu.utils.hugging_face.registry import model_tokens_cleaners

        return model_tokens_cleaners[self.model_name](split_token_ids, token_strings)

    def _post_process_sequence_embeddings(
        self, sequence_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sentence level representations and sequence level representations
        for relevant tokens.

        Args:
            sequence_embeddings: Sequence level dense features received as output from
            language model.

        Returns:
            Sentence and sequence level representations.
        """
        from rasa.nlu.utils.hugging_face.registry import (
            model_embeddings_post_processors,
        )

        sentence_embeddings = []
        post_processed_sequence_embeddings = []

        for example_embedding in sequence_embeddings:
            (
                example_sentence_embedding,
                example_post_processed_embedding,
            ) = model_embeddings_post_processors[self.model_name](example_embedding)

            sentence_embeddings.append(example_sentence_embedding)
            post_processed_sequence_embeddings.append(example_post_processed_embedding)

        return (
            np.array(sentence_embeddings),
            np.array(post_processed_sequence_embeddings),
        )

    def _tokenize_example(
        self, message: Message, attribute: Text
    ) -> Tuple[List[Token], List[int]]:
        """Tokenize a single message example.

        Many language models add a special char in front of (some) words and split
        words into sub-words. To ensure the entity start and end values matches the
        token values, tokenize the text first using the whitespace tokenizer. If
        individual tokens are split up into multiple tokens, we add this information
        to the respected token.

        Args:
            message: Single message object to be processed.
            attribute: Property of message to be processed, one of ``TEXT`` or
            ``RESPONSE``.

        Returns:
            List of token strings and token ids for the corresponding attribute of the
            message.
        """

        if self.component_config['language'] == "zh":
            # 在中文的时候， 使用 hf_transfromers_tokizer 替换 原来的 whitespace_tokenizer
            tokens_in = self.hf_transfromers_tokizer.tokenize(message, attribute)
        else:
            tokens_in = self.whitespace_tokenizer.tokenize(message, attribute)

        tokens_out = []

        token_ids_out = []

        for token in tokens_in:
            # use lm specific tokenizer to further tokenize the text
            split_token_ids, split_token_strings = self._lm_tokenize(token.text)

            split_token_ids, split_token_strings = self._lm_specific_token_cleanup(
                split_token_ids, split_token_strings
            )

            token_ids_out += split_token_ids

            token.set(NUMBER_OF_SUB_TOKENS, len(split_token_strings))

            tokens_out.append(token)

        return tokens_out, token_ids_out

    def _get_token_ids_for_batch(
        self, batch_examples: List[Message], attribute: Text
    ) -> Tuple[List[List[Token]], List[List[int]]]:
        """Compute token ids and token strings for each example in batch.

        A token id is the id of that token in the vocabulary of the language model.
        Args:
            batch_examples: Batch of message objects for which tokens need to be
            computed.
            attribute: Property of message to be processed, one of ``TEXT`` or
            ``RESPONSE``.

        Returns:
            List of token strings and token ids for each example in the batch.
        """
        batch_token_ids = []
        batch_tokens = []
        for example in batch_examples:

            example_tokens, example_token_ids = self._tokenize_example(
                example, attribute
            )
            batch_tokens.append(example_tokens)
            batch_token_ids.append(example_token_ids)

        return batch_tokens, batch_token_ids

    @staticmethod
    def _compute_attention_mask(
        actual_sequence_lengths: List[int], max_input_sequence_length: int
    ) -> np.ndarray:
        """Compute a mask for padding tokens.

        This mask will be used by the language model so that it does not attend to
        padding tokens.

        Args:
            actual_sequence_lengths: List of length of each example without any padding.
            max_input_sequence_length: Maximum length of a sequence that will be
                present in the input batch. This is
            after taking into consideration the maximum input sequence the model can
                handle. Hence it can never be
            greater than self.max_model_sequence_length in case the model applies
                length restriction.

        Returns:
            Computed attention mask, 0 for padding and 1 for non-padding tokens.
        """
        attention_mask = []

        for actual_sequence_length in actual_sequence_lengths:
            # add 1s for present tokens, fill up the remaining space up to max
            # sequence length with 0s (non-existing tokens)
            padded_sequence = [1] * min(
                actual_sequence_length, max_input_sequence_length
            ) + [0] * (
                max_input_sequence_length
                - min(actual_sequence_length, max_input_sequence_length)
            )
            attention_mask.append(padded_sequence)

        attention_mask = np.array(attention_mask).astype(np.float32)
        return attention_mask

    def _extract_sequence_lengths(
        self, batch_token_ids: List[List[int]]
    ) -> Tuple[List[int], int]:
        """Extracts the sequence length for each example and maximum sequence length.

        Args:
            batch_token_ids: List of token ids for each example in the batch.

        Returns:
            Tuple consisting of: the actual sequence lengths for each example,
            and the maximum input sequence length (taking into account the
            maximum sequence length that the model can handle.
        """
        # Compute max length across examples
        max_input_sequence_length = 0
        actual_sequence_lengths = []

        for example_token_ids in batch_token_ids:
            sequence_length = len(example_token_ids)
            actual_sequence_lengths.append(sequence_length)
            max_input_sequence_length = max(
                max_input_sequence_length, len(example_token_ids)
            )

        # Take into account the maximum sequence length the model can handle
        max_input_sequence_length = (
            max_input_sequence_length
            if self.max_model_sequence_length == NO_LENGTH_RESTRICTION
            else min(max_input_sequence_length, self.max_model_sequence_length)
        )

        return actual_sequence_lengths, max_input_sequence_length

    def _add_padding_to_batch(
        self, batch_token_ids: List[List[int]], max_sequence_length_model: int
    ) -> List[List[int]]:
        """Add padding so that all examples in the batch are of the same length.

        Args:
            batch_token_ids: Batch of examples where each example is a non-padded list
            of token ids.
            max_sequence_length_model: Maximum length of any input sequence in the batch
            to be fed to the model.

        Returns:
            Padded batch with all examples of the same length.
        """
        padded_token_ids = []

        # Add padding according to max_sequence_length
        # Some models don't contain pad token, we use unknown token as padding token.
        # This doesn't affect the computation since we compute an attention mask
        # anyways.
        for example_token_ids in batch_token_ids:

            # Truncate any longer sequences so that they can be fed to the model
            if len(example_token_ids) > max_sequence_length_model:
                example_token_ids = example_token_ids[:max_sequence_length_model]

            padded_token_ids.append(
                example_token_ids
                + [self.pad_token_id]
                * (max_sequence_length_model - len(example_token_ids))
            )
        return padded_token_ids

    @staticmethod
    def _extract_nonpadded_embeddings(
        embeddings: np.ndarray, actual_sequence_lengths: List[int]
    ) -> np.ndarray:
        """Use pre-computed non-padded lengths of each example to extract embeddings
        for non-padding tokens.

        Args:
            embeddings: sequence level representations for each example of the batch.
            actual_sequence_lengths: non-padded lengths of each example of the batch.

        Returns:
            Sequence level embeddings for only non-padding tokens of the batch.
        """
        nonpadded_sequence_embeddings = []
        for index, embedding in enumerate(embeddings):
            # 获取 每个 example token_ids 对应的 embedding:  (seq_len,embedding_size)
            unmasked_embedding = embedding[: actual_sequence_lengths[index]]
            nonpadded_sequence_embeddings.append(unmasked_embedding)
        return np.array(nonpadded_sequence_embeddings)

    def _compute_batch_sequence_features(
        self, batch_attention_mask: np.ndarray, padded_token_ids: List[List[int]]
    ) -> np.ndarray:
        """Feed the padded batch to the language model.

        Args:
            batch_attention_mask: Mask of 0s and 1s which indicate whether the token
            is a padding token or not.
            padded_token_ids: Batch of token ids for each example. The batch is padded
            and hence can be fed at once.

        Returns:
            Sequence level representations from the language model.
        """
        # TFBert output: ['last_hidden_state', 'pooler_output']
        model_outputs = self.model(
            np.array(padded_token_ids), attention_mask=np.array(batch_attention_mask)
        )
        # sequence hidden states is always the first output from all models:
        # sequence_hidden_states = [64, 37, 768]
        sequence_hidden_states = model_outputs[0]
        sequence_hidden_states = sequence_hidden_states.numpy()
        return sequence_hidden_states

    def _validate_sequence_lengths(
        self,
        actual_sequence_lengths: List[int],
        batch_examples: List[Message],
        attribute: Text,
        inference_mode: bool = False,
    ) -> None:
        """Validate if sequence lengths of all inputs are less the max sequence length.

        This method should throw an error during training, whereas log a debug message
        during inference if any of the input examples have a length greater than
        maximum sequence length allowed.

        Args:
            actual_sequence_lengths: original sequence length of all inputs
            batch_examples: all message instances in the batch
            attribute: attribute of message object to be processed
            inference_mode: Whether this is during training or during inferencing
        """
        if self.max_model_sequence_length == NO_LENGTH_RESTRICTION:
            # There is no restriction on sequence length from the model
            return

        for sequence_length, example in zip(actual_sequence_lengths, batch_examples):
            if sequence_length > self.max_model_sequence_length:
                if not inference_mode:
                    raise RuntimeError(
                        f"The sequence length of '{example.get(attribute)[:20]}...' "
                        f"is too long({sequence_length} tokens) for the "
                        f"model chosen {self.model_name} which has a maximum "
                        f"sequence length of {self.max_model_sequence_length} tokens. "
                        f"Either shorten the message or use a model which has no "
                        f"restriction on input sequence length like XLNet."
                    )
                else:
                    logger.debug(
                        f"The sequence length of '{example.get(attribute)[:20]}...' "
                        f"is too long({sequence_length} tokens) for the "
                        f"model chosen {self.model_name} which has a maximum "
                        f"sequence length of {self.max_model_sequence_length} tokens. "
                        f"Downstream model predictions may be affected because of this."
                    )

    def _add_extra_padding(
        self, sequence_embeddings: np.ndarray, actual_sequence_lengths: List[int]
    ) -> np.ndarray:
        """Adds extra zero padding to match the original sequence length.

        This is only done if the input was truncated during the batch preparation of
        input for the model.

        Args:
            sequence_embeddings: Embeddings returned from the model
            actual_sequence_lengths: original sequence length of all inputs

        Returns:
            Modified sequence embeddings with padding if necessary
        """
        if self.max_model_sequence_length == NO_LENGTH_RESTRICTION:
            # No extra padding needed because there wouldn't have been any truncation
            # in the first place
            return sequence_embeddings

        reshaped_sequence_embeddings = []
        for index, embedding in enumerate(sequence_embeddings):
            embedding_size = embedding.shape[-1]
            if actual_sequence_lengths[index] > self.max_model_sequence_length:
                embedding = np.concatenate(
                    [
                        embedding,
                        np.zeros(
                            (
                                actual_sequence_lengths[index]
                                - self.max_model_sequence_length,
                                embedding_size,
                            ),
                            dtype=np.float32,
                        ),
                    ]
                )
            reshaped_sequence_embeddings.append(embedding)

        return np.array(reshaped_sequence_embeddings)

    def _get_model_features_for_batch(
        self,
        batch_token_ids: List[List[int]],
        batch_tokens: List[List[Token]],
        batch_examples: List[Message],
        attribute: Text,
        inference_mode: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute dense features of each example in the batch.

        We first add the special tokens corresponding to each language model. Next, we
        add appropriate padding and compute a mask for that padding so that it doesn't
        affect the feature computation. The padded batch is next fed to the language
        model and token level embeddings are computed. Using the pre-computed mask,
        embeddings for non-padding tokens are extracted and subsequently sentence
        level embeddings are computed.

        Args:
            batch_token_ids: List of token ids of each example in the batch.
            batch_tokens: List of token objects for each example in the batch.
            batch_examples: List of examples in the batch.
            attribute: attribute of the Message object to be processed.
            inference_mode: Whether the call is during training or during inference.

        Returns:
            Sentence and token level dense representations.
        """
        # Let's first add tokenizer specific special tokens to all examples
        batch_token_ids_augmented = self._add_lm_specific_special_tokens(
            batch_token_ids
        )

        # Compute sequence lengths for all examples
        (
            actual_sequence_lengths,
            max_input_sequence_length,
        ) = self._extract_sequence_lengths(batch_token_ids_augmented)

        # Validate that all sequences can be processed based on their sequence lengths
        # and the maximum sequence length the model can handle
        self._validate_sequence_lengths(
            actual_sequence_lengths, batch_examples, attribute, inference_mode
        )

        # Add padding so that whole batch can be fed to the model
        padded_token_ids = self._add_padding_to_batch(
            batch_token_ids_augmented, max_input_sequence_length
        )

        # Compute attention mask based on actual_sequence_length
        batch_attention_mask = self._compute_attention_mask(
            actual_sequence_lengths, max_input_sequence_length
        )

        # Get token level features from the model
        sequence_hidden_states = self._compute_batch_sequence_features(
            batch_attention_mask, padded_token_ids
        )

        # Extract features for only non-padding tokens
        sequence_nonpadded_embeddings = self._extract_nonpadded_embeddings(
            sequence_hidden_states, actual_sequence_lengths
        )

        # Extract sentence level and post-processed features
        (
            sentence_embeddings,
            sequence_embeddings,
        ) = self._post_process_sequence_embeddings(sequence_nonpadded_embeddings)

        # Pad zeros for examples which were truncated in inference mode.
        # This is intentionally done after sentence embeddings have been extracted so
        # that they are not affected
        sequence_embeddings = self._add_extra_padding(
            sequence_embeddings, actual_sequence_lengths
        )

        # shape of matrix for all sequence embeddings
        batch_dim = len(sequence_embeddings)
        seq_dim = max(e.shape[0] for e in sequence_embeddings)
        feature_dim = sequence_embeddings[0].shape[1]
        shape = (batch_dim, seq_dim, feature_dim)

        # align features with tokens so that we have just one vector per token
        # (don't include sub-tokens)
        sequence_embeddings = train_utils.align_token_features(
            batch_tokens, sequence_embeddings, shape
        )

        # sequence_embeddings is a padded numpy array
        # remove the padding, keep just the non-zero vectors
        sequence_final_embeddings = []
        for embeddings, tokens in zip(sequence_embeddings, batch_tokens):
            sequence_final_embeddings.append(embeddings[: len(tokens)])
        sequence_final_embeddings = np.array(sequence_final_embeddings)

        return sentence_embeddings, sequence_final_embeddings

    def _get_docs_for_batch(
        self,
        batch_examples: List[Message],
        attribute: Text,
        inference_mode: bool = False,
    ) -> List[Dict[Text, Any]]:
        """Compute language model docs for all examples in the batch.

        Args:
            batch_examples: Batch of message objects for which language model docs
            need to be computed.
            attribute: Property of message to be processed, one of ``TEXT`` or
            ``RESPONSE``.
            inference_mode: Whether the call is during inference or during training.


        Returns:
            List of language model docs for each message in batch.
        """
        batch_tokens, batch_token_ids = self._get_token_ids_for_batch(
            batch_examples, attribute
        )
        # 获取 features 用于后续的 intent_classifer 和 ner
        # batch_sentence_features： bert 的 CLS 输出的 embedding =(1, 768)
        # batch_sequence_features: 每个 token 对应的 embedding  (num_tokens, 768)
        (
            batch_sentence_features,
            batch_sequence_features,
        ) = self._get_model_features_for_batch(
            batch_token_ids, batch_tokens, batch_examples, attribute, inference_mode
        )

        # A doc consists of
        # {'token_ids': ..., 'tokens': ..., 'sequence_features': ...,'sentence_features': ...}
        batch_docs = []
        for index in range(len(batch_examples)):
            doc = {
                # 在 2.0.0 版本中增加存在,我们仍旧加入，在 next component: LanguageModelTokenizer 中将会使用
                "token_ids": batch_token_ids[index],
                "tokens": batch_tokens[index],
                SEQUENCE_FEATURES: batch_sequence_features[index], # shape=(seq_len, embedding_size)
                SENTENCE_FEATURES: np.reshape(batch_sentence_features[index], (1, -1)),  # shape=(1, embedding_size)
            }
            batch_docs.append(doc)

        return batch_docs

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Compute tokens and dense features for each message in training data.

        Args:
            training_data: NLU training data to be tokenized and featurized
            config: NLU pipeline config consisting of all components.
        """
        batch_size = 64
        #  ['text', 'response', 'action_text'] 属性获取对应的 language_model_doc：tokens, token_ids, 'sequence_features', 'sentence_features'
        # TODO: 对 response 计算有什么作用呢？
        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:

            non_empty_examples = list(
                filter(lambda x: x.get(attribute), training_data.training_examples)
            )

            batch_start_index = 0

            while batch_start_index < len(non_empty_examples):

                batch_end_index = min(
                    batch_start_index + batch_size, len(non_empty_examples)
                )
                # Collect batch examples
                batch_messages = non_empty_examples[batch_start_index:batch_end_index]

                # Construct a doc with relevant features
                # extracted(tokens, dense_features)
                batch_docs = self._get_docs_for_batch(batch_messages, attribute)

                for index, ex in enumerate(batch_messages):
                    # 设置 LANGUAGE_MODEL_DOCS 用于后面的 lm_tokenizer 和 lm_featurizer
                    ex.set(LANGUAGE_MODEL_DOCS[attribute], batch_docs[index])

                batch_start_index += batch_size

    def process(self, message: Message, **kwargs: Any) -> None:
        """Process an incoming message by computing its tokens and dense features.

        Args:
            message: Incoming message object
        """
        # process of all featurizers operates only on TEXT and ACTION_TEXT attributes,
        # because all other attributes are labels which are featurized during training
        # and their features are stored by the model itself.
        for attribute in {TEXT, ACTION_TEXT}:
            if message.get(attribute):
                message.set(
                    LANGUAGE_MODEL_DOCS[attribute],
                    self._get_docs_for_batch(
                        [message], attribute=attribute, inference_mode=True
                    )[0],
                )
