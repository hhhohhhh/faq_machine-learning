#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/2/9 13:44 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/9 13:44   wangfc      1.0         None
"""

import copy
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import os
import scipy.sparse
import tensorflow as tf

from typing import Any, Dict, List, Optional, Text, Tuple, Union, Type

# from rasa.nlu.extractors.extractor import EntityTagSpec
# from rasa.nlu.classifiers.diet_classifier import DIETClassifier
# from rasa.shared.utils.io import create_directory_for_file

# from rasa.nlu.utils import bilou_utils
# from rasa.utils.io import create_temporary_directory
# from rasa.utils.tensorflow.models import RasaModel, TransformerRasaModel
# from rasa.utils import train_utils
from data_process.dataset.hsnlp_faq_knowledge_dataset import ATTRIBUTE
from data_process.training_data.message import Message
from data_process.training_data.rasa_model_data import RasaModelData, FeatureArray, create_data_generators, \
    FeatureSignature
from data_process.training_data.training_data import TrainingData

from models import train_utils
from models.model import Metadata
from models.temp_keras_modules import RasaModel, TransformerRasaModel
from models.classifiers.classifier import IntentClassifier
from models.components import Component
from models.extractors.extrator import EntityExtractor, EntityTagSpec
from models.featurizers.featurizer import Featurizer
from utils import bilou_utils
import utils.io as io_utils
from utils.exceptions import InvalidParameterException
from utils.io import raise_warning, dump_obj_as_json_to_file
from utils.tensorflow import rasa_layers
from utils.tensorflow.constants import *
from utils.constants import ENTITY_ATTRIBUTE_TYPE, ENTITY_ATTRIBUTE_ROLE, ENTITY_ATTRIBUTE_GROUP, TEXT, \
    SPLIT_ENTITIES_BY_COMMA, NO_ENTITY_TAG, INTENT, INTENT_RESPONSE_KEY, ENTITIES, LABEL_RANKING_LENGTH, TOKENS_NAMES, \
    DIAGNOSTIC_DATA, SUB_LABEL_TEXT
from utils.tensorflow.layers import DotProductLoss

logger = logging.getLogger(__name__)

SPARSE = "sparse"
DENSE = "dense"
LABEL_KEY = LABEL
LABEL_SUB_KEY = IDS
# NEW
SUB_LABEL_KEY = SUB_LABEL

POSSIBLE_TAGS = [ENTITY_ATTRIBUTE_TYPE, ENTITY_ATTRIBUTE_ROLE, ENTITY_ATTRIBUTE_GROUP]


class DIETClassifier(IntentClassifier, EntityExtractor):
    """A multi-task model for intent classification and entity extraction.

    DIET is Dual Intent and Entity Transformer.
    The architecture is based on a transformer which is shared for both tasks.
    A sequence of entity labels is predicted through a Conditional Random Field (CRF)
    tagging layer on top of the transformer output sequence corresponding to the
    input sequence of tokens. The transformer output for the ``__CLS__`` token and
    intent labels are embedded into a single semantic vector space. We use the
    dot-product loss to maximize the similarity with the target label and minimize
    similarities with negative samples.
    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Featurizer]

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # ## Architecture of the used neural network
        # Hidden layer sizes for layers before the embedding layers for user message
        # and labels.
        # The number of hidden layers is equal to the length of the corresponding list.
        HIDDEN_LAYERS_SIZES: {TEXT: [], LABEL: []},
        # Whether to share the hidden layer weights between user message and labels.
        SHARE_HIDDEN_LAYERS: False,
        # Number of units in transformer
        TRANSFORMER_SIZE: 256,
        # Number of transformer layers
        NUM_TRANSFORMER_LAYERS: 2,
        # Number of attention heads in transformer
        NUM_HEADS: 4,
        # If 'True' use key relative embeddings in attention
        KEY_RELATIVE_ATTENTION: False,
        # If 'True' use value relative embeddings in attention
        VALUE_RELATIVE_ATTENTION: False,
        # Max position for relative embeddings
        MAX_RELATIVE_POSITION: None,
        # Use a unidirectional or bidirectional encoder.
        UNIDIRECTIONAL_ENCODER: False,
        # ## Training parameters
        # Initial and final batch sizes:
        # Batch size will be linearly increased for each epoch.
        BATCH_SIZES: [64, 256],
        # Strategy used when creating batches.
        # Can be either 'sequence' or 'balanced'.
        BATCH_STRATEGY: BALANCED,
        # Number of epochs to train
        EPOCHS: 300,
        # Set random seed to any 'int' to get reproducible results
        RANDOM_SEED: None,
        # Initial learning rate for the optimizer
        LEARNING_RATE: 0.001,
        # ## Parameters for embeddings
        # Dimension size of embedding vectors
        EMBEDDING_DIMENSION: 20,
        # Dense dimension to use for sparse features.
        DENSE_DIMENSION: {TEXT: 128, LABEL: 20},
        # Default dimension to use for concatenating sequence and sentence features.
        CONCAT_DIMENSION: {TEXT: 128, LABEL: 20},
        # The number of incorrect labels. The algorithm will minimize
        # their similarity to the user input during training.
        NUM_NEG: 20,
        # Type of similarity measure to use, either 'auto' or 'cosine' or 'inner'.
        SIMILARITY_TYPE: AUTO,
        # The type of the loss function, either 'cross_entropy' or 'margin'.
        LOSS_TYPE: CROSS_ENTROPY,
        # Number of top intents to normalize scores for. Applicable with
        # loss type 'cross_entropy' and 'softmax' confidences. Set to 0
        # to turn off normalization.
        RANKING_LENGTH: 10,
        # Indicates how similar the algorithm should try to make embedding vectors
        # for correct labels.
        # Should be 0.0 < ... < 1.0 for 'cosine' similarity type.
        MAX_POS_SIM: 0.8,
        # Maximum negative similarity for incorrect labels.
        # Should be -1.0 < ... < 1.0 for 'cosine' similarity type.
        MAX_NEG_SIM: -0.4,
        # If 'True' the algorithm only minimizes maximum similarity over
        # incorrect intent labels, used only if 'loss_type' is set to 'margin'.
        USE_MAX_NEG_SIM: True,
        # If 'True' scale loss inverse proportionally to the confidence
        # of the correct prediction
        SCALE_LOSS: False,
        # ## Regularization parameters
        # The scale of regularization
        REGULARIZATION_CONSTANT: 0.002,
        # The scale of how important is to minimize the maximum similarity
        # between embeddings of different labels,
        # used only if 'loss_type' is set to 'margin'.
        NEGATIVE_MARGIN_SCALE: 0.8,
        # Dropout rate for encoder
        DROP_RATE: 0.2,
        # Dropout rate for attention
        DROP_RATE_ATTENTION: 0,
        # Fraction of trainable weights in internal layers.
        CONNECTION_DENSITY: 0.2,
        # If 'True' apply dropout to sparse input tensors
        SPARSE_INPUT_DROPOUT: True,
        # If 'True' apply dropout to dense input tensors
        DENSE_INPUT_DROPOUT: True,
        # ## Evaluation parameters
        # How often calculate validation accuracy.
        # Small values may hurt performance, e.g. model accuracy.
        EVAL_NUM_EPOCHS: 20,
        # How many examples to use for hold out validation set
        # Large values may hurt performance, e.g. model accuracy.
        EVAL_NUM_EXAMPLES: 0,
        # ## Model config
        # If 'True' intent classification is trained and intent predicted.
        INTENT_CLASSIFICATION: True,
        # If 'True' named entity recognition is trained and entities predicted.
        ENTITY_RECOGNITION: True,
        # If 'True' random tokens of the input message will be masked and the model
        # should predict those tokens.
        MASKED_LM: False,
        # 'BILOU_flag' determines whether to use BILOU tagging or not.
        # If set to 'True' labelling is more rigorous, however more
        # examples per entity are required.
        # Rule of thumb: you should have more than 100 examples per entity.
        BILOU_FLAG: True,
        # If you want to use tensorboard to visualize training and validation metrics,
        # set this option to a valid output directory.
        TENSORBOARD_LOG_DIR: None,
        # Define when training metrics for tensorboard should be logged.
        # Either after every epoch or for every training step.
        # Valid values: 'epoch' and 'batch'
        TENSORBOARD_LOG_LEVEL: "epoch",
        # Perform model checkpointing
        CHECKPOINT_MODEL: False,
        # Specify what features to use as sequence and sentence features
        # By default all features in the pipeline are used.
        FEATURIZERS: [],
        # Split entities by comma, this makes sense e.g. for a list of ingredients
        # in a recipie, but it doesn't make sense for the parts of an address
        SPLIT_ENTITIES_BY_COMMA: True,
        # If 'True' applies sigmoid on all similarity terms and adds
        # it to the loss function to ensure that similarity values are
        # approximately bounded. Used inside softmax loss only.
        CONSTRAIN_SIMILARITIES: False,
        # Model confidence to be returned during inference. Possible values -
        # 'softmax' and 'linear_norm'.
        MODEL_CONFIDENCE: SOFTMAX,
    }

    # init helpers
    def _check_masked_lm(self) -> None:
        if (
                self.component_config[MASKED_LM]
                and self.component_config[NUM_TRANSFORMER_LAYERS] == 0
        ):
            raise ValueError(
                f"If number of transformer layers is 0, "
                f"'{MASKED_LM}' option should be 'False'."
            )

    def _check_share_hidden_layers_sizes(self) -> None:
        if self.component_config.get(SHARE_HIDDEN_LAYERS):
            first_hidden_layer_sizes = next(
                iter(self.component_config[HIDDEN_LAYERS_SIZES].values())
            )
            # check that all hidden layer sizes are the same
            identical_hidden_layer_sizes = all(
                current_hidden_layer_sizes == first_hidden_layer_sizes
                for current_hidden_layer_sizes in self.component_config[
                    HIDDEN_LAYERS_SIZES
                ].values()
            )
            if not identical_hidden_layer_sizes:
                raise ValueError(
                    f"If hidden layer weights are shared, "
                    f"{HIDDEN_LAYERS_SIZES} must coincide."
                )

    def _check_config_parameters(self) -> None:
        # from rasa.utils import train_utils
        # from models import train_utils

        self.component_config = train_utils.check_deprecated_options(
            self.component_config
        )

        self._check_masked_lm()
        self._check_share_hidden_layers_sizes()

        self.component_config = train_utils.update_confidence_type(self.component_config)

        train_utils.validate_configuration_settings(self.component_config)

        self.component_config = train_utils.update_deprecated_loss_type(self.component_config)
        self.component_config = train_utils.update_deprecated_sparsity_to_density(self.component_config)
        self.component_config = train_utils.update_similarity_type(self.component_config)
        self.component_config = train_utils.update_evaluation_parameters(self.component_config)

    # package safety checks
    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow"]

    def __init__(
            self,
            component_config: Optional[Dict[Text, Any]] = None,
            index_label_id_mapping: Optional[Dict[int, Text]] = None,
            entity_tag_specs: Optional[List[EntityTagSpec]] = None,
            model: Optional[RasaModel] = None,
            finetune_mode: bool = False,
    ) -> None:
        """Declare instance variables with default values."""
        if component_config is not None and EPOCHS not in component_config:
            raise_warning(
                f"Please configure the number of '{EPOCHS}' in your configuration file."
                f" We will change the default value of '{EPOCHS}' in the future to 1. "
            )

        super().__init__(component_config)

        self._check_config_parameters()

        # transform numbers to labels
        self.index_label_id_mapping = index_label_id_mapping

        self._entity_tag_specs = entity_tag_specs

        self.model = model

        self.tmp_checkpoint_dir = None
        if self.component_config[CHECKPOINT_MODEL]:
            self.tmp_checkpoint_dir = Path(io_utils.create_temporary_directory())

        self._label_data: Optional[RasaModelData] = None
        self._data_example: Optional[Dict[Text, List[FeatureArray]]] = None

        self.split_entities_config = self.init_split_entities()

        self.finetune_mode = finetune_mode

        # from rasa.nlu.classifiers.diet_classifier import DIETClassifier
        if not self.model and self.finetune_mode:
            raise InvalidParameterException(
                f"{self.__class__.__name__} was instantiated "
                f"with `model=None` and `finetune_mode=True`. "
                f"This is not a valid combination as the component "
                f"needs an already instantiated and trained model "
                f"to continue training in finetune mode."
            )

    @property
    def label_key(self) -> Optional[Text]:
        """Return key if intent classification is activated."""
        return LABEL_KEY if self.component_config[INTENT_CLASSIFICATION] else None

    @property
    def label_sub_key(self) -> Optional[Text]:
        """Return sub key if intent classification is activated."""
        return LABEL_SUB_KEY if self.component_config[INTENT_CLASSIFICATION] else None

    @staticmethod
    def model_class() -> Type[RasaModel]:
        return DIET

    # training data helpers:
    # @staticmethod
    def _label_id_index_mapping(self,
                                training_data: TrainingData, attribute: Text
                                ) -> Dict[Text, int]:
        """Create label_id dictionary.
        attribute:
        intent:              意图识别的时候，将 intent 作为 label， 生成 label - id 对应的关系
        INTENT_RESPONSE_KEY: response selector 将 INTENT_RESPONSE_KEY 作为 label
        RESONSE:             response selector 将 resonse text 作为 label
        """
        # rasa 似乎直接将多个意图做为一个 另外的意图，使用分类方法来做的
        if attribute == 'intent' and self.component_config.get('intent_tokenization_flag') is not None:
            logging.info("生成 label_id - index 字典,为了生成 多标签的 intent,改写源代码")
            distinct_label_ids = set()
            for example in training_data.intent_examples:
                for intent_tokens in example.data.get('intent_tokens'):
                    distinct_label_ids.add(intent_tokens.text)
        else:
            distinct_label_ids = {
                                     example.get(attribute) for example in training_data.intent_examples
                                 } - {None}
        return {
            label_id: idx for idx, label_id in enumerate(sorted(distinct_label_ids))
        }

    @staticmethod
    def _invert_mapping(mapping: Dict) -> Dict:
        return {value: key for key, value in mapping.items()}

    def _create_entity_tag_specs(
            self, training_data: TrainingData
    ) -> List[EntityTagSpec]:
        """Create entity tag specifications with their respective tag id mappings."""

        _tag_specs = []

        for tag_name in POSSIBLE_TAGS:
            if self.component_config[BILOU_FLAG]:
                tag_id_index_mapping = bilou_utils.build_tag_id_dict(
                    training_data, tag_name
                )
            else:
                tag_id_index_mapping = self._tag_id_index_mapping_for(
                    tag_name, training_data
                )

            if tag_id_index_mapping:
                _tag_specs.append(
                    EntityTagSpec(
                        tag_name=tag_name,
                        tags_to_ids=tag_id_index_mapping,
                        ids_to_tags=self._invert_mapping(tag_id_index_mapping),
                        num_tags=len(tag_id_index_mapping),
                    )
                )

        return _tag_specs

    @staticmethod
    def _tag_id_index_mapping_for(
            tag_name: Text, training_data: TrainingData
    ) -> Optional[Dict[Text, int]]:
        """Create mapping from tag name to id."""
        if tag_name == ENTITY_ATTRIBUTE_ROLE:
            distinct_tags = training_data.entity_roles
        elif tag_name == ENTITY_ATTRIBUTE_GROUP:
            distinct_tags = training_data.entity_groups
        else:
            distinct_tags = training_data.entities

        distinct_tags = distinct_tags - {NO_ENTITY_TAG} - {None}

        if not distinct_tags:
            return None

        tag_id_dict = {
            tag_id: idx for idx, tag_id in enumerate(sorted(distinct_tags), 1)
        }
        # NO_ENTITY_TAG corresponds to non-entity which should correspond to 0 index
        # needed for correct prediction for padding
        tag_id_dict[NO_ENTITY_TAG] = 0

        return tag_id_dict

    @staticmethod
    def _find_example_for_label(
            label: Text, examples: List[Message], attribute: Text
    ) -> Optional[Message]:
        for ex in examples:
            if ex.get(attribute) == label:
                return ex
        return None

    def _check_labels_features_exist(
            self, labels_example: List[Message], attribute: Text
    ) -> bool:
        """Checks if all labels have features set."""

        return all(
            label_example.features_present(
                attribute, self.component_config[FEATURIZERS]
            )
            for label_example in labels_example
        )

    def _extract_features(
            self, message: Message, attribute: Text
    ) -> Dict[Text, Union[scipy.sparse.spmatrix, np.ndarray]]:

        (
            sparse_sequence_features,
            sparse_sentence_features,
        ) = message.get_sparse_features(attribute, self.component_config[FEATURIZERS])
        dense_sequence_features, dense_sentence_features = message.get_dense_features(
            attribute, self.component_config[FEATURIZERS]
        )

        if dense_sequence_features is not None and sparse_sequence_features is not None:
            if (
                    dense_sequence_features.features.shape[0]
                    != sparse_sequence_features.features.shape[0]
            ):
                raise ValueError(
                    f"Sequence dimensions for sparse and dense sequence features "
                    f"don't coincide in '{message.get(TEXT)}'"
                    f"for attribute '{attribute}'."
                )
        if dense_sentence_features is not None and sparse_sentence_features is not None:
            if (
                    dense_sentence_features.features.shape[0]
                    != sparse_sentence_features.features.shape[0]
            ):
                raise ValueError(
                    f"Sequence dimensions for sparse and dense sentence features "
                    f"don't coincide in '{message.get(TEXT)}'"
                    f"for attribute '{attribute}'."
                )

        # If we don't use the transformer and we don't want to do entity recognition,
        # to speed up training take only the sentence features as feature vector.
        # We would not make use of the sequence anyway in this setup. Carrying over
        # those features to the actual training process takes quite some time.
        if (
                self.component_config[NUM_TRANSFORMER_LAYERS] == 0
                and not self.component_config[ENTITY_RECOGNITION]
                and attribute not in [INTENT, INTENT_RESPONSE_KEY]
        ):
            sparse_sequence_features = None
            dense_sequence_features = None

        out = {}

        if sparse_sentence_features is not None:
            out[f"{SPARSE}_{SENTENCE}"] = sparse_sentence_features.features
        if sparse_sequence_features is not None:
            out[f"{SPARSE}_{SEQUENCE}"] = sparse_sequence_features.features
        if dense_sentence_features is not None:
            out[f"{DENSE}_{SENTENCE}"] = dense_sentence_features.features
        if dense_sequence_features is not None:
            out[f"{DENSE}_{SEQUENCE}"] = dense_sequence_features.features

        return out

    def _check_input_dimension_consistency(self, model_data: RasaModelData) -> None:
        """Checks if features have same dimensionality if hidden layers are shared."""
        if self.component_config.get(SHARE_HIDDEN_LAYERS):
            num_text_sentence_features = model_data.number_of_units(TEXT, SENTENCE)
            num_label_sentence_features = model_data.number_of_units(LABEL, SENTENCE)
            num_text_sequence_features = model_data.number_of_units(TEXT, SEQUENCE)
            num_label_sequence_features = model_data.number_of_units(LABEL, SEQUENCE)

            if (0 < num_text_sentence_features != num_label_sentence_features > 0) or (
                    0 < num_text_sequence_features != num_label_sequence_features > 0
            ):
                raise ValueError(
                    "If embeddings are shared text features and label features "
                    "must coincide. Check the output dimensions of previous components."
                )

    def _extract_labels_precomputed_features(
            self, label_examples: List[Message], attribute: Text = INTENT
    ) -> Tuple[List[FeatureArray], List[FeatureArray]]:
        """Collects precomputed encodings."""
        features = defaultdict(list)

        for e in label_examples:
            label_features = self._extract_features(e, attribute)
            for feature_key, feature_value in label_features.items():
                features[feature_key].append(feature_value)

        sequence_features = []
        sentence_features = []
        for feature_name, feature_value in features.items():
            if SEQUENCE in feature_name:
                sequence_features.append(
                    FeatureArray(np.array(feature_value), number_of_dimensions=3)
                )
            else:
                sentence_features.append(
                    FeatureArray(np.array(feature_value), number_of_dimensions=3)
                )

        return sequence_features, sentence_features

    @staticmethod
    def _compute_default_label_features(
            labels_example: List[Message],
    ) -> List[FeatureArray]:
        """Computes one-hot representation for the labels."""
        logger.debug("No label features found. Computing default label features.")
        # 生成 单位矩阵  (label_num, label_num)
        eye_matrix = np.eye(len(labels_example), dtype=np.float32)
        # add sequence dimension to one-hot labels: 对于每个 label_index 生产 one-hot 表示 (label_num,1,label_num)
        return [
            FeatureArray(
                np.array([np.expand_dims(a, 0) for a in eye_matrix]),
                number_of_dimensions=3,
            )
        ]

    def _create_label_data(
            self,
            training_data: TrainingData,
            label_id_dict: Dict[Text, int],
            attribute: Text,
            # sub_label_id_dict:Optional[Dict[Text, int]]=None,
            # sub_attribute: Optional[Text]=None
    ) -> RasaModelData:
        """Create matrix with label_ids encoded in rows as bag of words.
        Find a training example for each label and get the encoded features
        from the corresponding Message object.
        If the features are already computed, fetch them from the message object
        else compute a one hot encoding for the label as the feature vector.
        """
        # Collect one example for each label
        labels_idx_examples = []
        for label_name, idx in label_id_dict.items():
            label_example = self._find_example_for_label(
                label_name, training_data.intent_examples, attribute
            )
            labels_idx_examples.append((idx, label_example))

        # Sort the list of tuples based on label_idx
        labels_idx_examples = sorted(labels_idx_examples, key=lambda x: x[0])
        labels_example = [example for (_, example) in labels_idx_examples]

        # Collect features, precomputed if they exist, else compute on the fly
        if self._check_labels_features_exist(labels_example, attribute):
            (
                sequence_features,
                sentence_features,
            ) = self._extract_labels_precomputed_features(labels_example, attribute)
        else:
            sequence_features = None
            # 生产要给 3维的 one-hot 格式的 sentence_features: (label_num,1,label_num), 每个 label_id 对应一个 向量
            sentence_features = self._compute_default_label_features(labels_example)

        label_data = RasaModelData()
        label_data.add_features(LABEL, SEQUENCE, sequence_features)
        label_data.add_features(LABEL, SENTENCE, sentence_features)

        if label_data.does_feature_not_exist(
                LABEL, SENTENCE
        ) and label_data.does_feature_not_exist(LABEL, SEQUENCE):
            raise ValueError(
                "No label features are present. Please check your configuration file."
            )

        label_ids = np.array([idx for (idx, _) in labels_idx_examples])
        # explicitly add last dimension to label_ids to track correctly dynamic sequences
        # 增加 label/id feature = (3, 1)
        label_data.add_features(
            LABEL_KEY,
            LABEL_SUB_KEY,
            [FeatureArray(np.expand_dims(label_ids, -1), number_of_dimensions=2)],
        )

        label_data.add_lengths(LABEL, SEQUENCE_LENGTH, LABEL, SEQUENCE)

        return label_data

    def _use_default_label_features(self, label_ids: np.ndarray) -> List[FeatureArray]:
        all_label_features = self._label_data.get(LABEL, SENTENCE)[0]
        return [
            FeatureArray(
                np.array([all_label_features[label_id] for label_id in label_ids]),
                number_of_dimensions=all_label_features.number_of_dimensions,
            )
        ]

    def _create_model_data(
            self,
            training_data: List[Message],
            label_id_dict: Optional[Dict[Text, int]] = None,
            label_attribute: Optional[Text] = None,
            training: bool = True,
    ) -> RasaModelData:
        """Prepare data for training and create a RasaModelData object."""
        # from rasa.utils.tensorflow import model_data_utils
        from data_process.training_data import rasa_model_data_utils as  model_data_utils
        # 需要考虑的 attributes_to_consider：
        attributes_to_consider = [TEXT]
        if training and self.component_config[INTENT_CLASSIFICATION]:
            # we don't have any intent labels during prediction, just add them during training
            # label_attribute ：可能是 INTENT （意图分类） , RESPONSE or INTENT_RESPONSE_KEY （Retrieve Intent）
            attributes_to_consider.append(label_attribute)
        if (
                training
                and self.component_config[ENTITY_RECOGNITION]
                and self._entity_tag_specs
        ):
            # Add entities as labels only during training and only if there was
            # training data added for entities with DIET configured to predict entities.
            attributes_to_consider.append(ENTITIES)

        if training and label_attribute is not None:
            # only use those training examples that have the label_attribute set
            # during training
            training_data = [
                example for example in training_data if label_attribute in example.data
            ]

        if not training_data:
            # no training data are present to train
            return RasaModelData()

        features_for_examples = model_data_utils.featurize_training_examples(
            training_data,
            attributes_to_consider,
            entity_tag_specs=self._entity_tag_specs,
            featurizers=self.component_config[FEATURIZERS],
            bilou_tagging=self.component_config[BILOU_FLAG],
        )
        # 生成 attribute_data features
        attribute_data, _ = model_data_utils.convert_to_data_format(
            features_for_examples, consider_dialogue_dimension=False
        )
        # 生成模型训练的 RasaModelData
        model_data = RasaModelData(
            label_key=self.label_key, label_sub_key=self.label_sub_key
        )
        model_data.add_data(attribute_data)
        model_data.add_lengths(TEXT, SEQUENCE_LENGTH, TEXT, SEQUENCE)
        # 生成 label_features
        self._add_label_features(
            model_data, training_data, label_attribute, label_id_dict, training
        )

        # make sure all keys are in the same order during training and prediction
        # as we rely on the order of key and sub-key when constructing the actual
        # tensors from the model data
        model_data.sort()

        return model_data

    def _add_label_features(
            self,
            model_data: RasaModelData,
            training_data: List[Message],
            label_attribute: Text,
            label_id_dict: Dict[Text, int],
            training: bool = True,
    ) -> None:
        # 获取  label 对应的 ids
        label_ids = []
        if training and self.component_config[INTENT_CLASSIFICATION]:
            for example in training_data:
                if example.get(label_attribute):
                    label_ids.append(label_id_dict[example.get(label_attribute)])

            # explicitly add last dimension to label_ids
            # to track correctly dynamic sequences
            model_data.add_features(
                LABEL_KEY,
                LABEL_SUB_KEY,
                [FeatureArray(np.expand_dims(label_ids, -1), number_of_dimensions=2)],
            )

        if (
                label_attribute
                and model_data.does_feature_not_exist(label_attribute, SENTENCE)
                and model_data.does_feature_not_exist(label_attribute, SEQUENCE)
        ):
            # no label features are present, get default features from _label_data
            model_data.add_features(
                LABEL, SENTENCE, self._use_default_label_features(np.array(label_ids))
            )

        # as label_attribute can have different values, e.g. INTENT or RESPONSE,
        # copy over the features to the LABEL key to make
        # it easier to access the label features inside the model itself
        model_data.update_key(label_attribute, SENTENCE, LABEL, SENTENCE)
        model_data.update_key(label_attribute, SEQUENCE, LABEL, SEQUENCE)
        model_data.update_key(label_attribute, MASK, LABEL, MASK)

        model_data.add_lengths(LABEL, SEQUENCE_LENGTH, LABEL, SEQUENCE)

    # def _get_exclude_entities(self,training_data: TrainingData) -> List[Text]:
    #
    #     """
    #     # 过滤 regex 中的 entity
    #     """
    #     from rasa.nlu.utils.pattern_utils import extract_patterns
    #     patterns = extract_patterns(training_data)
    #     pattern_names = set([ pattern.get('name') for pattern in patterns])
    #     entities = training_data.entities
    #     exclude_entities = [entity for entity in entities if entity in  pattern_names]
    #     return exclude_entities

    # train helpers
    def preprocess_train_data(self, training_data: TrainingData) -> RasaModelData:
        """Prepares data for training.

        Performs sanity checks on training data, extracts encodings for labels.
        """
        logging.info("训练数据预处理：BILOU 实体标注 和 意图识别 标注")

        #  过滤 regex 中的 entity
        # exclude_entities = self._get_exclude_entities(training_data)

        if self.component_config[BILOU_FLAG]:
            bilou_utils.apply_bilou_schema(training_data)  # ,exclude_entities=exclude_entities)

        # 生成 label_id_index_mapping： 需要将 intent 进行分割进行多意图的识别
        label_id_index_mapping = self._label_id_index_mapping(
            training_data, attribute=INTENT
        )

        if not label_id_index_mapping:
            # no labels are present to train
            return RasaModelData()

        self.index_label_id_mapping = self._invert_mapping(label_id_index_mapping)

        self._label_data = self._create_label_data(
            training_data, label_id_index_mapping, attribute=INTENT
        )
        # EntityTagSpec(tag_name='entity', ids_to_tags={1: 'B-人均消费', 2: 'I-人均消费', 3: 'L-人均消费', 4: 'U-人均消费', 5: 'B-价格',
        self._entity_tag_specs = self._create_entity_tag_specs(training_data)

        label_attribute = (
            INTENT if self.component_config[INTENT_CLASSIFICATION] else None
        )
        # 增加多意图处理
        model_data = self._create_model_data(
            training_data.nlu_examples,
            label_id_index_mapping,
            label_attribute=label_attribute,
        )

        self._check_input_dimension_consistency(model_data)

        return model_data

    @staticmethod
    def _check_enough_labels(model_data: RasaModelData) -> bool:
        return len(np.unique(model_data.get(LABEL_KEY, LABEL_SUB_KEY))) >= 2

    def train(
            self,
            training_data: TrainingData,
            config=None,  # : Optional[RasaNLUModelConfig]
            **kwargs: Any,
    ) -> None:
        """Train the embedding intent classifier on a data set."""
        model_data = self.preprocess_train_data(training_data)
        if model_data.is_empty():
            logger.debug(
                f"Cannot train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the classifier."
            )
            return

        if self.component_config.get(INTENT_CLASSIFICATION):
            if not self._check_enough_labels(model_data):
                logger.error(
                    f"Cannot train '{self.__class__.__name__}'. "
                    f"Need at least 2 different intent classes. "
                    f"Skipping training of classifier."
                )
                return
        if self.component_config.get(ENTITY_RECOGNITION):
            self.check_correct_entity_annotations(training_data)

        # keep one example for persisting and loading
        self._data_example = model_data.first_data_example()

        if not self.finetune_mode:
            # No pre-trained model to load from. Create a new instance of the model.
            self.model = self._instantiate_model_class(model_data)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(self.component_config[LEARNING_RATE]),
                run_eagerly=self.component_config.get(RUN_EAGERLY, False)
            )

        data_generator, validation_data_generator = create_data_generators(
            model_data,
            self.component_config[BATCH_SIZES],
            self.component_config[EPOCHS],
            self.component_config[BATCH_STRATEGY],
            self.component_config[EVAL_NUM_EXAMPLES],
            self.component_config[RANDOM_SEED],
        )
        callbacks = train_utils.create_common_callbacks(
            self.component_config[EPOCHS],
            self.component_config[TENSORBOARD_LOG_DIR],
            self.component_config[TENSORBOARD_LOG_LEVEL],
            self.tmp_checkpoint_dir,
        )

        self.model.fit(
            data_generator,
            epochs=self.component_config[EPOCHS],
            validation_data=validation_data_generator,
            validation_freq=self.component_config[EVAL_NUM_EPOCHS],
            callbacks=callbacks,
            verbose=False,
            shuffle=False,  # we use custom shuffle inside data generator

        )

    # process helpers
    def _predict(
            self, message: Message
    ) -> Optional[Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]]:
        if self.model is None:
            logger.debug(
                f"There is no trained model for '{self.__class__.__name__}': The "
                f"component is either not trained or didn't receive enough training "
                f"data."
            )
            return None

        # create session data from message and convert it into a batch of 1
        model_data = self._create_model_data([message], training=False)
        return self.model.run_inference(model_data)

    def _predict_label(
            self, predict_out: Optional[Dict[Text, tf.Tensor]]
    ) -> Tuple[Dict[Text, Any], List[Dict[Text, Any]]]:
        """Predicts the intent of the provided message."""
        label: Dict[Text, Any] = {"name": None, "id": None, "confidence": 0.0}
        label_ranking = []

        if predict_out is None:
            return label, label_ranking
        # message 与 label 的相似度 shape = (batch_size, num_labels)
        message_sim = predict_out["i_scores"]

        message_sim = message_sim.flatten()  # sim is a matrix
        # 按照 similarity 从大大小排序后获得 index
        label_ids = message_sim.argsort()[::-1]

        if (
                self.component_config[RANKING_LENGTH] > 0
                and self.component_config[MODEL_CONFIDENCE] == SOFTMAX
        ):
            # TODO: This should be removed in 3.0 when softmax as
            #  model confidence and normalization is completely deprecated.
            message_sim = train_utils.normalize(
                message_sim, self.component_config[RANKING_LENGTH]
            )
        message_sim[::-1].sort()
        message_sim = message_sim.tolist()

        # if X contains all zeros do not predict some label
        if label_ids.size > 0:
            label = {
                "id": hash(self.index_label_id_mapping[label_ids[0]]),
                "name": self.index_label_id_mapping[label_ids[0]],
                "confidence": message_sim[0],
            }

            if (
                    self.component_config[RANKING_LENGTH]
                    and 0 < self.component_config[RANKING_LENGTH] < LABEL_RANKING_LENGTH
            ):
                output_length = self.component_config[RANKING_LENGTH]
            else:
                output_length = LABEL_RANKING_LENGTH

            ranking = list(zip(list(label_ids), message_sim))
            ranking = ranking[:output_length]
            label_ranking = [
                {
                    "id": hash(self.index_label_id_mapping[label_idx]),
                    "name": self.index_label_id_mapping[label_idx],
                    "confidence": score,
                }
                for label_idx, score in ranking
            ]

        return label, label_ranking

    def _predict_entities(
            self, predict_out: Optional[Dict[Text, tf.Tensor]], message: Message
    ) -> List[Dict]:
        if predict_out is None:
            return []

        predicted_tags, confidence_values = train_utils.entity_label_to_tags(
            predict_out, self._entity_tag_specs, self.component_config[BILOU_FLAG]
        )

        entities = self.convert_predictions_into_entities(
            message.get(TEXT),
            message.get(TOKENS_NAMES[TEXT], []),
            predicted_tags,
            self.split_entities_config,
            confidence_values,
        )

        entities = self.add_extractor_name(entities)
        entities = message.get(ENTITIES, []) + entities

        return entities

    def process(self, message: Message, **kwargs: Any) -> None:
        """Augments the message with intents, entities, and diagnostic data."""
        out = self._predict(message)

        if self.component_config[INTENT_CLASSIFICATION]:
            label, label_ranking = self._predict_label(out)

            message.set(INTENT, label, add_to_output=True)
            message.set("intent_ranking", label_ranking, add_to_output=True)

        if self.component_config[ENTITY_RECOGNITION]:
            entities = self._predict_entities(out, message)

            message.set(ENTITIES, entities, add_to_output=True)

        if out and DIAGNOSTIC_DATA in out:
            message.add_diagnostic_data(self.unique_name, out.get(DIAGNOSTIC_DATA))

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """
        import shutil

        if self.model is None:
            return {"file": None}

        model_dir = Path(model_dir)
        tf_model_file = model_dir / f"{file_name}.tf_model"

        io_utils.create_directory_for_file(tf_model_file)

        if self.component_config[CHECKPOINT_MODEL]:
            shutil.move(self.tmp_checkpoint_dir, model_dir / "checkpoints")
        self.model.save(str(tf_model_file))

        io_utils.pickle_dump(
            model_dir / f"{file_name}.data_example.pkl", self._data_example
        )
        io_utils.pickle_dump(
            model_dir / f"{file_name}.label_data.pkl", dict(self._label_data.data)
        )
        io_utils.json_pickle(
            model_dir / f"{file_name}.index_label_id_mapping.json",
            self.index_label_id_mapping,
        )

        entity_tag_specs = (
            [tag_spec._asdict() for tag_spec in self._entity_tag_specs]
            if self._entity_tag_specs
            else []
        )
        dump_obj_as_json_to_file(
            model_dir / f"{file_name}.entity_tag_specs.json", entity_tag_specs
        )

        return {"file": file_name}

    @classmethod
    def load(
            cls,
            meta: Dict[Text, Any],
            model_dir: Text,
            model_metadata: Metadata = None,
            cached_component: Optional["DIETClassifier"] = None,
            should_finetune: bool = False,
            **kwargs: Any,
    ) -> "DIETClassifier":
        """Loads the trained model from the provided directory."""
        if not meta.get("file"):
            logger.debug(
                f"Failed to load model for '{cls.__name__}'. "
                f"Maybe you did not provide enough training data and no model was "
                f"trained or the path '{os.path.abspath(model_dir)}' doesn't exist?"
            )
            return cls(component_config=meta)
        # 需要保存的文件
        (
            index_label_id_mapping,
            entity_tag_specs,
            label_data,
            meta,
            data_example,
        ) = cls._load_from_files(meta, model_dir)

        meta = train_utils.override_defaults(cls.defaults, meta)
        meta = train_utils.update_confidence_type(meta)
        meta = train_utils.update_similarity_type(meta)
        meta = train_utils.update_deprecated_loss_type(meta)
        # 加载 DIET 模型
        model = cls._load_model(
            entity_tag_specs,
            label_data,
            meta,
            data_example,
            model_dir,
            finetune_mode=should_finetune,
        )
        # 生成 DIETClassifier 对象
        return cls(
            component_config=meta,
            index_label_id_mapping=index_label_id_mapping,
            entity_tag_specs=entity_tag_specs,
            model=model,
            finetune_mode=should_finetune,
        )

    @classmethod
    def _load_from_files(
            cls, meta: Dict[Text, Any], model_dir: Text
    ) -> Tuple[
        Dict[int, Text],
        List[EntityTagSpec],
        RasaModelData,
        Dict[Text, Any],
        Dict[Text, Dict[Text, List[FeatureArray]]],
    ]:
        file_name = meta.get("file")

        model_dir = Path(model_dir)

        data_example = io_utils.pickle_load(model_dir / f"{file_name}.data_example.pkl")
        label_data = io_utils.pickle_load(model_dir / f"{file_name}.label_data.pkl")
        label_data = RasaModelData(data=label_data)
        index_label_id_mapping = io_utils.json_unpickle(
            model_dir / f"{file_name}.index_label_id_mapping.json"
        )
        entity_tag_specs = io_utils.read_json_file(
            model_dir / f"{file_name}.entity_tag_specs.json"
        )
        entity_tag_specs = [
            EntityTagSpec(
                tag_name=tag_spec["tag_name"],
                ids_to_tags={
                    int(key): value for key, value in tag_spec["ids_to_tags"].items()
                },
                tags_to_ids={
                    key: int(value) for key, value in tag_spec["tags_to_ids"].items()
                },
                num_tags=tag_spec["num_tags"],
            )
            for tag_spec in entity_tag_specs
        ]

        # jsonpickle converts dictionary keys to strings
        index_label_id_mapping = {
            int(key): value for key, value in index_label_id_mapping.items()
        }

        return (
            index_label_id_mapping,
            entity_tag_specs,
            label_data,
            meta,
            data_example,
        )

    @classmethod
    def _load_model(
            cls,
            entity_tag_specs: List[EntityTagSpec],
            label_data: RasaModelData,
            meta: Dict[Text, Any],
            data_example: Dict[Text, Dict[Text, List[FeatureArray]]],
            model_dir: Text,
            finetune_mode: bool = False,
    ) -> "RasaModel":
        file_name = meta.get("file")
        tf_model_file = os.path.join(model_dir, file_name + ".tf_model")

        label_key = LABEL_KEY if meta[INTENT_CLASSIFICATION] else None
        label_sub_key = LABEL_SUB_KEY if meta[INTENT_CLASSIFICATION] else None

        model_data_example = RasaModelData(
            label_key=label_key, label_sub_key=label_sub_key, data=data_example
        )

        model = cls._load_model_class(
            tf_model_file,
            model_data_example,
            label_data,
            entity_tag_specs,
            meta,
            finetune_mode=finetune_mode,
        )

        return model

    @classmethod
    def _load_model_class(
            cls,
            tf_model_file: Text,
            model_data_example: RasaModelData,
            label_data: RasaModelData,
            entity_tag_specs: List[EntityTagSpec],
            meta: Dict[Text, Any],
            finetune_mode: bool,
    ) -> "RasaModel":

        predict_data_example = RasaModelData(
            label_key=model_data_example.label_key,
            data={
                feature_name: features
                for feature_name, features in model_data_example.items()
                if TEXT in feature_name
            },
        )

        return cls.model_class().load(
            tf_model_file,
            model_data_example,
            predict_data_example,
            data_signature=model_data_example.get_signature(),
            label_data=label_data,
            entity_tag_specs=entity_tag_specs,
            config=copy.deepcopy(meta),
            finetune_mode=finetune_mode,
        )

    def _instantiate_model_class(self, model_data: RasaModelData) -> "RasaModel":
        return self.model_class()(
            data_signature=model_data.get_signature(),
            label_data=self._label_data,
            entity_tag_specs=self._entity_tag_specs,
            config=self.component_config,
        )


class DIET(TransformerRasaModel):
    def __init__(
            self,
            data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
            label_data: RasaModelData,
            entity_tag_specs: Optional[List[EntityTagSpec]],
            config: Dict[Text, Any],
    ) -> None:
        # create entity tag spec before calling super otherwise building the model
        # will fail
        super().__init__("DIET", config, data_signature, label_data)
        self._entity_tag_specs = self._ordered_tag_specs(entity_tag_specs)

        # 生成 predict_data_signature
        self.predict_data_signature = {
            feature_name: features
            for feature_name, features in data_signature.items()
            if TEXT in feature_name
        }

        # tf training
        self._create_metrics()
        self._update_metrics_to_log()

        # needed for efficient prediction
        self.all_labels_embed: Optional[tf.Tensor] = None

        self._prepare_layers()

    @staticmethod
    def _ordered_tag_specs(
            entity_tag_specs: Optional[List[EntityTagSpec]],
    ) -> List[EntityTagSpec]:
        """Ensure that order of entity tag specs matches CRF layer order."""
        if entity_tag_specs is None:
            return []

        crf_order = [
            ENTITY_ATTRIBUTE_TYPE,
            ENTITY_ATTRIBUTE_ROLE,
            ENTITY_ATTRIBUTE_GROUP,
        ]

        ordered_tag_spec = []

        for tag_name in crf_order:
            for tag_spec in entity_tag_specs:
                if tag_name == tag_spec.tag_name:
                    ordered_tag_spec.append(tag_spec)

        return ordered_tag_spec

    def _check_data(self) -> None:
        if TEXT not in self.data_signature:
            raise Exception(
                f"No text features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if self.config[INTENT_CLASSIFICATION]:
            if LABEL not in self.data_signature:
                raise Exception(
                    f"No label features specified. "
                    f"Cannot train '{self.__class__.__name__}' model."
                )

            if self.config[SHARE_HIDDEN_LAYERS]:
                different_sentence_signatures = False
                different_sequence_signatures = False
                if (
                        SENTENCE in self.data_signature[TEXT]
                        and SENTENCE in self.data_signature[LABEL]
                ):
                    different_sentence_signatures = (
                            self.data_signature[TEXT][SENTENCE]
                            != self.data_signature[LABEL][SENTENCE]
                    )
                if (
                        SEQUENCE in self.data_signature[TEXT]
                        and SEQUENCE in self.data_signature[LABEL]
                ):
                    different_sequence_signatures = (
                            self.data_signature[TEXT][SEQUENCE]
                            != self.data_signature[LABEL][SEQUENCE]
                    )

                if different_sentence_signatures or different_sequence_signatures:
                    raise ValueError(
                        "If hidden layer weights are shared, data signatures "
                        "for text_features and label_features must coincide."
                    )

        if self.config[ENTITY_RECOGNITION] and (
                ENTITIES not in self.data_signature
                or ENTITY_ATTRIBUTE_TYPE not in self.data_signature[ENTITIES]
        ):
            logger.debug(
                f"You specified '{self.__class__.__name__}' to train entities, but "
                f"no entities are present in the training data. Skipping training of "
                f"entities."
            )
            self.config[ENTITY_RECOGNITION] = False

    def _create_metrics(self) -> None:
        # self.metrics will have the same order as they are created
        # so create loss metrics first to output losses first
        self.mask_loss = tf.keras.metrics.Mean(name="m_loss")
        self.intent_loss = tf.keras.metrics.Mean(name="i_loss")
        self.entity_loss = tf.keras.metrics.Mean(name="e_loss")
        self.entity_group_loss = tf.keras.metrics.Mean(name="g_loss")
        self.entity_role_loss = tf.keras.metrics.Mean(name="r_loss")
        # create accuracy metrics second to output accuracies second
        self.mask_acc = tf.keras.metrics.Mean(name="m_acc")
        self.intent_acc = tf.keras.metrics.Mean(name="i_acc")
        self.entity_f1 = tf.keras.metrics.Mean(name="e_f1")
        self.entity_group_f1 = tf.keras.metrics.Mean(name="g_f1")
        self.entity_role_f1 = tf.keras.metrics.Mean(name="r_f1")

    def _update_metrics_to_log(self) -> None:
        # debug_log_level = logging.getLogger("rasa").level == logging.DEBUG
        debug_log_level = logging.getLogger(__name__).level == logging.DEBUG

        if self.config[MASKED_LM]:
            self.metrics_to_log.append("m_acc")
            # if debug_log_level:
            self.metrics_to_log.append("m_loss")
        if self.config[INTENT_CLASSIFICATION]:
            self.metrics_to_log.append("i_acc")
            # if debug_log_level:
            self.metrics_to_log.append("i_loss")
        if self.config[ENTITY_RECOGNITION]:
            for tag_spec in self._entity_tag_specs:
                if tag_spec.num_tags != 0:
                    name = tag_spec.tag_name
                    self.metrics_to_log.append(f"{name[0]}_f1")
                    # if debug_log_level:
                    self.metrics_to_log.append(f"{name[0]}_loss")

        self._log_metric_info()

    def _log_metric_info(self) -> None:
        metric_name = {
            "t": "total",
            "i": "intent",
            "e": "entity",
            "m": "mask",
            "r": "role",
            "g": "group",
        }
        logger.debug("Following metrics will be logged during training: ")
        for metric in self.metrics_to_log:
            parts = metric.split("_")
            name = f"{metric_name[parts[0]]} {parts[1]}"
            logger.debug(f"  {metric} ({name})")

    def _prepare_layers(self) -> None:
        # For user text, prepare layers that combine different feature types, embed
        # everything using a transformer and optionally also do masked language
        # modeling.
        self.text_name = TEXT
        self._tf_layers[
            f"sequence_layer.{self.text_name}"
        ] = rasa_layers.RasaSequenceLayer(
            self.text_name, self.data_signature[self.text_name], self.config
        )
        if self.config[MASKED_LM]:
            self._prepare_mask_lm_loss(self.text_name)

        # Intent labels are treated similarly to user text but without the transformer,
        # without masked language modelling, and with no dropout applied to the
        # individual features, only to the overall label embedding after all label
        # features have been combined.
        if self.config[INTENT_CLASSIFICATION]:
            self.label_name = TEXT if self.config[SHARE_HIDDEN_LAYERS] else LABEL

            # disable input dropout applied to sparse and dense label features
            label_config = self.config.copy()
            label_config.update(
                {SPARSE_INPUT_DROPOUT: False, DENSE_INPUT_DROPOUT: False}
            )
            # label 方面的 训练层: feature_combining_layer.label + ffnn + classification_layer
            self._tf_layers[
                f"feature_combining_layer.{self.label_name}"
            ] = rasa_layers.RasaFeatureCombiningLayer(
                self.label_name, self.label_signature[self.label_name], label_config
            )

            self._prepare_ffnn_layer(
                self.label_name,
                self.config[HIDDEN_LAYERS_SIZES][self.label_name],
                self.config[DROP_RATE],
            )

            self._prepare_label_classification_layers(predictor_attribute=TEXT)

        if self.config[ENTITY_RECOGNITION]:
            self._prepare_entity_recognition_layers()

    def _prepare_mask_lm_loss(self, name: Text) -> None:
        # for embedding predicted tokens at masked positions
        self._prepare_embed_layers(f"{name}_lm_mask")

        # for embedding the true tokens that got masked
        self._prepare_embed_layers(f"{name}_golden_token")

        # mask loss is additional loss
        # set scaling to False, so that it doesn't overpower other losses
        self._prepare_dot_product_loss(f"{name}_mask", scale_loss=False)

    def _create_bow(
            self,
            sequence_features: List[Union[tf.Tensor, tf.SparseTensor]],
            sentence_features: List[Union[tf.Tensor, tf.SparseTensor]],
            sequence_feature_lengths: tf.Tensor,
            name: Text,
    ) -> tf.Tensor:

        x, _ = self._tf_layers[f"feature_combining_layer.{name}"](
            (sequence_features, sentence_features, sequence_feature_lengths),
            training=self._training,
        )

        # convert to bag-of-words by summing along the sequence dimension
        x = tf.reduce_sum(x, axis=1)

        return self._tf_layers[f"ffnn.{name}"](x, self._training)

    def _create_all_labels(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_label_ids = self.tf_label_data[LABEL_KEY][LABEL_SUB_KEY][0]

        sequence_feature_lengths = self._get_sequence_feature_lengths(
            self.tf_label_data, LABEL
        )

        x = self._create_bow(
            self.tf_label_data[LABEL][SEQUENCE],
            self.tf_label_data[LABEL][SENTENCE],
            sequence_feature_lengths,
            self.label_name,
        )
        all_labels_embed = self._tf_layers[f"embed.{LABEL}"](x)

        return all_label_ids, all_labels_embed

    def _mask_loss(
            self,
            outputs: tf.Tensor,
            inputs: tf.Tensor,
            seq_ids: tf.Tensor,
            mlm_mask_boolean: tf.Tensor,
            name: Text,
    ) -> tf.Tensor:
        # make sure there is at least one element in the mask
        mlm_mask_boolean = tf.cond(
            tf.reduce_any(mlm_mask_boolean),
            lambda: mlm_mask_boolean,
            lambda: tf.scatter_nd([[0, 0, 0]], [True], tf.shape(mlm_mask_boolean)),
        )

        mlm_mask_boolean = tf.squeeze(mlm_mask_boolean, -1)

        # Pick elements that were masked, throwing away the batch & sequence dimension
        # and effectively switching from shape (batch_size, sequence_length, units) to
        # (num_masked_elements, units).
        outputs = tf.boolean_mask(outputs, mlm_mask_boolean)
        inputs = tf.boolean_mask(inputs, mlm_mask_boolean)
        ids = tf.boolean_mask(seq_ids, mlm_mask_boolean)

        tokens_predicted_embed = self._tf_layers[f"embed.{name}_lm_mask"](outputs)
        tokens_true_embed = self._tf_layers[f"embed.{name}_golden_token"](inputs)

        # To limit the otherwise computationally expensive loss calculation, we
        # constrain the label space in MLM (i.e. token space) to only those tokens that
        # were masked in this batch. Hence the reduced list of token embeddings
        # (tokens_true_embed) and the reduced list of labels (ids) are passed as
        # all_labels_embed and all_labels, respectively. In the future, we could be less
        # restrictive and construct a slightly bigger label space which could include
        # tokens not masked in the current batch too.
        return self._tf_layers[f"loss.{name}_mask"](
            inputs_embed=tokens_predicted_embed,
            labels_embed=tokens_true_embed,
            labels=ids,
            all_labels_embed=tokens_true_embed,
            all_labels=ids,
        )

    def _calculate_label_loss(
            self, text_features: tf.Tensor, label_features: tf.Tensor, label_ids: tf.Tensor
    ) -> tf.Tensor:
        all_label_ids, all_labels_embed = self._create_all_labels()

        text_embed = self._tf_layers[f"embed.{TEXT}"](text_features)
        label_embed = self._tf_layers[f"embed.{LABEL}"](label_features)

        return self._tf_layers[f"loss.{LABEL}"](
            text_embed, label_embed, label_ids, all_labels_embed, all_label_ids
        )

    def batch_loss(
            self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        """Calculates the loss for the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The loss of the given batch.
        """
        tf_batch_data = self.batch_to_model_data_format(batch_in, self.data_signature)

        sequence_feature_lengths = self._get_sequence_feature_lengths(
            tf_batch_data, TEXT
        )

        # 输入的数据格式： attribute =TEXT 的features: feature_type分为 SEQUENCE and SENTENCE
        # 每个 feature_type又是一个 List[Tensor] 或者 List[SparseTensor]，之后通过是否是 SparseTensor 进行操作
        (
            text_transformed,
            text_in,
            mask_combined_sequence_sentence,
            text_seq_ids,
            mlm_mask_boolean_text,
            _,
        ) = self._tf_layers[f"sequence_layer.{self.text_name}"](
            (
                tf_batch_data[TEXT][SEQUENCE],
                tf_batch_data[TEXT][SENTENCE],
                sequence_feature_lengths,
            ),
            training=self._training,
        )

        losses = []

        # Lengths of sequences in case of sentence-level features are always 1, but they
        # can effectively be 0 if sentence-level features aren't present.
        sentence_feature_lengths = self._get_sentence_feature_lengths(
            tf_batch_data, TEXT
        )

        combined_sequence_sentence_feature_lengths = (
                sequence_feature_lengths + sentence_feature_lengths
        )

        if self.config[MASKED_LM]:
            loss, acc = self._mask_loss(
                text_transformed, text_in, text_seq_ids, mlm_mask_boolean_text, TEXT
            )
            self.mask_loss.update_state(loss)
            self.mask_acc.update_state(acc)
            losses.append(loss)

        if self.config[INTENT_CLASSIFICATION]:
            loss = self._batch_loss_intent(
                combined_sequence_sentence_feature_lengths,
                text_transformed,
                tf_batch_data,
            )
            losses.append(loss)

        if self.config[ENTITY_RECOGNITION]:
            losses += self._batch_loss_entities(
                mask_combined_sequence_sentence,
                sequence_feature_lengths,
                text_transformed,
                tf_batch_data,
            )

        return tf.math.add_n(losses)

    def _batch_loss_intent(
            self,
            combined_sequence_sentence_feature_lengths_text: tf.Tensor,
            text_transformed: tf.Tensor,
            tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
    ) -> tf.Tensor:
        # get sentence features vector for intent classification
        sentence_vector = self._last_token(
            text_transformed, combined_sequence_sentence_feature_lengths_text
        )

        sequence_feature_lengths_label = self._get_sequence_feature_lengths(
            tf_batch_data, LABEL
        )

        label_ids = tf_batch_data[LABEL_KEY][LABEL_SUB_KEY][0]
        label = self._create_bow(
            tf_batch_data[LABEL][SEQUENCE],
            tf_batch_data[LABEL][SENTENCE],
            sequence_feature_lengths_label,
            self.label_name,
        )
        loss, acc = self._calculate_label_loss(sentence_vector, label, label_ids)

        self._update_label_metrics(loss, acc)

        return loss

    def _update_label_metrics(self, loss: tf.Tensor, acc: tf.Tensor) -> None:

        self.intent_loss.update_state(loss)
        self.intent_acc.update_state(acc)

    def _batch_loss_entities(
            self,
            mask_combined_sequence_sentence: tf.Tensor,
            sequence_feature_lengths: tf.Tensor,
            text_transformed: tf.Tensor,
            tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
    ) -> List[tf.Tensor]:
        losses = []

        entity_tags = None

        for tag_spec in self._entity_tag_specs:
            if tag_spec.num_tags == 0:
                continue

            tag_ids = tf_batch_data[ENTITIES][tag_spec.tag_name][0]
            # add a zero (no entity) for the sentence features to match the shape of
            # inputs
            tag_ids = tf.pad(tag_ids, [[0, 0], [0, 1], [0, 0]])

            loss, f1, _logits = self._calculate_entity_loss(
                text_transformed,
                tag_ids,
                mask_combined_sequence_sentence,
                sequence_feature_lengths,
                tag_spec.tag_name,
                entity_tags,
            )

            if tag_spec.tag_name == ENTITY_ATTRIBUTE_TYPE:
                # use the entity tags as additional input for the role
                # and group CRF
                entity_tags = tf.one_hot(
                    tf.cast(tag_ids[:, :, 0], tf.int32), depth=tag_spec.num_tags
                )

            self._update_entity_metrics(loss, f1, tag_spec.tag_name)

            losses.append(loss)

        return losses

    def _update_entity_metrics(
            self, loss: tf.Tensor, f1: tf.Tensor, tag_name: Text
    ) -> None:
        if tag_name == ENTITY_ATTRIBUTE_TYPE:
            self.entity_loss.update_state(loss)
            self.entity_f1.update_state(f1)
        elif tag_name == ENTITY_ATTRIBUTE_GROUP:
            self.entity_group_loss.update_state(loss)
            self.entity_group_f1.update_state(f1)
        elif tag_name == ENTITY_ATTRIBUTE_ROLE:
            self.entity_role_loss.update_state(loss)
            self.entity_role_f1.update_state(f1)

    def prepare_for_predict(self) -> None:
        """Prepares the model for prediction."""
        if self.config[INTENT_CLASSIFICATION]:
            _, self.all_labels_embed = self._create_all_labels()

    def batch_predict(
            self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, tf.Tensor]:
        """Predicts the output of the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The output to predict.
        """
        tf_batch_data = self.batch_to_model_data_format(
            batch_in, self.predict_data_signature
        )

        sequence_feature_lengths = self._get_sequence_feature_lengths(
            tf_batch_data, TEXT
        )
        sentence_feature_lengths = self._get_sentence_feature_lengths(
            tf_batch_data, TEXT,
        )

        text_transformed, _, _, _, _, attention_weights = self._tf_layers[
            f"sequence_layer.{self.text_name}"
        ](
            (
                tf_batch_data[TEXT][SEQUENCE],
                tf_batch_data[TEXT][SENTENCE],
                sequence_feature_lengths,
            ),
            training=self._training,
        )
        predictions = {
            DIAGNOSTIC_DATA: {
                "attention_weights": attention_weights,
                "text_transformed": text_transformed,
            }
        }

        if self.config[INTENT_CLASSIFICATION]:
            predictions.update(
                self._batch_predict_intents(
                    sequence_feature_lengths + sentence_feature_lengths,
                    text_transformed,
                )
            )

        if self.config[ENTITY_RECOGNITION]:
            predictions.update(
                self._batch_predict_entities(sequence_feature_lengths, text_transformed)
            )

        return predictions

    def _batch_predict_entities(
            self, sequence_feature_lengths: tf.Tensor, text_transformed: tf.Tensor
    ) -> Dict[Text, tf.Tensor]:
        predictions: Dict[Text, tf.Tensor] = {}

        entity_tags = None

        for tag_spec in self._entity_tag_specs:
            # skip crf layer if it was not trained
            if tag_spec.num_tags == 0:
                continue

            name = tag_spec.tag_name
            _input = text_transformed

            if entity_tags is not None:
                _tags = self._tf_layers[f"embed.{name}.tags"](entity_tags)
                _input = tf.concat([_input, _tags], axis=-1)

            _logits = self._tf_layers[f"embed.{name}.logits"](_input)
            pred_ids, confidences = self._tf_layers[f"crf.{name}"](
                _logits, sequence_feature_lengths
            )

            predictions[f"e_{name}_ids"] = pred_ids
            predictions[f"e_{name}_scores"] = confidences

            if name == ENTITY_ATTRIBUTE_TYPE:
                # use the entity tags as additional input for the role
                # and group CRF
                entity_tags = tf.one_hot(
                    tf.cast(pred_ids, tf.int32), depth=tag_spec.num_tags
                )

        return predictions

    def _batch_predict_intents(
            self,
            combined_sequence_sentence_feature_lengths: tf.Tensor,
            text_transformed: tf.Tensor,
    ) -> Dict[Text, tf.Tensor]:

        if self.all_labels_embed is None:
            raise ValueError(
                "The model was not prepared for prediction. "
                "Call `prepare_for_predict` first."
            )

        # get sentence feature vector for intent classification
        sentence_vector = self._last_token(
            text_transformed, combined_sequence_sentence_feature_lengths
        )

        # 使用 train的时候定义的 embed.text layer 进行编码
        sentence_vector_embed = self._tf_layers[f"embed.{TEXT}"](sentence_vector)

        # 计算 confidence = (batch_size,label_num)
        # sentence_vector_embed = (batch_size, embedding_size)
        # all_labels_embed = (label_num, embedding_size)
        _, scores = self._tf_layers[
            f"loss.{LABEL}"
        ].similarity_confidence_from_embeddings(
            sentence_vector_embed[:, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, :, :],
        )

        return {"i_scores": scores}


class HierarchicalDIETClassifier(DIETClassifier):

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 index_label_id_mapping: Optional[Dict[int, Text]] = None,
                 index_sub_label_id_mapping: Dict[int, Text] = None,
                 label_to_sub_label_mapping: Dict[Text, List[Text]] = None,
                 entity_tag_specs: Optional[List[EntityTagSpec]] = None,
                 model: Optional[RasaModel] = None,
                 finetune_mode: bool = False):
        """
        新增 参数：
        index_sub_label_id_mapping ： 属性对应的id
        intent_attribute_mapping： 意图和属性的映射关系
        """
        # 在DIETClassifier的 defaults 参数中新增 SUB_INTENT_CLASSIFICATION

        self.defaults.update({
            SUB_INTENT_CLASSIFICATION: False,
            LABEL_TO_SUB_LABEL_MAPPING: None,
            HIDDEN_LAYERS_SIZES: {TEXT: [], LABEL: [], SUB_LABEL: []},
            HARD_HIERARCHICAL: True
        })

        super(HierarchicalDIETClassifier, self).__init__(component_config=component_config,
                                                         index_label_id_mapping=index_label_id_mapping,
                                                         entity_tag_specs=entity_tag_specs,
                                                         model=model,
                                                         finetune_mode=finetune_mode)

        # 新增 load时候的初始化参数 label_to_sub_label_mapping，其在训练的时候生成，在 persist的时候保存，在 load的时候加载
        self.label_to_sub_label_mapping = label_to_sub_label_mapping

        # 新增 load时候的初始化参数 index_sub_label_id_mapping，其在训练的时候生成，在 persist的时候保存，在 load的时候加载
        self.index_sub_label_id_mapping = index_sub_label_id_mapping
        if self.index_sub_label_id_mapping:
            self.sub_label_id_index_mapping = self._invert_mapping(self.index_sub_label_id_mapping)
        # 使用 hard_hirarchical 的方式进行属性识别
        self.hard_hierarchical = self.component_config.get(HARD_HIERARCHICAL)

        # self.tmp_checkpoint_dir = Path(self.component_config[CHECKPOINT_MODEL_DIR])
        # self.component_config.update({"sub_intent_classification":sub_intent_classification})

    @staticmethod
    def model_class() -> Type[RasaModel]:
        return HierarchicalDIET

    def train(
            self,
            training_data: TrainingData,
            config=None,  # : Optional[RasaNLUModelConfig]
            **kwargs: Any,
    ) -> None:
        """Train the embedding intent classifier on a data set."""

        model_data = self.preprocess_train_data(training_data)

        if model_data.is_empty():
            logger.debug(
                f"Cannot train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the classifier."
            )
            return

        if self.component_config.get(INTENT_CLASSIFICATION):
            if not self._check_enough_labels(model_data):
                logger.error(
                    f"Cannot train '{self.__class__.__name__}'. "
                    f"Need at least 2 different intent classes. "
                    f"Skipping training of classifier."
                )
                return

        if self.component_config.get(ENTITY_RECOGNITION):
            self.check_correct_entity_annotations(training_data)

        # keep one example for persisting and loading
        self._data_example = model_data.first_data_example()

        if not self.finetune_mode:
            # No pre-trained model to load from. Create a new instance of the model.
            self.model = self._instantiate_model_class(model_data)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(self.component_config[LEARNING_RATE]),
                run_eagerly=self.component_config.get(RUN_EAGERLY, False)
            )

        data_generator, validation_data_generator = create_data_generators(
            model_data,
            self.component_config[BATCH_SIZES],
            self.component_config[EPOCHS],
            self.component_config[BATCH_STRATEGY],
            self.component_config[EVAL_NUM_EXAMPLES],
            self.component_config[RANDOM_SEED],
        )
        callbacks = train_utils.create_common_callbacks(
            self.component_config[EPOCHS],
            self.component_config[TENSORBOARD_LOG_DIR],
            self.component_config[TENSORBOARD_LOG_LEVEL],
            self.tmp_checkpoint_dir,
        )

        self.model.fit(
            data_generator,
            epochs=self.component_config[EPOCHS],
            validation_data=validation_data_generator,
            validation_freq=self.component_config[EVAL_NUM_EPOCHS],
            callbacks=callbacks,
            verbose=False,
            shuffle=False,  # we use custom shuffle inside data generator

        )

    # train helpers
    def preprocess_train_data(self, training_data: TrainingData) -> RasaModelData:
        """Prepares data for training.

        Performs sanity checks on training data, extracts encodings for labels.
        """
        logging.info("训练数据预处理：BILOU 实体标注 和 意图识别 标注")

        if self.component_config[BILOU_FLAG]:
            bilou_utils.apply_bilou_schema(training_data)  # ,exclude_entities=exclude_entities)

        # 生成 label_id_index_mapping： 需要将 intent 进行分割进行多意图的识别
        label_id_index_mapping = self._label_id_index_mapping(
            training_data, attribute=INTENT
        )

        if not label_id_index_mapping:
            # no labels are present to train
            return RasaModelData()

        self.index_label_id_mapping = self._invert_mapping(label_id_index_mapping)

        if self.component_config.get(SUB_INTENT_CLASSIFICATION):
            # 获取 label_to_sub_label_mapping
            self.label_to_sub_label_mapping = self._label_to_sub_label_mapping(training_data=training_data,
                                                                               label_attribute=INTENT,
                                                                               sub_label_attribute=ATTRIBUTE)

            # 生成 sub_label_id_index_mapping
            sub_label_id_index_mapping = self._label_id_index_mapping(training_data, attribute=ATTRIBUTE)
            self.index_sub_label_id_mapping = self._invert_mapping(sub_label_id_index_mapping)
            # 使用 sub_label_id_index_mapping 和 ATTRIBUTE 属性数据创建 _sub_label_data 对象
            # self._sub_label_data = self._create_sub_label_data(
            #     training_data, sub_label_id_index_mapping, attribute=ATTRIBUTE
            # )
        else:
            sub_label_id_index_mapping = None
            self.index_sub_label_id_mapping = None
            self.label_to_sub_label_mapping = None
        # 生成 _label_data，用于后面 _create_model_data 和 构建模型的时候
        self._label_data = self._create_label_data(
            training_data, label_id_index_mapping, attribute=INTENT,
            sub_label_id_dict=sub_label_id_index_mapping, sub_attribute=ATTRIBUTE,
        )

        # EntityTagSpec(tag_name='entity', ids_to_tags={1: 'B-人均消费', 2: 'I-人均消费', 3: 'L-人均消费', 4: 'U-人均消费', 5: 'B-价格',
        self._entity_tag_specs = self._create_entity_tag_specs(training_data)

        label_attribute = (INTENT if self.component_config[INTENT_CLASSIFICATION] else None)
        sub_label_attribute = (ATTRIBUTE if self.component_config[SUB_INTENT_CLASSIFICATION] else None)

        # 增加多意图处理
        model_data = self._create_model_data(
            training_data.nlu_examples,
            label_id_index_mapping,
            label_attribute=label_attribute,
            sub_label_attribute=sub_label_attribute,
            sub_label_id_dict=sub_label_id_index_mapping
        )

        self._check_input_dimension_consistency(model_data)

        return model_data

    def _create_model_data(
            self,
            training_data: List[Message],
            label_id_dict: Optional[Dict[Text, int]] = None,
            label_attribute: Optional[Text] = None,
            sub_label_attribute: Optional[Text] = None,
            sub_label_id_dict: Optional[Dict[Text, int]] = None,
            training: bool = True,

    ) -> RasaModelData:
        """Prepare data for training and create a RasaModelData object.
        重载 _create_model_data() method 加入 sub_intent 数据
        模型数据处理流程：training_data -> features_for_examples -> attribute_data -> model_data
        """
        # from rasa.utils.tensorflow import model_data_utils
        from data_process.training_data import rasa_model_data_utils as  model_data_utils

        # 需要考虑的 attributes_to_consider：
        attributes_to_consider = [TEXT]
        if training and self.component_config[INTENT_CLASSIFICATION]:
            # we don't have any intent labels during prediction, just add them during training
            # label_attribute ：可能是 INTENT （意图分类） , RESPONSE or INTENT_RESPONSE_KEY （Retrieve Intent）
            attributes_to_consider.append(label_attribute)

        if training_data and self.component_config[SUB_INTENT_CLASSIFICATION]:
            attributes_to_consider.append(sub_label_attribute)

        if (
                training
                and self.component_config[ENTITY_RECOGNITION]
                and self._entity_tag_specs
        ):
            # Add entities as labels only during training and only if there was
            # training data added for entities with DIET configured to predict entities.
            attributes_to_consider.append(ENTITIES)

        if training and label_attribute is not None:
            # only use those training examples that have the label_attribute set
            # during training
            training_data = [
                example for example in training_data if label_attribute in example.data
            ]

        if not training_data:
            # no training data are present to train
            return RasaModelData()
        # 将 training_examples 转换为 attribute_to_feautures 的列表：
        # 每个 example 对应 Dict[Text, List["Features"]]
        features_for_examples = model_data_utils.featurize_training_examples(
            training_data,
            attributes_to_consider,
            entity_tag_specs=self._entity_tag_specs,
            featurizers=self.component_config[FEATURIZERS],
            bilou_tagging=self.component_config[BILOU_FLAG],
        )

        # 生成 attribute_data: 每个 value 对应一个字典： key: ['sequence','sentence'] ,value = List[FeatureArray]
        # List[FeatureArray] 每个元素对应 sparse or dense
        # FeatureArray: 维度0 表示 examples, 维度1 表示 featurizers对应的 List[coo_matrix or np.ndarray], 维度2 表示 coo_matrix or np.ndarray
        attribute_data, _ = model_data_utils.convert_to_data_format(
            features_for_examples, consider_dialogue_dimension=False
        )

        # 生成模型训练的 RasaModelData
        model_data = RasaModelData(
            label_key=self.label_key, label_sub_key=self.label_sub_key
        )

        model_data.add_data(attribute_data)
        model_data.add_lengths(TEXT, SEQUENCE_LENGTH, TEXT, SEQUENCE)

        # 生成 label_features
        self._add_label_features(
            model_data, training_data, label_attribute, label_id_dict, training
        )

        # TODO: _add_sub_label_features
        if self.component_config[SUB_INTENT_CLASSIFICATION]:
            self._add_sub_label_features(
                model_data, training_data, sub_label_attribute, sub_label_id_dict, training
            )

        # make sure all keys are in the same order during training and prediction
        # as we rely on the order of key and sub-key when constructing the actual
        # tensors from the model data
        model_data.sort()

        return model_data

    def _label_to_sub_label_mapping(self,
                                    training_data: TrainingData, label_attribute: Text, sub_label_attribute: Text
                                    ) -> Dict[Text, List[Text]]:
        """Create label_id dictionary.
        label_attribute:
        sub_label_attribute:

        """
        label_to_sub_label_mapping = defaultdict(set)
        for example in training_data.intent_examples:
            label = example.get(label_attribute)
            sub_label = example.get(sub_label_attribute)
            label_to_sub_label_mapping[label].add(sub_label)
        label_to_sub_label_mapping = {label: list(sub_label_set)
                                      for label, sub_label_set in label_to_sub_label_mapping.items()}
        return label_to_sub_label_mapping

    def _create_label_data(
            self,
            training_data: TrainingData,
            label_id_dict: Dict[Text, int],
            attribute: Text,
            sub_label_id_dict: Optional[Dict[Text, int]] = None,
            sub_attribute: Optional[Text] = None
    ) -> RasaModelData:
        """Create matrix with label_ids encoded in rows as bag of words.
        Find a training example for each label and get the encoded features
        from the corresponding Message object.
        If the features are already computed, fetch them from the message object
        else compute a one hot encoding for the label as the feature vector.
        将 sub_label_id_dict 融入到 label_data 中
        """

        # Collect one example for each label
        labels_idx_examples = []
        for label_name, idx in label_id_dict.items():
            label_example = self._find_example_for_label(
                label_name, training_data.intent_examples, attribute
            )
            labels_idx_examples.append((idx, label_example))

        # Sort the list of tuples based on label_idx
        labels_idx_examples = sorted(labels_idx_examples, key=lambda x: x[0])
        labels_example = [example for (_, example) in labels_idx_examples]

        # Collect features, precomputed if they exist, else compute on the fly
        if self._check_labels_features_exist(labels_example, attribute):
            (
                sequence_features,
                sentence_features,
            ) = self._extract_labels_precomputed_features(labels_example, attribute)
        else:
            sequence_features = None
            # 生产要给 3维的 one-hot 格式的 sentence_features: (label_num,1,label_num), 每个 label_id 对应一个 向量
            sentence_features = self._compute_default_label_features(labels_example)

        label_data = RasaModelData()
        label_data.add_features(LABEL, SEQUENCE, sequence_features)
        label_data.add_features(LABEL, SENTENCE, sentence_features)

        if label_data.does_feature_not_exist(
                LABEL, SENTENCE
        ) and label_data.does_feature_not_exist(LABEL, SEQUENCE):
            raise ValueError(
                "No label features are present. Please check your configuration file."
            )

        label_ids = np.array([idx for (idx, _) in labels_idx_examples])
        # explicitly add last dimension to label_ids to track correctly dynamic sequences
        # 增加 label/id feature = (3, 1)
        label_data.add_features(
            LABEL_KEY,
            LABEL_SUB_KEY,
            [FeatureArray(np.expand_dims(label_ids, -1), number_of_dimensions=2)],
        )
        label_data.add_lengths(LABEL, SEQUENCE_LENGTH, LABEL, SEQUENCE)
        #  label_data : {'label': {'sentence': [FeatureSignature(is_sparse=False, units=3, number_of_dimensions=3)],
        #   'ids': [FeatureSignature(is_sparse=False, units=1, number_of_dimensions=2)]}}
        # 增加 sub_label 信息
        if self.component_config[SUB_INTENT_CLASSIFICATION]:
            self._add_sub_label_data(label_data, training_data, sub_label_id_dict, attribute=sub_attribute)

        return label_data

    def _add_sub_label_data(
            self,
            label_data: RasaModelData,
            training_data: TrainingData,
            label_id_dict: Dict[Text, int],
            attribute: Text,
    ) -> RasaModelData:
        """Create matrix with label_ids encoded in rows as bag of words.
        Find a training example for each label and get the encoded features
        from the corresponding Message object.
        If the features are already computed, fetch them from the message object
        else compute a one hot encoding for the label as the feature vector.
        """
        # Collect one example for each label
        labels_idx_examples = []
        for label_name, idx in label_id_dict.items():
            label_example = self._find_example_for_label(
                label_name, training_data.intent_examples, attribute
            )
            labels_idx_examples.append((idx, label_example))

        # Sort the list of tuples based on label_idx
        labels_idx_examples = sorted(labels_idx_examples, key=lambda x: x[0])
        labels_example = [example for (_, example) in labels_idx_examples]

        # Collect features, precomputed if they exist, else compute on the fly
        if self._check_labels_features_exist(labels_example, attribute):
            (
                sequence_features,
                sentence_features,
            ) = self._extract_labels_precomputed_features(labels_example, attribute)
        else:
            sequence_features = None
            # 生产要给 3维的 one-hot 格式的 sentence_features: (label_num,1,label_num), 每个 label_id 对应一个 向量
            sentence_features = self._compute_default_label_features(labels_example)

        # label_data = RasaModelData()
        label_data.add_features(SUB_LABEL, SEQUENCE, sequence_features, update_num_examples=False)
        label_data.add_features(SUB_LABEL, SENTENCE, sentence_features, update_num_examples=False)

        if label_data.does_feature_not_exist(
                SUB_LABEL, SENTENCE
        ) and label_data.does_feature_not_exist(SUB_LABEL, SEQUENCE):
            raise ValueError(
                "No label features are present. Please check your configuration file."
            )
        # 将  label_ids 转换为 array: (num_labels,1)
        label_ids = np.array([idx for (idx, _) in labels_idx_examples])
        # explicitly add last dimension to label_ids to track correctly dynamic sequences
        # 增加 label/id feature = (3, 1)
        label_data.add_features(
            SUB_LABEL_KEY,
            LABEL_SUB_KEY,
            [FeatureArray(np.expand_dims(label_ids, -1), number_of_dimensions=2)],
            update_num_examples=False
        )

        label_data.add_lengths(SUB_LABEL, SEQUENCE_LENGTH, SUB_LABEL, SEQUENCE)

        return label_data

    def _add_sub_label_features(
            self,
            model_data: RasaModelData,
            training_data: List[Message],
            sub_label_attribute: Text,
            sub_label_id_dict: Dict[Text, int],
            training: bool = True,
    ) -> None:
        # 获取  label 对应的 ids
        sub_label_ids = []
        if training and self.component_config[SUB_INTENT_CLASSIFICATION]:
            for example in training_data:
                if example.get(sub_label_attribute):
                    sub_label_ids.append(sub_label_id_dict[example.get(sub_label_attribute)])

            # explicitly add last dimension to label_ids
            # to track correctly dynamic sequences
            model_data.add_features(
                SUB_LABEL_KEY,
                LABEL_SUB_KEY,
                [FeatureArray(np.expand_dims(sub_label_ids, -1), number_of_dimensions=2)],
            )

        if (
                sub_label_attribute
                and model_data.does_feature_not_exist(sub_label_attribute, SENTENCE)
                and model_data.does_feature_not_exist(sub_label_attribute, SEQUENCE)
        ):
            # no label features are present, get default features from _label_data
            model_data.add_features(
                SUB_LABEL, SENTENCE, self._use_default_sub_label_features(np.array(sub_label_ids))
            )

        # as label_attribute can have different values, e.g. INTENT or RESPONSE,
        # copy over the features to the LABEL key to make
        # it easier to access the label features inside the model itself
        model_data.update_key(sub_label_attribute, SENTENCE, LABEL, SENTENCE)
        model_data.update_key(sub_label_attribute, SEQUENCE, LABEL, SEQUENCE)
        model_data.update_key(sub_label_attribute, MASK, LABEL, MASK)

        model_data.add_lengths(SUB_LABEL, SEQUENCE_LENGTH, SUB_LABEL, SEQUENCE)

    def _use_default_sub_label_features(self, label_ids: np.ndarray) -> List[FeatureArray]:
        """
        默认的 all_sub_label_features 为 array：（num_labels,1,num_labels）
        """
        all_sub_label_features = self._label_data.get(SUB_LABEL, SENTENCE)[0]
        return [
            FeatureArray(
                np.array([all_sub_label_features[label_id] for label_id in label_ids]),
                number_of_dimensions=all_sub_label_features.number_of_dimensions,
            )
        ]

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        增加 sub_label 信息
        """
        import shutil

        if self.model is None:
            return {"file": None}

        model_dir = Path(model_dir)
        tf_model_file = model_dir / f"{file_name}.tf_model"

        io_utils.create_directory_for_file(tf_model_file)

        if self.component_config[CHECKPOINT_MODEL]:
            shutil.move(self.tmp_checkpoint_dir, model_dir / "checkpoints")
        self.model.save(str(tf_model_file))

        io_utils.pickle_dump(
            model_dir / f"{file_name}.data_example.pkl", self._data_example
        )
        io_utils.pickle_dump(
            model_dir / f"{file_name}.label_data.pkl", dict(self._label_data.data)
        )

        # io_utils.json_pickle(
        #     model_dir / f"{file_name}.index_label_id_mapping.json",
        #     self.index_label_id_mapping,
        # )

        dump_obj_as_json_to_file(
            model_dir / f"{file_name}.index_label_id_mapping.json",
            self.index_label_id_mapping,
        )

        if self.component_config.get(SUB_INTENT_CLASSIFICATION):
            # 增加 sub_label 信息
            # io_utils.json_pickle(
            #     model_dir / f"{file_name}.index_sub_label_id_mapping.json",
            #     self.index_sub_label_id_mapping,
            # )
            # 增加 意图和属性的映射关系
            # io_utils.json_pickle(model_dir / f"{file_name}.{LABEL_TO_SUB_LABEL_MAPPING}.json",
            #                      obj=self.label_to_sub_label_mapping)

            dump_obj_as_json_to_file(
                model_dir / f"{file_name}.index_sub_label_id_mapping.json",
                self.index_sub_label_id_mapping,
            )

            dump_obj_as_json_to_file(model_dir / f"{file_name}.{LABEL_TO_SUB_LABEL_MAPPING}.json",
                                 obj=self.label_to_sub_label_mapping)

        entity_tag_specs = (
            [tag_spec._asdict() for tag_spec in self._entity_tag_specs]
            if self._entity_tag_specs
            else []
        )
        dump_obj_as_json_to_file(filename=
                                 model_dir / f"{file_name}.entity_tag_specs.json", obj=entity_tag_specs
                                 )

        return {"file": file_name}

    @classmethod
    def load(
            cls,
            meta: Dict[Text, Any],
            model_dir: Text,
            model_metadata: Metadata = None,
            cached_component: Optional["DIETClassifier"] = None,
            should_finetune: bool = False,
            **kwargs: Any,
    ) -> "HierarchicalDIETClassifier":
        """Loads the trained model from the provided directory.
        重载 self.load() 方法
        """
        if not meta.get("file"):
            logger.debug(
                f"Failed to load model for '{cls.__name__}'. "
                f"Maybe you did not provide enough training data and no model was "
                f"trained or the path '{os.path.abspath(model_dir)}' doesn't exist?"
            )
            return cls(component_config=meta)

        # TODO: 新增 提取 index_sub_label_id_mapping
        (
            index_label_id_mapping,
            index_sub_label_id_mapping,
            label_to_sub_label_mapping,
            entity_tag_specs,
            label_data,
            meta,
            data_example,
        ) = cls._load_from_files(meta, model_dir)

        meta = train_utils.override_defaults(cls.defaults, meta)
        meta = train_utils.update_confidence_type(meta)
        meta = train_utils.update_similarity_type(meta)
        meta = train_utils.update_deprecated_loss_type(meta)

        # 加载 DIET 模型
        model = cls._load_model(
            entity_tag_specs,
            label_data,
            meta,
            data_example,
            model_dir,
            finetune_mode=should_finetune,
        )

        # 生成 HierarchicalDIETClassifier 对象
        return cls(
            component_config=meta,
            index_label_id_mapping=index_label_id_mapping,
            index_sub_label_id_mapping=index_sub_label_id_mapping,
            label_to_sub_label_mapping=label_to_sub_label_mapping,
            entity_tag_specs=entity_tag_specs,
            model=model,
            finetune_mode=should_finetune,
        )

    @classmethod
    def _load_from_files(
            cls, meta: Dict[Text, Any], model_dir: Text
    ) -> Tuple[
        Dict[int, Text],
        List[EntityTagSpec],
        RasaModelData,
        Dict[Text, Any],
        Dict[Text, Dict[Text, List[FeatureArray]]],
    ]:
        file_name = meta.get("file")

        model_dir = Path(model_dir)

        data_example = io_utils.pickle_load(model_dir / f"{file_name}.data_example.pkl")
        label_data = io_utils.pickle_load(model_dir / f"{file_name}.label_data.pkl")

        escape_check_data_examples = False


        index_sub_label_id_mapping = None
        intent_attribute_mapping = None

        if meta.get(SUB_INTENT_CLASSIFICATION):
            escape_check_data_examples = True
            # index_sub_label_id_mapping = io_utils.json_unpickle(
            #     model_dir / f"{file_name}.index_sub_label_id_mapping.json")
            # label_to_sub_label_mapping = io_utils.json_unpickle(
            #     model_dir / f"{file_name}.{LABEL_TO_SUB_LABEL_MAPPING}.json")
            index_sub_label_id_mapping = io_utils.read_json_file(
                model_dir / f"{file_name}.index_sub_label_id_mapping.json")
            label_to_sub_label_mapping = io_utils.read_json_file(
                model_dir / f"{file_name}.{LABEL_TO_SUB_LABEL_MAPPING}.json")

            index_sub_label_id_mapping = {int(key): value for key, value in index_sub_label_id_mapping.items()}

        label_data = RasaModelData(data=label_data, escape_check_data_examples= escape_check_data_examples)
        # index_label_id_mapping = io_utils.json_unpickle(model_dir / f"{file_name}.index_label_id_mapping.json")
        index_label_id_mapping = io_utils.read_json_file(model_dir / f"{file_name}.index_label_id_mapping.json")



        entity_tag_specs = io_utils.read_json_file(model_dir / f"{file_name}.entity_tag_specs.json")

        entity_tag_specs = [
            EntityTagSpec(
                tag_name=tag_spec["tag_name"],
                ids_to_tags={
                    int(key): value for key, value in tag_spec["ids_to_tags"].items()
                },
                tags_to_ids={
                    key: int(value) for key, value in tag_spec["tags_to_ids"].items()
                },
                num_tags=tag_spec["num_tags"],
            )
            for tag_spec in entity_tag_specs
        ]

        # jsonpickle converts dictionary keys to strings
        index_label_id_mapping = {
            int(key): value for key, value in index_label_id_mapping.items()
        }

        return (
            index_label_id_mapping,
            index_sub_label_id_mapping,
            label_to_sub_label_mapping,
            entity_tag_specs,
            label_data,
            meta,
            data_example,
        )

    def process(self, message: Message, **kwargs: Any) -> None:
        """Augments the message with intents, entities, and diagnostic data."""
        out = self._predict(message)

        if self.component_config[INTENT_CLASSIFICATION]:
            label, label_ranking = self._predict_label(out)

            message.set(INTENT, label, add_to_output=True)
            message.set("intent_ranking", label_ranking, add_to_output=True)

        if self.component_config[SUB_INTENT_CLASSIFICATION]:
            sub_label, sub_label_ranking = self._predict_sub_label(out,
                                                                   label=label,
                                                                   hard_hierarchical= self.hard_hierarchical)

            message.set(ATTRIBUTE, sub_label, add_to_output=True)
            message.set("attribute_ranking", sub_label_ranking, add_to_output=True)

        if self.component_config[ENTITY_RECOGNITION]:
            entities = self._predict_entities(out, message)

            message.set(ENTITIES, entities, add_to_output=True)

        if out and DIAGNOSTIC_DATA in out:
            message.add_diagnostic_data(self.unique_name, out.get(DIAGNOSTIC_DATA))

    def _predict_sub_label(
            self, predict_out: Optional[Dict[Text, tf.Tensor]],
            label: Dict[Text, Any] = None, hard_hierarchical=True
    ) -> Tuple[Dict[Text, Any], List[Dict[Text, Any]]]:
        """Predicts the intent of the provided message.
        argus:
        predict_out:
        label: label 预测的信息
        """
        sub_label: Dict[Text, Any] = {"name": None, "id": None, "confidence": 0.0}
        sub_label_ranking = []

        if predict_out is None:
            return sub_label, sub_label_ranking

        # 获取 message 与 sub_label 的 score
        # message 与 label 的相似度 shape = (batch_size, num_labels)
        message_sim = predict_out["s_scores"]

        if hard_hierarchical and label:
            # 使用 label 和 sub_label的层级关系进行输出
            # 计算 label 对应的 sub_labels
            sub_labels = self.label_to_sub_label_mapping[label['name']]
            # 获取 sub_labels 对应的id
            sub_label_indexes = [self.sub_label_id_index_mapping[sub_label] for sub_label in sub_labels]
            # 获取 sub_label_indexes_mask
            sub_label_indexes_mask = np.ones(shape=(self.index_sub_label_id_mapping.__len__()), dtype=np.float32)
            sub_label_indexes_mask[sub_label_indexes] = 0
            # 沿着 batch 方向复制 batch_size
            batch_size = message_sim.shape[0]
            sub_label_indexes_mask = np.expand_dims(sub_label_indexes_mask, axis=0)
            sub_label_indexes_mask.repeat(repeats=batch_size, axis=0)

            # 生成 masked_message_sim
            masked_message_sim = sub_label_indexes_mask * -1e9 + message_sim
            # 重新计算 softmax
            message_sim = tf.math.softmax(masked_message_sim).numpy()

        message_sim = message_sim.flatten()  # sim is a matrix
        # 按照 similarity 从大大小排序后获得 index
        label_ids = message_sim.argsort()[::-1]

        if (
                self.component_config[RANKING_LENGTH] > 0
                and self.component_config[MODEL_CONFIDENCE] == SOFTMAX
        ):
            # TODO: This should be removed in 3.0 when softmax as
            #  model confidence and normalization is completely deprecated.
            message_sim = train_utils.normalize(
                message_sim, self.component_config[RANKING_LENGTH]
            )
        message_sim[::-1].sort()
        message_sim = message_sim.tolist()

        # if X contains all zeros do not predict some label
        if label_ids.size > 0:
            label = {
                "id": hash(self.index_sub_label_id_mapping[label_ids[0]]),
                "name": self.index_sub_label_id_mapping[label_ids[0]],
                "confidence": message_sim[0],
            }

            if (
                    self.component_config[RANKING_LENGTH]
                    and 0 < self.component_config[RANKING_LENGTH] < LABEL_RANKING_LENGTH
            ):
                output_length = self.component_config[RANKING_LENGTH]
            else:
                output_length = LABEL_RANKING_LENGTH

            ranking = list(zip(list(label_ids), message_sim))
            ranking = ranking[:output_length]
            label_ranking = [
                {
                    "id": hash(self.index_sub_label_id_mapping[label_idx]),
                    "name": self.index_sub_label_id_mapping[label_idx],
                    "confidence": score,
                }
                for label_idx, score in ranking
            ]

        return label, label_ranking


class HierarchicalDIET(DIET):
    def __init__(self, *args, **kwargs):
        super(HierarchicalDIET, self).__init__(*args, **kwargs)

        if self.config[SUB_INTENT_CLASSIFICATION]:
            # 增加 sub_intent_classification layer
            self._prepare_sub_label_layers()

            # 在原有的 metric 增加  sub_intent_classification metric
            # self._create_metrics()
            self._create_sub_intent_metrics()
            self._update_sub_intent_metrics_to_log()

            # needed for efficient prediction
            self.all_sub_labels_embed: Optional[tf.Tensor] = None

        # 在原来 loss 基础上 增加 sub_intent_classification loss
        # self._batch_loss_intent()

        # 在原来的 batch_predict 增加对于 sub_intent 的 predict
        # 可以保持 self.batch_predict()，在 _batch_predict_intents 中增加 sub_intent 的预测
        # self._batch_predict_intents()

    def _prepare_sub_label_layers(self):
        """
        准备 sub_label_layers
        """

        self.sub_label_name = SUB_LABEL_TEXT  if self.config[SHARE_HIDDEN_LAYERS] else SUB_LABEL

        # disable input dropout applied to sparse and dense label features
        label_config = self.config.copy()
        label_config.update(
            {SPARSE_INPUT_DROPOUT: False, DENSE_INPUT_DROPOUT: False}
        )
        # label 方面的 训练层: feature_combining_layer.label + ffnn + classification_layer
        self._tf_layers[
            f"feature_combining_layer.{self.sub_label_name}"
        ] = rasa_layers.RasaFeatureCombiningLayer(
            self.sub_label_name, self.label_signature[self.sub_label_name], label_config
        )

        self._prepare_ffnn_layer(
            self.sub_label_name,
            self.config[HIDDEN_LAYERS_SIZES][self.sub_label_name],
            self.config[DROP_RATE],
        )

        self._prepare_sub_label_classification_layer(predictor_attribute= SUB_LABEL_TEXT)

    def _prepare_sub_label_classification_layer(self,predictor_attribute:Text):
        """
        参考 ： self._prepare_label_classification_layers()
        在原来模型的基础上  self._prepare_layer() 上 增加 sub_intent_classification layer
        self._prepare_layers()
        """

        # layers: ['sequence_layer.text', 'feature_combining_layer.label', 'ffnn.label', 'embed.text', 'embed.label', 'loss.label']
        # 新增 embed.sub_label_text layer  来编码句子向量到 sub_label 空间用于匹配相似度  （vs embed.text）
        self._prepare_embed_layers(predictor_attribute)

        # 准备 子意图的 embed.sub_label layer :
        self._prepare_embed_layers(SUB_LABEL)

        # 准备 子意图的 loss layer,增加 model_confidence = HARD_HIERARCHICAL
        model_confidence = HARD_HIERARCHICAL if self.config[SUB_INTENT_CLASSIFICATION] else self.config[MODEL_CONFIDENCE]
        #
        self._prepare_sub_label_dot_product_loss(SUB_LABEL, self.config[SCALE_LOSS],model_confidence=model_confidence)


    def _prepare_sub_label_dot_product_loss(
        self, name: Text, scale_loss: bool, prefix: Text = "loss",model_confidence:Text=None
    ) -> None:
        """
        重在 _prepare_dot_product_loss
        """
        self._tf_layers[f"{prefix}.{name}"] = DotProductLoss(
            self.config[NUM_NEG],
            self.config[LOSS_TYPE],
            self.config[MAX_POS_SIM],
            self.config[MAX_NEG_SIM],
            self.config[USE_MAX_NEG_SIM],
            self.config[NEGATIVE_MARGIN_SCALE],
            scale_loss,
            similarity_type=self.config[SIMILARITY_TYPE],
            constrain_similarities=self.config[CONSTRAIN_SIMILARITIES],
            # sub_label model_confidence 设置为
            model_confidence= model_confidence,
        )



    def _create_sub_intent_metrics(self):
        # TODO:
        self.s_loss = tf.keras.metrics.Mean(name="s_loss")
        self.s_acc = tf.keras.metrics.Mean(name="s_acc")

    def _update_sub_intent_metrics_to_log(self) -> None:
        """
        重写 _update_metrics_to_log 新增 sub_intent metrics to metrics_to_log
        """
        debug_log_level = logging.getLogger(__name__).level == logging.DEBUG

        # 新增 sub_intent metrics
        # 根据 _get_metric_results() 方法：  metric.name in self.metrics_to_log
        self.metrics_to_log.append('s_acc')
        # if debug_log_level:
        self.metrics_to_log.append("s_loss")

        self._log_metric_info()

    def _log_metric_info(self) -> None:
        metric_name = {
            "t": "total",
            "i": "intent",
            "s": "subintent",
            "e": "entity",
            "m": "mask",
            "r": "role",
            "g": "group",
        }

        logger.debug("Following metrics will be logged during training: ")
        for metric in self.metrics_to_log:
            # metric= 's_loss'
            # parts = ['s','loss']
            # name = sub_intent loss
            parts = metric.split("_")
            name = f"{metric_name[parts[0]]} {parts[1]}"
            logger.debug(f"  {metric} ({name})")

    def _batch_loss_intent(
            self,
            combined_sequence_sentence_feature_lengths_text: tf.Tensor,
            text_transformed: tf.Tensor,
            tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
    ) -> tf.Tensor:
        """
        在原来 loss 基础上 增加 sub_intent_classification loss
        可以保留  self.batch_loss() ，只重写 self._batch_loss_intent() 中 增加 sub_intent_classification loss
        self.batch_loss()
        """
        # get sentence features vector for intent classification
        sentence_vector = self._last_token(
            text_transformed, combined_sequence_sentence_feature_lengths_text
        )

        sequence_feature_lengths_label = self._get_sequence_feature_lengths(
            tf_batch_data, LABEL
        )

        # 获取 batch 中的 label_id
        label_ids = tf_batch_data[LABEL_KEY][LABEL_SUB_KEY][0]

        # 获取 label features
        label = self._create_bow(
            tf_batch_data[LABEL][SEQUENCE],
            tf_batch_data[LABEL][SENTENCE],
            sequence_feature_lengths_label,
            self.label_name,
        )

        sub_label_ids = None
        sub_label_features = None
        if self.config[SUB_INTENT_CLASSIFICATION]:
            sub_label_ids = tf_batch_data[SUB_LABEL_KEY][LABEL_SUB_KEY][0]
            sub_label_features = self._create_bow(
                tf_batch_data[SUB_LABEL][SEQUENCE],
                tf_batch_data[SUB_LABEL][SENTENCE],
                sequence_feature_lengths_label,
                self.sub_label_name,
            )

        loss, acc, sub_label_loss, sub_label_acc = self._calculate_label_loss(sentence_vector, label, label_ids,
                                                                              sub_label_features,sub_label_ids, )
        self._update_label_metrics(loss, acc)

        # 获取   sub_label, sub_label_ids
        if self.config[SUB_INTENT_CLASSIFICATION]:
            self._update_sub_label_metrics(loss=sub_label_loss, acc=sub_label_acc)
            # 累加 label_loss 和 sub_label loss
            # Bug?! 之前使用 loss  = add_n([loss,sub_label_loss]) 是否存在梯度传递问题
            all_intent_loss = tf.math.add_n([loss, sub_label_loss])
        else:
            all_intent_loss = tf.identity(loss)

        return all_intent_loss

    def _calculate_label_loss(
            self, text_features: tf.Tensor, label_features: tf.Tensor, label_ids: tf.Tensor,
            sub_label_features: tf.Tensor = None, sub_label_ids: tf.Tensor = None,
    ) -> tf.Tensor:
        """
        重写 _calculate_label_loss，增加 sub_label loss 和 acc 计算
        """
        # all_label_ids = [11, 1], all_labels_embed = [11, 20]
        all_label_ids, all_labels_embed = self._create_all_labels()
        # 计算  text_embed 和  label_embed ，后面需要计算其相似度
        text_embed = self._tf_layers[f"embed.{TEXT}"](text_features)
        # label_features 是 根据 all_label 转换过来的
        label_embed = self._tf_layers[f"embed.{LABEL}"](label_features)

        label_loss, label_acc = self._tf_layers[f"loss.{LABEL}"](
            text_embed, label_embed, label_ids, all_labels_embed, all_label_ids
        )
        # 新增 计算  sub_label loss 和 acc
        sub_label_loss, sub_label_acc = None, None
        if self.config[SUB_INTENT_CLASSIFICATION]:
            # all_sub_label_ids =[39, 1] , all_sub_labels_embed = [39, 20]
            all_sub_label_ids, all_sub_labels_embed = self._create_all_sub_labels()
            # 计算  sub_label_text_embed 和  label_embed ，后面需要计算其相似度
            sub_label_text_embed = self._tf_layers[f"embed.{SUB_LABEL_TEXT}"](text_features)

            # 训练的时候，使用 embed.sub_label 来对 sub_label_features 进行编码到同一个向量空间
            sub_label_embed = self._tf_layers[f"embed.{SUB_LABEL}"](sub_label_features)
            # 使用 _tf_layers["loss.sub_label"] layer（使用 _prepare_dot_product_loss 来定义的层） 来计算 sub intent 的 loss
            sub_label_loss, sub_label_acc = self._tf_layers[f"loss.{SUB_LABEL}"](
                sub_label_text_embed, sub_label_embed, sub_label_ids, all_sub_labels_embed, all_sub_label_ids
            )

        return label_loss, label_acc, sub_label_loss, sub_label_acc

    def _create_all_sub_labels(self):
        """
        创建 label的 embedding
        """
        # 获取所有标签的 ids
        all_sub_label_ids = self.tf_label_data[SUB_LABEL_KEY][LABEL_SUB_KEY][0]

        sequence_feature_lengths = self._get_sequence_feature_lengths(
            self.tf_label_data, SUB_LABEL
        )
        # TODO: _create_bow() 使用 的 feature_combining_layer.label layer 和 ffnn.label layer 来进行 embedding，不是很合理
        # 但是因为这两层默认都不对 label 做变换因而没有报错
        # x.shape = [39,39]
        x = self._create_bow(
            self.tf_label_data[SUB_LABEL][SEQUENCE],
            self.tf_label_data[SUB_LABEL][SENTENCE],
            sequence_feature_lengths,
            self.sub_label_name,
        )
        # 对所有的 x 做 embdding : shape = [39,20]
        all_sub_labels_embed = self._tf_layers[f"embed.{SUB_LABEL}"](x)

        return all_sub_label_ids, all_sub_labels_embed

    def _update_sub_label_metrics(self, loss: tf.Tensor, acc: tf.Tensor) -> None:
        """
        用于在 self.batch_loss 中更新 metric
        """
        self.s_loss.update_state(loss)
        self.s_acc.update_state(acc)

    def prepare_for_predict(self) -> None:
        """Prepares the model for prediction.
        构建 label 或者 sub label的 embedding
        """
        if self.config[INTENT_CLASSIFICATION]:
            _, self.all_labels_embed = self._create_all_labels()
        if self.config[SUB_INTENT_CLASSIFICATION]:
            _, self.all_sub_labels_embed = self._create_all_sub_labels()

    def _batch_predict_intents(
            self,
            combined_sequence_sentence_feature_lengths: tf.Tensor,
            text_transformed: tf.Tensor,
    ) -> Dict[Text, tf.Tensor]:
        """
        重载 _batch_predict_intents, ,新增 对 sub_label 的预测
        classifier.process() -> classifier._predict() -> Model.run_inference() ->
        RasaModel._rasa_predict() -> predict_step() -> batch_predict()
        """

        if self.all_labels_embed is None:
            raise ValueError(
                "The model was not prepared for prediction. "
                "Call `prepare_for_predict` first."
            )

        # get sentence feature vector for intent classification
        sentence_vector = self._last_token(
            text_transformed, combined_sequence_sentence_feature_lengths
        )

        # 使用 train的时候定义的 embed.text layer 对 sentence_vector 进行编码
        sentence_vector_embed = self._tf_layers[f"embed.{TEXT}"](sentence_vector)

        # 计算 confidence = (batch_size,label_num)
        # sentence_vector_embed = (batch_size, embedding_size) -> [batch_size, 1, 20]
        # all_labels_embed = (label_num, embedding_size) ->  [1,num_labels, 20]
        _, scores = self._tf_layers[f"loss.{LABEL}"].similarity_confidence_from_embeddings(
            sentence_vector_embed[:, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, :, :],
        )

        # TODO : 使用 sentence_vector_embed 和 all_sub_labels_embed 计算 sub_label_scores
        if self.config[SUB_INTENT_CLASSIFICATION] :
            # 使用 embed.SUB_LABEL_TEXT 来编码 sentence_vector 到 sub_label 空间
            sub_label_sentence_vector_embed = self._tf_layers[f"embed.{SUB_LABEL_TEXT}"](sentence_vector)
            _, sub_label_scores = self._tf_layers[f"loss.{SUB_LABEL}"].similarity_confidence_from_embeddings(
                sub_label_sentence_vector_embed[:, tf.newaxis, :],
                self.all_sub_labels_embed[tf.newaxis, :, :],
            )
        # TODO: 根据层次结构，强制 对不匹配的 sub_intent_score mask 后归零

        return {"i_scores": scores, "s_scores": sub_label_scores}