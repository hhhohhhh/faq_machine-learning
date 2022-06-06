#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/19 13:54 

rasa/utils/train_utils
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/19 13:54   wangfc      1.0         None
"""
import copy
from pathlib import Path
from typing import Optional, Dict, Text, Any, Union, List, Tuple
from typing import Counter
import math

import numpy as np
import logging

from utils.bilou_utils import ensure_consistent_bilou_tagging
from utils.constants import SPLIT_ENTITIES_BY_COMMA, NUMBER_OF_SUB_TOKENS
from utils.io import raise_deprecation_warning, raise_warning

from utils.tensorflow.constants import  *
from utils.exceptions import InvalidConfigException
logger = logging.getLogger(__name__)



def check_deprecated_options(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """Update the config according to changed config params.

    If old model configuration parameters are present in the provided config, replace
    them with the new parameters and log a warning.

    Args:
        config: model configuration

    Returns: updated model configuration
    """
    # note: call _replace_deprecated_option() here when there are options to deprecate

    return config


def override_defaults(
    defaults: Optional[Dict[Text, Any]], custom: Optional[Dict[Text, Any]]
) -> Dict[Text, Any]:
    """Override default config with the given config.

    We cannot use `dict.update` method because configs contain nested dicts.

    Args:
        defaults: default config
        custom: user config containing new parameters

    Returns:
        updated config
    """
    if defaults:
        config = copy.deepcopy(defaults)
    else:
        config = {}

    if custom:
        for key in custom.keys():
            if isinstance(config.get(key), dict):
                config[key].update(custom[key])
            else:
                config[key] = custom[key]

    return config






def validate_configuration_settings(component_config: Dict[Text, Any]) -> None:
    """Validates that combination of parameters in the configuration are correctly set.

    Args:
        component_config: Configuration to validate.
    """
    _check_loss_setting(component_config)
    _check_similarity_loss_setting(component_config)
    _check_confidence_setting(component_config)





def _check_loss_setting(component_config: Dict[Text, Any]) -> None:
    if not component_config[CONSTRAIN_SIMILARITIES] and component_config[LOSS_TYPE] in [
        SOFTMAX,
        CROSS_ENTROPY,
    ]:
        raise_warning(
            f"{CONSTRAIN_SIMILARITIES} is set to `False`. It is recommended "
            f"to set it to `True` when using cross-entropy loss. It will be set to "
            f"`True` by default, "
            f"Rasa Open Source 3.0.0 onwards.",
            category=UserWarning,
        )


def _check_similarity_loss_setting(component_config: Dict[Text, Any]) -> None:
    if (
        component_config[SIMILARITY_TYPE] == COSINE
        and component_config[LOSS_TYPE] == CROSS_ENTROPY
        or component_config[SIMILARITY_TYPE] == INNER
        and component_config[LOSS_TYPE] == MARGIN
    ):
        raise_warning(
            f"`{SIMILARITY_TYPE}={component_config[SIMILARITY_TYPE]}`"
            f" and `{LOSS_TYPE}={component_config[LOSS_TYPE]}` "
            f"is not a recommended setting as it may not lead to best results."
            f"Ideally use `{SIMILARITY_TYPE}={INNER}`"
            f" and `{LOSS_TYPE}={CROSS_ENTROPY}` or"
            f"`{SIMILARITY_TYPE}={COSINE}` and `{LOSS_TYPE}={MARGIN}`.",
            category=UserWarning,
        )


def _check_confidence_setting(component_config: Dict[Text, Any]) -> None:
    if component_config[MODEL_CONFIDENCE] == COSINE:
        # from rasa.shared.exceptions import InvalidConfigException
        raise InvalidConfigException(
            f"{MODEL_CONFIDENCE}={COSINE} was introduced in Rasa Open Source 2.3.0 "
            f"but post-release experiments revealed that using cosine similarity can "
            f"change the order of predicted labels. "
            f"Since this is not ideal, using `{MODEL_CONFIDENCE}={COSINE}` has been "
            f"removed in versions post `2.3.3`. "
            f"Please use either `{SOFTMAX}` or `{LINEAR_NORM}` as possible values."
        )
    if component_config[MODEL_CONFIDENCE] == INNER:

        raise InvalidConfigException(
            f"{MODEL_CONFIDENCE}={INNER} is deprecated as it produces an unbounded "
            f"range of confidences which can break the logic of assistants in various "
            f"other places. "
            f"Please use `{MODEL_CONFIDENCE}={LINEAR_NORM}` which will produce a "
            f"linearly normalized version of dot product similarities with each value "
            f"in the range `[0,1]`."
        )
    if component_config[MODEL_CONFIDENCE] not in [SOFTMAX, LINEAR_NORM, AUTO]:
        raise InvalidConfigException(
            f"{MODEL_CONFIDENCE}={component_config[MODEL_CONFIDENCE]} is not a valid "
            f"setting. Possible values: `{SOFTMAX}`, `{LINEAR_NORM}`."
        )
    if component_config[MODEL_CONFIDENCE] == SOFTMAX:
        raise_warning(
            f"{MODEL_CONFIDENCE} is set to `softmax`. It is recommended "
            f"to try using `{MODEL_CONFIDENCE}={LINEAR_NORM}` to make it easier to "
            f"tune fallback thresholds.",
            category=UserWarning,
        )
        if component_config[LOSS_TYPE] not in [SOFTMAX, CROSS_ENTROPY]:
            raise InvalidConfigException(
                f"{LOSS_TYPE}={component_config[LOSS_TYPE]} and "
                f"{MODEL_CONFIDENCE}={SOFTMAX} is not a valid "
                f"combination. You can use {MODEL_CONFIDENCE}={SOFTMAX} "
                f"only with {LOSS_TYPE}={CROSS_ENTROPY}."
            )
        if component_config[SIMILARITY_TYPE] not in [INNER, AUTO]:
            raise InvalidConfigException(
                f"{SIMILARITY_TYPE}={component_config[SIMILARITY_TYPE]} and "
                f"{MODEL_CONFIDENCE}={SOFTMAX} is not a valid "
                f"combination. You can use {MODEL_CONFIDENCE}={SOFTMAX} "
                f"only with {SIMILARITY_TYPE}={INNER}."
            )



def update_confidence_type(component_config: Dict[Text, Any]) -> Dict[Text, Any]:
    """Set model confidence to auto if margin loss is used.

    Option `auto` is reserved for margin loss type. It will be removed once margin loss
    is deprecated.

    Args:
        component_config: model configuration

    Returns:
        updated model configuration
    """
    if component_config[LOSS_TYPE] == MARGIN:
        raise_warning(
            f"Overriding defaults by setting {MODEL_CONFIDENCE} to "
            f"{AUTO} as {LOSS_TYPE} is set to {MARGIN} in the configuration. This means that "
            f"model's confidences will be computed as cosine similarities. "
            f"Users are encouraged to shift to cross entropy loss by setting `{LOSS_TYPE}={CROSS_ENTROPY}`."
        )
        component_config[MODEL_CONFIDENCE] = AUTO
    return component_config


def update_similarity_type(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """
    If SIMILARITY_TYPE is set to 'auto', update the SIMILARITY_TYPE depending
    on the LOSS_TYPE.
    Args:
        config: model configuration

    Returns: updated model configuration
    """
    if config.get(SIMILARITY_TYPE) == AUTO:
        if config[LOSS_TYPE] == CROSS_ENTROPY:
            config[SIMILARITY_TYPE] = INNER
        elif config[LOSS_TYPE] == MARGIN:
            config[SIMILARITY_TYPE] = COSINE

    return config


def update_deprecated_loss_type(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """Updates LOSS_TYPE to 'cross_entropy' if it is set to 'softmax'.

    Args:
        config: model configuration

    Returns:
        updated model configuration
    """
    if config.get(LOSS_TYPE) == SOFTMAX:
        raise_deprecation_warning(
            f"`{LOSS_TYPE}={SOFTMAX}` is deprecated. "
            f"Please update your configuration file to use"
            f"`{LOSS_TYPE}={CROSS_ENTROPY}` instead.",
            # warn_until_version=NEXT_MAJOR_VERSION_FOR_DEPRECATIONS,
        )
        config[LOSS_TYPE] = CROSS_ENTROPY

    return config


def update_deprecated_sparsity_to_density(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """Updates `WEIGHT_SPARSITY` to `CONNECTION_DENSITY = 1 - WEIGHT_SPARSITY`.

    Args:
        config: model configuration

    Returns:
        Updated model configuration
    """
    if WEIGHT_SPARSITY in config:
        raise_deprecation_warning(
            f"`{WEIGHT_SPARSITY}` is deprecated."
            f"Please update your configuration file to use"
            f"`{CONNECTION_DENSITY}` instead.",
            # warn_until_version=NEXT_MAJOR_VERSION_FOR_DEPRECATIONS,
            # docs=DOCS_URL_MIGRATION_GUIDE_WEIGHT_SPARSITY,
        )
        config[CONNECTION_DENSITY] = 1.0 - config[WEIGHT_SPARSITY]

    return config



def update_evaluation_parameters(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """
    If EVAL_NUM_EPOCHS is set to -1, evaluate at the end of the training.

    Args:
        config: model configuration

    Returns: updated model configuration
    """

    if config[EVAL_NUM_EPOCHS] == -1:
        config[EVAL_NUM_EPOCHS] = config[EPOCHS]
    elif config[EVAL_NUM_EPOCHS] < 1:
        raise ValueError(
            f"'{EVAL_NUM_EXAMPLES}' is set to "
            f"'{config[EVAL_NUM_EPOCHS]}'. "
            f"Only values > 1 are allowed for this configuration value."
        )

    return config



def entity_label_to_tags(
    model_predictions: Dict[Text, Any],
    entity_tag_specs: List["EntityTagSpec"],
    bilou_flag: bool = False,
    prediction_index: int = 0,
) -> Tuple[Dict[Text, List[Text]], Dict[Text, List[float]]]:
    """Convert the output predictions for entities to the actual entity tags.

    Args:
        model_predictions: the output predictions using the entity tag indices
        entity_tag_specs: the entity tag specifications
        bilou_flag: if 'True', the BILOU tagging schema was used
        prediction_index: the index in the batch of predictions
            to use for entity extraction

    Returns:
        A map of entity tag type, e.g. entity, role, group, to actual entity tags and
        confidences.
    """
    predicted_tags = {}
    confidence_values = {}

    for tag_spec in entity_tag_specs:
        predictions = model_predictions[f"e_{tag_spec.tag_name}_ids"]
        confidences = model_predictions[f"e_{tag_spec.tag_name}_scores"]

        if not np.any(predictions):
            continue

        confidences = [float(c) for c in confidences[prediction_index]]
        tags = [tag_spec.ids_to_tags[p] for p in predictions[prediction_index]]

        if bilou_flag:
            (
                tags,
                confidences,
            ) = ensure_consistent_bilou_tagging(
                tags, confidences
            )

        predicted_tags[tag_spec.tag_name] = tags
        confidence_values[tag_spec.tag_name] = confidences

    return predicted_tags, confidence_values



def init_split_entities(
    split_entities_config: Union[bool, Dict[Text, Any]], default_split_entity: bool
) -> Dict[Text, bool]:
    """Initialise the behaviour for splitting entities by comma (or not).

    Returns:
        Defines desired behaviour for splitting specific entity types and
        default behaviour for splitting any entity types for which no behaviour
        is defined.
    """
    if isinstance(split_entities_config, bool):
        # All entities will be split according to `split_entities_config`
        split_entities_config = {SPLIT_ENTITIES_BY_COMMA: split_entities_config}
    else:
        # All entities not named in split_entities_config will be split
        # according to `split_entities_config`
        split_entities_config[SPLIT_ENTITIES_BY_COMMA] = default_split_entity
    return split_entities_config


def align_token_features(
    list_of_tokens: List[List["Token"]],
    in_token_features: np.ndarray,
    shape: Optional[Tuple] = None,
) -> np.ndarray:
    """Align token features to match tokens.

    ConveRTTokenizer, LanguageModelTokenizers might split up tokens into sub-tokens.
    We need to take the mean of the sub-token vectors and take that as token vector.

    Args:
        list_of_tokens: tokens for examples
        in_token_features: token features from ConveRT
        shape: shape of feature matrix

    Returns:
        Token features.
    """
    if shape is None:
        shape = in_token_features.shape
    out_token_features = np.zeros(shape)

    for example_idx, example_tokens in enumerate(list_of_tokens):
        offset = 0
        for token_idx, token in enumerate(example_tokens):
            number_sub_words = token.get(NUMBER_OF_SUB_TOKENS, 1)

            if number_sub_words > 1:
                token_start_idx = token_idx + offset
                token_end_idx = token_idx + offset + number_sub_words
                # 对应 token 被 language model 分割为 sub token 的情况，取平均值
                mean_vec = np.mean(
                    in_token_features[example_idx][token_start_idx:token_end_idx],
                    axis=0,
                )

                offset += number_sub_words - 1

                out_token_features[example_idx][token_idx] = mean_vec
            else:
                out_token_features[example_idx][token_idx] = in_token_features[
                    example_idx
                ][token_idx + offset]

    return out_token_features


def normalize(values: np.ndarray, ranking_length: Optional[int] = 0) -> np.ndarray:
    """Normalizes an array of positive numbers over the top `ranking_length` values.

    Other values will be set to 0.
    """
    new_values = values.copy()  # prevent mutation of the input
    if 0 < ranking_length < len(new_values):
        ranked = sorted(new_values, reverse=True)
        new_values[new_values < ranked[ranking_length - 1]] = 0

    if np.sum(new_values) > 0:
        new_values = new_values / np.sum(new_values)

    return new_values


def create_common_callbacks(
    epochs: int,
    tensorboard_log_dir: Optional[Text] = None,
    tensorboard_log_level: Optional[Text] = None,
    checkpoint_dir: Optional[Path] = None,
    steps_per_epoch:int =None
) -> List["Callback"]:
    """Create common callbacks.

    The following callbacks are created:
    - RasaTrainingLogger callback
    - Optional TensorBoard callback
    - Optional RasaModelCheckpoint callback

    Args:
        epochs: the number of epochs to train
        tensorboard_log_dir: optional directory that should be used for tensorboard
        tensorboard_log_level: defines when training metrics for tensorboard should be
                               logged. Valid values: 'epoch' and 'batch'.
        checkpoint_dir: optional directory that should be used for model checkpointing

    Returns:
        A list of callbacks.
    """
    import tensorflow as tf

    # from rasa.utils.tensorflow.callback import RasaTrainingLogger
    # from rasa.utils.tensorflow.callback import RasaModelCheckpoint

    from utils.tensorflow.callback import RasaTrainingLogger
    from utils.tensorflow.callback import RasaModelCheckpoint

    callbacks = [RasaTrainingLogger(epochs, silent=False,steps_per_epoch=steps_per_epoch)]

    if tensorboard_log_dir:
        if tensorboard_log_level == "minibatch":
            tensorboard_log_level = "batch"
            raise_deprecation_warning(
                "You set 'tensorboard_log_level' to 'minibatch'. This value should not "
                "be used anymore. Please use 'batch' instead."
            )

        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=tensorboard_log_dir,
                update_freq=tensorboard_log_level,
                write_graph=True,
                write_images=True,
                histogram_freq=10,
            )
        )

    if checkpoint_dir:
        callbacks.append(RasaModelCheckpoint(checkpoint_dir))

    return callbacks



class ClassWeight():
    """
    from sklearn.utils import class_weight
    """

    def __init__(self, class_weight_strategy: Text,
                 label_counter: Union[Counter,Dict[int,int]] = None,
                 y_train: np.ndarray = None,
                 mu:float=0.15):
        self.class_weight_strategy = class_weight_strategy
        # 使用 sklearn.utils 的 class_weight
        self.y_train = y_train

        self.mu = mu  # parameter to tune
        self.label_counter = label_counter
        self.n_samples = np.sum([count for label, count in self.label_counter.items()])
        self.n_classes = self.label_counter.__len__()

    def get_class_weights(self) -> Dict[int,float]:
        """
        @time:  2022/1/8 20:50
        @author:wangfc
        @version:
        @description:
        balanced: 1/ count * total/n_classes
            每类标签的 weight 与其样本数量成反比，并且 所有 weights 之和等于 total

        log: log(mu * 1/count * total)
           对数格式的 weight

        @params:
        @return: 字典格式的  class_weights
        """

        class_weights = dict()
        if self.class_weight_strategy == 'balanced':
            """
            weight = n_samples / (n_classes * np.bincount(y))
            """
            # class_weights = class_weight.compute_class_weight(class_weight='balanced',
            #                                                   classes=np.unique(self.y_train),
            #                                                   y=self.y_train)
            default_weight =1.0
            for label, count in self.label_counter.items():
                if count > 0 :
                    weight = self.n_samples / float(self.n_classes * count)
                else:
                    weight = default_weight
                class_weights[label] = weight
        elif self.class_weight_strategy == 'log':
            default_weight = math.log(self.mu * self.n_classes)
            for label, count in self.label_counter:
                if count>0:
                    score = math.log(self.mu * self.n_samples / float(count))
                else:
                    score = default_weight
                class_weights[label] = score if score > 1.0 else 1.0
        else:
            class_weights = None

        logger.info(f"class_weight_strategy={self.class_weight_strategy},class_weights={class_weights}")
        return class_weights

