#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:

from rasa.constant

@time: 2021/5/19 22:27 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/19 22:27   wangfc      1.0         None
"""


ENV_GPU_CONFIG = "TF_GPU_MEMORY_ALLOC"
ENV_CPU_INTER_OP_CONFIG = "TF_INTER_OP_PARALLELISM_THREADS"
ENV_CPU_INTRA_OP_CONFIG = "TF_INTRA_OP_PARALLELISM_THREADS"


CUDA_VISIBLE_DEVICES ="CUDA_VISIBLE_DEVICES"


BATCH_SIZES = "batch_size"
BATCH_STRATEGY = "batch_strategy"
EPOCHS = "epochs"
RANDOM_SEED = "random_seed"
LEARNING_RATE = "learning_rate"
# 新增参数，控制 eager模式
RUN_EAGERLY = "run_eagerly"


LABEL = "label"
#  新增 SUB_LABEL 用于识别 sub_intent 或者说 业务上说的 attribute
SUB_LABEL = 'sub_label'
IDS = "ids"
HIDDEN_LAYERS_SIZES = "hidden_layers_sizes"
SHARE_HIDDEN_LAYERS = "share_hidden_layers"

TRANSFORMER_SIZE = "transformer_size"
NUM_TRANSFORMER_LAYERS = "number_of_transformer_layers"
NUM_HEADS = "number_of_attention_heads"
UNIDIRECTIONAL_ENCODER = "unidirectional_encoder"
KEY_RELATIVE_ATTENTION = "use_key_relative_attention"
VALUE_RELATIVE_ATTENTION = "use_value_relative_attention"
MAX_RELATIVE_POSITION = "max_relative_position"


DENSE_DIMENSION = "dense_dimension"
CONCAT_DIMENSION = "concat_dimension"
EMBEDDING_DIMENSION = "embedding_dimension"
ENCODING_DIMENSION = "encoding_dimension"

SIMILARITY_TYPE = "similarity_type"
LOSS_TYPE = "loss_type"
NUM_NEG = "number_of_negative_examples"
MAX_POS_SIM = "maximum_positive_similarity"
MAX_NEG_SIM = "maximum_negative_similarity"
USE_MAX_NEG_SIM = "use_maximum_negative_similarity"

SCALE_LOSS = "scale_loss"
REGULARIZATION_CONSTANT = "regularization_constant"
NEGATIVE_MARGIN_SCALE = "negative_margin_scale"
DROP_RATE = "drop_rate"
DROP_RATE_ATTENTION = "drop_rate_attention"
DROP_RATE_DIALOGUE = "drop_rate_dialogue"
DROP_RATE_LABEL = "drop_rate_label"
CONSTRAIN_SIMILARITIES = "constrain_similarities"

WEIGHT_SPARSITY = "weight_sparsity"  # Deprecated and superseeded by CONNECTION_DENSITY
CONNECTION_DENSITY = "connection_density"

EVAL_NUM_EPOCHS = "evaluate_every_number_of_epochs"
EVAL_NUM_EXAMPLES = "evaluate_on_number_of_examples"

INTENT_CLASSIFICATION = "intent_classification"
# 新增子意图的识别
SUB_INTENT_CLASSIFICATION = "sub_intent_classification"
LABEL_TO_SUB_LABEL_MAPPING = "label_to_sub_label_mapping"
HARD_HIERARCHICAL = "hard_hierarchical"

ENTITY_RECOGNITION = "entity_recognition"
MASKED_LM = "use_masked_language_model"

SPARSE_INPUT_DROPOUT = "use_sparse_input_dropout"
DENSE_INPUT_DROPOUT = "use_dense_input_dropout"

RANKING_LENGTH = "ranking_length"
MODEL_CONFIDENCE = "model_confidence"

BILOU_FLAG = "BILOU_flag"

RETRIEVAL_INTENT = "retrieval_intent"

USE_TEXT_AS_LABEL = "use_text_as_label"

SOFTMAX = "softmax"
MARGIN = "margin"
AUTO = "auto"
INNER = "inner"
LINEAR_NORM = "linear_norm"
COSINE = "cosine"
CROSS_ENTROPY = "cross_entropy"

BALANCED = "balanced"

SEQUENCE = "sequence"
SEQUENCE_LENGTH = f"{SEQUENCE}_lengths"
SENTENCE = "sentence"

POOLING = "pooling"
MAX_POOLING = "max"
MEAN_POOLING = "mean"

TENSORBOARD_LOG_DIR = "tensorboard_log_directory"
TENSORBOARD_LOG_LEVEL = "tensorboard_log_level"

SEQUENCE_FEATURES = "sequence_features"
SENTENCE_FEATURES = "sentence_features"

FEATURIZERS = "featurizers"
CHECKPOINT_MODEL = "checkpoint_model"
CHECKPOINT_MODEL_DIR = "checkpoint_model_dir"
MASK = "mask"

