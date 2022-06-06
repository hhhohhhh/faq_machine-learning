#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/10 9:35 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/10 9:35   wangfc      1.0         None
"""
import os
from typing import List, Text

import numpy as np
import tensorflow as tf

from data_process.dataset.datasets_utils import get_conll_data
from models import train_utils
from models.extractors.bilstm_crf_model import BilstmCrfModel
from models.extractors.extrator import EntityExtractor
from tasks.tf_task import BaseTfTask
from utils.hugging_face.hf_datasets import Conll2003Dataset, HfDataset, create_ner_data_generators
from utils.tensorflow.preprocessing_layers import create_lookup_layer


class BilstmCrfEntityExtrator(EntityExtractor, BaseTfTask):
    """
    继承 rasa EntityExtractor
    """

    def __init__(self,
                 dataset='conll2003',
                 vocab_size: int = 10000, tags: List[Text] = None, vocabulary: List[Text] = None,
                 *args, **kwargs):
        BaseTfTask.__init__(self, dataset=dataset, *args, **kwargs)
        self.vocab_size = vocab_size
        self.tags = tags
        self.vocabulary = vocabulary

        self.optimizer_name = 'adam'
        self.learning_rate = 1e-3

    def train(self, train_epochs=3, batch_size=64):
        """

        """
        dataset = self._prepare_data(training=True)

        train_data_generator, validation_data_generator = create_ner_data_generators(dataset=dataset,
                                                                                     batch_sizes=batch_size,
                                                                                     epochs=train_epochs)

        train_step_per_epoch = dataset._get_steps(data_type='train', batch_size=batch_size)
        train_steps = train_epochs * train_step_per_epoch

        self.model = BilstmCrfModel(tag_size=len(dataset.tags), vocabulary=dataset.lookup_layer.get_vocabulary())
        optimizer = self._build_optimizer(optimizer_name=self.optimizer_name, learning_rate=self.learning_rate,
                                          train_steps=train_steps, train_step_per_epoch=train_step_per_epoch)
        self.model.compile(optimizer=optimizer, run_eagerly=self.run_eagerly)

        callbacks = train_utils.create_common_callbacks(epochs=train_epochs,steps_per_epoch=train_step_per_epoch)
        self.model.fit(train_data_generator,
                       epochs=train_epochs,
                       validation_data=validation_data_generator,
                       validation_freq=None,
                       callbacks=callbacks,
                       verbose=False,
                       shuffle=False)

    def _load_dataset(self, dataset_name='conll2003', vocab_size=None) -> Conll2003Dataset:
        cache_dir = os.path.join('corpus', 'conll2003')
        dataset = Conll2003Dataset(dataset_name=dataset_name,
                                   cache_dir=cache_dir,
                                   vocab_size=vocab_size,
                                   debug = self.debug)
        return dataset

    def _prepare_data(self, training=True, ) -> Conll2003Dataset:
        dataset = self._load_dataset(vocab_size=self.vocab_size)
        return dataset
