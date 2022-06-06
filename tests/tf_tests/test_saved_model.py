#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/25 13:50 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/25 13:50   wangfc      1.0         None

Saved model :
作用 ：Models and layers can be loaded from this representation without actually making an instance of
      the Python class that created it.

组成： graph + weight
      graph:  operations, or ops, that implement the function

call:
    tf.saved_model.save()
        - saved_model.pb :
            a protocol buffer describing the functional tf.Graph
            stores the actual TensorFlow program, or model, and a set of named signatures,
            each identifying a function that accepts tensor inputs and produces tensor outputs.
        - assets:
            files used by the TensorFlow graph

        - variables :
            The variables directory contains a standard training checkpoint
            - variables.data-00000-of-00001
            - variables.index

"""

import os
import tempfile

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


class DemoSavedModelTask():
    """
    from  : https://www.tensorflow.org/guide/saved_model?hl=zh-tw#creating_a_savedmodel_from_keras
    """

    def __init__(self):
        # tmpdir = tempfile.mkdtemp()
        self.output_dir = 'output'
        self.mobilenet_save_path = os.path.join(self.output_dir, "mobilenet/1/")

    def prepare_data(self):
        img = self._get_image()
        x = self._process_img(img)
        return x

    def _get_image(self):
        file = tf.keras.utils.get_file(
            "grace_hopper.jpg",
            "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
        img = tf.keras.utils.load_img(file, target_size=[224, 224])
        plt.imshow(img)
        plt.axis('off')

    def _process_img(self, img):
        x = tf.keras.utils.img_to_array(img)
        x = tf.keras.applications.mobilenet.preprocess_input(
            x[tf.newaxis, ...])
        return x

    def _get_pretrained_model(self):
        pretrained_model = tf.keras.applications.MobileNet()
        return pretrained_model

    def _get_labels(self):
        labels_path = tf.keras.utils.get_file(
            'ImageNetLabels.txt',
            'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
        imagenet_labels = np.array(open(labels_path).read().splitlines())
        return imagenet_labels

    def save(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

        self.x = self.prepare_data()
        self.imagenet_labels = self._get_labels()
        self.pretrained_model = self._get_pretrained_model()

        result_before_save = self.pretrained_model(self.x)
        decoded = self.imagenet_labels[np.argsort(result_before_save)[0, ::-1][:5] + 1]

        print("Result before saving:\n", decoded)
        tf.saved_model.save(self.pretrained_model, self.mobilenet_save_path)

    def load_saved_model(self):
        loaded = tf.saved_model.load(self.mobilenet_save_path)

        print(list(loaded.signatures.keys()))  # ["serving_defaul

        infer = loaded.signatures["serving_default"]
        print(infer.structured_outputs)
        return infer

    def predict(self):
        labeling = self.infer(tf.constant(self.x))[self.pretrained_model.output_names[0]]

        decoded = self.imagenet_labels[np.argsort(labeling)[0, ::-1][:5] + 1]

        print("Result after saving and loading:\n", decoded)




class CustomModule(tf.Module):

    def __init__(self):
        super(CustomModule, self).__init__()
        self.v = tf.Variable(1.)

    @tf.function
    def __call__(self, x):
        print('Tracing with', x)
        return x * self.v

    @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
    def mutate(self, new_v):
        self.v.assign(new_v)




