#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/28 18:15 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/28 18:15   wangfc      1.0         None
"""
from datetime import datetime

from utils.tensorflow.environment import setup_tf_environment

setup_tf_environment(gpu_memory_config=None)

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization


class SimpleModule(tf.Module):
    """
    https://www.tensorflow.org/guide/intro_to_modules

    Most models are made of layers. Layers are functions with a known mathematical structure
    that can be reused and have trainable variables.
    In TensorFlow, most high-level implementations of layers and models,
    such as Keras or Sonnet, are built on the same foundational class: tf.Module.
    """

    def __init__(self, name=None):
        super().__init__(name=name)
        self.a_variable = tf.Variable(5.0, name="train_me")
        self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")

    def __call__(self, x):
        return self.a_variable * x + self.non_trainable_variable


class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


class SequentialModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = Dense(in_features=3, out_features=3)
        self.dense_2 = Dense(in_features=3, out_features=2)

    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


class FlexibleDenseModule(tf.Module):
    # Note: No need for `in_features`
    def __init__(self, out_features, name=None):
        super().__init__(name=name)
        self.is_built = False
        self.out_features = out_features

    def __call__(self, x):
        # Create variables on first call.
        if not self.is_built:
            self.w = tf.Variable(
                tf.random.normal([x.shape[-1], self.out_features]), name='w')
            self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
            self.is_built = True

        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


class MySequentialModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = Dense(in_features=3, out_features=3)
        self.dense_2 = Dense(in_features=3, out_features=2)

    """
    adding the @tf.function decorator to indicate that this code should run as a graph.
    """

    @tf.function
    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


class MyDense(tf.keras.layers.Layer):
    # Adding **kwargs to support base Keras layer arguments
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(**kwargs)

        # This will soon move to the build step; see below
        self.w = tf.Variable(
            tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')

    def call(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


class FlexibleDense(tf.keras.layers.Layer):
    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features

    def build(self, input_shape):
        """
        # TODO: buid 何时被调用
        Create the state of the layer (weights)
        The build step :
        wait to create variables until you are sure of the input shape.
        """
        self.w = tf.Variable(
            tf.random.normal([input_shape[-1], self.out_features]), name='w')
        self.b = tf.Variable(tf.zeros([self.out_features]), name='b')

    def call(self, inputs):  # Defines the computation from inputs to outputs
        return tf.matmul(inputs, self.w) + self.b


class MySequentialKerasModel(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)

        self.dense_1 = FlexibleDense(out_features=3)
        self.dense_2 = FlexibleDense(out_features=2)

    def call(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


class Model_v1(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.unit = 10
        self.layer_norm = tf.keras.layers.LayerNormalization(name='my_layer_norm')
        self.dense = tf.keras.layers.Dense(self.unit)

    def call(self, inputs):
        x = inputs
        for index in range(2):
            with tf.compat.v1.variable_scope(name_or_scope='test_layer', reuse=True):
                x = self.dense(x)
                x = self.layer_norm(x)
        return x


class Model_Build_v2():
    def __init__(self):
        self.unit = 10

    def build_network(self):
        input = Input(shape=(10,), name="my_input")
        x = input
        for index in range(2):
            with tf.compat.v1.variable_scope(name_or_scope='test_layer', reuse=True):
                x = Dense(units=self.unit)(x)
                x = LayerNormalization()(x)
        return tf.keras.models.Model(inputs=input, outputs=x)


if __name__ == '__main__':
    # model_v1 = Model_v1()
    # inputs = Input(shape=(10,), name="input")
    # model_v1(inputs)
    # trainable_variables_v1 = model_v1.trainable_variables
    # tf_names_v1 = [tv.name for tv in trainable_variables_v1]
    # print(f"model_v1: {tf_names_v1}")
    #
    # model_build_v2 = Model_Build_v2()
    # model_v2 = model_build_v2.build_network()
    # trainable_variables_v2 = model_v2.trainable_variables
    # tf_names_v2 = [tv.name for tv in trainable_variables_v2]
    # print(f"model_v2: {tf_names_v2}")

    # tf.executing_eagerly()
    # simple_module = SimpleModule(name="simple")
    # simple_module(tf.constant(1.0))
    # # All trainable variables
    # print("trainable variables:", simple_module.trainable_variables)
    # # Every variable
    # print("all variables:", simple_module.variables)
    import os
    from utils.tensorflow.tensorboard import trace_graph_to_tensorboad

    # inputs = tf.constant([[2.0, 2.0, 2.0]])
    # log_dir = os.path.join('output', 'tensorboard', 'trace_funcs')
    # trace_graph_to_tensorboad(logdir=log_dir,
    #                           module=MySequentialModule,
    #                           inputs=inputs
    #                           )

    my_sequential_keras_model = MySequentialKerasModel()
    inputs = tf.constant([[2.0, 2.0, 2.0]])
    print("Model results:", my_sequential_keras_model(tf.constant([[2.0, 2.0, 2.0]])))
    output_model = os.path.join('output','model','my_sequential_keras_model')
    # tf.keras.models.save_model(model=my_sequential_keras_model,filepath=output_model)

    loaded_model = tf.keras.models.load_model(filepath = output_model)
    results = loaded_model(inputs)
    print(f"results ={results}")

