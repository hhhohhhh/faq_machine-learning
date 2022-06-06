#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/6/27 10:36 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/27 10:36   wangfc      1.0         None

learning rate schedule:  learning rate annealing or adaptive learning rates.

With learning rate decay, the learning rate is calculated each iterations  as follows:
Current learning_rate = initial_learning_rate * (1 / (1 + decay * iterations))



Module: tf.keras.optimizers.schedules
Classes
class LearningRateSchedule: A serializable learning rate decay schedule.
class ExponentialDecay: A LearningRateSchedule that uses an exponential decay schedule.
class InverseTimeDecay: A LearningRateSchedule that uses an inverse time decay schedule.
class PiecewiseConstantDecay: A LearningRateSchedule that uses a piecewise constant decay schedule.
class PolynomialDecay: A LearningRateSchedule that uses a polynomial decay schedule.
"""
from typing import Callable, List, Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
from tensorflow.python.types.core import Tensor
import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

# 示范通过Callback子类化编写回调函数（LearningRateScheduler的源代码）



if __name__ == '__main__':


    ipt = Input((12,))
    out = Dense(12)(ipt)
    model = Model(ipt, out)
    model.compile(SGD(1e-4, decay=1e-2), loss='mse')

    x = y = np.random.randn(32, 12)  # dummy data
    for iteration in range(10):
        model.train_on_batch(x, y)
        print("lr at iteration {}: {}".format(
            iteration + 1, model.optimizer._decayed_lr('float32').numpy()))