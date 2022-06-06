#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file:
@version:
@desc:
@author: wangfc
@time: 2020/11/9 14:02

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/11/9 14:02   wangfc      1.0         None


"""
from utils.tensorflow.environment import setup_tf_environment
setup_tf_environment("0:1024")
import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest


def test_reduce_sum():
    # 默认数据类型 为 tf.int32
    a = tf.reshape(tf.range(6), (3, 2))
    assert a.dtype == tf.int32
    # warning: 如果 datatype = tf.int32,reduce_mean 会自动舍去小数部分
    assert tf.reduce_mean(a) == 2

    a_float = tf.cast(a, tf.float32)
    assert tf.reduce_mean(a_float) == 2.5

    print(f'a={a}')
    aa = np.array([[0, 1],
                   [2, 3],
                   [4, 5]], dtype=np.int32)
    assert np.mean(aa) == 2.5  # numpy.float64

    assert np.all(a == aa)
    # 按照不同的方式进行 reduce_sum
    assert tf.reduce_sum(a) == 15
    assert np.all(tf.reduce_sum(a, axis=1) == [1, 5, 9])
    assert np.all(tf.reduce_sum(a, axis=0) == [6, 9])
    assert np.all(tf.reduce_max(a, axis=1) == [1, 3, 5])


def test_tf_max():
    a = np.array([[0, 1],
                  [2, 3],
                  [4, 5]], dtype=np.int32)
    # tf.maximum : Returns the max of x and y (i.e. x > y ? x : y) element-wise
    assert tf.maximum(a, 3) == [[3, 3],
                                [3, 3],
                                [4, 5]]

    assert tf.reduce_max(a) == 5
    assert tf.reduce_sum(a,axis=-1) == (1,3,5)

    b = np.arange(24).reshape(2,3,4)
    print(b)
    tf.reduce_max(b,axis=(1,2),keepdims=True)


def test_reduce_mean():
    a = tf.reshape(tf.range(6), (3, 2))
    c = tf.reshape(tf.random.shuffle(tf.range(6)), shape=(3, 2))
    tf.reduce_mean(tf.cast(tf.math.equal(a, c), tf.float32))


# # 抽样
# x = tf.reshape(tf.range(32),(32,1))
# a= tf.reduce_sum(x, axis=1)
# print(a.shape)
# b= self._tf_layers['ffnn.label'](a)
# print(b.shape)
#
def test_tf_random():
    """
    对 每个 batch 的各个样本抽样对应的 num_neg 个负样本的 index
    """
    batch_size = 2
    maxval = 10
    num_neg = 3
    negative_indies = tf.random.uniform(shape=(batch_size,num_neg),maxval=maxval,dtype=tf.dtypes.int32)
    print(negative_indies)


def test_tf_tile():
    # 抽样
    population_size=3
    emb_size=5
    batch_size =2
    x = tf.reshape(tf.range(population_size*emb_size),(population_size,emb_size))
    print(x.shape)
    tiled = tf.tile(tf.expand_dims(x, 0), (batch_size, 1, 1))
    print(tiled.shape)

    neg_num=2
    idxs = np.random.randint(0,2,size=(batch_size,neg_num))
    tf.gather(tiled, idxs, batch_dims=1)

def test_tf_gather():
    (batch_size, sample_index)= 2,2
    # a = (batch_size, target_size, emb_size)
    a = tf.reshape(tf.range(24),(2,6,2))
    print(a)
    # b= (batch_size,sample_index)
    b = np.array([[0,0],
                 [0,1],
                  [1,1],
                  [0,1]])
    c = tf.gather(params=a,indices=b,batch_dims=1)
    print(c.shape)
    assert c.shape == (batch_size,sample_index)
    print(c)



def test_tf_slice():
    """
    https://www.tensorflow.org/guide/tensor_slicing
    """
    t1 = tf.constant([0, 1, 2, 3, 4, 5, 6, 7])

    print(tf.slice(t1,begin=[1],size=[3]))
    # tf.Tensor([1 2 3], shape=(3,), dtype=int32)

    t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                     [[3, 3, 3], [4, 4, 4]],
                     [[5, 5, 5], [6, 6, 6]]])
    t1 = tf.slice(t, [1, 0, 0], [1, 1, 3])

    print(t1)
    # tf.Tensor([[[3 3 3]]], shape=(1, 1, 3), dtype=int32)
    t2 = tf.slice(t,[0,0,0],[-1,-1,-1])




def test_tf_gather():

    """
    tf.gather :Use tf.gather to extract specific indices from a single axis of a tensor.
    tf.gather_nd:
    To extract slices from multiple axes of a tensor, use tf.gather_nd.
    This is useful when you want to gather the elements of a matrix as opposed to just its rows or columns.

    """
    t4 = tf.constant([[0, 5],
                      [1, 6],
                      [2, 7],
                      [3, 8],
                      [4, 9]])
    t = tf.gather_nd(t4,indices=[[2], [3], [0]])
    print(t)
    # tf.Tensor(
    # [[2 7]
    #  [3 8]
    #  [0 5]], shape=(3, 2), dtype=int32)

    t5 = np.reshape(np.arange(18), [2, 3, 3])

    print(tf.gather_nd(t5,
                       indices=[[0, 0, 0], [1, 2, 1]]))
    # tf.Tensor([ 0 16], shape=(2,), dtype=int64)

    # Return a list of two matrices
    print(tf.gather_nd(t5,
                       indices=[[[0, 0], [0, 2]], [[1, 0], [1, 2]]]))

    # Return one matrix
    print(tf.gather_nd(t5,
                       indices=[[0, 0], [0, 2], [1, 0], [1, 2]]))


def test_nest_flatten():
    """
    函数作用：将嵌套结构压平，返回Python的list。
    Tensor对象、和NumPy的ndarray对象，都看做是一个元素，不会被压平。

    注意和tf.contrib.layers.flatten()的区别
    """
    a = np.ones((2, 3))
    b = np.zeros((3, 2))
    inputs = {'a': a, 'b': b}
    result = nest.flatten(inputs)
    assert result == [a, b]


def test_tf_concat():
    batch_size = 4
    neg_num = 2
    a = np.ones(shape=(batch_size, 1))
    b = np.zeros((batch_size, neg_num))
    c = tf.concat([a, b], axis=-1)

    assert c.shape == (batch_size, 1 + neg_num)



def test_tf_scan():
    """
    从elems的第0维开始遍历elems，每次取出一个数据，与fn上次的输出一起再次送进fn，直到遍历完elems的所有第一维元素。
    （1）fn是一个函数，必须接收两个参数，这里记为fn(a,b)。
    （2）elems是一个数据体，可以是多维的、立体的、一维的
    （3）initializer，默认值为none，但是在实际使用中一般会传入值。
    """
    """
    
    第一次时，没有 initializer 传入，那么initializer=elems[0]=1，即a=initializer=1，并作为fn的第1次输出。
    第二次时，第一次的输出作为a，即a=1，elems[1]作为b，b=2，1+2=3作为第2次输出。
    第三次时，第二次的输出作为a，即a=3，elems[2]作为b，b=3, 3+3=6作为第3次输出。
    第四次时，第三次的输出作为a，即a=6，elems[3]作为b，b=4, 6+4=10作为第4次输出。
    第五次时，第四次的输出作为a，即a=10，elems[4]作为b，b=5, 10+5=15作为第5次输出。
    第六次时，第五次的输出作为a，即a=15，elems[4]作为b，b=6, 15+6=21作为第6次输出。
    最终的输出是所有输出组成的[ 1 3 6 10 15 21],类型是<class 'numpy.ndarray'>，和elems一样。
    
    elems:       1    2    3    4     5     6
                      |    |    |     |     |
    initializer: 1 -> 3 -> 6 -> 10 -> 15 -> 21
    """
    elems = np.array([1, 2, 3, 4, 5, 6])
    sum = tf.scan(lambda a, x: a + x, elems=elems,initializer=None)
    print(sum)

    sum = tf.scan(lambda a, x: a + x, elems, reverse=True)
