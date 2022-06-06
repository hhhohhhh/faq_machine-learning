#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/12/6 17:34
tf.v1
from : https://cs230.stanford.edu/blog/datapipeline/


https://www.tensorflow.org/guide/data#basic_mechanics


@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/12/6 17:34   wangfc      1.0         None

tf.v1:
The one_shot_iterator method creates an iterator that will be able to iterate once over the dataset.
In other words, once we reach the end of the dataset, it will stop yielding elements and raise an Exception.

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
for element in iterator.get_next():
    print(element)

tf.v2
# 读取数据
tf.data.Dataset.from_tensors()
tf.data.Dataset.from_tensor_slices()
tf.data.TFRecordDataset().
tf.data.TextLineDataset()

# 转换数据:

tf_dataset.map( map_func )
tf_dataset.filter( func )
tf_dataset.reduce()
# reduce



dataset = dataset.map(lambda string: tf.string_split([string]).values)
"""
import numpy as np
import tensorflow as tf
from pprint import pprint



def build_dataset_from_memory(data):
    """
    tf.data.Dataset.from_tensors()
    tf.data.Dataset.from_tensor_slices()

    Dataset structure:
    The Dataset.element_spec property allows you to inspect the type of each element component.
    The property returns a nested structure of tf.TypeSpec objects, matching the structure of the element, which may be a single component, a tuple of components, or a nested tuple of components.


    """
    dataset = tf.data.Dataset.from_tensor_slices(data)

    # The Dataset object is a Python iterable. This makes it possible to consume its elements using a for loop:
    for e in dataset:
        pprint(e.numpy())

    # Or by explicitly creating a Python iterator using iter and consuming its elements using next
    for e in iter(dataset):
        pprint(e.numpy())

    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
    dataset1.element_spec
    # TensorSpec(shape=(10,), dtype=tf.float32, name=None)

    dataset2 = tf.data.Dataset.from_tensor_slices(
        (tf.random.uniform([4]),
         tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
    dataset2.element_spec
    # (TensorSpec(shape=(), dtype=tf.float32, name=None),
    #  TensorSpec(shape=(100,), dtype=tf.int32, name=None))


    dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
    dataset3.element_spec
    # (TensorSpec(shape=(10,), dtype=tf.float32, name=None),
    #  (TensorSpec(shape=(), dtype=tf.float32, name=None),
    #   TensorSpec(shape=(100,), dtype=tf.int32, name=None)))

    # Dataset containing a sparse tensor.
    dataset4 = tf.data.Dataset.from_tensors(
        tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))

    dataset4.element_spec
    # SparseTensorSpec(TensorShape([3, 4]), tf.int32)

    # Use value_type to see the type of value represented by the element spec
    dataset4.element_spec.value_type
    # tensorflow.python.framework.sparse_tensor.SparseTensor

    return dataset

def build_dataset_from_numpy():
    train, test = tf.keras.datasets.fashion_mnist.load_data()
    images, labels = train
    images = images / 255
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset

def build_dataset_from_generator():
    """
    https://www.tensorflow.org/guide/data#consuming_python_generators
    While this is a convienient approach it has limited portability and scalibility.
    It must run in the same python process that created the generator, and is still subject to the Python GIL.

    The output_types argument is required because tf.data builds a tf.Graph internally, and graph edges require a tf.dtype.
    """
    def count(stop):
        i = 0
        while i < stop:
            yield i
            i += 1

    ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes=(), )
    # repeat -> batch -> take


    for count_batch in ds_counter.repeat().batch(10).take(10):
        print(count_batch.numpy())

    def gen_series():
        i = 0
        while True:
            size = np.random.randint(0, 10)
            yield i, np.random.normal(size=(size,))
            i += 1

    ds_series = tf.data.Dataset.from_generator(
        gen_series,
        # The first output is an int32 the second is a float32.
        # The first item is a scalar, shape (), and the second is a vector of unknown length, shape (None,)
        output_types=(tf.int32, tf.float32),
        output_shapes=((), (None,)))

    # batch: stacks n consecutive elements of a dataset into a single element.
    # for each component i, all elements must have a tensor of the exact same shape.
    # padded_batch: when batching a dataset with a variable shape, you need to use Dataset.padded_batch
    # padded_shapes: 确定输出的 padded shape

    for count_batch in ds_series.padded_batch(10, padded_shapes=((),(20,))).take(10):
        print([x.numpy() for x in count_batch])





def build_dataset_from_file():
    dataset = tf.data.TextLineDataset("file.txt")
    iterator = dataset.as_numpy_iterator()
    for element in iterator:
        pprint(element)


def calculate_dataset_mean_and_std():
    data = [8, 3, 0, 8, 2, 1]
    dataset = build_dataset_from_memory(data)

    # 求和
    dataset = dataset.map(map_func=lambda x: tf.cast(x, tf.float32))
    dataset_sum  = dataset.reduce(tf.cast(0, tf.float32), lambda x, y: tf.reduce_sum([x, y]))
    dataset_size = dataset.reduce(tf.cast(0,tf.float32), lambda x,_: x+1)
    dataset_mean = dataset_sum/dataset_size

    dataset_variance_sum = dataset.map(lambda x: tf.square(x-dataset_mean)).reduce(tf.cast(0,tf.float32), lambda x,y:tf.reduce_sum([x,y]))
    dataset_variance = dataset_variance_sum/dataset_size

    dataset_size = 0
    dataset_sum = 0
    for batch in dataset.batch(batch_size=2):
        dataset_size += batch.shape[0]
        dataset_sum  +=tf.reduce_sum(batch)

    for batch in dataset.batch(batch_size=2):
        dataset_variance_sum+= tf.reduce_sum(tf.square(batch -  dataset_mean))



    # 使用 numpy 计算
    data_na = np.array(data,dtype=np.float32)
    np.std(data_na)
    np.var(data_na)

    dataset_variance_sum = dataset.reduce(tf.cast(0,tf.float32),lambda x,y:
    tf.reduce_sum(tf.square([x,y]-dataset_mean)))

    t = tf.convert_to_tensor(data)
    for e in t:
        pprint(e)

    tf.reduce_sum(t)
    tf.reduce_mean(tf.cast(t,tf.float64))
    tf.math.reduce_mean(dataset)


def main():
    build_dataset_from_generator()


if __name__ == '__main__':
    from utils.tensorflow.environment import setup_tf_environment
    setup_tf_environment(None)
    main()





