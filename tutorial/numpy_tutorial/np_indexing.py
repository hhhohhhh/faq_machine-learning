#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/25 9:29

https://numpy.org/doc/stable/user/basics.indexing.html
https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/AdvancedIndexing.html


@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/25 9:29   wangfc      1.0         None
"""
import numpy as  np
from pprint import pprint


def single_element_indexing():
    """
    基础索引：
    The array you get back when you index or slice a numpy array is a view of the original array.
    It is the same data, just accessed in a different order. If you change the view, you will change the corresponding elements in the original array.

    Single element indexing works exactly like that for other standard Python sequences.
    It is 0-based, and accepts negative indices for indexing from the end of the array.
    """
    size = (4, 5)
    x = np.arange(20)
    x.resize(size)
    pprint(x)
    # 选取元素
    # Notice the syntax - the i and j values are both inside the square brackets,
    # separated by a comma (the index is actually a tuple (2, 1), but tuple packing is used).
    y = x[1, 3]
    print(f'y.shape={y.shape},y=\n', y)
    is_share_memory = np.shares_memory(x, y)
    print(f'x and y shares_memory: {is_share_memory}')

    y = x[(1, 3)]
    print(f'y.shape={y.shape},y=\n', y)

    # 基本索引: 选取某行
    y = x[1]
    is_share_memory = np.shares_memory(x, y)
    print(f'y.shape={y.shape},shares_memory={is_share_memory}, y=\n', y)

    # 高级索引: 选取不同的行
    y = x[[1]]
    is_share_memory = np.shares_memory(x, y)
    print(f'y.shape={y.shape},shares_memory={is_share_memory}, y=\n', y)

    # 高级索引: 选取不同的行
    y = x[[1, 3]]
    is_share_memory = np.shares_memory(x, y)
    print(f'y.shape={y.shape},shares_memory={is_share_memory}, y=\n', y)

    # 选取不同的列
    y = x[:, [1, 3]]
    print(f'y.shape={y.shape},y=\n', y)


def integer_array_indexing_1d_array():
    """
    Advanced indexing always returns a copy of the data (contrast with basic slicing that returns a view).

    """
    # indexing a 1-d array with 2-d indexing
    x = np.arange(6)
    y = x[[1, 3]]
    is_share_memory = np.shares_memory(x, y)
    print(f'x.shape={x.shape},y.shape={y.shape},shares_memory={is_share_memory},y=\n', y)


    # utilizing a 2D-array as an index
    index_2d = np.array([[1, 2, 0],
                         [5, 5, 5],
                         [2, 3, 4]])
    #  # the resulting shape matches the shape of the indexing array
    y= x[index_2d]
    is_share_memory = np.shares_memory(x, y)
    print(f'x.shape={x.shape},y.shape={y.shape},shares_memory={is_share_memory},y=\n', y)

def integer_array_indexing_2d_array():
    """
    When the index consists of as many integer arrays as dimensions of the array being indexed,
    the indexing is straightforward, but different from slicing.
    Advanced indices always are broadcast and iterated as one:
    """
    size = (4, 5)
    x = np.arange(20)
    x.resize(size)
    # 高级索引:  使用索引 a tuple with at least one sequence object

    y = x[(1, 2, 3),]
    is_share_memory = np.shares_memory(x, y)
    print(f'x.shape={x.shape},y.shape={y.shape},shares_memory={is_share_memory}y=\n', y)

    # advanced indexing returns a copy
    # 报错,不能使用tuple :x[(1,2,3)],因为这个是 基础索引中 single_element_indexing，需要维度和x一致或者小于x.ndim

    # 高级索引：使用索引 an ndarray (of data type integer or bool)
    y = x[np.array([1, 2, 3])]
    print(f'y.shape={y.shape},y=\n', y)
    # 高级索引：使用 a non-tuple sequence object
    y = x[[1, 2, 3]]
    print(f'y.shape={y.shape},y=\n', y)
    # 报错: x[1, 2, 3] 使用基础索引选取值，index各个轴的数值必须在 x各个轴的范围内 一致
    # y = x[1, 2, 3]

    # 报错 ：
    # y = x[np.array([1, 2, 3]),np.array([2,3])]
    # 因为所选索引形状不匹配：索引数组无法与该形状一起广播。
    # 当访问numpy多维数组时，用于索引的数组需要具有相同的形状（行和列）。numpy将有可能去广播

    # 1. if the index arrays  have the same shape
    y = x[np.array([1, 2, 3]), np.array([2, 3, 4])]
    # np.array([1, 2, 3]) 和  np.array([2, 3,4]) 形状相同 ，获取 index 数据 (1,2),(2,3),(3,4)
    print(f'y.shape={y.shape},y=\n', y)

    # 2. if the index arrays do not have the same shape
    #  there is an attempt to broadcast them to the same shape. If they cannot be broadcast to the same shape, an exception is raised
    y = x[np.array([1, 2, 3]), np.array([3])]
    # np.array([1, 2, 3]) 和  np.array([3]) 形状不相同 ，进行 broadcast 获取数据 (1,3),(2,3),(3,3)
    print(f'y.shape={y.shape},y=\n', y)

    # The broadcasting mechanism permits index arrays to be combined with scalars for other indices.
    # The effect is that the scalar value is used for all the corresponding values of the index arrays:

    y = x[np.array([1, 2, 3]), 3]
    print(f'y.shape={y.shape},y=\n', y)

    # 实现不同维度的选择: 可分两步对数组进行选择。
    y = x[np.array([1, 2, 3]), :][:, np.array([2, 3])]
    print(f'y.shape={y.shape},y=\n', y)

    # 如何选取 corner
    # # Using integer index-arrays to produce a shape(2, 2) result
    # To use advanced indexing one needs to select all elements explicitly



    rows = np.array([[0, 0],
                     [3, 3]])
    columns = np.array([[0, 4],
                        [0, 4]])
    #  行索引是 [0,0] 和 [3,3]，而列索引是 [0,2] 和 [0,2]
    # 第一维
    # [x[0,0], x[0,4],
    #  x[3,0], x[3,4]]
    # rows 和 columns 形状相同，获得 index为
    # [rows[0], columns[0]],  [rows[0],column[1]]
    y = x[rows, columns]
    print(f'y.shape={y.shape},y=\n', y)

    # broadcasting can be used (compare operations such as rows[:, np.newaxis] + columns) to simplify this:
    rows = np.array([0, 3])
    columns = np.array([0, 4])
    rows_expanded = rows[:, np.newaxis]
    y = x[rows_expanded, columns]
    print(f'y.shape={y.shape},y=\n', y)

    # This broadcasting can also be achieved using the function ix_:
    index = np.ix_(rows, columns)
    y = x[index]
    print(f'y.shape={y.shape},y=\n', y)






def indexing_Nd_array():
    """
    in order to perform this variety of indexing on an N-dimensional array,
    we must specify  index-arrays; one for each dimension
    """
    size = (2, 3, 4)
    x = np.arange(24)
    x.resize(size)

    # specifies subsequent sheets to access
    ind0 = np.array([0, 1, 0])

    # specifies subsequent rows to access
    ind1 = np.array([0, 2, 1])

    # specifies subsequent columns to access
    ind2 = np.array([3, 3, 0])

    y = x[ind0, ind1, ind2]
    is_share_memory = np.shares_memory(x, y)
    print(f'x.shape={x.shape},y.shape={y.shape},shares_memory={is_share_memory}y=\n', y)

    # Using integer index-arrays to produce a shape(2, 2) result
    ind0 = np.array([1, 1, 0, 1]).reshape(2, 2)
    ind1 = np.array([1, 2, 0, 0]).reshape(2, 2)
    ind2 = np.array([1, 3, 1, 3]).reshape(2, 2)
    y = x[ind0, ind1, ind2]
    is_share_memory = np.shares_memory(x, y)
    print(f'x.shape={x.shape},y.shape={y.shape},shares_memory={is_share_memory}y=\n', y)



def broadcasting():
    size = (3, 2)
    x = np.arange(6)
    x.resize(size)

    ones = np.ones(shape=(1, 2))
    y = x + ones
    print(f'y.shape={y.shape},y=\n', y)


def main():
    indexing_Nd_array()


if __name__ == '__main__':
    main()
