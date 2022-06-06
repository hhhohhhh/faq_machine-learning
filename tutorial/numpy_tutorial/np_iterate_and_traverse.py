#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file:
@version:
@desc:
@time: 2022/3/28 16:37

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/28 16:37   wangfc      1.0         None
"""
import numpy as np


def np_ndenumerate():
    size = (2, 3, 4)
    x = np.arange(24)
    x.resize(size)

    for i in np.ndenumerate(x):
        print(i)


def traverse_2d_array():
    """
    NumPy always defaults to row-major ordering whenever one of its functions involves array traversal.

    row-major ordering:  traverse the columns within a row, and then proceed to the next row.

    array([[0, 1, 2],
     [3, 4, 5]])
    or
    column-major ordering: traverse the rows within a given column , and then transition to the next column.
    array([[0, 2, 4],
     [1, 3, 5]])

    """
    size = (2, 3)
    x = np.arange(6)


def traverse_nd_array():
    """
    Row-major ordering (C ordering) {NumPy’s default}: traverse an array by advancing the index of the last axis, first,
    until the end of that axis is reached, and then advance the index of the second-to last axis, and so on.

    Column-major ordering (F ordering): traverse an array by advancing the index of the first axis, first,
    until the end of that axis is reached, and then advance the index of the second axis, and so on.
    """
