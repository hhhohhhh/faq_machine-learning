#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/28 16:55 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/28 16:55   wangfc      1.0         None
"""
import numpy as np


def basic_broadcasting():
    """
    To determine if two arrays are broadcast-compatible, align the entries of their shapes such that their trailing dimensions are aligned, and then check that each pair of aligned dimensions satisfy either of the following conditions:
        the aligned dimensions have the same size
        one of the dimensions has a size of 1

    The two arrays are broadcast-compatible if either of these conditions are satisfied for each pair of aligned dimensions.


    """
    # a shape-(3, 4) array
    shape_a = (4, 3)
    a = np.arange(12).reshape(shape_a)
    # broadcast-incompatible arrays
    # b = np.arange(4)
    b = np.arange(3)
    """
    array-1: 4 x 3
    array-2:     3
    result-shape: 4 x 3
    """
    y = a + b
    print(f"a.shape ={a.shape},b.shape={b.shape}, y.shape={y.shape}\n", y)

    # using np.broadcast_to()
    b_broadcasted = np.broadcast_to(b, shape_a)
    z = a + b_broadcasted
    print(f"a.shape ={a.shape},b.shape={b.shape}, y.shape={y.shape}\n", y)

    is_array_equal = np.array_equal(y, z)
    # np.equal(y,z).all()

    shape = (3, 1, 2)
    a = np.arange(6).reshape(shape)
    b_shape = (3, 1)
    b = np.arange(3).reshape(b_shape)
    """
    array-1:      3 x 1 x 2
    array-2:          3 x 1
    result-shape: 3 x 3 x 2
    """

    y = a + b
    print(f"a.shape ={a.shape},b.shape={b.shape}, y.shape={y.shape}\n", y)


def insert_size_1_dimension():
    shape = (1, 3, 1, 1)
    x = np.arange(3)
    a = x.reshape(shape)

    b = x[np.newaxis, :, np.newaxis, np.newaxis]
    np.array_equal(a, b)


def utilize_size_1_dimesion_for_broadcasting():
    """
     we want to multiply all possible pairs of entries between two arrays

    """
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6, 7])

    x = a[:, np.newaxis] * b[np.newaxis, :]

    y = a[:, np.newaxis] * b

    np.array_equal(x, y)


def normalize_data():
    """
    An RGB-image can thus be stored as a 3D NumPy array of shape- (V,H,3).
    V is the number of pixels along the vertical direction,
    H is the number of pixels along the horizontal,
    and the size-3 dimension stores the red, blue, and green color values for a given pixel.
     Thus a  array would be a 32x32 RGB image.

    Using the sequential function np.max and broadcasting, normalize images such that the largest value
    within each color-channel of each image is 1.
    """
    images = np.random.rand(500, 48, 48, 3)

    images_reshaped = images.reshape([-1, 3])
    max_value_in_channels = np.max(images_reshaped, axis=0)

    print(f"images_reshaped.shape ={images_reshaped.shape},max_value_in_channels.shape={max_value_in_channels.shape}")

    images_normalized = images / max_value_in_channels

    print(f"images[0][0][0]= {images[0][0][0]},images_normalized[0][0][0] ={images_normalized[0][0][0]}")


class PairWiseDistance():
    """
    https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/Broadcasting.html

    Suppose we have two, 2D arrays. x has a shape of (M,D) and y has a shape of (N,D).
    We want to compute the Euclidean distance (a.k.a. the L2-distance) between each pair of rows between the two arrays.


    Let’s proceed by performing this computation in three different ways:
        1. Using explicit for-loops
        2. Using straight-forward broadcasting
        3. Refactoring the problem and then using broadcasting

    """

    def pairwise_dists_looped(self, x, y):
        """  Computing pairwise distances using for-loops

         Parameters
         ----------
         x : numpy.ndarray, shape=(M, D)
         y : numpy.ndarray, shape=(N, D)

         Returns
         -------
         numpy.ndarray, shape=(M, N)
             The Euclidean distance between each pair of
             rows between `x` and `y`."""
        # `dists[i, j]` will store the Euclidean
        # distance between  `x[i]` and `y[j]`
        m = x.shape[0]
        d = x.shape[1]
        n = y.shape[0]
        assert x.shape[1] == y.shape[1]
        dists = np.empty((m, n))
        # 可以通过 enumerate 函数对 array 进行迭代
        for i,row_x in enumerate(x):
            for j,row_y in enumerate(y):
                distance = euclidean_distance(row_x,row_y)
                dists[i,j]=distance
        # 将 np.sqrt() 对整个矩阵进行计算
        dists_a = np.sqrt(dists)
        return dists_a

    def pairwise_dists_crude(self,x,y):
        """
        Computing pairwise distances using vectorization.

         This method uses memory-inefficient broadcasting.

         Parameters
         ----------
         x : numpy.ndarray, shape=(M, D)
         y : numpy.ndarray, shape=(N, D)

         Returns
         -------
         numpy.ndarray, shape=(M, N)
             The Euclidean distance between each pair of
             rows between `x` and `y`.

        x = (m,d)
        y = (n,d)
        如何进行 broadcasting ?
        x   -> (m,1,d)
        y   -> (1,n,d)
        x-y -> (m,n,d) :  不够节省内存空间
        (x-y)^2 -> (m,n,d)
        sum() -> (m,n)
        sqr() -> (m,n)
        """
        m = x.shape[0]
        d = x.shape[1]
        n = y.shape[0]

        # The use of `np.newaxis` here is equivalent to our use of the `reshape` function
        # x_expanded = x.reshape((m,1,d))
        # error: x_expanded = x[m,np.newaxis,d]
        x_expanded =  x[:,np.newaxis]

        # y_expanded = y.reshape((1, n, d))
        # error: y_expanded = y[n,np.newaxis,d]
        y_expanded = y[np.newaxis]
        # important: diffs[i, j] stores x[i] - y[j]
        diffs = x_expanded-y_expanded
        dists_b = np.sqrt(np.sum(np.square(diffs),axis=-1))
        # np.array_equal(dists_a,dists_b)
        return dists_b

    def pairwise_dists(self,x, y):
        """ Computing pairwise distances using memory-efficient
        vectorization.

        Parameters
        ----------
        x : numpy.ndarray, shape=(M, D)
        y : numpy.ndarray, shape=(N, D)

        Returns
        -------
        numpy.ndarray, shape=(M, N)
            The Euclidean distance between each pair of
            rows between `x` and `y`.

        分解欧几里得距离：
        euclidean_distance(a,b) = sum(a_i-b_i)^2 = sum(a_i^2+b_i^2- 2*a_i*b_i)
        前两项可以直接计算，后一项可以通过矩阵乘法计算得到 (m,n)
        x        = (m,d)
        y        = (n,d)
        x^2      = (m,d)
        sum(x^2) = (m,)
                -> (m,1)
        y^2      = (n,d)
        sum(y^2) = (n,)
                -> (1,n)
        sum      = (m,n)

        matmul(x,y)= (m,n)
        substract =(m,n)
        sqrt      =(m,n)

        """
        m = x.shape[0]
        d = x.shape[1]
        n = y.shape[0]
        x_squared = np.square(x)
        x_sum = np.sum(x_squared,axis=-1,keepdims=True)
        y_squared = np.square(y)
        y_sum = np.sum(y_squared,axis=-1,keepdims=False)
        y_sum = y_sum[np.newaxis]
        squred_sum = x_sum + y_sum

        cross_mat = np.matmul(x,y.T)
        # the strange behavior is that x_y_sqrd - x_y_prod can produce negative numbers

        # dists_c = np.sqrt(squred_sum - 2* cross_mat)
        cliped = np.clip(squred_sum - 2*cross_mat,a_min=0,a_max=None)
        dicts_c = np.sqrt(cliped)

        # np.allclose(dists_a,dists_c)
        return dicts_c





def euclidean_distance(a: np.ndarray,b:np.ndarray) :
    assert a.shape ==b.shape
    dis = np.sum(np.square(a-b))
    # 将 np.sqrt() 对整个矩阵进行计算，而不是放在两个向量元素的运算中
    return dis


if __name__ == '__main__':
    # a shape-(5, 3) array
    x = np.array([[8.54, 1.54, 8.12],
                  [3.13, 8.76, 5.29],
                  [7.73, 6.71, 1.31],
                  [6.44, 9.64, 8.44],
                  [7.27, 8.42, 5.27]])

    # a shape-(6, 3) array
    y = np.array([[8.65, 0.27, 4.67],
                  [7.73, 7.26, 1.95],
                  [1.27, 7.27, 3.59],
                  [4.05, 5.16, 3.53],
                  [4.77, 6.48, 8.01],
                  [7.85, 6.68, 6.13]])

    pair_wist_distance = PairWiseDistance()
    pair_wist_distance.pairwise_dists_looped(x=x,y=y)
