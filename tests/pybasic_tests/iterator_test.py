#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/3/7 11:06 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/7 11:06   wangfc      1.0         None

"""

from collections.abc import Iterable,Iterator


class RangeIterator(object):
    def __init__(self,stop_threshold,start_num=0):
        self.stop_threshold = stop_threshold
        self.start_num = start_num
        self.out_num = start_num

    def __iter__(self):
        """
        @author:wangfc
        @desc:
        判断对象是否可迭代
        原生函数iter(instance) 可以判断某个对象是否可迭代，它的工作流程大概分为以下3个步骤：
            检查对象instance是否实现了__iter__方法，并调用它获取返回的迭代器(iterator)。
            如果对象没有实现__iter__方法，但是实现了__getitem__方法，Python会生成一个迭代器。
            如果上述都失败，则编译器则抛出TypeError错误，‘xxx' Object is not iterable。

        该示例类中 实现了__iter__方法，通过iter()调用，返回 self(其自身，因为定义了 __next__() 函数所以是一个迭代器并返回
        因此该类实现了迭代器协议，因此可以通过for-in等方式对该对象进行迭代。

        第二条通常都是针对Python中的序列(sequence)而定义，例如list，为了实现sequence协议，需要实现__getitem__方法。

        @version：
        @time:2021/3/8 13:43

        Parameters
        ----------

        Returns
        -------
        """
        return self

    def __next__(self):
        """
        迭代器是个迭代值生产工厂，它保存迭代状态，并通过next()函数产生下一个迭代值。
        返回下一个可用的元素，如果无可用元素则抛出StopIteration异常
        """
        if self.out_num <self.stop_threshold:
            self.out_num = self.out_num +1
            return self.out_num -1
        else:
            raise StopIteration


class RangeGenerator(object):
    def __init__(self,stop_threshold,start_num=0):

        self.stop_threshold = stop_threshold
        self.start_num = start_num
        self.out_num = start_num

    def __iter__(self):
        """
        利用生成器代替可迭代中的__iter__迭代器
        因为yield存在于__iter__，因此__iter__变成了生成器函数，调用它测返回一个生成器，同时生成器又实现了迭代器协议，
        因此满足了可迭代的需求。

        迭代器通过next()不断产出下一个元素直到迭代器耗尽，
        而Python中的生成器可以理解为一个更优雅的迭代器(不需要实现__iter__和__next__方法)，实现了迭代器协议，它也可以通过next()产出元素。
        Python中的生成器主要分为两种类型：
            生成器函数(generator function)返回得到的生成器:  包含yield关键字的函数称为生成器函数
            生成器表达式(generator expression)返回得到的生成器

        """
        while self.out_num <self.stop_threshold:
            self.out_num = self.out_num +1
            yield self.out_num -1
        else:
            raise StopIteration


if __name__ == '__main__':
    ls = [1,2,3]
    iter_ls = iter(ls)
    isinstance(iter_ls,Iterator)

    range_iterator = RangeIterator(10)
    range_generator =RangeGenerator(10)
    isinstance(range_iterator,Iterator)
    isinstance(range_generator,Iterator)
    isinstance(range_generator,Iterable)
    # next(range_iterator)
    for generator  in range_generator:
        print(generator)
