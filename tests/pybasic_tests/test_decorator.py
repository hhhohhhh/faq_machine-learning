#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在代码运行期间动态增加功能的方式，称之为“装饰器”（Decorator）
本质上，decorator就是一个返回函数的高阶函数。

装饰器的一大特性是: 接受一个函数作为参数，并返回一个函数。
第二大特性是，装饰器在加载模块时立即执行。



"""
import os
import functools
import time
from utils.time import get_current_time


def log(func):
    """
    1. 定义 装饰器

    2. 使用 该装饰器定义的函数： 借助Python的@语法，把decorator置于函数的定义处
    @log
    def func()

    3. 调用 该装饰器定义的函数的时候：接受一个函数作为参数，并返回一个函数
    func() -> log(func)
    相当于执行 log(func)，返回 wrapper 函数

    现在同名的now变量指向了新的函数，于是调用now()将执行新函数，即在log()函数中返回的wrapper()函数。
    但是经过decorator装饰之后的函数，它们的__name__已经从原来的'now'变成了'wrapper'

    @functools.wraps(func)
    保留函数原来的 __name__等属性


    :param func:
    :return:
    """
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        print("call {}".format(func.__name__))
        return func(*args,**kwargs)
    return wrapper


def log_with_parameter(text=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args,**kwargs):
            print("{} {}".format(text,func.__name__))
            return func(*args, **kwargs)
        return wrapper
    return decorator


def timeit(func):
    """
    @author:wangfc27441
    @desc:  对程序计时的装饰器
    @version：
    @time:2021/2/7 10:01

    Parameters
    ----------

    Returns
    -------
    """
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        start_time,current_time = get_current_time(with_time_stamp=True)
        print(f'pid:{os.getpid()} {func.__name__} 开始运行 at:{current_time}')
        result = func(*args,**kwargs)
        end_time,current_time = get_current_time(with_time_stamp=True)
        print(f'pid:{os.getpid()} {func.__name__} 结束运行 at:{current_time}，共耗时 {end_time- start_time} secs')
        return result
    return wrapper




@log
def now(params):
    print("2020-01-01")
    return params


@log_with_parameter('execute')
def now_with_parameter():
    print("2020-01-01")

@timeit
def run(delay=3):
    time.sleep(delay)
    print('run finished!')
    return delay



def test_decorator():
    # 相当于执行 log(now)
    result = now(1)
    print(now.__name__)


    now_with_parameter()
    now_with_parameter.__name__

    result = run(3)
