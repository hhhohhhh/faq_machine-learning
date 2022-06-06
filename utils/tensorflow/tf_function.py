#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/6/25 11:59 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/25 11:59   wangfc      1.0         None


关于Tensorflow 2.0，最令我觉得有意思的功能就是 tf.function 和 AutoGraph
AutoGraph是TF提供的一个非常具有前景的工具, 它能够将一部分python语法的代码转译成高效的图表示代码.把 Python风格的代码转为效率更好的Tensorflow计算图
由于从TF 2.0开始, TF将会默认使用动态图(eager execution), 因此利用AutoGraph, 在理想情况下, 能让我们实现用动态图写(方便, 灵活), 用静态图跑(高效, 稳定).

在Tensorflow 2.0中，我们会自动的对被@tf.function装饰的函数进行AutoGraph优化．那么AutoGraph将会被自动调用, 从而将python函数转换成可执行的图表示.

我们来粗浅的看一下被tf.function装饰的函数第一次执行时都做了什么：
1. 函数被执行并且被跟踪(tracing)．Eager execution处于关闭状态，所有的Tensorflow函数被当做tf.Operation进行图的创建．
2. AutoGraph被唤醒，去检测Python代码可以转为Tensorflow的逻辑，比如while > tf.while, for > tf.while, if > tf.cond, assert > tf.assert．
3. 通过以上两部，我们对函数进行建图，为了保证Python代码中每一行的执行顺序，tf.control_dependencies被自动加入到代码中．保证第i行执行完后我们会执行第i+1行．
4. 返回tf.Graph，根据函数名和输入参数，创建唯一ID并将其与定义好的计算图相关联。计算图被缓存到一个映射表中：map [id] = graph. 我们将这个graph存到一个cache中．
对于任何一个该函数的调用，我们会重复利用cache中的计算图进行计算．


"""