#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
@file: 
@version: 
@desc:
@time: 2021/9/14 17:35

https://www.huaweicloud.com/articles/12603947.html

from tensorflow.python.util.tf_export import tf_export

@tf_export是一个修饰符。修饰符的本质是一个函数
tf_export的实现在 tensorflow/python/util/tf_export.py 中：


等号的右边的理解分两步：
  1.functools.partial
  2.api_export
  functools.partial是偏函数,它的本质简而言之是为函数固定某些参数。如：functools.partial(FuncA, p1)的作用是把函数FuncA的第一个参数固定为p1；
  又如functools.partial(FuncB, key1="Hello")的作用是把FuncB中的参数key1固定为“Hello"。
  functools.partial(api_export, api_name=TENSORFLOW_API_NAME)的意思是把api_export的api_name这个参数固定为TENSORFLOW_API。
  其中TENSORFLOW_API_NAME = 'tensorflow'。

  api_export是实现了__call__()函数的类

  tf_export= functools.partial(api_export, api_name=TENSORFLOW_API_NAME)
  写法等效于：
  funcC = api_export(api_name=TENSORFLOW_API_NAME)
  tf_export = funcC

  对于funcC = api_export(api_name=TENSORFLOW_API_NAME)，会导致__init__(api_name=TENSORFLOW_API_NAME)被调用:
  然后调用像函数一样调用funcC()实际上就会调用__call__()

  @tf_export("app.run")最终的结果是用上面这个__call__()来作为修饰器。



@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/14 17:35   wangfc      1.0         None
"""

from tensorflow.python.util.tf_export import tf_export, API_ATTRS, TENSORFLOW_API_NAME


@tf_export('test_export_api_name')
class TestDemo():
    def __init__(self,a):
        self.a =a

    def __call__(self, x):
        return 0


def test_tf_api_export():
    test_demo = TestDemo(a=1)
    print(test_demo.__name__)
    # 获取 api_names_attr api 名称的属性
    api_names_attr = API_ATTRS[TENSORFLOW_API_NAME].names
    # tf_api_names 是一个 tuple
    tf_api_names = test_demo.__getattribute__(api_names_attr)
    assert tf_api_names == ('test_export_api_name',)