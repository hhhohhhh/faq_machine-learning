#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/22 17:30 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/22 17:30   wangfc      1.0         None

https://docs.python.org/zh-cn/3.10/library/io.html#raw-i-o
https://www.liaoxuefeng.com/wiki/1016959663602400/1017606916795776

file-like Object:
    file-like Object不要求从特定类继承，只要写个read()方法就行。
    文本 IO : TextIOBase , TextIOWrapper, StringIO
        TextIOWrapper: 像open()函数返回的这种有个read()方法的对象
        StringIO:
            很多时候，数据读写不一定是文件，也可以在内存中读写。StringIO顾名思义就是在内存中读写str。
            在内存中创建的 file-like Object，常用作临时缓冲。

    二进制 IO:
        BytesIO: 内存中字节流,创建二进制流的最简单方法是使用 open()，并在模式字符串中指定 'b'
        BufferedReader:
        BufferedWriter:
        BufferedRandom:
        BufferedRWPair:

        网络流，自定义流

    原始 I/O:
        原始 I/O（也称为 非缓冲 I/O）通常用作二进制和文本流的低级构建块。用户代码直接操作原始流的用法非常罕见。
        不过，可以通过在禁用缓冲的情况下以二进制模式打开文件来创建原始流：

"""
import os
print(os.getcwd())

from io import StringIO
from io import BytesIO

def test_file_io():
    filename = 'test.csv'
    with open(file=filename) as f:
        print(f.__class__.name)
        f.read()


def test_string_io():
    """
    要把str写入StringIO，我们需要先创建一个StringIO，然后，像文件一样写入即可：
    StringIO操作的只能是str，如果要操作二进制数据，就需要使用BytesIO。
    """
    f = StringIO()
    f.write('hello')
    f.write(' ')
    f.write('world')
    print(f.getvalue())
    f.close()

def test_bytes_io():
    """
    BytesIO实现了在内存中读写bytes，我们创建一个BytesIO，然后写入一些bytes：
    """
    fb = BytesIO()
    bytes= '中文'.encode('utf-8')
    # 请注意，写入的不是str，而是经过UTF-8编码的bytes。
    fb.write(bytes)
    print(fb.getvalue())
    fb.close()

