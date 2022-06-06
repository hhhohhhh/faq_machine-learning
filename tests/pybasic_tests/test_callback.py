#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/17 13:51 
What is a Callback Function?
A callback is a function that is passed as an argument to other function.
This other function is expected to call this callback function in its definition.
The point at which other function calls our callback function depends on the requirement and nature of other function.
Callback Functions are generally used with asynchronous functions.

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/17 13:51   wangfc      1.0         None
"""

try:
    from urllib.request import Request, urlopen  # Python 3
except ImportError:
    from urllib2 import Request, urlopen  # Python 2

import time

def callbackFunc(s):
    print('Length of the text file is : ', s)


def callbackFunc1(s):
    print('Callback Function 1: Length of the text file is : ', s)

def callbackFunc2(s):
    print('Callback Function 2: Length of the text file is : ', s)


def printFileLength(path, callback):
    """
    define a function named printFileLength() that takes file path and callback functions as arguments.
    printFileLength() reads the file, gets the length of the file and at the end makes a call to the callback function.
    """
    f = open(path, "r")
    length = len(f.read())
    f.close()
    callback(length)





def test_example1():
    """"""
    printFileLength("sample.txt", callbackFunc)

    printFileLength('sample.txt', callbackFunc1)
    printFileLength('sample.txt',callbackFunc2)


def http_blockway(url1, url2):
    """
    python的urllib是同步的，即当一个请求结束之后才能发起下一个请求，我们知道http请求基于tcp，
    tcp又需要三次握手建立连接（https的握手会更加复杂），
    在这个过程中，程序很多时候都在等待IO，CPU空闲，但是又不能做其他事情。
    但同步模式的优点是比较直观，符合人类的思维习惯: 那就是一件一件的来，干完一件事再开始下一件事。
    在同步模式下，要想发挥duo多喝CPU的威力，可以使用多进程或者多线程
    """
    begin = time.time()

    data1 = urlopen(url1).read()
    data2 = urlopen(url2).read()
    print(f'len(data1)={len(data1)}, len(data2) ={len(data2)}, \nhttp_blockway cost={time.time() - begin}')

def test_http_blockway():
    url_list = ['http://www.baidu.com', 'https://www.bing.com']
    http_blockway(*url_list)