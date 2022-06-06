#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/3/16 16:50 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/16 16:50   wangfc      1.0         None

"""

from time import ctime
import threading

def coding(language):
    print('thread %s is running...' % threading.current_thread().name)
    for i in range(5):
        print('I\'m coding ',language, ' program at ', ctime() )

def music():
    print('thread %s is running...' % threading.current_thread().name)
    for i in range(5):
        print('I\'m listening music at ', ctime())


def run_multiple_thread():
    # 由于任何进程默认就会启动一个线程，我们把该线程称为主线程，主线程又可以启动新的线程，
    # 主线程实例的名字叫MainThread
    # Python的threading模块有个current_thread() 函数，它永远返回当前线程的实例。
    print('thread %s is running...' % threading.current_thread().name)

    thread_list = []
    # 创建 coding 线程和music 线程
    t1 = threading.Thread(target=coding, args=('Python',))
    t2 = threading.Thread(target=music)
    thread_list.append(t1)
    thread_list.append(t2)

    for t in thread_list:
        t.setDaemon(True)  # 设置为守护线程
        t.start()
        t.join()  # 在这个子线程完成运行之前，主线程将一直被阻塞

    print('thread %s ended.' % threading.current_thread().name)


if __name__ == '__main__':
    run_multiple_thread()