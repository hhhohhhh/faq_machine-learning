#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/17 15:52 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/17 15:52   wangfc      1.0         None
"""
import os
from multiprocessing import Process, Pool
import subprocess
from tests.pybasic_tests.test_asyncio import block_task


def test_fork():
    """
    Unix/Linux操作系统提供了一个fork()系统调用，它非常特殊。
    普通的函数调用，调用一次，返回一次，
    但是fork()调用一次，返回两次，因为操作系统自动把当前进程（称为父进程）复制了一份（称为子进程），
    然后，分别在父进程和子进程内返回。

    子进程永远返回 0，而父进程返回子进程的ID。
    这样做的理由是，一个父进程可以fork出很多子进程，所以，父进程要记下每个子进程的ID，
    而子进程只需要调用 getppid()就可以拿到父进程的ID

    有了fork调用，一个进程在接到新任务时就可以复制出一个子进程来处理新任务，
    常见的Apache服务器就是由父进程监听端口，每当有新的http请求时，就fork出子进程来处理新的http请求。

    """
    print('Process (%s) start...' % os.getpid())
    # Only works on Unix/Linux/Mac:
    pid = os.fork()
    if pid == 0:
        print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
    else:
        print('I (%s) just created a child process (%s).' % (os.getpid(), pid))


# 子进程要执行的代码
def run_proc(name):
    """
    multiprocessing模块提供了一个Process类来代表一个进程对象
    """
    print('Run child process %s (%s)...' % (name, os.getpid()))

def test_process():
    """
    创建子进程时，只需要传入一个执行函数和函数的参数，创建一个Process实例，
    用start()方法启动，这样创建进程比fork()还要简单。

    join()方法可以等待子进程结束后再继续往下运行，通常用于进程间的同步。
    """
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print('Child process will start.')
    p.start()
    p.join()
    print('Child process end.')


def test_pool():
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(block_task, args=(i,))
    print('Waiting for all subprocesses done...')
    # 对Pool对象调用join()方法会等待所有子进程执行完毕，调用join()之前必须先调用close()，
    # 调用close()之后就不能继续添加新的Process了。
    p.close()
    p.join()
    print('All subprocesses done.')

def test_subprocess():
    print('$ nslookup www.python.org')
    r = subprocess.call(['nslookup', 'www.python.org'])
    print('Exit code:', r)