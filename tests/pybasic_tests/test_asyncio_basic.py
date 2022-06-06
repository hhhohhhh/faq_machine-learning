#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@time: 2021/2/5 11:47

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/2/5 11:47   wangfc      1.0         None

"""
import random
import threading
import asyncio
import time
# from decorator import timeit
from typing import List

from utils.time import timeit, get_current_time

"""
阻塞的 ： 程序未得到所需计算资源时被挂起的状态。程序在等待某个操作完成期间，自身无法继续干别的事情，则称该程序在该操作上是阻塞的。
常见的阻塞形式有：网络I/O阻塞、磁盘I/O阻塞、用户输入阻塞等。

非阻塞的 ：程序在等待某操作过程中，自身不被阻塞，可以继续运行干别的事情，则称该程序在该操作上是非阻塞的。


0. 可等待对象
如果一个对象可以在 await 语句中使用，那么它就是 可等待 对象。许多 asyncio API 都被设计为接受可等待对象。
可等待 对象有三种主要类型: 协程, 任务 和 Future.
await后面的对象需要是一个 Awaitable，或者实现了相关的协议。

查看Awaitable抽象类的代码，表明了只要一个类实现了__await__方法，那么通过它构造出来的实例就是一个Awaitable：

class Awaitable(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def __await__(self):
        yield

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Awaitable:
            return _check_methods(C, "__await__")
        return NotImplemented
        
        
1. 协程 函数的定义
 A coroutine is a specialized version of a Python generator function. 
 Coroutine 类也继承了Awaitable，而且实现了send，throw和close方法。

class Coroutine(Awaitable):
    __slots__ = ()

    @abstractmethod
    def send(self, value):
        ...

    @abstractmethod
    def throw(self, typ, val=None, tb=None):
        ...

    def close(self):
        ...
        
    @classmethod
    def __subclasshook__(cls, C):
        if cls is Coroutine:
            return _check_methods(C, '__await__', 'send', 'throw', 'close')
        return NotImplemented
        

2. task对象和 Future 对象
task是Futures的子类

3. 事件循环 Eventloop

loop.run_until_complete 是一个阻塞方法，只有当它里面的协程运行结束以后这个方法才结束，才会运行之后的代码。




"""

@timeit
@asyncio.coroutine
def helloV1(secs=3):
    print(f"{get_current_time()}:Hello world!")
    # 异步调用asyncio.sleep(1):
    """
    @asyncio.coroutine
    Python 3.5 时的装饰器写法已经过时
    
    Python中 time.sleep 是阻塞的，都知道使用它要谨慎，但在多线程编程中，time.sleep 并不会阻塞其他线程。
    阻塞的 ： 程序未得到所需计算资源时被挂起的状态。
    asyncio.sleep 是非阻塞的 ：非阻塞就是在做一件事的时候，不阻碍调用它的程序做别的事情。
    """
    r = yield from asyncio.sleep(secs)
    print(f"{get_current_time()}:Hello again!")
    return r

def run_single_coro_with_event_loop():
    """
    In 3.5-3.6, your example is roughly equivalent to:
    import asyncio
    futures = [...]
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(futures))
    loop.close()

    Python 3.7 addition.
    最简单的运行 单个 协程对象 的形式
    asyncio.run
    """
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait([helloV1()]))
    loop.close()


def run_single_coro_with_asyncio_run():
    """
    Python 3.7 addition. 更加简单地 运行 coroutine的方式
    """

    asyncio.run(helloV1())


# 定义 coroutine
@timeit
async def helloV2(secs):
    """
    @author:wangfc27441
    @desc:
    coroutine：
        使用async修饰将 普通函数 和 生成器函数 包装成 协程 函数 (coroutine object)和 协程 生成器。
        协程函数: 定义形式为 async def 的函数;
        协程对象: 调用 协程函数 所返回的对象。
    task：
        任务 被用来“并行的”调度协程
        当一个协程通过 asyncio.create_task() 等函数被封装为一个 任务，该协程会被自动调度执行:
    @version：
    @time:2021/2/5 16:03

    # 异步调用 asyncio.sleep(1),由于 asyncio.sleep()也是一个coroutine，所以线程不会等待 asyncio.sleep()，
    # 而是直接中断并执行下一个消息循环。
    # return_secs= await asyncio.sleep(secs)

    await语法只能出现在通过async修饰的函数中,否则会报SyntaxError错误
    await后面的对象需要是一个Awaitable，或者实现了相关的协议。
    asyncio.sleep() is used to stand in for a non-blocking call (but one that also takes some time to complete).
    其返回为 None
    """
    await asyncio.sleep(secs)
    return secs



@timeit
def creat_single_task_with_loop(loop,secs=3,use_task=True):
    """
    @author:wangfc
    @desc:

    @version：
    @time:2021/3/16 10:47

    """
    # 生成 coroutine 对象
    coro = helloV2(secs)
    # 运行
    """
    接着说 Future，它代表了一个「未来」对象，异步操作结束后会把最终结果设置到这个 Future 对象上。Future 是对协程的封装，
    不过日常开发基本是不需要直接用这个底层 Future 类的。
    """
    if use_task:
        # 创建 任务 对象: loop.create_task()方法创建一个task对象, 将 task 加入的 running loop 中
        # 用来并发运行作为 asyncio 任务 的多个协程。
        print(f"loop with task")
        task = loop.create_task(coro)
    else:
        task = coro
    return task


@timeit
def run_single_task_with_event_loop(use_task=True):
    """
    asyncio.wait():  参数是协程的列表
    coroutine asyncio.wait(aws, *, loop=None, timeout=None, return_when=ALL_COMPLETED)
    并发地运行 aws 可迭代对象中的 可等待对象 并进入阻塞状态直到满足 return_when 所指定的条件。
    aws 可迭代对象必须不为空。
    返回两个 Task/Future 集合: (done, pending)
    """
    # 获取EventLoop:
    loop = asyncio.get_event_loop()
    # 创建 task
    task = creat_single_task_with_loop(loop=loop,secs=3,use_task=use_task)

    # 执行coroutine
    loop.run_until_complete(asyncio.wait([task]))



@timeit
def run_two_coro_with_event_loop(secs=3):
    """
    @author:wangfc
    @desc:
    @version：
    @time:2021/3/16 10:47

    """
    # 生成 coroutine 对象
    tasks = [helloV1(), helloV1()]
    # 生成 event loop
    loop = asyncio.get_event_loop()
    # 运行
    result = loop.run_until_complete(asyncio.wait(tasks))
    print(f"result={result} at {get_current_time()}")
    loop.close()
    return result





async def create_and_gather_multiple_tasks(delays:List[int]):
    """
    @author:wangfc27441
    @desc:
    awaitable asyncio.gather(*aws, loop=None, return_exceptions=False)
         将 多个 aws 对象 加入 event loop, 每次 aws 遇到 await ,函数会挂起将控制权交给 event loop
        并发 运行 aws 序列中的 可等待对象。
        如果 aws 中的某个可等待对象为协程，它将自动作为一个任务加入 日程。
        如果所有可等待对象都成功完成，结果将是一个由所有返回值聚合而成的列表。结果值的顺序与 aws 中可等待对象的顺序一致。

    asyncio.gather()函数来执行协程函数
    asyncio.wait()函数，它的参数是协程的列表。

    使用wait和gather有哪些区别呢？
    首先，gather是需要所有任务都执行结束，如果某一个协程函数崩溃了，则会抛异常，都不会有结果。
    wait可以定义函数返回的时机，可以是FIRST_COMPLETED(第一个结束的), FIRST_EXCEPTION(第一个出现异常的), ALL_COMPLETED(全部执行完，默认的)

    @version：
    @time:2021/2/5 17:11

    Parameters
    ----------

    Returns
    -------
    """
    coros = []
    for delay in delays:
        # 创建任务：返回一个task对象，此时task进入pending状态，并没有执行
        # task = asyncio.ensure_future(hello(secs=request))
        coro = helloV2(secs=delay)
        coros.append(coro)
    # 将 多个 aws 对象 加入 event loop, 每次 aws 遇到 await ,函数会挂起将控制权交给 event loop
    results = await asyncio.gather(*coros)
    return results


def run_gather_two_tasks_with_event_loop(delays=[3,5]):
    """
    @author:wangfc27441
    @desc:  使用协程 并发运行 for_loop
    @version：
    @time:2021/2/5 17:13

    Parameters
    ----------

    Returns
    -------
    """
    # 并发运行任务
    tasks = create_and_gather_multiple_tasks(delays)
    # 获取 EventLoop:
    loop = asyncio.get_event_loop()
    # 执行 coroutine 对象
    results = loop.run_until_complete(tasks)
    # 打印结果
    for delay,result in zip(delays,results):
        print(f'delay={delay},result={result} at {get_current_time()}')



@timeit
def run_gather_two_coro_with_asyncio_run(secs=3):
    """
    2 个 hello 任务是 coroutine,可以异步执行，在执行完后返回结果，因为是异步，所以时间不用累加起来，总统只要 3 secs
    :param secs:
    :return:
    """

    async def main():
        results = await asyncio.gather(helloV2(secs=secs),helloV2(secs=secs))
        return results
    result = asyncio.run(main())

    # 在 3秒后 获取 result
    print(f"result={result} at {get_current_time()}")

@timeit
async def create_two_task_by_sequence():

    task1 = helloV2(3)
    print(f"{get_current_time()}:已经生成 task1={task1},开始做别的事情 ")
    task2 = helloV2(5)
    print(f"{get_current_time()}:已经生成 task2={task2},开始做别的事情")
    print(f"{get_current_time()}:现在需要task1的结果进行计算")
    # result1 会等待 task1 的结果才会继续往下执行
    result1= await task1
    resutl1_add = result1+1
    print(f"{get_current_time()}:resutl1_add={resutl1_add}")

    print(f"{get_current_time()}:现在需要task2的结果进行计算")
    result2 = await task2
    resutl2_mul= result2*10
    print(f"{get_current_time()}:resutl2_mul={resutl2_mul}")
    return result1,result2


def run_two_sequence_task_by_syn():
    """
    共耗时 8 秒
    """
    asyncio.run(create_two_task_by_sequence())


@timeit
async  def creat_two_sequence_task():
    """
    共耗时 5 秒
    """
    coro1 = helloV2(3)
    # task1 = asyncio.create_task(coro1)
    print(f"{get_current_time()}:已经生成 task1={coro1},开始做别的事情 ")

    coro2 = helloV2(5)
    # task2 = asyncio.create_task(coro2)
    print(f"{get_current_time()}:已经生成 task2={coro2},开始做别的事情")

    return await asyncio.gather(*[coro1,coro2])


def run_two_sequence_task_by_asyncio():
    gather_tasks = creat_two_sequence_task()
    print(f"{get_current_time()}:完成 gather tasks")
    results = asyncio.run(gather_tasks)
    print(f"{get_current_time()}:完成 tasks的计算结果={results}")


# @timeit
def run_executor(secs):
    """
    @author:wangfc
    @desc: 
    @version：
    @time:2021/3/16 11:33
    
    Parameters
    ----------
    
    Returns
    -------
    """
    loop = asyncio.get_event_loop()
    # 将同步函数改为协程： # 虽然 ,sleep(secs) 已经执行了，但是状态还是 pending。
    # <Future pending cb=[_chain_future.<locals>._call_check_cancel() at C:\ProgramData\Anaconda3\lib\asyncio\futures.py:403]>
    future = loop.run_in_executor(None,sleep(secs))
    print(f'feature={future}')
    future.done()  # 还没有完成



# 定义一个回调函数
def callbackfunc(task):
    print("task 运行结束,它的结果是:",task.result())



def run_single_async_task_with_callback(delay=0.5):
    """
    @author:wangfc27441
    @desc:

    @version：
    @time:2021/2/5 17:36

    Parameters
    ----------

    Returns
    -------
    """
    start_time = time.time()
    print(f'开始 添加回调函数的单个协程task 任务: main thread={threading.currentThread()}')
    # 创建 coroutine 对象
    async_func = hello(secs =delay)
    # 获取 EventLoop:
    loop = asyncio.get_event_loop()
    # 创建 任务 对象: loop.create_task()方法创建一个task对象
    task = loop.create_task(async_func)
    # 添加 调用函数
    task.add_done_callback(callbackfunc)
    # 执行 coroutine 对象
    loop.run_until_complete(task)

    end_time = time.time()
    duration = end_time - start_time
    print(f'总共耗时为{duration}')


def run_async_stream_tasks():
    """
    @author:wangfc27441
    @desc:  使用协程 并发处理 流数据 （或者网络请求）
    @version：
    @time:2021/2/5 17:15

    Parameters
    ----------

    Returns
    -------
    """
    pass





async def foo(text):
    print(f"{get_current_time()}:print text= {text} in foo")
    await asyncio.sleep(3)

    print(f"{get_current_time()}:get result= {text} in foo")
    return text

async def foo_task(sec=1,result=None):
    print(f"{get_current_time()}:start")
    task = asyncio.create_task(foo('hello'))
    await asyncio.sleep(sec)
    # TODO: task 如果没有 await ，sleep 之后为什么会没有打印呢
    # result = await task
    print(f"{get_current_time()}:finished and result = {result}")
    return task


def run_foo_task(sec=1):
    asyncio.run(foo_task())


async def fetch_data():
    print('start fetching')
    await asyncio.sleep(2)
    print('done fetching')
    return {'data':1}

async def print_numbers():
    for i in range(10):
        print(i)
        await  asyncio.sleep(0.25)

async def create_fetch_and_print_tasks():
    task1 = asyncio.create_task(fetch_data())
    print('build task1')
    task2 = asyncio.create_task(print_numbers())
    print('build task2')
    value = await task1
    print(f"{get_current_time()}:get value={value}")
    await task2

def run_fetch_and_print_tasks():
    asyncio.run(create_fetch_and_print_tasks())


if __name__ == '__main__':
    # helloV1()
    # run_single_coro_with_event_loop()
    # run_single_coro_with_asyncio_run()

    # test_two_hello_task()
    # helloV2(3)
    # test_helloV2(secs=3)

    # run_single_task_with_event_loop()
    # run_gather_two_tasks_with_event_loop()

    run_two_sequence_task_by_syn()
    run_two_sequence_task_by_asyncio()


    # run_async_chain_coroutines()

    # run_single_coroutine(secs=3)
    run_executor(secs=3)