#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/15 13:52 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/15 13:52   wangfc      1.0         None

https://zhuanlan.zhihu.com/p/20953544 谈谈python的GIL、多线程、多进程
https://cloud.tencent.com/developer/news/743497  Python进阶——为什么GIL让多线程变得如此鸡肋？

1、GIL是什么？GIL的全称是Global Interpreter Lock(全局解释器锁)，来源是python设计之初的考虑，为了数据安全所做的决定。
In CPython, the global interpreter lock, or GIL, is a mutex that protects access to Python objects,
preventing multiple threads from executing Python bytecodes at once.
This lock is necessary mainly because CPython's memory management is not thread-safe.
 (However, since the GIL exists, other features have grown to depend on the guarantees that it enforces.)




2、每个CPU在同一时间只能执行一个线程（在单核CPU下的多线程其实都只是并发，不是并行，并发和并行从宏观上来讲都是同时处理多路请求的概念。
但并发和并行又有区别，并行是指两个或者多个事件在同一时刻发生；而并发是指两个或多个事件在同一时间间隔内发生。

在Python多线程下，每个线程的执行方式：
1.获取GIL
2.执行代码直到sleep或者是python虚拟机将其挂起。
3.释放GIL


可见，某个线程想要执行，必须先拿到GIL，我们可以把GIL看作是“通行证”，并且在一个python进程中，GIL只有一个。拿不到通行证的线程，就不允许进入CPU执行。


释放GIL锁:
    由于 Python 的线程就是 C 语言的 pthread，它是通过操作系统调度算法调度执行的。
    python2.x里，GIL的释放逻辑是当前线程遇见 系统IO操作 或者 ticks计数(opcode 数量) 达到 100
    （ticks可以看作是python自身的一个计数器，专门做用于GIL，每次释放后归零，这个计数可以通过 sys.setcheckinterval 来调整），进行释放。
    而每次释放GIL锁，线程进行锁竞争、切换线程，会消耗资源。
    并且由于GIL锁存在，python里一个进程永远只能同时执行一个线程(拿到GIL的线程才能执行)，这就是为什么在多核CPU上，python的多线程效率并不高。

    python3.x中，GIL不使用ticks计数，改为使用计时器（执行时间达到阈值 (15ms)后，当前线程释放GIL） 或遇到系统 IO 时，强制释放 GIL，触发系统的线程调度。
    这样对CPU密集型程序更加友好， 但依然没有解决GIL导致的同一时间只能执行一个线程的问题，所以效率依然不尽如人意。






那么是不是python的多线程就完全没用了呢？

在这里我们进行分类讨论：
1、CPU密集型代码(各种循环处理、计数等等)
在这种情况下，ticks计数很快就会达到阈值，然后触发GIL的释放与再竞争（多个线程来回切换当然是需要消耗资源的），所以python下的多线程对CPU密集型代码并不友好。

2、IO密集型代码(文件处理、网络爬虫等)
多线程能够有效提升效率(单线程下有IO操作会进行IO等待，造成不必要的时间浪费，而开启多线程能在线程A等待时，自动切换到线程B，可以不浪费CPU的资源，
从而能提升程序执行效率)。所以python的多线程对IO密集型代码比较友好。




多核多线程比单核多线程更差，原因是单核下多线程，每次释放GIL，唤醒的那个线程都能获取到GIL锁，所以能够无缝执行，
但多核下，CPU0释放GIL后，其他CPU上的线程都会进行竞争，但GIL可能会马上又被CPU0拿到，导致其他几个CPU上被唤醒后的线程会醒着等待到切换时间后又进入待调度状态，
这样会造成线程颠簸(thrashing)，导致效率更低

“python下想要充分利用多核CPU，就用多进程”，原因是什么呢？
原因是：每个进程有各自独立的GIL，互不干扰，这样就可以真正意义上的并行执行，所以在python中，多进程的执行效率优于多线程(仅仅针对多核CPU而言)。

https://python3-cookbook.readthedocs.io/zh_CN/latest/c12/p01_start_stop_thread.html
由于全局解释锁（GIL）的原因，Python 的线程被限制到同一时刻只允许一个线程执行这样一个执行模型。
所以，Python 的线程更适用于处理I/O和其他需要并发执行的阻塞操作（比如等待I/O、等待从数据库获取数据等等），而不是需要多处理器并行的计算密集型任务。


解决方案
https://cloud.tencent.com/developer/news/743497
既然 GIL 的存在会导致这么多问题，那我们在开发时，需要注意哪些地方，避免受到 GIL 的影响呢？
我总结了以下几个方案：

    IO 密集型任务场景，可以使用多线程可以提高运行效率
    CPU 密集型任务场景，不使用多线程，推荐使用多进程方式部署运行
    更换没有 GIL 的 Python 解释器，但需要提前评估运行结果是否与 CPython 一致
    编写 Python 的 C 扩展模块，把 CPU 密集型任务交给 C 模块处理，但缺点是编码较为复杂
    更换其他语言 :)


"""