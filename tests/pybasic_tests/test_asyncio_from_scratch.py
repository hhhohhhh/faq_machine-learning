#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/17 15:32 

from:  深入理解Python异步编程（上） https://www.jianshu.com/p/fe146f9781d2


@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/17 15:32   wangfc      1.0         None
"""
import socket
from concurrent import futures
from selectors import DefaultSelector, EVENT_READ, EVENT_WRITE
# from typing import List, Text
# import asyncio
# import aiohttp

from tests.pybasic_tests.test_decorator import timeit


@timeit
def blocking_task(task_index=None, url='example.com', port=80):
    """
    建立 socket 连接，发送HTTP请求，然后从 socket 读取HTTP响应并返回数据。示例中我们请求了 example.com 的首页。


    sock.connect() : 创建网络连接，多久能创建完成不是客户端决定的，而是由网络状况和服务端处理能力共同决定。
    sock.recv(): 服务端什么时候返回了响应数据并被客户端接收到可供程序读取，也是不可预测的
    所以 sock.connect()和 sock.recv()这两个调用在默认情况下是阻塞的

    sock.send()函数并不会阻塞太久，它只负责将请求数据拷贝到TCP/IP协议栈的系统缓冲区中就返回，并不等待服务端返回的应答确认。


    """
    if task_index:
        print(f"task_index={task_index}")
    sock = socket.socket()
    # 向 example.com 主机的80端口发起网络连接请求
    sock.connect((url, port))
    request = 'GET / HTTP/1.0\r\nHost:example.com\r\n\r\n'
    sock.send(request.encode('ascii'))
    response = b''
    # 从socket上读取4K字节数据
    chunk = sock.recv(4096)
    while chunk:
        response += chunk
        chunk = sock.recv(4096)
    return response


@timeit
def test_sync_tasks():
    """
    共耗时 5.111061334609985 secs
    """
    res = []
    for i in range(10):
        res.append(blocking_task())
    return len(res)


@timeit
def test_multiprocess_way():
    """
    在一个程序内，依次执行10次太耗时，那开10个一样的程序同时执行不就行了。于是我们想到了多进程编程。
    共耗时 0.6143355369567871 secs or  1sec

    """
    workers = 10
    with futures.ProcessPoolExecutor(workers) as executor:
        future_results = {executor.submit(blocking_task) for i in range(10)}
    return len([fut.result for fut in future_results])


@timeit
def test_multithread_way():
    """
    共耗时 0.5884056091308594 secs

    首先，Python中的多线程因为 GIL的存在，它们并不能利用CPU多核优势，一个Python进程中，只允许有一个线程处于运行状态.
    因为在做阻塞的系统调用时，例如sock.connect(),sock.recv()时，当前线程会释放GIL，让别的线程有执行机会。但是单个线程内，在阻塞调用上还是阻塞的。

    """
    threads = 10
    with futures.ThreadPoolExecutor(threads) as executor:
        future_results = {executor.submit(blocking_task) for i in range(10)}
    return len([fut.result for fut in future_results])


@timeit
def raw_noneblocking_task(task_index=None, url='example.com', port=80):
    """


    """
    if task_index:
        print(f"task_index={task_index}")
    sock = socket.socket()
    # 让socket上阻塞调用都改为非阻塞的方式： 默认 True ，如果设置 False ，那么accept和recv时一旦无数据，则报错。
    sock.setblocking(False)
    # 向example.com主机的80端口发起网络连接请求
    try:
        # 要放在try语句内，是因为socket在发送非阻塞连接请求过程中，系统底层也会抛出异常
        sock.connect((url, port))
    except BlockingIOError:
        pass

    request = 'GET / HTTP/1.0\r\nHost:example.com\r\n\r\n'

    data = request.encode('ascii')


    # 判断非阻塞调用是否就绪:
    # 需要while循环不断尝试 send()，是因为connect()已经非阻塞，在send()之时并不知道 socket 的连接是否就绪，
    # 只有不断尝试，尝试成功为止，即发送数据成功了。
    # 虽然 connect() 和 recv() 不再阻塞主程序，空出来的时间段CPU没有空闲着，但并没有利用好这空闲去做其他有意义的事情，
    # 而是在循环尝试读写 socket （不停判断非阻塞调用的状态是否就绪）。
    # 还得处理来自底层的可忽略的异常。也不能同时处理多个 socket 。
    while True:
        try:
            sock.send(data)
            # 直到 send 不抛出异常，则发送完成
            break
        except OSError:
            pass

    response = b''
    while True:
        try:
            # 从socket上读取4K字节数据
            chunk = sock.recv(4096)
            while chunk:
                response += chunk
                chunk = sock.recv(4096)
            break
        except IOError:
            pass

    return response


@timeit
def test_raw_noneblocking_tasks():
    """
    共耗时 5.420908689498901 secs
    虽然 connect() 和 recv() 不再阻塞主程序，空出来的时间段CPU没有空闲着，但并没有利用好这空闲去做其他有意义的事情，
    而是在循环尝试读写 socket （不停判断非阻塞调用的状态是否就绪）。还得处理来自底层的可忽略的异常。也不能同时处理多个 socket 。

    """
    res = []
    for i in range(10):
        res.append(raw_noneblocking_task(task_index=i))
    return len(res)


class Crawler:
    def __init__(self, url='example.com', port=80):

        self.url = url
        self.port = port
        self.sock = None
        self.response = b''
        print(f"{self.__class__.__name__}.__init__(url={self.url})")

    def fetch(self):
        """
        select 模块: OS将I/O状态的变化都封装成了事件，如可读事件、可写事件,并且提供了专门的系统模块让应用程序可以接收事件通知。
        让应用程序可以通过select 注册文件描述符和回调函数。
        当文件描述符的状态发生变化时，select 就调用事先注册的回调函数。

        创建了一个DefaultSelector 实例。
        Python标准库提供的selectors模块是对底层select/poll/epoll/kqueue的封装。
        DefaultSelector类会根据 OS 环境自动选择最佳的模块，那在 Linux 2.5.44 及更新的版本上都是epoll了
        让epoll代替应用程序监听socket状态时，得告诉epoll：
        如果socket状态变为可以往里写数据（连接建立成功了），请调用HTTP请求发送函数。
        如果socket 变为可以读数据了（客户端已收到响应），请调用响应处理函数。”

        """
        global selector
        self.sock = socket.socket()
        self.sock.setblocking(False)
        try:
            self.sock.connect(("example.com", self.port))
        except BlockingIOError:
            pass
        # 注册了socket 可写事件(EVENT_WRITE)发生后 应该采取的回调函数。
        selector.register(fileobj=self.sock.fileno(), events=EVENT_WRITE, data=self.connected)

    def connected(self, key, mask):
        selector.unregister(key.fd)
        request = 'GET / HTTP/1.0\r\nHost:example.com\r\n\r\n'
        data = request.encode('ascii')
        self.sock.send(data)
        # 注册了socket 可读事件(EVENT_READ)发生后 应该采取的 回调函数。
        selector.register(key.fd, EVENT_READ, self.read_response)

    def read_response(self, key, mask):
        # stopped 全局变量控制事件循环何时停止。当 urls_todo消耗完毕后，会标记stopped为True。
        global stopped
        global urls
        chunk = self.sock.recv(4096)
        if chunk:
            self.response += chunk
        else:
            selector.unregister(key.fd)
            urls.remove(self.url)
            if not urls:
                stopped = True


# 定义事件循环
def events_loop(version=1):
    global selector
    global stopped

    while not stopped:
        # 如何从selector里获取当前正发生的事件，并且得到对应的回调函数去执行呢？
        # 去访问selector模块，等待它告诉我们当前是哪个事件发生了，应该对应哪个回调。这个等待事件通知的循环，称之为事件循环。
        # selector.select() 是一个阻塞调用，因为如果事件不发生，那应用程序就没事件可处理，所以就干脆阻塞在这里等待事件发生
        events = selector.select()
        for event_key, event_mask in events:
            callback = event_key.data
            if version==1:
                callback(event_key, event_mask)
            elif version ==2:
                callback()


@timeit
def run_selectors_loop():
    """
    共耗时 0.6915977001190186 secs

    基于 事件循环 + 回调的基础 的异步执行过程：

    创建 Crawler 实例；
    1.调用fetch方法，会创建socket连接和在 selector上注册可写事件；
    2. fetch内并无阻塞操作，该方法立即返回；
    3. 重复上述3个步骤，将10个不同的下载任务都加入事件循环；
    4. 启动事件循环，进入第1轮循环，阻塞在事件监听上；
    5. 当某个下载任务EVENT_WRITE被触发，回调其connected方法，第一轮事件循环结束；
    6. 进入第2轮事件循环，当某个下载任务有事件触发，执行其回调函数；此时已经不能推测是哪个事件发生，因为有可能是上次connected里的EVENT_READ先被触发，
    也可能是其他某个任务的EVENT_WRITE被触发；（此时，原来在一个下载任务上会阻塞的那段时间被利用起来执行另一个下载任务了）
    7. 循环往复，直至所有下载任务被处理完成
    8. 退出事件循环，结束整个下载程序

    """
    # 默认使用epoll去注册事件
    global selector
    selector= DefaultSelector()
    # stopped 全局变量控制 事件循环 何时停止
    global stopped
    stopped = False
    global urls
    urls = {f'/{i}' for i in range(10)}

    for url in urls:
        # 虽然有一个for 循环顺序地创建Crawler 实例并调用 fetch 方法，但是fetch 内仅有connect()和注册可写事件
        crawler = Crawler(url)
        crawler.fetch()
    # 维护一个事件循环
    events_loop()


class Future():
    """
    不用回调的方式了，怎么知道异步调用的结果呢？
    先设计一个对象，异步调用执行完的时候，就把结果放在它里面。这种对象称之为未来对象。
    """

    def __init__(self):
        # result属性，用于存放未来的执行结果
        self.result = None
        self._callbacks = []

    def add_done_callback(self, fn):
        self._callbacks.append(fn)

    def set_result(self, result):
        # 是用于设置result的，并且会在给result绑定值以后运行事先给future添加的回调

        self.result = result
        for fn in self._callbacks:
            fn(self)


class CrawlerV2:
    """
    用Future来重构爬虫代码
    """

    def __init__(self, url='example.com', port=80):

        self.url = url
        self.port = port
        # self.sock = None
        self.response = b''

    def fetch(self):

        global selector
        global stopped
        global urls

        sock = socket.socket()
        sock.setblocking(False)
        try:
            sock.connect(("example.com", self.port))
        except BlockingIOError:
            pass

        # 初始化一个 Future 对象
        f = Future()

        def on_connected():
            f.set_result(None)

        # 注册了socket可写事件(EVENT_WRITE)发生后应该采取的回调函数。回调相当简单，就是给对应的future对象绑定结果值
        selector.register(fileobj=sock.fileno(), events=EVENT_WRITE, data=on_connected)

        # fetch 方法内有了yield表达式，使它成为了生成器，返回对应的future对象
        # 那这fetch生成器如何再次恢复执行呢？
        yield f

        selector.unregister(fileobj=sock.fileno())

        request = f'GET {self.url} / HTTP/1.0\r\nHost:example.com\r\n\r\n'
        data = request.encode('ascii')
        sock.send(data)

        while True:
            f = Future()

            def on_readable():
                f.set_result(sock.recv(4096))

            selector.register(fileobj=sock.fileno(), events=EVENT_READ, data=on_readable())
            chunk = yield f
            selector.unregister(sock.fileno())
            if chunk:
                self.response += chunk
            else:
                urls.remove(self.url)
                if not urls:
                    stopped = True
                break


class Task:
    """
    没人来恢复这个生成器的执行么？没人来管理生成器的状态么？
    创建一个，就叫Task
    Task封装了coro对象，即初始化时传递给他的对象，被管理的任务是待执行的协程，故而这里的 coro 就是 fetch()生成器。
    """

    def __init__(self, coro):
        self.coro = coro
        f = Future()
        f.set_result(None)
        self.step(f)

    def step(self, future):
        """
        在初始化的时候就会执行一遍。
        step()内会调用生成器的send()方法，初始化第一次发送的是None就驱动了coro即fetch()的第一次执行。
        send()完成之后，得到下一次的future，然后给下一次的future添加step()回调。
        """
        try:
            next_future = self.coro.send(future.result)
        except StopIteration:
            return
        next_future.add_done_callback(self.step)


@timeit
def run_selectors_loopV2():
    # TODO: 代码还存在错误

    global selector
    selector= DefaultSelector()
    global stopped
    stopped = False
    global urls
    urls = {f'/{i}' for i in range(10)}

    for url in urls:
        crawler = CrawlerV2(url)
        Task(crawler.fetch())
    events_loop(version=2)



async def async_fetch(url):
    pass



def run_aiohttp():
    urls = {f'/{i}' for i in range(10)}



if __name__ == '__main__':
    # run_selectors_loop()
    run_selectors_loopV2()
