#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/15 14:02 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/15 14:02   wangfc      1.0         None

https://zhuanlan.zhihu.com/p/477826233 : Python多线程、多进程最全整理


进程是资源分配的最小单位，线程是CPU调度的最小单位
线程基本上是一个独立的执行流程。
单个进程可以包含多个线程。
程序中的每个线程都执行特定任务。


threading.currentThread(): 返回当前的线程变量
threading.enumerate(): 返回一个包含正在运行的线程的list
threading.activeCount(): 返回正在运行的线程数量，与len(threading.enumerate())有相同的结果


Thread对象
    threading.Thread和threading.current_thread()都创建了一个Thread对象，Thread对象有如下属性和方法

    getName() .name 获取线程名
    setName() 设置线程名
    start() join()这两个之前说过了
    join()有一个timeout参数，表示等待这个线程结束时，如果等待时间超过这个时间，就不再等，继续进行下面的代码，但是这个线程不会被中断
    run() 也是运行这个线程，但是必须等到这个线程运行结束才会继续执行之后的代码（如果将上面的start全换成run则相当于没有开多线程）
    is_alive()如果该线程还没运行完，就是True否则False
    daemon 返回该线程的daemon
    setDaemon(True)设置线程的daemon


Lock
多线程和多进程最大的不同在于，多进程中，同一个变量，各自有一份拷贝存在于每个进程中，互不影响，而多线程中，所有变量都由所有线程共享，
所以，任何一个变量都可以被任何一个线程修改，因此，线程之间共享数据最大的危险在于多个线程同时改一个变量，把内容给改乱了。


"""

# Create and launch a thread
from threading import Thread, current_thread,Lock

# Code to execute in an independent thread
import time
import multiprocessing
from utils.time import get_current_time, timeit


@timeit
def countdown(n):
    """
    IO密集型 ： 运行时间主要在于每次 sleep的那一秒， n-1 的计算是不会耗多少时间的。这种情况可以用多线程提高效率。
    """
    print(current_thread().getName())
    sleep_time = 1.1
    cost_time = 0
    while n > 0:
        print('T-minus', n)
        n -= 1
        time.sleep(sleep_time)
        cost_time += sleep_time
    return cost_time


@timeit
def run_thread(daemon=False):
    """
    Python中的多线程也可以在不创建类的情况下完成:
    threading 库可以在单独的线程中执行任何的在 Python 中可以调用的对象。
    1. 创建一个 Thread 对象并将你要执行的对象以 target 参数的形式提供给该对象
    2. .start()
    当你创建好一个线程对象后，该对象并不会立即执行，除非你调用它的 start() 方法:
    当你调用 start() 方法时，它会调用你传递进来的函数，并把你传递进来的参数传递给该函数
    Python中的线程会在一个单独的系统级线程中执行（比如说一个 POSIX 线程或者一个 Windows 线程），这些线程将由操作系统来全权管理。
    线程一旦启动，将独立执行直到目标函数返回。

    Python解释器直到所有线程都终止前仍保持运行。对于需要长时间运行的线程或者需要一直运行的后台任务，你应当考虑使用后台线程。
    t = Thread(target=countdown, args=(10,), daemon=True)

    后台线程无法等待，不过，这些线程会在主线程终止时自动销毁。 除了如上所示的两个操作，并没有太多可以对线程做的事情。
    你无法结束一个线程，无法给它发送信号，无法调整它的调度，也无法执行其他高级操作。

    """
    # 由于任何进程默认就会启动一个线程，默认名称为MainThread，也就是主程序占一个线程，我们把该线程称为主线程，这个线程和之后用Thread新加的线程是相互独立的，
    # 主线程又可以启动新的线程，不会等待其余线程运行结束就会继续往下运行。如果不用join()无法计算运行时间就是因为主线程先运行完了。

    # 主线程创建一个正在执行该函数的子线程: 创建一个 Thread 对象并将你要执行的对象以 target 参数的形式提供给该对象
    thread_name = None

    # args指定target对应函数的参数，用元组传入，比如args = (3, )

    t = Thread(target=countdown, args=(9,), daemon=daemon, name=thread_name)
    # 由主线程开始执行该线程
    t.start()
    # join()函数使主线程等待子进程完成
    cost_time = t.join()
    # 主线程打印
    print(f"cost_time ={cost_time}")


@timeit
def run_multiple_thread(num=5):
    print(f"同时运行{num}个 thread")
    thread_ls = []
    for _ in range(num):
        t = Thread(target=countdown,args=(9,))
        t.start()
        # 线程的join()方法表示等这个线程运行完毕，程序再往下运行。不能在这里使用 .join(),否则会逐个等待每个线程完成才开始下一个线程
        # t.join()
        thread_ls.append(t)
    # 定义ths列表存储这些线程，最后用循环确保每一个线程都已经运行完成再计算时间差。
    for th in thread_ls:
        th.join()




class CountdownTask:
    """
    t.start() 之后无法结束一个线程，无法给它发送信号，无法调整它的调度，也无法执行其他高级操作。
    如果需要这些特性，你需要自己添加。
    比如说，如果你需要终止线程，那么这个线程必须通过编程在某个特定点 轮询 来退出。你可以像下边这样把线程放入一个类中：
    """

    def __init__(self):
        self._running = True

    def terminate(self):
        self._running = False
        self.cost_time = 0

    @timeit
    def run(self, n):
        print(current_thread().getName())
        sleep_time = 1.1
        cost_time = 0
        while self._running and n > 0:
            print('T-minus', n)
            n -= 1
            time.sleep(sleep_time)
            cost_time += sleep_time
        self.cost_time = cost_time


@timeit
def run_countdown_task_in_thread():
    c = CountdownTask()
    t = Thread(target=c.run, args=(10,))
    t.start()
    # c.terminate()  # Signal termination
    # 可以将一个线程加入到当前线程，并等待它终止：
    t.join()  # Wait for actual termination (if needed)
    # 主线程打印
    print(f"cost_time ={c.cost_time}")


class CountdownThread(Thread):
    """
    通过扩展Thread类创建子类时，子类表示新线程正在执行某个任务。
    扩展Thread类时，子类只能覆盖两个方法，即__init __（）方法和run（）方法。
    除了这两种方法之外，没有其他方法可以被覆盖。

    尽管这样也可以工作，但这使得你的代码依赖于 threading 库，所以你的这些代码只能在线程上下文中使用。

    """

    def __init__(self, n):
        super().__init__()
        self.n = n

    def run(self):
        while self.n > 0:
            print('T-minus', self.n)
            self.n -= 1
            time.sleep(5)


def run_countdownthread():
    c = CountdownThread(5)
    c.start()


def run_countdown_task_in_multiprocess():
    c = CountdownTask(5)
    p = multiprocessing.Process(target=c.run)
    p.start()


class IOTask:
    """
    IO 密集型任务，Python 多线程是可以提高运行效率的。

    如果线程执行一些像I/O这样的阻塞操作，那么通过轮询来终止线程将使得线程之间的协调变得非常棘手。
    比如，如果一个线程一直阻塞在一个I/O操作上，它就永远无法返回，也就无法检查自己是否已经被结束了。
    要正确处理这些问题，你需要利用超时循环来小心操作线程。

    """

    def terminate(self):
        self._running = False

    def run(self, sock):
        # sock is a socket
        sock.settimeout(5)  # Set timeout period
        while self._running:
            # Perform a blocking I/O operation w/ timeout
            try:
                data = sock.recv(8192)
                break
            except socket.timeout:
                continue
            # Continued processing
            ...
        # Terminated
        return

balance = 0
lock = Lock()


def change_it(n):
    # 先存后取，结果应该为0:
    global balance
    balance = balance + n
    balance = balance - n

def run_change_loop(n):
    for i in range(2000000):
        change_it(n)


@timeit
def run_change_loop_with_lock(n):
    """
    https://zhuanlan.zhihu.com/p/75780308 深入理解Python中的GIL（全局解释器锁）。

    GIL是否意味着线程安全:
    GIL 是为了让解释器在执行 Python 代码时，同一时刻只有一个线程在运行，以此保证内存管理是安全的。
    现在很多 Python 项目已经习惯依赖 GIL（开发者认为 Python 内部对象 就是线程安全的，写代码时对共享资源的访问不会加锁）

    Python中的线程也会放弃GIL。既然一个线程可能随时会失去GIL，那么这就一定会涉及到线程安全的问题。
    CPthon 中是粗粒度的锁，即语言层面本身维护着一个全局的锁机制,用来保证线程安全,即不需要程序员自己对线程进行加锁处理
    细粒度就是指程序员需要自行加、解锁来保证线程安全，典型代表是 Java


    当多个线程同时执行lock.acquire()时，只有一个线程能成功地获取锁，然后继续执行代码，其他线程就继续等待直到获得锁为止。
    获得锁的线程用完后一定要释放锁，否则那些苦苦等待锁的线程将永远等待下去，成为死线程。所以我们用try...finally来确保锁一定会被释放。

    锁的好处就是确保了某段关键代码只能由一个线程从头到尾完整地执行.
    坏处当然也很多:
    首先是阻止了多线程并发执行，包含锁的某段代码实际上只能以单线程模式执行，效率就大大地下降了。
    其次，由于可以存在多个锁，不同的线程持有不同的锁，并试图获取对方持有的锁时，可能会造成死锁，导致多个线程全部挂起，既不能执行，也无法结束，只能靠操作系统强制终止。
        """
    for i in range(20000000):
        # 先获取全局锁
        lock.acquire()
        # 放心地修改
        try:
            change_it(n)
        except Exception as e:
            print(e)
        finally:
            # 修改万后释放
            lock.release()



@timeit
def run_change_balance(with_lock=True):
    """
    典型的多个线程操作同一个全局变量造成的线程不安全的问题

    https://www.liaoxuefeng.com/wiki/1016959663602400/1017629247922688

    https://cloud.tencent.com/developer/news/743497 Python进阶——为什么GIL让多线程变得如此鸡肋？

    我们定义了一个共享变量balance，初始值为0，并且启动两个线程，先存后取，理论上结果应该为0，
    但是，由于线程的调度是由操作系统决定的，当t1、t2交替执行时，只要循环次数足够多，balance的结果就不一定是0了。

    gil 问题：
    在这个例子中，虽然我们开启了 2 个线程去执行 loop，但我们观察 CPU 的使用情况，发现这个程序只能跑满一个（多） CPU 核心，没有利用到多核。
    """
    if with_lock:
        thread_fn = run_change_loop_with_lock
    else:
        thread_fn = run_change_loop
    t1 = Thread(target=thread_fn, args=(5,))
    t2 = Thread(target=thread_fn, args=(8,))
    t3 = Thread(target=thread_fn, args=(3,))
    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()
    print(balance)






def main():
    # run_thread()
    # run_multiple_thread()
    # run_countdown_task_in_thread()
    run_change_balance()


if __name__ == '__main__':
    main()
