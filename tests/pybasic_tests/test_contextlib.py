#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/17 9:49 

原文作者：Python 技术论坛文档：《Python 3 标准库实例教程（）》
转自链接：https://learnku.com/docs/pymotw/contextlib-context-manager-tool/3379
版权声明：翻译文档著作权归译者和 LearnKu 社区所有。转载请保留原文链接


@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/17 9:49   wangfc      1.0         None
"""
import contextlib


class BasicContext:
    """
    实现 上下文管理 是通过__enter__和__exit__这两个方法实现的
    with Context() as c:
        c.do_something()

    每个上下文管理器都允许使用 with 语句来执行，在写的时候也要包含两个必须的方法。

     __enter__() 方法: with 进入代码时所执行的方法，一般要返回一个对象以让处在代码块中的代码使用。
     __exit__() 方法: 离开 with 代码块后，上下文管理器中的 __exit__() 方法就会被调用以清理一些用过的资源。

    """

    def __init__(self):
        print(f'{self.__class__.__name__}__init__()')

    def __enter__(self):
        """
        __enter__() 方法可以返回任意对象，它返回的任何对象都会被赋给 with 语句中的 as 所指向的变量。
        __enter__() 所返回的值 可以 在被装饰的函数中使用
        """
        print(f'{self.__class__.__name__}__enter__()')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'{self.__class__.__name__}__exit__()')


def test_basic_context():
    with BasicContext() as c:
        print('just do something')


class DoSomething():

    def __init__(self, *args):
        print('DoSomething.__init__({})'.format(*args))

    def do_something(self):
        print('DoSomething.do_something()')

    def __del__(self):
        print('DoSomething.__del__')


def test_do_something_in_context():
    with BasicContext() as c:
        do = DoSomething(1)
        do.do_something()


class Context:
    """
    with Context() as c:
        c.do_something()

    每个上下文管理器都允许使用 with 语句来执行，在写的时候也要包含两个必须的方法。
     __enter__() 方法是 with 进入代码时所执行的方法，一般要返回一个对象以让处在代码块中的代码使用。
     离开 with 代码块后，上下文管理器中的 __exit__() 方法就会被调用以清理一些用过的资源。

    """

    def __init__(self):
        print('__init__()')

    def __enter__(self):
        """
        __enter__() 方法可以返回任意对象，它返回的任何对象都会被赋给 with 语句中的 as 所指向的变量。
        __enter__() 所返回的值 可以 在被装饰的函数中使用
        """
        print('__enter__()')

        return WithinContext(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('__exit__()')


class WithinContext:

    def __init__(self, context):
        print('WithinContext.__init__({})'.format(context))

    def do_something(self):
        print('WithinContext.do_something()')

    def __del__(self):
        print('WithinContext.__del__')


class ContextHandleError:

    def __init__(self, handle_error):
        print('__init__({})'.format(handle_error))
        self.handle_error = handle_error

    def __enter__(self):
        print('__enter__()')
        # raise ValueError("within content 异常")
        return WithinContext(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        __exit__() 方法所接受的参数是任何在 with 代码块中产生的异常的详细信息
        __exit__() 执行后 返回 True or False
        如果返回的是 True，则该异常就会就此消失，不会再进行传播，否则就会一直传递下去
        如果返回的是 False，则该异常会在 __exit__() 执行后重新抛出
        """
        print('__exit__()')
        print('  exc_type =', exc_type)
        print('  exc_val  =', exc_val)
        print('  exc_tb   =', exc_tb)

        return self.handle_error


def test_handle_error_context():
    try:
        with ContextHandleError(True) as c:
            """
            如果上下文管理器可以处理这个异常， __exit__() 应该返回 True 表示这个异常并没有造成麻烦，不必管它。
            """
            c.do_something()
            raise RuntimeError('error message propagated')
    # __exit__() 执行后 返回 True or False
    except Exception as e:
        # 如果返回的是 False，则该异常会在 __exit__() 执行后重新抛出
        print(e)


class MyContextDecorator(contextlib.ContextDecorator):
    """
    函数装饰器方式的上下文管理器
    ContextDecorator 类可以让标准的上下文管理器类变成一个可以作为函数装饰器方式使用的上下文管理器

    把上下文管理器作为函数装饰器使用的不同之处在于 __enter__() 所返回的值无法在被装饰的函数中使用，不能像 with 和 as 一样

    """

    def __init__(self, how_used):
        self.how_used = how_used
        print('__init__({})'.format(how_used))

    def __enter__(self):
        print('__enter__({})'.format(self.how_used))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('__exit__({})'.format(self.how_used))


@MyContextDecorator('my context decorator')
def func(params):
    print(f"doing something with params={params}")


@MyContextDecorator('my context decorator')
class DoSomethingWithContextDecorator:
    def __init__(self, *args):
        print('WithinContextDecorator.__init__({})'.format(args))

    def do_something(self):
        print('WithinContextDecorator.do_something()')

    def __del__(self):
        print('WithinContextDecorator.__del__')


def test_context_decorator_on_func():
    func('playing a context decorator')


def test_context_decorator_on_class():
    # TODO: 不知道这种方法是否合理
    doer = DoSomethingWithContextDecorator()
    doer.do_something()


"""
example from  https://www.liaoxuefeng.com/wiki/1016959663602400/1115615597164000
"""
class QueryContextClass(object):

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print('Begin')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            print('Error')
        else:
            print('End')

    def query(self):
        print('Query info about %s...' % self.name)


class Query(object):

    def __init__(self, name):
        self.name = name
        print(' query init')

    def query(self):
        print('Query info about %s...' % self.name)


@contextlib.contextmanager
def create_query(name):
    print('Begin')
    q = Query(name)
    yield q
    print('End')



def test_query_with_context():
    with create_query('Bob') as q:
        q.query()




@contextlib.contextmanager
def context_from_generator(value=None):
    """
    创建一个上下文管理器的传统方式是写一个类，然后写它的 __enter__() 和 __exit__() 方法，这并不难写，不过有时候把所有的东西都写全并没有必要。
    在这种情况下，可以使用 contextmanager() 装饰器将一个 生成器函数 变成一个上下文管理器。
    """
    print('  entering')
    try:
        """
        通过 Yield 控制，
        作为装饰器使用时，无需返回值,上下文管理器所返回的值并不会被使用到。
        """
        yield {value}
    except RuntimeError as err:
        print('  ERROR:', err)
    finally:
        print('  exiting\n')


def test_context_from_generator():
    """
    生成器应首先初始化上下文，确保只生成一次，最后应清理上下文内容。所生成的内容可以被 with 语句的 as 所赋值给一个变量。
    """
    print('Normal:')
    with context_from_generator() as value:
        print('  inside with statement:', value)


    """
    with 语句中的异常也会在生成器内部重新被抛出，这样在我们就可以处理这个异常。
    """
    print('\nHandled error:')
    with context_from_generator() as value:
        raise RuntimeError('showing example of handling an error')



    print('\n Do something:')
    with context_from_generator() as value:
        do = DoSomething(value)
        do.do_something()


    print('\nUnhandled error:')
    with context_from_generator() as value:
        raise ValueError('this exception is not handled')


def test_context_from_generator_as_decarator():
    @context_from_generator()
    def normal():
        print('  inside with statement')

    @context_from_generator()
    def throw_error(err):
        raise err

    print('Normal:')
    normal()

    print('\nHandled error:')
    throw_error(RuntimeError('showing example of handling an error'))

    print('\nUnhandled error:')
    throw_error(ValueError('this exception is not handled'))


@contextlib.contextmanager
def make_context_generator(i):
    """
    很多时候，我们希望在某段代码执行前后自动执行特定代码，

    """
    print('{} entering'.format(i))
    yield {}
    print('{} exiting'.format(i))


def test_make_content_before_code():
    with make_context_generator():
        print("hello world")


def variable_stack(n=None, msg=None,*args):
    """
    可能需要创建一个未知数量的上下文，同时希望控制流退出上下文时这些上下文管理器也全部执行清理功能。
    ExitStack 就是用来处理这些动态情况的。

    ExitStack 实例维护一个包含清理回调的栈。这些回调都会被放在上下文中，任何被注册的回调都会在控制流退出上下文时以倒序方式被调用。
    这有点像嵌套了多层的 with 语句，除了它们是被动态创建的。

    variable_stack() 把上下文管理器放到 ExitStack 中使用，逐一建立起上下文。
    """

    with contextlib.ExitStack() as stack:
        if n is not None:
            for i in range(n):
                # 有几种填充 ExitStack 的方式。本例使用 enter_context() 来将一个新的上下文管理器添加入栈。
                # enter_context() 首先会调用上下文管理器中的__enter__() 方法，然后把它的 __exit__() 注册为一个回调以便让栈调用。
                context = make_context_generator(i)
                stack.enter_context(context)
        else:
            for context in args:
                stack.enter_context(context)
        print(msg)


def test_varialble_stack():
    variable_stack(2, 'inside context')


class Tracker:
    "用于提醒上下文信息的基础类"

    def __init__(self, i):
        self.i = i

    def msg(self, s):
        print('  {}({}): {}'.format(
            self.__class__.__name__, self.i, s))

    def __enter__(self):
        self.msg('entering')


class HandleError(Tracker):
    "处理任何接收到的异常."

    def __exit__(self, *exc_details):
        received_exc = exc_details[1] is not None
        if received_exc:
            self.msg('handling exception {!r}'.format(
                exc_details[1]))
        self.msg('exiting {}'.format(received_exc))
        # 返回布尔类型的值代表是否已经处理了该异常。
        return received_exc


class PassError(Tracker):
    "传递任何接收到的异常。"

    def __exit__(self, *exc_details):
        received_exc = exc_details[1] is not None
        if received_exc:
            self.msg('passing exception {!r}'.format(
                exc_details[1]))
        self.msg('exiting')
        # 返回False，表示没有处理这个异常。
        return False


class ErrorOnExit(Tracker):
    "抛出个异常"

    def __exit__(self, *exc_details):
        self.msg('throwing error')
        raise RuntimeError('from {}'.format(self.i))


class ErrorOnEnter(Tracker):
    "抛出个异常."

    def __enter__(self):
        self.msg('throwing error on enter')
        raise RuntimeError('from {}'.format(self.i))

    def __exit__(self, *exc_info):
        self.msg('exiting')

def test_contextlib_context_managers():
    """
    ExitStack 中的上下文管理器会像一系列嵌套的 with 一样。 任何发生在上下文中的错误都会交给上下文管理器的正常错误处理系统去处理。下面的上下文管理器类们可以说明传递方式。

    """

    print('No errors: outside of stack, any errors were handled')
    variable_stack(None,None,*[
        HandleError(1),
        PassError(2),
    ])

    print('\nError at the end of the context stack:')
    variable_stack([
        HandleError(1),
        HandleError(2),
        ErrorOnExit(3),
    ])


