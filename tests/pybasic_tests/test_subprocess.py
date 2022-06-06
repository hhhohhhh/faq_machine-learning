#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/3 11:15 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/3 11:15   wangfc      1.0         None

直接调用操作系统的方法
    os.system()方法：
        利用os模块可以直接调用操作系统的方法，里面可以直接填写操作系统的一个方法，这里用的方法就和我们平常在终端开启程序是一样的命令，
        使用 python 文件名.py 指令就可以了。
        这个方法会接受一个字符串命令，然后在 subshell 中执行，通常是 linux/OSX 下的 bash ，或者 Windows 下面的 cmd.exe。
        根据官方的文档，os.system() 方法时使用标准 C 方法 system() 来调用实现的，所以存在和 C 方法中一样的限制。
        ex:  os.system("python –version")

    os.popen() 方法
        利用os模块的popen方法，用python解释器以读的模式打开文件，打开后还得加上读的方法才可以运行。


subprocess(python3.7)
    subprocess 主要是为了替换以下的操作系统模块函数，允许你执行一些命令，并获取返回的状态码和 输入，输出和错误信息。
        Python 3.5 之前:
       .call() : 运行命令。该函数将一直等待到子进程运行结束，并返回进程的 returncode。如果子进程不需要进行交互,就可以使用该函数来创建。
            ex: subprocess.call(["python", "–version"])

        python3.5 以后官方推荐使用run:

       .Popen: 可以使用Popen来创建进程，并与进程进行复杂的交互。
       subprocess.Popen(args, bufsize=0, executable=None, stdin=None, stdout=None, stderr=None, preexec_fn=None, close_fds=False, shell=False, cwd=None, env=None, universal_newlines=False, startupinfo=None, creationflags=0)
            args：shell命令，可以是字符串，或者序列类型，如list,tuple。
            bufsize：缓冲区大小，可不用关心
            参数executable用于指定可执行程序。一般情况下我们通过args参数来设置所要运行的程序。如果将参数shell设为 True，executable将指定程序使用的shell。在windows平台下，默认的shell由COMSPEC环境变量来指定。
            stdin,stdout,stderr：分别表示程序的标准输入，标准输出及标准错误
                subprocess.PIPE: 在创建Popen对象时，subprocess.PIPE可以初始化stdin, stdout或stderr参数。表示与子进程通信的标准流。
                subprocess.STDOUT: 创建Popen对象时，用于初始化stderr参数，表示将错误通过标准输出流输出。
            shell：与上面方法中用法相同
            cwd：用于设置子进程的当前目录
            preexec_fn只在Unix平台下有效，用于指定一个可执行对象（callable object），它将在子进程运行之前被调用。
            Close_sfs：在windows平台下，如果close_fds被设置为True，则新创建的子进程将不会继承父进程的输入、输出、错误管 道。我们不能将close_fds设置为True同时重定向子进程的标准输入、输出与错误(stdin, stdout, stderr)。
            shhell设为true，程序将通过shell来执行。
            参数cwd用于设置子进程的当前目录。
            env：用于指定子进程的环境变量。如果env=None，则默认从父进程继承环境变量

            universal_newlines：不同系统的的换行符不同，当该参数设定为true时，则表示使用\n作为换行符

Popen的方法：
    Popen.poll()
    用于检查子进程是否已经结束。None:正在运行 0：进程正常结束  1：出现错误

    Popen.wait()
    等待子进程结束。设置并返回returncode属性。

    Popen.communicate(input=None)
    与子进程进行交互。向 stdin发送数据，可选参数input指定发送到子进程的参数
    或从stdout和stderr中读取数据，如使用PIPE时可以用来读取数据，解决子进程管道数据太多的阻塞：
    　　stdout,stderr = p.communicate()
    Communicate()返回一个元组：(stdoutdata, stderrdata)
    注意：如果希望通过进程的stdin向其发送数据，在创建Popen对象的时候，参数stdin必须被设置为PIPE。
    同样，如 果希望从stdout和stderr获取数据，必须将stdout和stderr设置为PIPE。

    Popen.send_signal(signal)
    向子进程发送信号。

    Popen.terminate()
    停止(stop)子进程。在windows平台下，该方法将调用Windows API TerminateProcess（）来结束子进程。

    Popen.kill()
    杀死子进程。

    Popen.stdin
    如果在创建Popen对象是，参数stdin被设置为PIPE，Popen.stdin将返回一个文件对象用于策子进程发送指令。否则返回None。

    Popen.stdout
    如果在创建Popen对象是，参数stdout被设置为PIPE，Popen.stdout将返回一个文件对象用于策子进程发送指令。否则返回 None。

    Popen.stderr
    如果在创建Popen对象是，参数stdout被设置为PIPE，Popen.stdout将返回一个文件对象用于策子进程发送指令。否则返回 None。

    Popen.pid
    获取子进程的进程ID。

    Popen.returncode




Process
    from multiprocess import process

"""
