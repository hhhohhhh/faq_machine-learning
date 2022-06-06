#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/17 14:05 


临时文件通常用来保存无法保存在内存中的数据，或者传递给必须从文件读取的外部程序。
一般我们会在/tmp目录下生成唯一的文件名，但是安全的创建临时文件并不是那么简单，需要遵守许多规则。
永远不要自己去尝试做这件事，而是要借助库函数实现。而且也要小心清理临时文件。

创建临时文件一般使用的模块就是tempfile，此模块库函数常用的有以下几个：
tempfile.TemporaryFile # 内存中创建文件，文件不会存储在磁盘，关闭后即删除（可以使用）
tempfile.NamedTemporaryFile(delete=True) 当delete=True时，作用跟上面一样，当是False时，会存储在磁盘（可以使用）
tempfile.TemporaryDirectory()
函数 应该是处理临时文件目录的最简单的方式了，因为它们会自动处理所有的创建和清理步骤。

在一个更低的级别，你可以使用 mkstemp() 和 mkdtemp() 来创建临时文件和目录。
tempfile.mktemp # 不安全，禁止使用
tempfile.mkstemp # 随机创建tmp文件,默认创建的文件在/tmp目录，当然也可以指定（可以使用）
tempfile.mkdtemp

函数 mkstemp() 仅仅就返回一个原始的OS文件描述符，你需要自己将它转换为一个真正的文件对象。 同样你还需要自己清理这些文件。

通常来讲，临时文件在系统默认的位置被创建，比如 /var/tmp 或类似的地方。
为了获取真实的位置，可以使用 tempfile.gettempdir() 函数



@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/17 14:05   wangfc      1.0         None
"""
import tempfile


def test_tempfile():
    for i in range(5):
        temp_file = tempfile.mkdtemp()
        temp_dir = tempfile.gettempdir()
        print(f"i={i},temp_file={temp_file},temp_dir={temp_dir}")

