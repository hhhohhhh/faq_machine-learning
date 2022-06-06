#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/5/14 14:02 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/14 14:02   wangfc      1.0         None
"""

import jpype
from jpype import java,startJVM,shutdownJVM, getDefaultJVMPath,isJVMStarted

class TestJpype():
    def hello_world(self):
        print("启动 JVM")
        # 判读 JVM 是否已经启动
        if not isJVMStarted():
            #
            startJVM(jpype.getDefaultJVMPath())

        java.lang.System.out.println("Hello World")
        shutdownJVM()
        print("关闭 JVM")

if __name__ == '__main__':
    jpype_testor = TestJpype()
    jpype_testor.hello_world()