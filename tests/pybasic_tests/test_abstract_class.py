#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/30 15:42 

Abstract classes are classes that contain one or more abstract methods.
An abstract method is a method that is declared, but contains no implementation.
Abstract classes cannot be instantiated, and require subclasses to provide implementations for the abstract methods.


Abstract Base Classes (ABCs)

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/30 15:42   wangfc      1.0         None
"""
from abc import ABC,abstractmethod


class AbstractClassExample(ABC):
    """
    A class that is derived from an abstract class cannot be instantiated unless all of its abstract methods are overridden.
    require subclasses to provide implementations for the abstract methods.

    """
    def __init__(self,value):
        self.value = value
        super().__init__()

    @abstractmethod
    def do_something(self):
        pass




