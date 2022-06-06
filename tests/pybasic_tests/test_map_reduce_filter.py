#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/8/20 17:44 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/20 17:44   wangfc      1.0         None
"""
def test_map():
    def square(x):
        return x*x

    def multiply(x,y):
        return x*y

    ls = [1,2,3]
    result = map(square,ls)
    print(result)
    assert list(result) == [1,4,9]

    from functools import reduce
    result = reduce(multiply,ls)
    assert result == 6

if __name__ == '__main__':
    test_map()