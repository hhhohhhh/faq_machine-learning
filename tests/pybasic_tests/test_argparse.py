#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@time: 2020/11/5 17:32

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/11/5 17:32   wangfc      1.0         None



"""
import argparse


def operation_fun():
    parser = argparse.ArgumentParser(description="operation_fun : make some operation(default: square) to num ")
    # 我们add_argument() 方法，增加了一个位置参数，该方法用于指定程序能够接受哪些命令行选项。
    # 解析 argument,现在调用我们的程序必须要指定一个选项。
    parser.add_argument("num",help="input a num here",type=int)

    # 可选参数
    parser.add_argument("-o","--operation",
                        default='square',
                        help="do an operation of a given number",
                        type=str)
    # 现在，这一选项更多地是一个标志，而非需要接受一个值的什么东西
    # 现在指定了一个新的关键词 action，并赋值为 "store_true"。这意味着，当这一选项存在时，为 args.verbose 赋值为 True。没有指定时则隐含地赋值为 False。
    parser.add_argument("-v","--verbose", help="increase output verbosity",action="store_true")
    # Namespace 对象
    args = parser.parse_args()
    if args.operation =='square':
        answer = args.num ** 2
    if args.verbose:
        print("verbosity turned on")
        print(f"{args.operation} of num {args.num} = {answer}")
    else:
        print(answer)

    return answer

def operation_fun_with_subparsers():
    parser = argparse.ArgumentParser(description="operation_fun_with_subparsers: make some operation(default: square) to num ")

    subparsers = parser.add_subparsers(help='operation sub-command help')
    # 子命令 a 的解析器
    square_subparser = subparsers.add_parser('square', help='square operation subparser')
    square_subparser.add_argument("num", help="input a num here", type=int)

    # 请注意 parse_args() 返回的对象将只包含主解析器和由命令行所选择的子解析器的属性（而没有任何其他子解析器）
    args = parser.parse_args()
    print(args)
    answer = args.num ** 2
    print(f"square of num {args.num} = {answer}")
    return answer



# subparsers = parser.add_subparsers(help='sub-command help')
# # 子命令 a 的解析器
# parser_a = subparsers.add_parser('a', help='a help')
# parser_a.add_argument('--arg', action="store_true",
#                       help='a-argument help')
# # 子命令 b 的解析器
# parser_a = subparsers.add_parser('b', help='b help')
# parser_a.add_argument('--bar', type=int, help='b-argument help',)
#
# args = parser.parse_args(['b','--bar', '100'])
# print(args)
# if hasattr(args,'square') and args.square is not None:
#     answer = args.square ** 2
#     print('square of {} = {}'.format(args.square, answer))
#
# if hasattr(args,'arg'):
#     print('command a arg={}'.format(args.arg))
# if hasattr(args,'bar'):
#     print('command b arg={}'.format(args.bar))

if __name__ == '__main__':
    operation_fun_with_subparsers()