#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/4/13 9:42 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/4/13 9:42   wangfc      1.0         None

"""

"""
--help 选项，也可缩写为 -h，是唯一一个可以直接使用的选项（即不需要指定该选项的内容）。指定任何内容都会导致错误。即便如此，我们也能直接得到一条有用的用法信息。

"""
import argparse


def test_parser():
    parser = argparse.ArgumentParser()
    # 位置参数
    # add_argument() 方法，该方法用于指定程序能够接受哪些命令行选项,
    # help参数：增加解释
    # type参数： argparse 会把我们传递给它的选项视作为字符串，除非我们告诉它别这样
    parser.add_argument("square", help='对数值做平方', type=int)

    # 可选参数介绍
    # 默认情况下如果一个可选参数没有被指定，它的值会是 None，并且它不能和整数值相比较
    parser.add_argument('-m', "--model", default='polyencoder', help='设置模型名称')

    # 关键词 action，并赋值为 "store_true"。这意味着，当这一选项存在时，为 args.verbose 赋值为 True。没有指定时则隐含地赋值为 False
    # 当你为其指定一个值时，它会报错，符合作为标志的真正的精神。
    parser.add_argument('-v', "--verbose", help="increase output verbosity",
                        action="store_true")

    # choices: 限制
    parser.add_argument("--no_cuda", help="设置cuda", type=int, choices=[0, 1]
                        )
    # 解析参数
    # 可以看出 parse_args()函数返回的是一个命名空间（NameSpace），这个NameSpace中有一些变量，就是我们add_argument()的那些参数。
    args = parser.parse_args(['10', '-m', 'biencoder', '-v'])

    # parse_known_args()返回的是一个有两个元素的元组，第一个元素是NameSpace，和parge_args()返回的NameSpace完全相同，第二个是一个列表
    parser.parse_known_args(['10', '--model-file'])

    answer = args.square ** 2
    if args.verbose:
        print("verbosity turned on")
        print(f'args={args},square**2= {answer}')
    else:
        print(answer)


def test_subparsers():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')
    # 子命令 a 的解析器
    parser_a = subparsers.add_parser('a', help='a help')
    parser_a.add_argument('--arg', action="store_true",
                          help='a-argument help')
    # 子命令 b 的解析器
    parser_a = subparsers.add_parser('b', help='b help')
    parser_a.add_argument('--bar', type=int, help='b-argument help', )

    args = parser.parse_args(['b', '--bar', '100'])
    print(args)
    if hasattr(args, 'square') and args.square is not None:
        answer = args.square ** 2
        print('square of {} = {}'.format(args.square, answer))

    if hasattr(args, 'arg'):
        print('command a arg={}'.format(args.arg))
    if hasattr(args, 'bar'):
        print('command b arg={}'.format(args.bar))
if __name__ == '__main__':
    test_parser()


