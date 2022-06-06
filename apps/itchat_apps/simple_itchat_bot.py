#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/10/22 16:39 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/22 16:39   wangfc      1.0         None
"""

#coding=utf8
import itchat
from itchat.content import  TEXT,PICTURE,RECORDING,CARD

# tuling plugin can be get here:
# https://github.com/littlecodersh/EasierLife/tree/master/Plugins/Tuling
# from tuling import get_response


@itchat.msg_register(TEXT)
def text_reply(msg):
    """
    想要自动接收消息，需要先对不同类型的消息进行注册，如果没有注册，对应类型的消息将不会被接收.
    注册为处理文本消息的函数。
    文本     itchat.content.Text
    图片对应 itchat.content.PICTURE
    语音对应 itchat.content.RECORDING
    名片对应 itchat.content.CARD

    itchat.send():
    消息的内容与接受者的UserName，即标识符。
    文件传输助手:

    """

    if u'作者' in msg[TEXT] or u'主人' in msg[TEXT]:
        return u'你可以在这里了解他：https://github.com/littlecodersh'
    elif u'源代码' in msg[TEXT] or u'获取文件' in msg[TEXT]:
        itchat.send('@fil@main.py', msg['FromUserName'])
        return u'这就是现在机器人后台的代码，是不是很简单呢？'
    elif u'获取图片' in msg[TEXT]:
        itchat.send('@img@applaud.gif', msg['FromUserName']) # there should be a picture
    else:
        return u"收到信息"
        # return get_response(msg[TEXT]) or u'收到：' + msg[TEXT]

@itchat.msg_register(['Picture', 'Recording', 'Attachment', 'Video'])
def atta_reply(msg):
    return ({ 'Picture': u'图片', 'Recording': u'录音',
        'Attachment': u'附件', 'Video': u'视频', }.get(msg['Type']) +
        u'已下载到本地') # download function is: msg[TEXT](msg['FileName'])

@itchat.msg_register(['Map', 'Card', 'Note', 'Sharing'])
def mm_reply(msg):
    if msg['Type'] == 'Map':
        return u'收到位置分享'
    elif msg['Type'] == 'Sharing':
        return u'收到分享' + msg[TEXT]
    elif msg['Type'] == 'Note':
        return u'收到：' + msg[TEXT]
    elif msg['Type'] == 'Card':
        return u'收到好友信息：' + msg[TEXT]['Alias']

# @itchat.msg_register(TEXT, isGroupChat = True)
# def group_reply(msg):
#     if msg['isAt']:
#         return u'@%s\u2005%s' % (msg['ActualNickName'],
#             get_response(msg[TEXT]) or u'收到：' + msg[TEXT])

@itchat.msg_register('Friends')
def add_friend(msg):
    itchat.add_friend(**msg[TEXT])
    itchat.send_msg(u'项目主页：github.com/littlecodersh/ItChat\n'
        + u'源代码  ：回复源代码\n' + u'图片获取：回复获取图片\n'
        + u'欢迎Star我的项目关注更新！', msg['RecommendInfo']['UserName'])

if __name__ == '__main__':
    # hotReload==Tru:下次登录不用再扫二维码
    # enableCmdQR=2: 如部分的linux系统，块字符的宽度为一个字符（正常应为两字符），故赋值为2
    import os
    pic_dir = os.path.join(os.getcwd(),"qr_code.png")
    print(f"pic_dir={pic_dir}")
    itchat.auto_login(hotReload=True,  enableCmdQR=2,picDir=pic_dir)
    itchat.run()