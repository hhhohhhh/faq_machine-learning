#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@file: base.py
#@version: 02
#@author: $ {USER}
#@contact: wangfc27441@***
#@time: 2019/10/22 17:24 
#@desc:

from tornado.web import RequestHandler

class BaseHandler(RequestHandler):
    """
    Tornado 的 Web 程序会将 URL 或者 URL 范式映射到 tornado.web.RequestHandler 的子类上去。
    在其子类中定义了 get() 或 post() 方法，用以处理不同的 HTTP 请求。
    """
    pass

class NotFoundHandler(BaseHandler):
    pass
