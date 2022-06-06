#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/13 9:36 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/13 9:36   wangfc      1.0         None
"""


from typing import Tuple, Text, Dict, Optional, Any
import tornado
from tornado.options import define,options
import logging
from tornado.web import RequestHandler
from utils.host import get_host_ip

logger = logging.getLogger(__name__)


class Application(tornado.web.Application):
    """
    @author:wangfc27441
    @time:  2019/10/25  11:25
    @desc: 创建 Tornado 的 Application 类
    """
    # 初始化 Application 对象
    def __init__(self,url_handler_tuple:Tuple[Text,RequestHandler], settings:Dict[Text,Text]):
        super(Application,self).__init__(url_handler_tuple,**settings)



class TornadoServer():
    """
    @time:  2022/1/13 9:48
    @author:wangfc
    @version:
    @description:

    @params:
    @return:
    """

    def __init__(self,url:Text,port:int,gpu_memory_config="0:5120", mode=None,autoreload=True,
                 num_processes = 1,*args,**kwargs):
        self.mode =mode
        self.url =url
        self.port =port
        self.autoreload = autoreload

        # 启动单进程服务
        self.num_processes = num_processes
        self.args =args
        self.kwargs =kwargs

        # 设置 tensorflow 环境变量
        from utils.tensorflow.environment import setup_tf_environment
        setup_tf_environment(gpu_memory_config=gpu_memory_config)


    def run(self):
        # 设置 setting
        settings = self._get_settings()
        # 1. 初始化 路由信息映射元组的列表
        url_handler_tuple = self._build_url_handler_tuple()

        # 2. 构建一个 Application 对象的实例: 初始化接收的第一个参数就是一个路由信息映射元组的列表
        application =self._build_app(url_handler_tuple=url_handler_tuple,settings=settings)

        # 定义 HTTPServer 监听HTTP请求的端口
        define('port', default=self.port, help='run on the given port', type=int)

        # 使用 tornado 的 options 模块解析命令行：转换命令行参数，并将转换后的值对应的设置到全局options对象相关属性上
        tornado.options.parse_command_line()

        # 3. 创建HTTPServer 对象：将 Application实例传递给 Tornado 的  HTTPServer 对象
        http_server = tornado.httpserver.HTTPServer(application)

        # 4. 启动IOLoop
        hostname, ip = get_host_ip()
        logger.info(
            f'启动 tornado 应用: num_processes={self.num_processes} 进程 with at url={ip}:{self.port}{self.url}')
        if self.num_processes==1:
            # 服务器绑定到指定端口
            http_server.bind(options.port)  # http_server.listen()这个方法只能在单进程模式中使用
            # 开启num_processes进程（默认值为1）
            http_server.start(num_processes= self.num_processes)
            # 启动当前线程的IOLoop
            tornado.ioloop.IOLoop.current().start()
            # tornado.ioloop.IOLoop.instance().start()
        else:
            # 启动高级多进程 HTTPServer 服务
            sockets = tornado.netutil.bind_sockets(options.port)
            # 启动多进程
            tornado.process.fork_processes(num_processes=self.num_processes)
            http_server.add_sockets(sockets)
            # IOLoop.current() : 返回当前线程的IOLoop实例
            # IOLoop.start():  启动IOLoop实例的I / O循环, 同时服务器监听被打开。
            tornado.ioloop.IOLoop.current().start()




    def _get_settings(self)-> Dict[Text,Text]:
        # 初始化 web.Application 的配置
        settings = dict(
            title='HSNLP',
            autoreload=self.autoreload,  # 自动加载主要用于开发和测试阶段，要不每次修改，都重启tornado服务 ,
            debug=self.mode,  # 通过 option 参数 debug 来选择是否 autoreload
            login_url='/login'
        )
        return settings

    def _build_url_handler_tuple(self) -> Tuple[Text,RequestHandler,Optional[Dict[Text,Any]]]:
        """
        @author:wangfc27441
        @time:  2019/10/25  11:22
        @desc:
        url + handler + initialize params
        路由信息映射元组的列表，用于Application对象初始化:
        每个handler 处理对应特定HTTP请求的响应 :

        Tornado 的 Web 程序会将 URL 或者 URL 范式映射到 tornado.web.RequestHandler 的子类上去。
        在其子类中定义了 get() 或 post() 方法，用以处理不同的 HTTP 请求。

        在父类中定义的私有方法，其作用范围仅在当前类，若在子类中重写，实际上并不会起效果，
        原因：以双下划线前缀开头的属性或方法，Python解释器会重写其名称，以便在类被扩展的时候不容易产生冲突，这被称之为名称修饰(name mangling)

        :return: url + handler + initialize params:
        """
        raise NotImplementedError


    def _build_app(self,url_handler_tuple:Tuple[Text,RequestHandler], settings:Dict[Text,Text]):
        app = Application(url_handler_tuple=url_handler_tuple,settings=settings)
        return app

def main():
    # 设置环境变量
    tornado_server = TornadoServer()
    tornado_server.run()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(e,exc_info=True)