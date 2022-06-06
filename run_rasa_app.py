#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file:
@version:
@desc:
@time: 2021/8/13 14:11

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/13 14:11   wangfc      1.0         None
"""


def run_bulk_labeling_server():
    # from visualization.bokeh.server_embed.tonado_server.main_tornado import main
    # from visualization.bokeh.server_embed.tornado_embed import main,bk_workder
    # from visualization.bokeh.server_embed.flask_embed import main
    from data_process.bulk_labelling.bulk_labeling_ui import main
    tornado_server_port = 8003
    bokeh_server_port = 5007
    print(f"启动 tornado_app_port={tornado_server_port}，bokeh_app_port={bokeh_server_port} ")
    # bk_workder(tornado_server_port=tornado_server_port,bokeh_server_port = bokeh_server_port)
    main(tornado_server_port=tornado_server_port, bokeh_server_port=bokeh_server_port)


if __name__ == '__main__':
    from utils.tensorflow.environment import setup_tf_environment
    GPU_MEMORY_CONFIG = "0:1024"
    setup_tf_environment(GPU_MEMORY_CONFIG)
    from apps.rasa_apps.rasa_application import  RasaApplication
    from conf.config_parser import *
    from utils.common import init_logger
    logger = init_logger(output_dir=OUTPUT_DIR,log_level='debug')
    rasa_application = RasaApplication(taskbot_dir=OUTPUT_DIR,
                                       config_filename=RASA_CONFIG_FILENAME,
                                       models_subdir=RASA_MODELS_SUBDIRNAME,
                                       num_of_workers=NUM_OF_WORKERS,
                                       gpu_memory_per_worker=GPU_MEMORY_CONFIG,
                                       agent_port=5005,
                                       if_build_nlu_application=False,
                                       if_build_agent_application=True,
                                       if_enable_api=False,
                                       output_with_nlu_parse_data=True)
