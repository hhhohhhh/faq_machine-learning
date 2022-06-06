#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/8/6 14:05 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/6 14:05   wangfc      1.0         None

pip install ipywidgets_bokeh

"""

import pathlib
import numpy as np
import pandas as pd
import tornado
from bokeh.server.server import Server
from tornado.options import define, options
from tornado.web import Application

from data_process.dataset.intent_classifier_dataset import CombinedRawIntentDataset
from data_process.data_processing import remove_punctuation


import ipywidgets as widgets

from whatlies import EmbeddingSet
from whatlies.transformers import Pca, Umap
from hulearn.preprocessing import InteractivePreprocessor
from hulearn.experimental.interactive import InteractiveCharts
from whatlies.language import UniversalSentenceLanguage, LaBSELanguage, SentenceTFMLanguage
from bokeh.resources import INLINE, CDN
from IPython.core.display import display

import logging
logger = logging.getLogger(__name__)

# Here's the global state object
class BulkLabelingUI():
    def __init__(self, notebook_url='10.20.33.3:8888',sample_size=20,lang=None,
                 raw_intent_data:pd.DataFrame=None,df:pd.DataFrame=None,embset=None
                 ):
        self.notebook_url = notebook_url
        self.sample_size = sample_size
        self.lang = SentenceTFMLanguage('paraphrase-multilingual-mpnet-base-v2')
        self.raw_intent_data = raw_intent_data
        self.df = df
        self.embset = embset


    def prepare_data(self):
        self.raw_intent_data = self.read_raw_data()
        self.df,self.embset = self.sample_data(raw_intent_data=self.raw_intent_data)


    def read_raw_data(self):
        from conf.config_parser import VOCAB_FILE, STOPWORDS_PATH
        combined_raw_intent_data = CombinedRawIntentDataset(output_data_subdir='intent_classifier_data_20210812',
                                                            vocab_file=VOCAB_FILE)
        raw_intent_data = combined_raw_intent_data.raw_intent_data
        logger.info(f"读取数据共 {raw_intent_data.shape} 条")

    def sample_data(self,raw_intent_data, sample_size = 2000):
        test_data = raw_intent_data.sample(sample_size, random_state=1234)
        texts = test_data.loc[:, 'question'].apply(lambda x: remove_punctuation(x))
        texts.drop_duplicates(inplace=True)
        texts = texts.loc[~(texts == '')].copy()
        logger.info(f"We're going to plot {len(texts)} texts.")

        # This is where we prepare all of the state
        embset = self.lang[texts]
        df = embset.transform(Umap(2)).to_dataframe().reset_index()
        df.columns = ['text', 'd1', 'd2']
        df['label'] = ''
        df.index = texts.index
        return df,embset


    def create_bkapp(self):
        df = self.df.copy()
        pd.set_option('display.max_colwidth', None)
        # Here's the global state object
        interactive_charts = InteractiveCharts(df.loc[lambda d: d['label'] == ''], labels=['group'], resources=INLINE,
                                                notebook_url=self.notebook_url)
        chart = interactive_charts.add_chart(x='d1', y='d2', legend=False)
        return chart.app


    def build_ui_on_notebook(self):

        df = self.df.copy()
        pd.set_option('display.max_colwidth', None)
        # Here's the global state object
        state = {}
        state['df'] = df
        state['chart'] = InteractiveCharts(df.loc[lambda d: d['label'] == ''], labels=['group'], resources=INLINE,
                                                notebook_url=self.notebook_url)

        out_table = widgets.Output()
        out_chart = widgets.Output()
        out_counter = widgets.Output()
        label_name = widgets.Text("label name")

        # 定义 button
        btn_examples = widgets.Button(
            description='Show Examples',
            icon='eye'
        )

        btn_label = widgets.Button(
            description='Add label',
            icon='check'
        )

        btn_retrain = widgets.Button(
            description='Retrain',
            icon='coffee'
        )

        btn_redraw = widgets.Button(
            description='Redraw',
            icon='check'
        )

        def show_draw_chart(b=None):
            with out_table:
                out_table.clear_output()
            with out_chart:
                out_chart.clear_output()
                state['chart'].dataf = state['df'].loc[lambda d: d['label'] == '']
                state['chart'].charts = []
                chart = state['chart'].add_chart(x='d1', y='d2', legend=False)
                chart.show(notebook_url=self.notebook_url)

        def show_examples(b=None):
            with out_table:
                out_table.clear_output()
                tfm = InteractivePreprocessor(json_desc=state['chart'].data())
                subset = state['df'].pipe(tfm.pandas_pipe).loc[lambda d: d['group'] != 0]
                display(subset.sample(min(15, subset.shape[0]))[['text']])

        def assign_label(b=None):
            tfm = InteractivePreprocessor(json_desc=state['chart'].data())
            idx = state['df'].pipe(tfm.pandas_pipe).loc[lambda d: d['group'] != 0].index
            state['df'].iloc[idx, 3] = label_name.value
            with out_counter:
                out_counter.clear_output()
                n_lab = state['df'].loc[lambda d: d['label'] != ''].shape[0]
                print(f"{n_lab}/{state['df'].shape[0]} labelled")

        def retrain_state(b=None):
            keep = list(state['df'].loc[lambda d: d['label'] == '']['text'])
            umap = Umap(2)
            new_df = EmbeddingSet(*[e for e in self.embset if e.name in keep]).transform(
                umap).to_dataframe().reset_index()
            new_df.columns = ['text', 'd1', 'd2']
            new_df['label'] = ''
            state['df'] = pd.concat([new_df, state['df'].loc[lambda d: d['label'] != '']])
            show_draw_chart(b)

        btn_examples.on_click(show_examples)
        btn_label.on_click(assign_label)
        btn_redraw.on_click(show_draw_chart)
        btn_retrain.on_click(retrain_state)

        show_draw_chart()
        display(widgets.VBox([widgets.HBox([btn_retrain, btn_examples, btn_redraw]),
                              widgets.HBox([out_chart, out_table])]),
                label_name,
                widgets.HBox([btn_label, out_counter]))

        return state



def main(host_ip='10.20.33.3', tornado_server_port=8000, bokeh_server_port=5006):
    define("port", default=tornado_server_port, help="run on the given port", type=int)
    # 生成 tornado_application
    application = Application()
    # 建立 http_server
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(options.port)
    # 获取当前 的 IOLoop
    io_loop = tornado.ioloop.IOLoop.current()

    # bokeh_app = bokeh.application.Application(FunctionHandler(MainHandler.modify_doc))
    bulk_labeler = BulkLabelingUI()
    bulk_labeler.prepare_data()
    bkapp = bulk_labeler.create_bkapp()


    bokeh_server = Server({'/bkapp': bkapp},
                          io_loop=io_loop,
                          port=bokeh_server_port,
                          # extra_patterns=[('/', IndexHandler)],
                          allow_websocket_origin=[f'{host_ip}:{bokeh_server_port}'])
    bokeh_server.start() # start_loop=False
    # io_loop.add_callback(view, f"http://{host_ip}:{tornado_server_port}")
    io_loop.start()