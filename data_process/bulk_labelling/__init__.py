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
"""
import pathlib
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.core.display import display

from whatlies import EmbeddingSet
from whatlies.transformers import Pca, Umap

from hulearn.preprocessing import InteractivePreprocessor
from hulearn.experimental.interactive import InteractiveCharts

state = {}


def show_draw_chart(b=None):
    with out_table:
        out_table.clear_output()
    with out_chart:
        out_chart.clear_output()
        state['chart'].dataf = state['df'].loc[lambda d: d['label'] == '']
        state['chart'].charts = []
        state['chart'].add_chart(x='d1', y='d2', legend=False)

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
    new_df = EmbeddingSet(*[e for e in embset if e.name in keep]).transform(umap).to_dataframe().reset_index()
    new_df.columns = ['text', 'd1', 'd2']
    new_df['label'] = ''
    state['df'] = pd.concat([new_df, state['df'].loc[lambda d: d['label'] != '']])
    show_draw_chart(b)



if __name__ == '__main__':
    out_table = widgets.Output()
    out_chart = widgets.Output()
    out_counter = widgets.Output()
    label_name = widgets.Text("label name")

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

    btn_examples.on_click(show_examples)
    btn_label.on_click(assign_label)
    btn_redraw.on_click(show_draw_chart)
    btn_retrain.on_click(retrain_state)


    show_draw_chart()
    display(widgets.VBox([widgets.HBox([btn_retrain, btn_examples, btn_redraw]),
                          widgets.HBox([out_chart, out_table])]),
            label_name,
            widgets.HBox([btn_label, out_counter]))