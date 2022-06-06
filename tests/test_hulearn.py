#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/8/6 10:52 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/6 10:52   wangfc      1.0         None
"""
if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from hulearn.experimental.interactive import InteractiveCharts

    from bokeh.resources import INLINE, CDN
    import bokeh.io
    # 在notbook中展示
    bokeh.io.output_notebook(resources=INLINE)


    data = np.random.random((5,2))
    df = pd.DataFrame(data=data,columns=['d1','d2'])
    df.loc[:,'label']=''

    charts = InteractiveCharts(df.loc[lambda d: d['label'] == ''], labels=['group'])
    charts.add_chart(x='d1', y='d2')