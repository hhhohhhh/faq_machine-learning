#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/25 15:20 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/25 15:20   wangfc      1.0         None
"""
import tensorflow as tf

def trace_graph_to_tensorboad(logdir,module:tf.Module,inputs=None,step=0,name="my_func_trace"):
    # Set up logging.
    # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # logdir = "logs/func/%s" % stamp
    writer = tf.summary.create_file_writer(logdir)

    # Create a new model to get a fresh trace
    # Otherwise the summary will not see the graph.
    model = module()

    # Bracket the function call with
    # tf.summary.trace_on() and tf.summary.trace_export().
    tf.summary.trace_on(graph=True)
    tf.profiler.experimental.start(logdir)
    # Call only one tf.function when tracing.
    # inputs = tf.constant([[2.0, 2.0, 2.0]])
    z = print(model(inputs))
    with writer.as_default():
        tf.summary.trace_export(
            name=name,
            step=step,
            profiler_outdir=logdir)
