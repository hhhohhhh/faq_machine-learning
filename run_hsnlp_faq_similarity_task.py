#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/30 14:07 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/30 14:07   wangfc      1.0         None
"""
from hsnlp_faq_similarity.hsnlp_faq_similarity_task import HsnlpFaqSimilarityTask

if __name__ == '__main__':
    from utils.common import init_logger
    from utils.tensorflow.environment import setup_tf_environment
    setup_tf_environment("-1:5120")
    logger = init_logger(output_dir='output/hsnlp_faq_similarity_task')
    hanlp_faq_similarity_task = HsnlpFaqSimilarityTask()
    while True:
        # input_query = input()
        input_query = '我想开通创业板'
        recall_question, recall_std_question, recall_std_id, recall_score, \
        yy_filter, ss_filter, yy_std_raw, recall_logits = \
        hanlp_faq_similarity_task.similar_calculate_process(query=input_query)