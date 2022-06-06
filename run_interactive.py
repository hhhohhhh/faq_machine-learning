#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/4/9 16:35 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/4/9 16:35   wangfc      1.0         None

"""
import os
from parlai.scripts.interactive import Interactive

if __name__ == '__main__':
    from conf.polyencoder_config_parser import *
    home_dir = os.environ.get('HOME')
    # model_file ='faq/output/polyencoder_output/polyencoder.checkpoint'
    # model_file = 'faq/output/bert_ranker_biencoder_sentence_pairdata_to_dstc7_output_002/biencoder.checkpoint'
    # model_file = os.path.join(home_dir,model_file)
    kwargs = {'model_file':MODEL_FILE,
              'datapath':DATAPATH,
              # 'dict_file':'/home/wangfc/faq/output/polyencoder_output/polyencoder.checkpoint',
              # 'dict_tokenizer': 'bpe',
              # 'dict_lower': True,
              'data_parallel':DATA_PARALLEL,
              'fp16': FP16,
              'gpu':GPU,
              # candidates source 的设置
              'eval_candidates': 'fixed',
              # 设置 fixed-candidates-path 的路径
              'fixed_candidates_path': FIXED_CANDIDATES_PATH,
              # # torch_ranker_agent 设置 预测的时候,输出为 max or topk
              # 'inference':'topk',
              # # preds 返回个数： max的时候只返回最大的那个，topk返回topk个preds
              # 'topk':10,
              'display_add_fields':'text_candidates'
              }


    Interactive.main(**kwargs)
