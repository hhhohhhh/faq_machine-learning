#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/4/26 8:56 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/4/26 8:56   wangfc      1.0         None
"""
import os
from parlai.scripts.eval_model import EvalModel

from conf.polyencoder_config_parser import *
from utils.common import init_logger
logger = init_logger(output_dir=OUTPUT_DIR, log_filename=LOG_FILENAME)


if __name__ == '__main__':
    try:
        from conf.polyencoder_config_parser import *
        home_dir = os.environ.get('HOME')
        # model_file ='faq/output/polyencoder_output/polyencoder.checkpoint'
        # model_file = 'faq/output/bert_ranker_biencoder_sentence_pairdata_to_dstc7_output_002/biencoder.checkpoint'
        # model_file = os.path.join(home_dir,MODEL_FILE)

        # fixed_candidates_filename = 'test_fixed_candidates_without_next_candidate.txt' # 'test_fixed_candidates.txt'
        # fixed_candidates_path = os.path.join(DATAPATH,fixed_candidates_filename)
        kwargs = {'task':'sentence_similarity',
                  'datatype':'test',
                  'model_file':MODEL_FILE,
                  'datapath':DATAPATH,
                  # 'dict_file':'/home/wangfc/faq/output/polyencoder_output/polyencoder.checkpoint',
                  # 'dict_tokenizer': 'bpe',
                  # 'dict_lower': True,
                  'data_parallel':DATA_PARALLEL,
                  'fp16': FP16,
                  'gpu':1,
                  'report_filename': '.report',
                  'display_examples':True,
                  # torch_ranker_agent 对 scores 进行排序： rank_top_k 或者 全部排序
                  'rank_top_k':100,
                  # torch_ranker_agent 设置 预测的时候,输出为 max or topk
                  # 'inference':'topk',
                  # preds 返回个数： max的时候只返回最大的那个，topk返回topk个preds
                  # 'topk':100,
                  # candidates source 的设置
                  'eval_candidates':'fixed',
                  # 设置 fixed-candidates-path 的路径
                  'fixed_candidates_path':FIXED_CANDIDATES_PATH

                  }


        EvalModel.main(**kwargs)
    except Exception as e:
        logger.error(e,exc_info=True)