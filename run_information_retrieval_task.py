#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/15 23:15 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/15 23:15   wangfc      1.0         None
"""


def run_semantic_similarity_task():
    from conf.semantic_similarity_config_parser import CORPUS
    from data_process.dataset.sentence_pair_data_reader import SentencePairDataReader
    from tasks.semantic_textual_similarity_task import SemanticTextualSimilarityTask


    # 读取并转换原始数据： 蚂蚁金服的数据集 和 相似度
    data_files = [{"dataset": 'atec', "data_filenames": ['atec_nlp_sim_train.csv', 'atec_nlp_sim_train_add.csv']}]
                  # {"dataset": 'security_similary_data', "data_filenames": ["六家评估集.xlsx"]}
    sentence_pair_data_reader = SentencePairDataReader(data_dir=CORPUS, data_files=data_files,
                                                       if_read_data=True, if_overwrite=False,
                                                       if_build_triplet_data=False,
                                                       if_build_sentence_label_data=True,
                                                       train_size=0.8, dev_size=0.1, test_size=0.1)


    semantic_similarity_task = SemanticTextualSimilarityTask()

    semantic_similarity_task.train()




def run_semantic_retrieval_agent(mode='train'):

    import os
    from pathlib import Path
    from apps.retrieval_apps.semantic_search import SemanticRetrievalAgent
    output_dir = os.path.join('output', 'semantic_search_app')
    # model_name = "sentence-transformers_all-MiniLM-L6-v2"
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'

    if mode == 'train':
        document_path = os.path.join('examples', 'vector_engine', 'data', 'misinformation_papers.csv')
        # semantic_model_name_or_path = os.path.join('pretrained_model',model_name)
        semantic_retrieval_agent = SemanticRetrievalAgent(component_config=dict(semantic_model_name_or_path=model_name))
        semantic_retrieval_agent.train(document_path=document_path)
        meta_data = semantic_retrieval_agent.persist(output_dir=output_dir)
    elif mode=='evaluate':
        from matplotlib.font_manager import json_load
        meta_data_path = Path(output_dir) / "meta_data.json"
        meta_data = json_load(filename= meta_data_path)
        semantic_retrieval_agent = SemanticRetrievalAgent.load(meta=meta_data, output_dir=output_dir)
        query = 'query'
        results = semantic_retrieval_agent.process(query=query)



def load_sentence_transformer_model(name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    """Instantiate a sentence-level DistilBERT model."""
    return SentenceTransformer(name,cache_folder='pretrained_model')



def train_polyencoder_dstc7():
    from data_process.sentence_pair_data_reader import SentencePairDataReader
    # 蚂蚁金服的数据集
    data_dir = 'data'
    data_files = [{"dataset": 'atec', "data_filenames": ['atec_nlp_sim_train.csv', 'atec_nlp_sim_train_add.csv']},
                  {"dataset": 'security_similary_data', "data_filenames": ["六家评估集.xlsx"]}]

    output_dirname = 'sentence_pair_to_dstc7'
    sentence_pair_data_reader = SentencePairDataReader(data_dir=data_dir, data_files=data_files,
                                                       output_dirname=output_dirname,
                                                       if_overwrite=False,if_convert_to_dstc7=True,
                                                       train_size=0.8, dev_size=0.1, test_size=0.1)


# def train_polyencoder():
    # from conf.parlai_config_parser import BIENCODER_KWARGS
    # from parlai.scripts.train_model import TrainModel
    # # try:
    #     # logger.info(f'{BIENCODER_KWARGS}')
    #     TrainModel.main(**BIENCODER_KWARGS)
        # logger.info(f"训练完成")
    # except Exception as e:
        # logger.error(e, exc_info=True)

def run_parlai_task():
    """
    parlai interactive_web --model-file "zoo:tutorial_transformer_generator/model"
    """
    from conf.parlai_config_parser import PARLAI_MODE,ZOO_MODEL, MODEL_NAME,DATAPATH,OUTPUT_DIR
    from parlai.__main__ import superscript_main as parlai_main

    argv = [PARLAI_MODE]
    if PARLAI_MODE =='interactive_web':
        argv.extend(['--model-file', ZOO_MODEL,'--datapath',OUTPUT_DIR,
                     '--host','0.0.0.0', '--port',"5001"])

    parlai_main(argv)


if __name__ == '__main__':

    # run_semantic_retrieval_task(mode='train')
    run_parlai_task()