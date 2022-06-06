#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/3/9 16:03 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/9 16:03   wangfc      1.0         None

"""
import shutil
import os
from pathlib import Path
from typing import Text, Dict, Any, Optional
import pandas as pd
from matplotlib.font_manager import json_dump

from rasa.utils.io import pickle_dump, pickle_load
from sentence_transformers import SentenceTransformer
import torch
from apps.retrieval_apps.retrieval_application import RetrievalAgent
from utils.faiss_utils import FaissIndexer
from utils.io import get_file_stem
from utils.torch.environment import load_model_to_device


class SemanticRetrievalAgent(RetrievalAgent):
    """
    (1) index documents (ie store them in an easily retrievable form)
    (2) vectorise text data and
    (3) measure how relevant a document is to a query.
    """

    def __init__(self,component_config: Optional[Dict[Text, Any]] = None,
                 model:SentenceTransformer=None,indexer:FaissIndexer=None):

        self.pretrained_model_dir = 'pretrained_model'
        self.model_prefix = "sentence-transformers"
        self.semantic_model_name = component_config.get("semantic_model_name", "all-MiniLM-L6-v2")
        self.model_subdir = 'models'
        self.meta_data_filename = 'meta_data.json'
        self.semantic_model_name_or_path_filename = 'semantic_model_name_or_path.json'
        self.model = model
        self.indexer = indexer

    def _load_semantic_similarity_model(self)->SentenceTransformer:
        model_name_or_path =  Path(self.pretrained_model_dir)/f"{self.model_prefix}_{self.semantic_model_name}"
        if model_name_or_path.exists():
            sentence_transformer = SentenceTransformer(model_name_or_path=model_name_or_path)
        else:
            sentence_transformer = SentenceTransformer(model_name_or_path=self.semantic_model_name_or_path,
                                                       cache_folder=self.pretrained_model_dir
                                                       )
        load_model_to_device(sentence_transformer)
        return sentence_transformer


    def train(self,document_path):
        # 加载相似度模型
        self.model = self._load_semantic_similarity_model()
        # 读取 documents
        data_df = self._get_documents(document_path)

        # 转换为 embeddings
        documents = data_df.loc[:,'document'].values
        document_embeddings = self.model.encode(sentences=documents)
        # 转换为新的 数据
        document_data = data_df.copy()
        # document_data['embedding'] = document_embeddings
        # 建立 indexer
        faiss_indexer = FaissIndexer()
        # 训练 indexer
        faiss_indexer.train(document_data=document_data,embeddings = document_embeddings )
        self.indexer = faiss_indexer


    def _get_documents(self,document_path,new_columns= ['original_title', 'document', 'year', 'citations', 'id', 'is_EN']):
        df = pd.read_csv(document_path)
        df.columns = new_columns
        return df


    def persist(self,output_dir):
        if Path(output_dir).exists():
            shutil.rmtree(output_dir)

        model_dir = Path(output_dir) / self.model_subdir
        meta_data_path = Path(output_dir) /self.meta_data_filename
        meta = self.indexer.persist(model_dir=model_dir)
        file_name = meta.get("file")
        # semantic_model_name_or_path_file = Path(model_dir) / f"{file_name}.{self.semantic_model_name_or_path_filename}.pkl"

        semantic_model_name = get_file_stem(self.semantic_model_name_or_path)
        semantic_model_file = Path(model_dir) / f"{file_name}.{semantic_model_name}.pkl"
        pickle_dump(filename=semantic_model_file,obj=self.model)
        # 将 semantic_model_name_or_path 加入 meta 作为返回
        meta.update({"semantic_model_name_or_path":semantic_model_name})
        json_dump(data=meta,filename=meta_data_path)
        return meta

    @classmethod
    def load(cls,meta: Dict[Text, Any],output_dir: Text,):
        model_dir = Path(output_dir)/ 'models'
        indexer = FaissIndexer.load(meta=meta,model_dir=model_dir)
        file_name = meta.get("file")
        semantic_model_name_or_path = meta.get("semantic_model_name_or_path")
        semantic_model_name = get_file_stem(semantic_model_name_or_path)
        semantic_model_file = Path(model_dir) / f"{file_name}.{semantic_model_name}.pkl"
        model = pickle_load(filename=semantic_model_file)
        return cls(meta,model=model,indexer=indexer)


    def process(self,query:Text,top_k=10):
        query_embedding = self.model.encode(sentences=[query])
        results = self.indexer.process(query_embedding,top_k=top_k)
        return results

