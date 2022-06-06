#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/16 8:54 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/16 8:54   wangfc      1.0         None

It should not be so：
pip install faiss

should do：
CPU version only
conda install faiss-cpu -c pytorch

GPU version
conda install faiss-gpu cudatoolkit=8.0 -c pytorch # For CUDA8
conda install faiss-gpu cudatoolkit=9.0 -c pytorch # For CUDA9
conda install faiss-gpu cudatoolkit=10.0 -c pytorch # For CUDA10


"""
import os
from pathlib import Path
from typing import Union, Text, Any, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import faiss
import pickle

import rasa
from rasa.shared.nlu.training_data.message import Message

from rasa.nlu.config import RasaNLUModelConfig

from rasa.shared.nlu.training_data.training_data import TrainingData

from utils.io import dump_json, load_json
from rasa.nlu.components import Component
import logging
logger = logging.getLogger(__name__)


class FaissIndexer(Component):
    """
    To create an index with the misinformation abstract vectors, we will:
    1. Change the data type of the abstract vectors to float32.
    2. Build an index and pass it the dimension of the vectors it will operate on.
    3. Pass the index to IndexIDMap, an object that enables us to provide a custom list of IDs for the indexed vectors.
    4. Add the abstract vectors and their ID mapping to the index. In our case, we will map vectors to their paper IDs from MAG.

    """
    def __init__(self, component_config: Optional[Dict[Text, Any]] = {},
                 id_to_document_mapping: Dict[Text,Any] = None,index:faiss.IndexIDMap =None):
        super(FaissIndexer, self).__init__(component_config)
        self.faiss_index_type =  component_config.get('faiss_index_type', 'IndexFlatL2')
        self.id_to_document_mapping_filename = component_config.get('id_to_document_mapping_filename', 'id_to_document_mapping.json')
        self.faiss_indexer_pickle_filename = component_config.get('id_to_document_mapping_filename', 'faiss_indexer.pkl')
        self.id_to_document_mapping = id_to_document_mapping or {}
        self.index = index


    def train(
        self,
        document_data:pd.DataFrame,
        embeddings:np.ndarray,
        training_data: TrainingData=None,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        self.id_to_document_mapping, self.index = self._build_index(document_data=document_data,embeddings=embeddings)
        # self.persist(id_to_document_mapping, index)



    def process(self, query_embedding : np.ndarray,top_k=10, message: Message=None, **kwargs: Any) -> None:
        # Retrieve the 10 nearest neighbours

        distances, ids = self.index.search(np.array(query_embedding).astype("float32"), k=top_k)
        print(f'L2 distance: {distances.flatten().tolist()}\n\nMAG paper IDs: {ids.flatten().tolist()}')
        results ={index: {'id':str(id),"document":self.id_to_document_mapping[str(id)], "distance":distance}
                    for index,(id,distance) in enumerate(zip(ids.flatten().tolist(),distances.flatten().tolist()))}
        return results



    def _extract_ids(self,document_data)-> np.ndarray:
        ids = document_data.loc[:,'id'].values
        return ids

    def _extract_documents(self,document_data):
        documents = document_data.loc[:, 'document'].values
        return documents

    # def _extract_embeddings(self,document_data):
    #     embeddings = document_data.loc[:, 'embedding'].values
    #     return embeddings

    def _get_id_to_document_mapping(self,document_data):
        id_to_document_mapping = {id:document for id,document in document_data.loc[:,['id','document']].values}
        return id_to_document_mapping


    def _extract_data(self,document_data):
        self.ids = self._extract_ids(document_data)
        self.documents = self._extract_documents(document_data)
        # self.embeddings = self._extract_embeddings()
        id_to_document_mapping = self._get_id_to_document_mapping(document_data)
        return id_to_document_mapping



    def get_index(self):
        if os.path.exists(self.persist_dir):
            id_to_document_mapping,index = self.load(self.persist_dir)
        else:
            id_to_document_mapping, index = self.build_index()
        return id_to_document_mapping,index



    def _build_index(self,document_data:pd.DataFrame,embeddings:np.ndarray) -> \
            Tuple[Dict[Text,Text],faiss.IndexIDMap]:
        id_to_document_mapping = self._extract_data(document_data)
        # Step 1: Change data type
        embeddings = np.array([embedding for embedding in embeddings]).astype("float32")

        # Step 2: Instantiate the index
        index = faiss.IndexFlatL2(embeddings.shape[1])

        # Step 3: Pass the index to IndexIDMap
        index = faiss.IndexIDMap(index)

        # Step 4: Add vectors and their IDs
        index.add_with_ids(embeddings, self.ids)

        print(f"Number of vectors in the Faiss index: {index.ntotal}")
        return id_to_document_mapping, index

    def persist(self, model_dir: Text,file_name: Text="faiss_indexer" ) -> Optional[Dict[Text, Any]]:
        """
        保存 id_to_document_mapping 和 index 相关信息
        """
        id_to_document_mapping_filename = f"{file_name}.{self.id_to_document_mapping_filename}"
        id_to_document_mapping_path = os.path.join(model_dir,id_to_document_mapping_filename)
        faiss_indexer_pickle_filename = f"{file_name}.{self.faiss_indexer_pickle_filename}"
        index_path = os.path.join(model_dir,faiss_indexer_pickle_filename)

        dump_json(self.id_to_document_mapping, id_to_document_mapping_path)
        pickle_dump_faiss_index(filename=index_path,index=self.index)

        # 返回 模型的meta data
        meta ={'file':file_name,
               'faiss_index_type':self.faiss_index_type,
               "id_to_document_mapping_filename":self.id_to_document_mapping_filename,
               "faiss_indexer_pickle_filename":self.faiss_indexer_pickle_filename
               }
        return meta

    # def load(self):
    #     """
    #     提取 index 相关信息
    #     """
    #     id_to_document_mapping = load_json(self.id_to_document_mapping_path)
    #     index = pickle_load_faiss_index(filename=self.index_path)
    #     return id_to_document_mapping,index

    @classmethod
    def load(
            cls,
            meta: Dict[Text, Any],
            model_dir: Text,
            model_metadata: Optional["Metadata"] = None,
            cached_component: Optional["Component"] = None,
            **kwargs: Any,
    ) -> "Component":
        """Loads the trained model from the provided directory."""
        if not meta.get("file"):
            logger.debug(
                f"Failed to load model for '{cls.__name__}'. "
                f"Maybe you did not provide enough training data and no model was "
                f"trained or the path '{os.path.abspath(model_dir)}' doesn't exist?"
            )
            return cls(component_config=meta)
        file_name = meta.get("file")
        id_to_document_mapping_filename = meta.get('id_to_document_mapping_filename')
        id_to_document_mapping_file = Path(model_dir) / f"{file_name}.{id_to_document_mapping_filename}"

        faiss_indexer_pickle_filename = meta.get('faiss_indexer_pickle_filename')
        index_file = Path(model_dir) / f"{file_name}.{faiss_indexer_pickle_filename}"

        if os.path.exists(id_to_document_mapping_file) and os.path.exists(index_file):
            id_to_document_mapping =load_json(id_to_document_mapping_file)
            index = pickle_load_faiss_index(index_file)
            return FaissIndexer(meta, id_to_document_mapping=id_to_document_mapping,index=index)
        else:
            return FaissIndexer(meta)





def pickle_dump_faiss_index(filename: Union[Text, Path], index: Any):
    # Serialise index and store it as a pickle
    with open(filename, "wb") as h:
        pickle.dump(faiss.serialize_index(index), h)


def pickle_load_faiss_index(filename: Union[Text, Path]):
    """Load and deserialize the Faiss index."""
    with open(filename, "rb") as h:
        data = pickle.load(h)
    return faiss.deserialize_index(data)



if __name__ == '__main__':
    # 训练模型
    document_data = None
    model_dir = None
    faiss_indexer = FaissIndexer(document_data=document_data)
    faiss_indexer.train(document_data=document_data)
    meta = faiss_indexer.persist(model_dir=model_dir)

    # 加载模型
    faiss_indexer = FaissIndexer.load(meta=meta,model_dir=model_dir)

