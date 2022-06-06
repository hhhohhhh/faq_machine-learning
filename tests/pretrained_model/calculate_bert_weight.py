#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/3 14:57 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/3 14:57   wangfc      1.0         None
"""

BERT_BASE_CONFIG =\
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 30522
}


ALBERT_TINY_CONFIG = {
  "attention_probs_dropout_prob": 0.0,
  "bos_token_id": 2,
  "classifier_dropout_prob": 0.1,
  "down_scale_factor": 1,
  "embedding_size": 128,
  "eos_token_id": 3,
  "gap_size": 0,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.0,
  "hidden_size": 312,
  "initializer_range": 0.02,
  "inner_group_num": 1,
  "intermediate_size": 1248,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "net_structure_type": 0,
  "num_attention_heads": 12,
  "num_hidden_groups": 1,
  "num_hidden_layers": 4,
  "num_memory_blocks": 0,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 30522
}

class PretrainedModelSizeCalculator():
    """
    bert-large模型（24-layer, 1024-hidden, 16-heads）: 3亿参数, 340M
    bert-base模型（12-layer, 768-hidden, 12-heads）:110M

    
    """
    def __init__(self,model_name='bert_base',
                 share_hidden_layers= None,
                 factorized_embedding=None,
                 max_position_embeddings=512,
                 vocab_size= 768,
                 type_vocab_size=2,
                 embedding_size=768,
                 num_hidden_layers=12,hidden_size=768,
                 num_attention_heads=12,intermediate_size = 3072,
                 ):
        self.model_name =model_name

        self.share_hidden_layers = share_hidden_layers
        self.factorized_embedding = factorized_embedding

        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

        self.model_config = self._get_model_config(model_name)
        for k,v in self.model_config.items():
            self.__setattr__(k,v)

        self.embedding_token_num = self.vocab_size + self.max_position_embeddings + self.type_vocab_size
        self.layer_normalization_layer_size = self._calculate_layer_normalization_layer_size()


        self.embedding_layer_size = self._calculate_embedding_layer_size(factorized_embedding=factorized_embedding)

        self.self_attention_head_size = self._calculate_self_attention_head_size()
        self.multihead_size = self._calculate_multihead_size()
        self.self_attention_output_dense_layer_size = self._calculate_self_attention_output_dense_layer_size()
        self.self_attention_layer_size = self._calculate_self_attention_layer_size()

        self.self_attention_layers_size = self.self_attention_layer_size* self.num_hidden_layers
        # 768 * 12 * 64 * 3 + 12 * 64 * 3

        self.fullconnection_layer_size = self._calculate_fullconnection_layer_size()
        self.fullconnection_layers_size = self.fullconnection_layer_size*self.num_hidden_layers

        self.transform_layer_size = self._calculate_transform_layer_size()
        self.pooler_size = self._calculate_pooler_size()

        self.transform_model_size = self._calculate_transform_model_size(share_hidden_layers=share_hidden_layers)
        self.encoder_size = self._calculate_encoder_size()




    def _get_model_config(self,model_name):
        if model_name == 'bert_base':
            model_config = BERT_BASE_CONFIG
        elif model_name == 'albert_tiny':
            model_config = ALBERT_TINY_CONFIG
            self.share_hidden_layers =True
            self.factorized_embedding =True
        return model_config

    def _calculate_embedding_layer_size(self,factorized_embedding = None):
        """
        输入embeding层对应的参数，分别对应的是token-embedding,position-embedding，segment-embedding。
        https://www.cnblogs.com/jiangxinyang/p/11715678.html

        """
        factorized_embedding = factorized_embedding if factorized_embedding is not None else self.factorized_embedding
        if factorized_embedding:
            embedding_size = self.embedding_token_num * self.embedding_size + self.embedding_size*self.hidden_size
        else:
            embedding_size = self.embedding_token_num * self.hidden_size
        return embedding_size

    def _calculate_layer_normalization_layer_size(self):
        """
        如何计算 ：
        wrong : self.max_position_embeddings *2
        correct :
        """
        return self.hidden_size *2


    def _calculate_self_attention_head_size(self):
        """
        三个矩阵 W_Q,W_K,W_V: W= ( hidden_size/num_heads, hidden_sizes ),B= ( hidden_size/num_heads,)
        """
        return 3*(self.hidden_size/self.num_attention_heads * self.hidden_size + self.hidden_size/self.num_attention_heads)

    def _calculate_multihead_size(self):
        return self.self_attention_head_size* self.num_attention_heads

    def _calculate_self_attention_output_dense_layer_size(self):
        """
        将自注意层12个多头输出进行拼接后接全连接层输出对应的参数量
        """
        return self.hidden_size* self.hidden_size + self.hidden_size

    def _calculate_self_attention_layer_size(self):
        return self.multihead_size + self.self_attention_output_dense_layer_size + self.layer_normalization_layer_size
        

    def _calculate_fullconnection_layer_size(self):
        return self.intermediate_size*self.hidden_size + self.intermediate_size+\
               self.hidden_size* self.intermediate_size + self.hidden_size \
               +self.layer_normalization_layer_size

    def _calculate_transform_layer_size(self):
        return self.self_attention_layer_size +self.fullconnection_layer_size

    def _calculate_transform_model_size(self,share_hidden_layers=None):
        share_hidden_layers =share_hidden_layers  if share_hidden_layers is not None else self.share_hidden_layers
        if share_hidden_layers:
            return self.transform_layer_size
        else:
            return self.num_hidden_layers * self.transform_layer_size



    def _calculate_pooler_size(self):
        """
        pooler output，对应的[CLS]的输出，
        sequence output，对应的是序列中的所有字的最后一层hidden输出。
        所以BERT主要可以处理两种，一种任务是分类/回归任务（使用的是 pooler output），一种是序列任务（sequence output）。
        BertPooler模块的全连接层只接收BertEncoder模块中第一个token（[CLS]）对应的输出
        在全连接 + tanh
        """
        return self.hidden_size*self.hidden_size+ self.hidden_size

    def _calculate_encoder_size(self):
        return self.embedding_layer_size + self.layer_normalization_layer_size + self.transform_model_size \
        +self.pooler_size


    def _calculate_classification_layer_size(self,output_size):
        return output_size * self.hidden_size + output_size

    def get_encoder_size(self):
        print(f"{self.model_name} share_hidden_layers ={self.share_hidden_layers} factorized_embedding={self.factorized_embedding}"
              f"参数量共{ self.encoder_size}，\n"
              f"embedding_layer_size={self.embedding_layer_size},\n"
              f"transform_model_size={self.transform_model_size},\n"
              f"self_attention_layers_size={self.self_attention_layers_size},\n"
              f" fullconnection_layers_size ={self.fullconnection_layers_size}")
        return self.encoder_size


if __name__ == '__main__':
    calculator = PretrainedModelSizeCalculator(model_name='albert_tiny')
    encoder_size = calculator.get_encoder_size()
    calculator._calculate_embedding_layer_size(factorized_embedding=False)