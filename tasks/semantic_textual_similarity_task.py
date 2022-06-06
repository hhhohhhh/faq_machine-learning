#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/3/2 16:37 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/2 16:37   wangfc      1.0         None

"""
import os
from typing import List, Dict, Set

import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from data_process.sentence_pair_data_reader import SentencePairDataReader
from data_process.dataset.sentence_label_dataset import SentenceLabelExample, SentenceLabelFeature,SentenceLabelDatasetV2
from data_process.dataset.triplet_dataset import TripletDataset, TripletExample

from utils.torch.triplet_loss import TripletLoss,DistanceMetrics,BatchHardTripletLoss
from models.representation_based_model.siamese_bert import SiameseBert
from evaluation.InformationRetrievalEvaluator import InformationRetrievalEvaluator
from .torch_task import BasicTorchTask



import logging
logger = logging.getLogger(__name__)




class SemanticTextualSimilarityTask(BasicTorchTask):
    def _build_loss_fn(self):
        if self.loss_name == 'naive_triplet_loss':
            loss_fn = TripletLoss(margin=self.margin)
        elif self.loss_name == 'hard_triplet_loss':
            loss_fn = BatchHardTripletLoss(margin=self.margin)
        return loss_fn



class SiameseTask(BasicTorchTask):
    def __init__(self, data_dir: str, data_reader:SentencePairDataReader,
                 output_dir: str,
                 overwrite_saved_datafile =False,
                 tokenizer : BertTokenizer=None,
                 max_length:int=128,
                 pretrained_dir: str = None,
                 optimizer: str= 'adamw',learning_rate = 0.005,
                 warmup_proportion= 0.1,weight_decay_rate = 0.01,
                 loss_fn_name='hard_triplet_loss',distance_metric= 'cosine', margin = 0.5,
                 num_train_epochs=1, train_batch_size=8,dev_batch_size=8,test_batch_size=8,
                 gradient_accumulation_steps=1,
                 custom_collate_fn =None,
                 seed=1234,
                 parallel_decorate=False, local_rank=-1, no_cuda=0, device_ids=None,
                 summary_filename=None,
                 fp16=False,
                 evaluate_train_data=True,metric_name='recall@k',metric_at_k=100,
                 evaluate_every_epoch=None,save_checkpoint_every_epoch=1,
                 test_mode=False,
                 print_evaluate_step_interval=100,log_summary_every_step=100,
                 ):
        logger.info(f"num_train_epochs={num_train_epochs}\ntrain_batch_size={train_batch_size}\nlearning_rate={learning_rate}\n"
                    f"output_dir={output_dir}\nmax_length={max_length}\noptimizer={optimizer}\n")

        super(SiameseTask, self).__init__(data_dir=data_dir,output_dir=output_dir,summary_filename=summary_filename,
                                          seed=seed,
                                          num_train_epochs=num_train_epochs,train_batch_size=train_batch_size,
                                          dev_batch_size=dev_batch_size,test_batch_size=test_batch_size,
                                          learning_rate=learning_rate,
                                          warmup_proportion=warmup_proportion,
                                          weight_decay_rate =weight_decay_rate,
                                          gradient_accumulation_steps=gradient_accumulation_steps,
                                          parallel_decorate=parallel_decorate, # 分布式训练或者单机多卡
                                          local_rank=local_rank,no_cuda=no_cuda,fp16=fp16,device_ids=device_ids,
                                          evaluate_train_data=evaluate_train_data,
                                          evaluate_every_epoch=evaluate_every_epoch,
                                          metric_name= metric_name,metric_at_k=metric_at_k ,
                                          save_checkpoint_every_epoch= save_checkpoint_every_epoch,
                                          test_mode=test_mode,
                                          print_evaluate_step_interval=print_evaluate_step_interval,
                                          log_summary_every_step=log_summary_every_step
        )
        self.data_reader = data_reader
        self.sentence_label_data_df = data_reader.sentence_label_data_df
        self.relevant_docs_ids_ls = data_reader.relevant_docs_ids_ls # 相关句子id的集合

        self.max_length = max_length


        # 初始化 tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif pretrained_dir:
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_dir)

        self.custom_collate_fn = custom_collate_fn

        # 加载训练所需数据： 分别生成对应data_type的 examples,features,dataset
        self.train_examples :List[SentenceLabelExample] =None
        self.dev_examples:List[SentenceLabelExample]  =None

        # 生成 examples -> features -> dataset
        self._load_data(load_train=True,load_dev=True,load_test=True,
                        overwrite_saved_datafile=overwrite_saved_datafile)

        # 构造 搜索评估用的语料
        self.queries: Dict[str, str] = None
        self.corpus : Dict[str, str]  = None
        self.relevant_docs : Dict[str, Set[str]]  = None
        self._build_InformationRetrievalEvaluator_data()


        if pretrained_dir is not None:
            self.pretrained_dir = pretrained_dir
            self.model = SiameseBert(pretrained_bert_dir=pretrained_dir,tokenizer=self.tokenizer,device=self.device)
        # 加载模型到 device 或者 分布式或者
        self._decorate_model()

        # 默认使用 cosine距离
        self.distance_metric = DistanceMetrics.COSINE
        self.margin = margin

        self.loss_fn_name = loss_fn_name
        if self.loss_fn_name == 'naive_triplet_loss':
            self.loss_fn = TripletLoss(margin=margin)
            self.evaluate_on_batch = self.evaluate_trinplet_embedings
        elif self.loss_fn_name == 'hard_triplet_loss':
            self.loss_fn = BatchHardTripletLoss(margin=margin)
            # self.evaluate_on_batch = self.evaluate_embedding_label

        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer == 'adamw':
            self._init_bert_optimizer()


    def load_example_func(self,file_path)->List[SentenceLabelExample]:
        # 获取有 data_reader 读取的 sentence_label_data_df
        sentence_label_data_df = SentencePairDataReader.read_pd_json_data(file_path,test_mode=self.test_mode)
        # 将 sentence_label_data_df 转为 example
        example_ls = []
        for index in  sentence_label_data_df.index:
            id = sentence_label_data_df.loc[index,'id']
            text = sentence_label_data_df.loc[index,'text']
            lable = sentence_label_data_df.loc[index,'label']
            example = SentenceLabelExample(guid=index,id=id,text=text,label=lable)
            example_ls.append(example)
        logger.info(f"共读取 {example_ls.__len__()} 个examples从 {file_path}")
        return example_ls


    # 在子类中的具体实现形式
    def load_triplet_example_func(self, file_path):
        # 使用 SentencePairDataset的读取json文件的方法读取数据
        triplet_data_df = SentencePairDataReader.read_pd_json_data(file_path,test_mode=self.test_mode)
        # 将 triplet_data_df 转换为 examples
        triplet_example_ls = []
        for index in triplet_data_df.index:
            anchor = triplet_data_df.loc[index, 'anchor']
            positive = triplet_data_df.loc[index, 'positive']
            negative = triplet_data_df.loc[index, 'negative']
            example = TripletExample(guid=index, anchor=anchor, positive=positive, negative=negative)
            triplet_example_ls.append(example)
        logger.info(f"共读取 {triplet_example_ls.__len__()} 个examples从 {file_path}")
        return triplet_example_ls


    def convert_example_to_feature_func(self, examples:List[SentenceLabelExample])->List[SentenceLabelFeature]:
        features = []
        for ex_index,example in enumerate(examples):
            if ex_index %1000==0:
                logger.info(f"转换第{ex_index}个example到feature")

            # feature_dict = {}
            # feature_dict.update({'guid':example.guid})
            # feature_dict.update({f"text": example.text})
            # feature_dict.update({'label':example.label})
            feature = SentenceLabelFeature(guid=example.guid,id=example.id, text=example.text,label=example.label)
            features.append(feature)
        return features


    def convert_triplet_example_to_feature_func(self, examples:List[TripletExample]):
        features = []
        for ex_index,example in enumerate(examples):
            if ex_index %1000==0:
                logger.info(f"转换第{ex_index}个example到feature")

            feature_dict = {}
            feature_dict.update({'guid':example.guid})
            for key in ['anchor','positive','negative']:
                sentence = example.__getattribute__(key)
                # # 先对句子进行 tokenizer
                # tokens = self.tokenizer.tokenize(sentence)
                # # 获取 bert的feature
                # bert_feature = self.tokenizer([sentence],padding=True,max_length=self.max_length )
                # # 对每个句子生成 sentence_feature
                # sentence_feature = SentenceFeature(tokens=tokens,
                #                                    input_ids=bert_feature['input_ids'],
                #                                    token_type_ids=bert_feature['token_type_ids'],
                #                                    attention_mask=bert_feature['attention_mask'])
                # feature_dict.update({f"{key}_feature":sentence_feature})

                # 不进行 tokenize，直接输入句子,在输出  batch 之后进行tokenize处理
                feature_dict.update({f"{key}": sentence})

            # feature = TripletFeature(**feature_dict)
            # 转换为dict 形式
            # feature_dict = feature.get_feature_dict()
            features.append(feature_dict)
        return features


    def convert_to_dataset_func(self, features,data_type='train')->SentenceLabelDatasetV2:
        assert len(features)>0  #and isinstance(features[0],TripletFeature)
        dataset = SentenceLabelDatasetV2(examples=features,data_type=data_type)
        logger.info(f"features={type(features[0])}转换dataset={SentenceLabelDatasetV2}")
        return dataset

    def tokenizer_fn(self,batch,print_first_n_step=0):
        output =  SiameseBert.tokenizer_fn(batch,tokenizer=self.tokenizer,max_length=self.max_length)
        sentences = batch.get('text')
        labels = batch.get('label')
        # <class 'transformers.tokenization_utils_base.BatchEncoding'>
        sentence_encodings = output.get('sentence_encodings')
        if self.run_step <= print_first_n_step:
            for i in range(sentences.__len__()):
                sentence =sentences[i]
                label = labels[i]
                tokens = self.tokenizer.tokenize(sentence)
                tokens_to_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                logger.info(f'sentence={sentence},label={label}\ntokens={tokens}\ntokens_to_ids={tokens_to_ids}')
                for key,value in sentence_encodings.items():
                    v = sentence_encodings.get(key)[i]
                    logger.info(f'{key}={v}')

        return output


    def convert_to_triplet_dataset_func(self, features,data_type='train'):
        assert len(features)>0  #and isinstance(features[0],TripletFeature)
        dataset = TripletDataset(features=features,data_type=data_type)
        return dataset



    def IR_evaluator(self,queries,corpus,relevant_docs,**kwargs)->InformationRetrievalEvaluator:
        return InformationRetrievalEvaluator(queries,corpus,relevant_docs,**kwargs)


    def _build_InformationRetrievalEvaluator_data(self):
        """
        构造用于 信息检索评估 的数据
        1. 将 dev数据集的 examples 转换为 queries
        2. 将所有的训练数据转换为 corpus
        3. 将 整个 corpus中 相关的数据转换为 relevant_docs，
           我们测试的时候测试 query 是否可以召回所有相关的 relevant_docs
        """
        self.InformationRetrievalEvaluator_data_path = os.path.join(self.output_dir,"InformationRetrievalEvaluator_data.json")
        if not os.path.exists(self.InformationRetrievalEvaluator_data_path):
            self.queries = {int(example.id): example.text for example in self.dev_examples}
            self.corpus = {int(self.sentence_label_data_df.loc[index,'id']): self.sentence_label_data_df.loc[index,'text'] for index in self.sentence_label_data_df.index}
            relevant_docs = {}
            for index,query_id in enumerate(self.queries.keys()):
                for relevant_docs_ids in self.relevant_docs_ids_ls:
                    if query_id in relevant_docs_ids:
                        relevant_docs.update({query_id:relevant_docs_ids})
                        break
                if index %100==0:
                    logger.info(f"index={index},query_id={query_id},")
            self.relevant_docs = relevant_docs
            InformationRetrievalEvaluator_data = [self.queries,self.corpus,self.relevant_docs]
            default_dump_json(InformationRetrievalEvaluator_data,self.InformationRetrievalEvaluator_data_path)
        else:
            self.queries,self.corpus,self.relevant_docs = default_load_json(self.InformationRetrievalEvaluator_data_path)
        logger.info(f"build_InformationRetrievalEvaluator_data:len(queries)={self.queries.__len__()},"
                    f"len(corpus)={self.corpus.__len__()}")

    def evaluate(self,epoch,global_step,evaluator_name = "InformationRetrievalEvaluator",**kwargs):
        if evaluator_name == 'InformationRetrievalEvaluator':
            # 初始化 evaluator
            self.evaluator = self.IR_evaluator(self.queries,self.corpus,self.relevant_docs,
                                               **kwargs)
            # 使用 evaluator 进行评估后得到 metrics 字典
            evaluate_metrics = self.evaluator.__call__(model=self.model,  output_dir=self.evaluate_metrics_output_dir,
                                                       epoch=epoch,steps=global_step)
        return evaluate_metrics


    def evaluate_trinplet_embedings(self,triplet_embedding,batch):
        # 具体写评估函数
        assert triplet_embedding.shape[1] == 3
        # 提取对应的 embedding
        anchor_embedding = triplet_embedding[:, 0, :]
        pos_embedding = triplet_embedding[:, 1, :]
        neg_embedding = triplet_embedding[:, 2, :]

        batch_size = anchor_embedding.shape[0]

        anchor2pos_distances = self.distance_metric(anchor_embedding, pos_embedding)
        anchor2neg_distances = self.distance_metric(anchor_embedding, neg_embedding)
        # 如果 anchor2pos_distances - anchor2neg_distances + self.margin >0, 说明预测错误，否则说明正确
        margin = F.relu(anchor2pos_distances - anchor2neg_distances + self.margin)
        # 计算正确和错误的个数
        correct_tensor = torch.eq(margin,0)
        corrent_num = correct_tensor.sum()
        accuray = torch.true_divide(corrent_num,batch_size)
        return accuray



def train_siamese_task(mode='train'):
    # 蚂蚁金服的数据集
    data_dir = 'data'
    data_files = [{"dataset": 'atec', "data_filenames": ['atec_nlp_sim_train.csv', 'atec_nlp_sim_train_add.csv']},
                  {"dataset": 'security_similary_data', "data_filenames": ["六家评估集.xlsx"]}]
    sentence_pair_data_reader = SentencePairDataReader(data_dir=data_dir, data_files=data_files,
                                                       if_read_data=True, if_overwrite=False,
                                                       if_build_triplet_data=False,
                                                       if_build_sentence_label_data=True,
                                                       train_size=0.8, dev_size=0.1, test_size=0.1)

    transformer_model = 'albert-chinese-tiny'
    pretrained_bert_dir = os.path.join(os.getcwd(), 'pretrained_model', transformer_model)

    siamese_task = SiameseTask(data_dir=sentence_pair_data_reader.sentence_label_data_dir,
                               data_reader=sentence_pair_data_reader,
                               output_dir=MODEL_OUTPUT_DIR,
                               overwrite_saved_datafile=False,
                               tokenizer=None, pretrained_dir=pretrained_bert_dir,
                               no_cuda=GPU_NO, num_train_epochs=NUM_TRAIN_EPOCHS, learning_rate=LEARNING_RATE,
                               train_batch_size=32, dev_batch_size=32, test_batch_size=32,
                               evaluate_train_data=False, metric_name='recall@k', metric_at_k=100,
                               evaluate_every_epoch=1, save_checkpoint_every_epoch=None,
                               local_rank=-1, seed=1234,
                               test_mode=False,
                               print_evaluate_step_interval=100,
                               summary_filename='summary',
                               log_summary_every_step=100,
                               )

    # 开始训练
    if mode == 'train' or mode == 'train_and_evaluate':
        siamese_task.train()
    elif mode == 'evaluate' or mode == 'train_and_evaluate':
        siamese_task.evaluate(epoch=3)

    # 在测试集上进行测试工作
    # siamese_task.test()