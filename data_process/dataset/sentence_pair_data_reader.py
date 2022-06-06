#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/3/3 9:10 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/3 9:10   wangfc      1.0         None

"""
import os
import random
from pathlib import Path
from typing import List
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import deepcopy
from utils.io import load_json,dump_json,dataframe_to_file
import logging

logger = logging.getLogger(__name__)


class SentenceNode(object):
    def __init__(self,id,text):
        """
        定义每个句子节点的信息
        """
        self.id = id
        self.text =text
        self.connect_node_ids = set()
        # self.connect_nodes= []
        self.have_processed= False
        self.cluster_index = None # 记录属于哪一个 cluster

    def add_connect_node(self,connect_node):
        self.connect_node_ids.add(connect_node.id)
        # 不要记录连接的 node，因为该node可能自身还没有更新，只要记录 id
        # self.connect_nodes.append(connect_node)


    def __repr__(self):
        return f"id={self.id},text={self.text},cluster_index={self.cluster_index},have_processed={self.have_processed}," \
               f"connect_node_ids={self.connect_node_ids}"


class SentenceCluster():
    def __init__(self,cluster_index):
        self.cluster_index= cluster_index
        self.cluster_num = 0
        self.cluster_members = []

    def add(self,sentence_node:SentenceNode):
        self.cluster_members.append(sentence_node)
        self.cluster_num += 1

    def __repr__(self):
        return f"cluster_index={self.cluster_index},cluster_num={self.cluster_num},cluster_members={self.cluster_members}"


def dfs(current_node:SentenceNode,cluster_index:int,id_to_node_dict):
    # 根据当前节点进行 DFS  遍历
    # 新建一个cluster

    cluster = SentenceCluster(cluster_index)
    # 将该 node 加入 cluster
    current_node.have_processed = True
    cluster.add(current_node)
    stack = []
    # 将 node 加入 stack
    stack.append(current_node)
    while stack.__len__()>0:
        node = stack.pop()
        for connect_node_id in node.connect_node_ids:
            connect_node = id_to_node_dict.get(connect_node_id)
            # 判断连接的节点是否已经处理过
            if connect_node.have_processed == False:
                # 标记该 node
                connect_node.have_processed = True
                # 将该node 加入
                cluster.add(connect_node)
                stack.append(connect_node)
    # 遍历结束的时候，返回
    if cluster_index %100==1:
        logger.info(f"{'*' * 10}\n新建第{cluster_index}个cluster的遍历搜索{'*' * 10}")
        logger.info(f"第{cluster_index}个cluster的搜索结束，共有{cluster.cluster_num}节点")
        for node in cluster.cluster_members:
            logger.info(f'id={node.id},sentence={node.text}')
    return cluster,cluster_index


def sentence_node_clustering(sentence_nodes:List[SentenceNode])->List[SentenceCluster]:
    id_to_node_dict ={sentence_node.id:sentence_nodes for sentence_node in sentence_nodes}
    cluster_ls = []
    # cluster_index 从 1看开始标记，其他的句子使用 0，表示其他
    cluster_index = 1
    id_to_node_dict = {sentence_node.id:sentence_node for sentence_node in sentence_nodes}
    for sentence_node in sentence_nodes:
        if sentence_node.have_processed is False:
            cluster,cluster_index = dfs(current_node=sentence_node,cluster_index=cluster_index,id_to_node_dict=id_to_node_dict)
            cluster_index +=1
            cluster_ls.append(cluster)
    return cluster_ls












def get_labeled_data(data_dir, dataset1, dataset2, data_filenames1, data_filenames2):
    data_df_ls = []
    for dataset, data_filenames in zip([dataset1, dataset2], [data_filenames1, data_filenames2]):
        data_lines = read_txt_data(data_dir, dataset, data_filenames)
        if dataset == 'ccks':
            line_with_index = False
        else:
            line_with_index = True
        data_df = convert_sentence_pair_lines2df(data_lines, line_with_index=line_with_index)
        data_df_ls.append(data_df)

    data_df = pd.concat(data_df_ls)
    logger.info(f"data.shape={data_df.shape}")
    logger.info(data_df.label.value_counts())
    logger.info(data_df.head())
    return data_df



class SentencePairDataReader(object):
    def __init__(self, data_dir, data_files,
                 output_dirname ='sentence_label_data',
                 if_split=True,
                 train_size=0.9,dev_size=0.1,test_size=0):
        self.data_dir = data_dir
        self.data_files = data_files
        self.output_dirname  = output_dirname
        self.data_output_dir = os.path.join(self.data_dir, output_dirname)
        self.train_size = train_size
        self.dev_size = dev_size
        self.test_size = test_size

        # 创建 sentence_label_data
        self.sentence_label_data_path = os.path.join(self.data_output_dir, f'data.json')
        # self.sentence_label_train_data_path = os.path.join(self.sentence_label_data_dir, f'train.json')
        self.relevant_docs_ids_ls_path = os.path.join(self.data_output_dir, f'relevant_docs_ids_ls.json')

    def process(self,if_overwrite=False,
                if_build_triplet_data=False,
                if_build_sentence_label_data=True,
                if_convert_to_dstc7=False):
        # 1. 读取原始数据
        self.data_df = self._read_raw_data(data_dir=self.data_dir)

        # 2. 读取或者转换数据
        self._get_transform_data(if_overwrite=if_overwrite,if_build_triplet_data=if_build_triplet_data,
                                 if_build_sentence_label_data=if_build_sentence_label_data)

        # 保存处理后的数据
        # self.processed_data_dir = self.sentence_label_data_path
        if if_convert_to_dstc7:
            self._transform_to_dstc7()


    def _read_raw_data(self,data_dir)-> pd.DataFrame:
        data_df_ls = []
        for data_file in self.data_files:
            dataset =  data_file.get('dataset')
            data_filenames = data_file.get('data_filenames')
            if dataset =='atec':
                #  dataframe："index",'sentence1','sentence2','label'
                data_df = self._get_sentence_pair_data(data_dir=data_dir, dataset=dataset,
                                                       data_filenames=data_filenames)
            elif dataset =='security_similary_data':
                for data_filename in data_filenames:
                    data_path = Path().joinpath(data_dir,dataset,data_filename)
                    data_df = pd.read_excel(data_path,engine='openpyxl')
                    logger.info(f"读取原始数据 data_df.shape={data_df.shape} from {data_path} ")
                    data_df.columns = ['sentence2', 'sentence1']
                    data_df.loc[:,'label']=1
            data_df_ls.append(data_df)
        total_data_df = pd.concat(data_df_ls,ignore_index=True)
        logger.info(f"读取所有原始数据 total_data_df.shape={total_data_df.shape}")
        return total_data_df

    def _get_transform_data(self,if_overwrite=False,if_build_triplet_data=False, if_build_sentence_label_data=True):
        if if_overwrite or not os.path.exists(self.sentence_label_data_path) or  \
            os.path.exists(self.relevant_docs_ids_ls_path):
            self._transform_data(if_build_triplet_data, if_build_sentence_label_data)
        else:
            logger.info(f"数据已经处理存放在{self.sentence_label_data_path}")
            self.sentence_label_data_df = dataframe_to_file(path=self.sentence_label_data_path,mode='r')
            logger.info(
                f"读取 sentence_label_data_df数据：shape={self.sentence_label_data_df.shape},sentence_label_data_path={self.sentence_label_data_path}")
            self.sentence_label_data_df.label.value_counts(ascending=True)
            self.relevant_docs_ids_ls = load_json(self.relevant_docs_ids_ls_path)
            logger.info(
                f"读取 relevant_docs_ids_ls.len={len(self.relevant_docs_ids_ls)} 从 {self.relevant_docs_ids_ls_path}")




    def _get_sentence_pair_data(self, data_dir, dataset, data_filenames):
        data_lines = self._read_txt_data(data_dir, dataset, data_filenames)
        if dataset == 'ccks':
            line_with_index = False
        else:
            line_with_index = True
        data_df = self._convert_sentence_pair_lines2df(data_lines, line_with_index=line_with_index)
        return data_df

    def _read_txt_data(self,data_dir:str,dataset:str,data_filenames:List[str],encoding='utf-8', mode:str='r'):
        total_lines = []
        for data_filename in data_filenames:
            data_path = os.path.join(data_dir, dataset, data_filename)
            with open(data_path,encoding=encoding,mode=mode) as f:
                lines = f.readlines()
                logger.info(f"读取{lines.__len__()}条数据从{data_path} with encoding={encoding}")
                total_lines.extend(lines)
        logger.info(f"读取全部数据共 {total_lines.__len__()}条")#
        return total_lines


    def _convert_sentence_pair_lines2df(self, data_lines, line_with_index=True, test_mode=None):
        """
        @author:wangfc
        @desc:
        @version：
        @time:2021/4/2 14:16

        Parameters
        ----------

        Returns
        -------
        返回 dataframe：  "index",'sentence1','sentence2','label'
        """
        datalines_ls = []
        for index in range(data_lines.__len__()):
            line = data_lines[index]
            line_splited_ls = line.split(sep='\t')
            if line_with_index and line_splited_ls.__len__() == 4:
                line_index, sentence1, sentence2, label_index = line_splited_ls
            elif not line_with_index and line_splited_ls.__len__() == 3:
                sentence1, sentence2, label_index = line_splited_ls
                line_index = index
            else:
                continue
            label_index_matched = re.match('(\d)\n$', label_index)
            if label_index_matched:
                label_index = int(label_index_matched.groups()[0])
                data_labeled = {"index": line_index,
                                'sentence1': sentence1,
                                'sentence2': sentence2,
                                'label': label_index}
                datalines_ls.append(data_labeled)
        data_df = pd.DataFrame(datalines_ls)
        logger.info(f"data.shape={data_df.shape}")
        logger.info(data_df.head())
        logger.info(data_df.label.value_counts())
        return data_df



    def _convert_to_sentence_label_dataV1(self):
        logger.info("数据转变为sentence_label_data")
        data_df = self.data_df

        filter_data_df = data_df[data_df.label == 1]
        # 根据重复的情况，找到相似句子
        data_count = filter_data_df.groupby(by=['sentence1']).count().sort_values(by='sentence2')
        duplicate_data_ls = data_count[data_count.sentence2 > 1].index.tolist()
        logger.info(f"属于相似问label=1并且重复的句子共有{duplicate_data_ls.__len__()}")

        sentence_label_data_ls = []

        for label_index,data in enumerate(duplicate_data_ls):
            rows = filter_data_df[filter_data_df.sentence1==data]
            sentences1 = rows.loc[:,'sentence1']
            sentences2 = rows.loc[:,'sentence2']
            sentences = sentences1.tolist() + sentences2.tolist()
            sentences_ls = list(set(sentences))
            for sentence in sentences_ls:
                # label_index 从 1 起始
                label_index +=1
                example = dict(text=sentence, label=label_index)
                sentence_label_data_ls.append(example)

        noduplicate_data_df = filter_data_df[~filter_data_df.sentence1.isin(duplicate_data_ls)]

        for idx, index in enumerate(noduplicate_data_df.index):
            label = data_df.loc[index, 'label']
            sentence1 = data_df.loc[index, 'sentence1']
            sentence2 = data_df.loc[index, 'sentence2']
            for sentence in [sentence1,sentence2]:
                example = dict(text=sentence,label=int(idx+label_index+1))
                sentence_label_data_ls.append(example)
        sentence_label_data_df = pd.DataFrame(sentence_label_data_ls)

        # 不是相似的句子
        # other_data_df = data_df[~(data_df.label == 1)]
        return sentence_label_data_df

    def _build_connect_sentence_nodes(self):
        # 筛选同义的句子对
        filter_data_df = self.data_df[self.data_df.label == 1]
        logger.info(f"使用原始数据共{self.data_df.__len__()}行中选择label=1的数据共{filter_data_df.__len__()}行筛选关联的句子对")
        filter_data_df_revert = filter_data_df.copy().loc[:, ['sentence2','sentence1']]
        filter_data_df_revert.columns = ['sentence1','sentence2']
        # 合并在一起进行处理
        combined_df = pd.concat([filter_data_df,filter_data_df_revert])

        sentences1 = combined_df.loc[:, 'sentence1']
        sentences = sentences1.drop_duplicates()
        sentences.reset_index(inplace=True,drop=True)
        sentences_df = pd.DataFrame(sentences,columns=["sentence1"])
        # 建立 sentence2id 映射
        sentence2id = {sentences_df.loc[index,'sentence1']: str(index) for index in sentences_df.index}

        sentence_nodes =[]
        for num,index in enumerate(sentences_df.index):
            # 当前的 sentence_node
            sentence = sentences_df.loc[index,'sentence1']
            id = sentence2id[sentence]
            current_sentence_node = SentenceNode(id=id,text=sentence)
            connect_sentences_df = combined_df[combined_df.sentence1==sentence]
            connect_sentences= connect_sentences_df.loc[:,'sentence2'].tolist()
            # 排除自己
            connect_sentences_set = set(connect_sentences).difference({sentence})
            if num%100==0:
                logger.info(f"开始匹配第{num}/{sentences_df.__len__()}条数据："
                            f"\nsentence={sentence}"
                            f"\nconnect_sentences_set={connect_sentences_set}")

            for connect_sentence in connect_sentences_set:
                # 获取 connect_sentence 的id
                connect_sentence_id = sentence2id[connect_sentence]
                connect_sentence_node = SentenceNode(id=connect_sentence_id,text=connect_sentence)
                current_sentence_node.add_connect_node(connect_sentence_node)

            sentence_nodes.append(current_sentence_node)
        logger.info(f"共建立 len(sentence_nodes)={len(sentence_nodes)}")
        return sentence_nodes


    def _convert_to_sentence_label_dataV2(self):
        """
        # 根据传导性，找到相似句子
        1) 筛选可传导的句子对
        2）按照句子建立 node
        3）为每个句子计算其连接点
        4）使用 DFS 进行遍历，找到各个节点作为一个 cluster
        """
        logger.info("数据转变为 sentence_label_data")
        # 按照句子建立 node
        sentence_nodes = self._build_connect_sentence_nodes()
        # 使用 DFS 方法进行 clustering：cluster_index 从 1看开始标记，其他的句子使用 0，表示其他
        cluster_ls = sentence_node_clustering(sentence_nodes=sentence_nodes)
        # 解析 cluster_ls
        sentence_label_data_ls = []
        max_cluster_index=-1
        max_node_id = -1
        cluster_sentences_set = set()
        # 相关句子id的集合
        relevant_docs_ids_ls:List[List[int]] = []
        for cluster in cluster_ls:
            relevant_docs_ids = []
            for node in cluster.cluster_members:
                node_id = int(node.id)
                example = dict(id= node_id,text= node.text, label= cluster.cluster_index)
                sentence_label_data_ls.append(example)
                cluster_sentences_set.add(node.text)
                relevant_docs_ids.append(node_id)

                if cluster.cluster_index > max_cluster_index:
                    max_cluster_index = cluster.cluster_index
                if node_id > max_node_id:
                    max_node_id = node_id
            relevant_docs_ids_ls.append(relevant_docs_ids)
        logger.info(f"cluster_num ={relevant_docs_ids_ls.__len__()},max_cluster_index={max_cluster_index},max_node_id={max_node_id}")


        # 原来是不包括 cluster_index = 0 的数据的
        # TODO: 我们需要把所有其他的句子加入，标记为 label = 0，作为整个 corpus
        # 注意：此时在后面分割数据的时候，需要筛选 label != 0 的数据作为训练和评估数据，因为 label=0 表示全部其他不相关的句子，并不表示 label=0 的时候两个句子同意
        sentences_set  = set(self.data_df.loc[:,'sentence1'].tolist() + self.data_df.loc[:,'sentence2'].tolist())
        nonclustering_sentences_set = sentences_set.difference(cluster_sentences_set)
        logger.info(f'len(sentences_set)={len(sentences_set)},len(cluster_sentences_set)={len(cluster_sentences_set)},len(nonclustering_sentences_set)={len(nonclustering_sentences_set)}')
        nonclustering_sentence_label_data_ls = []
        for index,sentence in enumerate(nonclustering_sentences_set):
            sentence_id = max_node_id+1+index
            example = dict(id= sentence_id, text=sentence, label=0)
            nonclustering_sentence_label_data_ls.append(example)

        sentence_label_data_df = pd.DataFrame(sentence_label_data_ls + nonclustering_sentence_label_data_ls)
        logger.info(f"sentence_label_data_df.shape={sentence_label_data_df.shape}")

        return sentence_label_data_df,relevant_docs_ids_ls


    def _get_random_negative(self, positive_index, data_df, ):
        # 随机选取 其他的data的中 sentence2 作为 negative
        indexes = data_df.index
        negative_indexes = indexes.difference([positive_index])
        random_neg_index = random.randint(a=0, b=negative_indexes.__len__() - 1)
        negative_index = negative_indexes[random_neg_index]
        negative = data_df.loc[negative_index, 'sentence2']
        return negative





    def _transform_data(self,if_build_triplet_data=False,if_build_sentence_label_data=False,data_type='data'):
        os.makedirs(self.data_output_dir, exist_ok=True)
        if if_build_triplet_data:
            # 转换为 triplet 格式的数据
            self.triplet_data_path = os.path.join(self.data_output_dir, f'{data_type}.json')
            self.triplet_data_df = self._transform_to_triplet_data()
            self.triplet_data_df.to_json(self.triplet_data_path, orient='records', force_ascii=False, lines=True)
            # self.triplet_data_df = self.read_pd_json_data(self.triplet_data_path)
            logger.info(f"data_type={data_type},triplet_data 保存到{self.triplet_data_path}")
        elif if_build_sentence_label_data:
            self.sentence_label_data_df, self.relevant_docs_ids_ls = self._convert_to_sentence_label_dataV2()
            # 增加 label = 0 表示那些不相关的 sentence 到 sentence_label_data_df
            dataframe_to_file(path=self.sentence_label_data_path,data=self.sentence_label_data_df)
            logger.info(
                f"data_type={data_type},shape={self.sentence_label_data_df.shape},sentence_label_data保存到{self.sentence_label_data_path}")
            dump_json(self.relevant_docs_ids_ls,self.relevant_docs_ids_ls_path)
            logger.info(
                f"relevant_docs_ids_ls.len={len(self.relevant_docs_ids_ls)} 保存到{self.relevant_docs_ids_ls_path}")


    def _transform_to_triplet_data(self):
        """
        @author:wangfc
        @desc:
        @version：
        @time:2021/3/3 9:07

        Parameters
        ----------

        Returns
        -------
        """
        data_df = self.data_df
        filter_data_df = data_df[data_df.label == 1]
        triplet_ls = []
        for index in filter_data_df.index:
            label = data_df.loc[index, 'label']
            sentence1 = data_df.loc[index, 'sentence1']
            sentence2 = data_df.loc[index, 'sentence2']

            # 随机选取 其他的data的中 sentence2 作为 negative
            negative = self._get_random_negative(positive_index=index, data_df=data_df)

            triplet = dict(anchor=sentence1, positive=sentence2, negative=negative)
            triplet_ls.append(triplet)
        triplet_df = pd.DataFrame(triplet_ls)
        return triplet_df





    def split_data(self,train_size,dev_size,test_size):
        test_indexes = None
        test_data = None
        # 对数据进行筛选，只选取 label > 0 的数据作为 online_triplet_training 训练，评估和测试数据
        # 对数据进行分割
        all_indexes = self.sentence_label_data_df[self.sentence_label_data_df.label>0].index
        logger.info(f"进行数据分割：原始数据共有{self.sentence_label_data_df.__len__()}，筛选其中label>0的数据共{all_indexes.__len__()}")

        train_indexes, dev_indexes = train_test_split(all_indexes, train_size=train_size)
        if int(train_size + dev_size) != 1:
            dev_size_in_left = dev_size / (1 - train_size)
            dev_indexes, test_indexes = train_test_split(dev_indexes, train_size=dev_size_in_left)

        train_data = self.sentence_label_data_df.loc[train_indexes, :]
        dev_data = self.sentence_label_data_df.loc[dev_indexes, :]
        if test_indexes is not None:
            test_data = self.sentence_label_data_df.loc[test_indexes, :]

        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        for data_type, data in zip(['train', 'dev', 'test'], [train_data, dev_data, test_data]):
            data_path = os.path.join(self.sentence_label_data_dir, f'{data_type}.json')
            data.to_json(data_path, orient='records', force_ascii=False, lines=True)
            logger.info(f"data_type={data_type},len={data.__len__()},sentence_label_data保存到{data_path}")

    def _transform_to_dstc7(self,none_relevant_sample_num=99,scenario=1)->List[dict]:

        """
        将数据转换为 parlai_format_data：dstc7
        key =['data-split', 'example-id', 'messages-so-far', 'options-for-correct-answers', 'options-for-next', 'scenario'])

        'options-for-correct-answers'： 其实可以设置为多个的
        """
        # 1. 转化为 parlai_format_data：dstc7
        data_ls = self._convert_to_dstc7_track1_format_data(none_relevant_sample_num=none_relevant_sample_num, scenario=scenario,
                                                           only_one_correct_answers=False,over_write=False)

        # 2. 分割为 train：dev 数据集
        split_data = []
        json_file_path = os.path.join(self.data_output_dir,f'train.json')
        if os.path.exists(json_file_path):
            for data_type in ['train', 'dev', 'test']:
                json_file_path = os.path.join(self.data_output_dir, f'{data_type}.json')
                data = load_json(json_file_path)
                split_data.append(data)
        else:
            train_data,dev_data,test_data = self._split_list_data(data_ls,self.train_size,self.dev_size,self.test_size)
            for data_type ,data in zip(['train','dev','test'],[train_data,dev_data,test_data]):
                if data is not None:
                    # 更新名称
                    for index in range(data.__len__()):
                        data_in_index = data[index]
                        data_in_index.update({'data-split':data_type})
                    json_file_path = os.path.join(self.data_output_dir, f'{data_type}.json')
                    dump_json(data,json_file_path)
                split_data.append(data)

        # build fixed candidates
        test_data = split_data[-1]
        self._build_fixed_candidates(test_data=test_data,fixed_candidates_filename = 'fixed_candidates.json')

        return split_data

    def _split_list_data(self,data_ls,train_size,dev_size,test_size):
        random.shuffle(data_ls)
        data_len = len(data_ls)
        test_data = None
        train_index = int(data_len* train_size)
        dev_len = int(data_len*dev_size)
        train_data = data_ls[:train_index]
        if test_size == 0 or test_size is None:
            dev_data = data_ls[train_index:]
        else:
            dev_index = train_index + dev_len
            dev_data = data_ls[train_index:dev_index]
            test_data = data_ls[dev_index:]
        return train_data,dev_data,test_data

    def _build_fixed_candidates(self,test_data,fixed_candidates_filename = 'test_fixed_candidates.txt',
                                fixed_candidates_without_next_candidate_filename='test_fixed_candidates_without_next_candidate.txt',
                                candidate_all=False,over_write=False):

        # 包括 next_candidate 在内所有的语句作为 fixed_candidates
        text_file_path = os.path.join(self.data_output_dir, fixed_candidates_filename)

        # 不包括 next_candidate： fixed_candidates 少
        text_without_next_candidate_file_path = os.path.join(self.data_output_dir, fixed_candidates_without_next_candidate_filename)

        if not over_write and os.path.exists(text_file_path) and  os.path.exists(text_without_next_candidate_file_path) :
            with open(text_file_path,mode='r',encoding='utf-8') as f:
                fixed_candidates = f.readlines()
        else:
            if candidate_all:
                fixed_candidates = self.sentence_label_data_df.loc['text'].tolist()
            else:
                fix_candidate_ids_set = set()
                text_without_next_candidate_set =set()
                for data in test_data:
                    options_for_correct_answers = data['options-for-correct-answers']
                    options_for_next = data['options-for-next']
                    correct_answer_candidate_ids = [ correct_answer['candidate-id'] for correct_answer in options_for_correct_answers]

                    options_for_next_candidate_ids = [option_for_next['candidate-id'] for option_for_next in
                                                    options_for_next]
                    correct_answer_candidate_ids_set = set(correct_answer_candidate_ids)
                    candidate_ids = correct_answer_candidate_ids_set.union(set(options_for_next_candidate_ids))
                    fix_candidate_ids_set = fix_candidate_ids_set.union(candidate_ids)

                    # 不包括 next_candidate 的情况
                    text_without_next_candidate_set = text_without_next_candidate_set.union(correct_answer_candidate_ids_set)



                fixed_candidates = self.sentence_label_data_df.loc[self.sentence_label_data_df.id.isin(fix_candidate_ids_set),'text'].tolist()
                text_without_next_candidate = self.sentence_label_data_df.loc[self.sentence_label_data_df.id.isin(text_without_next_candidate_set),'text'].tolist()

            with open(text_file_path,mode='w',encoding='utf-8') as f:
                f.write('\n'.join(fixed_candidates))
                logger.info(f"保存 fixed_candidates 共 {fixed_candidates.__len__()} 个 ")
            with open(text_without_next_candidate_file_path,mode='w',encoding='utf-8') as f:
                f.write('\n'.join(text_without_next_candidate))
                logger.info(f"保存 text_without_next_candidate 共 {text_without_next_candidate.__len__()} 个 ")

        return fixed_candidates



    def _convert_to_dstc7_track1_format_data(self,none_relevant_sample_num=99,scenario=1, only_one_correct_answers=False,over_write=False)->List[dict]:
        """"
        'options-for-correct-answers'： 其实可以设置为多个的
        """
        json_file_path = os.path.join(self.data_output_dir, f'{self.output_dirname}.json')
        if not over_write and  os.path.exists(json_file_path):
            data_ls = load_json(json_file_path)
        else:
            ids_set = set(self.sentence_label_data_df.id.values)
            example_id = int(1e10 +1)
            data_ls = []
            for index, relevant_docs_ids in enumerate(self.relevant_docs_ids_ls):
                if index % 1000==0:
                    logger.info(f"开始转换第 {index}/{self.relevant_docs_ids_ls.__len__()} 个 relevant_docs_ids")
                if index >=0:
                    # logger.info(f"开始转换第 {index}/{self.relevant_docs_ids_ls.__len__()} 个 relevant_docs_ids")
                    none_relevant_ids_set = ids_set.difference(set(relevant_docs_ids))
                    for doc_id_index,doc_id in enumerate(relevant_docs_ids[:-1]):
                        doc_values = self.sentence_label_data_df.loc[
                            self.sentence_label_data_df.id == doc_id,'text'].values
                        assert doc_values.shape== (1,)
                        doc = doc_values[0]
                        messages_so_far =  [{'speaker':'participant_1', 'utterance': doc}]
                        # 选取后面的作为 relevant_docs
                        other_relevant_docs_ids = deepcopy(relevant_docs_ids)
                        other_relevant_docs_ids.remove(doc_id)
                        if  only_one_correct_answers:
                            # 只是随机抽样一个
                            other_relevant_doc_id = random.sample(population=other_relevant_docs_ids,k=1)[0]
                            # for other_relevant_doc_id in other_relevant_docs_ids:
                            # 将其作为 correct_answers
                            relevant_doc_values = self.sentence_label_data_df.loc[self.sentence_label_data_df.id == other_relevant_doc_id,'text'].values
                            assert relevant_doc_values.shape == (1,)
                            relevant_doc = relevant_doc_values[0]
                            options_for_correct_answers = [{'candidate-id':other_relevant_doc_id, 'utterance': relevant_doc}]
                        else:
                            relevant_docs_df = self.sentence_label_data_df.loc[
                                self.sentence_label_data_df.id.isin(other_relevant_docs_ids),['id','text']].copy()
                            relevant_docs_df.columns = ['candidate-id','utterance']

                            options_for_correct_answers = relevant_docs_df.to_dict(orient='record')

                        # options_for_next
                        none_relevant_sample_ids = random.sample(population=none_relevant_ids_set,k=none_relevant_sample_num)
                        # 非相关的 + 1个正确的:共 100个
                        options_for_next_ids = none_relevant_sample_ids + [options_for_correct_answers[0]['candidate-id']]

                        option_docs_df = self.sentence_label_data_df.loc[self.sentence_label_data_df.id.isin(options_for_next_ids),['id','text']]
                        assert option_docs_df.shape[0] == none_relevant_sample_num +1
                        option_docs_df.columns = ['candidate-id', 'utterance']
                        options_for_next = option_docs_df.to_dict(orient='records')
                        data = {'data-split':'data','example-id':example_id,
                                'messages-so-far': messages_so_far,
                                'options-for-correct-answers':options_for_correct_answers,
                                'options-for-next':options_for_next,
                                'scenario':scenario}
                        example_id +=1
                        data_ls.append(data)
            dump_json(obj=data_ls,json_file_path=json_file_path)
        return data_ls


if __name__ == '__main__':
    # 蚂蚁金服的数据集
    # data_dir = 'data'
    # dataset1 = 'atec'
    # data_filename1 = 'atec_nlp_sim_train.csv'
    # data_filename2 = 'atec_nlp_sim_train_add.csv'
    # data_filenames1 = [data_filename1, data_filename2]
    #
    # dataset2 = 'ccks'
    # data_filename1 = 'task3_train.txt'
    # data_filename2 = 'task3_dev.txt'
    # data_filenames2 = [data_filename1]

    # dstc7
    # dataset ='dstc7'
    # train_data_filename= 'ubuntu_train_subtask_1.json'
    # datapath = os.path(os.getcwd(),'data',dataset,train_data_filename)
    # train_data = default_load_json(json_file_path=datapath)


    # 蚂蚁金服的数据集
    data_dir = 'data'
    data_files = [{"dataset": 'atec', "data_filenames": ['atec_nlp_sim_train.csv', 'atec_nlp_sim_train_add.csv']},
                  {"dataset": 'security_similary_data', "data_filenames": ["六家评估集.xlsx"]}]

    sentence_pair_data_reader = SentencePairDataReader(data_dir=data_dir, data_files=data_files,
                                                       output_dirname='sentence_pair_to_dst7',
                                                       if_read_data=True, if_overwrite=False,
                                                       if_build_triplet_data=False,
                                                       if_build_sentence_label_data=True,
                                                       train_size=0.8, dev_size=0.1, test_size=0.1)

