from . import SentenceEvaluator
import torch
import logging
from tqdm import tqdm, trange
from utils.utils import pytorch_cos_sim
import os
import numpy as np
from typing import List, Tuple, Dict, Set


class InformationRetrievalEvaluator(SentenceEvaluator.SentenceEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.

    1. Given a set of queries ：
    2. a large corpus set (documents)： 多少大小比较合适？ 50 Information Need
    另外： A set of relevance judgments：
    It will retrieve for each query the top-k most similar document.
    It measures Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)

    precision = P( relevant| retrieved)
    recall = P(retrieved | relevant)

    搜索召回算法的评价指标:
    1. recall


    搜索排序算法的评价指标:
    1. MAP（mean average precision）

    2. Mean Reciprocal Rank (MRR)： 倒数排序法。
    这个是最简单的一个，因为他的评估假设是基于唯一的一个相关结果，比如q1的最相关是排在第3位，q2的最相关是在第4位，那么MRR=(1/3+1/4)/2，
    MRR方法主要用于寻址类检索（Navigational Search）或问答类检索（Question Answering）

    3. Normalized Discounted Cumulative Gain (NDCG)
    Mean Average Precision (MAP) : the arithmetic mean of average precision values for individual information needs.
    """

    def __init__(self,
                 queries: Dict[str, str]=None,  #qid => query
                 corpus: Dict[str, str]=None,  #cid => doc
                 relevant_docs: Dict[str, Set[str]]=None,  #qid => Set[cid]
                 corpus_chunk_size: int = 50000,
                 mrr_at_k: List[int] = [10],
                 ndcg_at_k: List[int] = [10],
                 accuracy_at_k: List[int] = [1, 3, 5, 10,100],
                 precision_recall_at_k: List[int] = [1, 3, 5, 10,100],
                 map_at_k: List[int] = [100],
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 name: str = '',
                 evaluate_metric_name = 'map@k'):
        # super(self,InformationRetrievalEvaluator).__init__()

        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)
        self.id_to_query_mapping = queries
        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name

        if name:
            self.name = "_" + name

        self.csv_headers = ["epoch", "steps"]


        for k in accuracy_at_k:
            self.csv_headers.append("Accuracy@{}".format(k))

        for k in precision_recall_at_k:
            self.csv_headers.append("Precision@{}".format(k))
            self.csv_headers.append("Recall@{}".format(k))

        for k in mrr_at_k:
            self.csv_headers.append("MRR@{}".format(k))

        for k in ndcg_at_k:
            self.csv_headers.append("NDCG@{}".format(k))

        for k in map_at_k:
            self.csv_headers.append("MAP@{}".format(k))

    def __call__(self, model, output_dir: str = None, epoch: int = -1, steps: int = -1,if_logging=False) -> float:
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("Information Retrieval Evaluation on " + self.name + " dataset" + out_txt)

        max_k = max(max(self.mrr_at_k), max(self.ndcg_at_k), max(self.accuracy_at_k), max(self.precision_recall_at_k), max(self.map_at_k))

        # Compute embedding for the queries
        query_embeddings = model.encode(sentences= self.queries,show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)
        # 记录每个 query 对应的匹配结果
        queries_result_list = [[] for _ in range(len(query_embeddings))]

        itr = range(0, len(self.corpus), self.corpus_chunk_size)

        if self.show_progress_bar:
            itr = tqdm(itr, desc='Corpus Chunks')

        # Iterate over chunks of the corpus
        for corpus_start_idx in itr:
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            #Encode chunk of corpus
            sub_corpus_embeddings = model.encode(self.corpus[corpus_start_idx:corpus_end_idx], show_progress_bar=False, batch_size=self.batch_size, convert_to_tensor=True)

            # Compute cosine similarites ： (query_size,sub_corpus_embeddings)
            cos_scores = pytorch_cos_sim(query_embeddings, sub_corpus_embeddings)
            del sub_corpus_embeddings

            #Get top-k values: 计算每个 query 的前 K 个 cos_scores,cos_scores 越大越相似
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(max_k, len(cos_scores[0]) - 1), dim=1, largest=True, sorted=False)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
            del cos_scores

            for query_itr in range(len(query_embeddings)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = self.corpus_ids[corpus_start_idx+sub_corpus_id]
                    queries_result_list[query_itr].append({'corpus_id': corpus_id, 'score': score})


        # Compute scores
        scores = self.compute_metrics(queries_result_list)

        # Output
        if if_logging:
            self.logging_scores(scores)

        logging.info("Queries: {}".format(len(self.queries)))
        logging.info("Corpus: {}\n".format(len(self.corpus)))

        csv_file: str = f"Information-Retrieval_evaluation_{self.name}_results_epoch_{epoch}.csv"

        if output_dir is not None:
            csv_path = os.path.join(output_dir, csv_file)
            os.makedirs(output_dir,exist_ok=True)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for k in self.accuracy_at_k:
                output_data.append(scores['accuracy@k'][k])

            for k in self.precision_recall_at_k:
                output_data.append(scores['precision@k'][k])
                output_data.append(scores['recall@k'][k])

            for k in self.mrr_at_k:
                output_data.append(scores['mrr@k'][k])

            for k in self.ndcg_at_k:
                output_data.append(scores['ndcg@k'][k])

            for k in self.map_at_k:
                output_data.append(scores['map@k'][k])

            fOut.write(",".join(map(str,output_data)))
            fOut.write("\n")
            fOut.close()

        # return scores['map@k'][max(self.map_at_k)]
        return scores

    def compute_metrics(self, queries_result_list: List[object]):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            # query 相关的docs
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is accross the top-k documents
            # accuracy @ k = num_hits_at_k / len(queries)
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    # top_k 中含有 query 相关的doc
                    if hit['corpus_id'] in query_relevant_docs:
                        # 对于 k_val 和 query_id 标记为 1，否则标记为0，
                        # 后面 按照 accuracy@k = num_hits_at_k/len(queries)
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit['corpus_id'] in query_relevant_docs:
                        # top_k 中记录相关的个数
                        num_correct += 1
                # 在 k_val的时候，对于每个 query_id 计算 precisions_at_k = num_correct / k_val
                precisions_at_k[k_val].append(num_correct / k_val)
                # 在 k_val的时候，对于每个 query_id 计算 precisions_at_k = num_correct / len(query_relevant_docs)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))
                # 最后，对于 k_val的时候，取平均值作为 precisions_at_k 和 recall_at_k

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        # 在 k_val的时候，对于 query 计算 最相关的倒数
                        #
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                # 对于 top_k个 doc 分别判断是否为 relevant
                predicted_relevance = [1 if top_hit['corpus_id'] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(true_relevances, k_val)
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0
                # 在 k_val 的时候，对于每个 query 计算 avg_precision
                for rank, hit in enumerate(top_hits[0:k_val]):
                    # 对于 前 k个预测当中，计算 avg_precision
                    if hit['corpus_id'] in query_relevant_docs:
                        num_correct += 1
                        # 当前 rank 的 precision = 前面预测准确的个数/截止当前rank的数量
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))

                AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])


        return {'accuracy@k': num_hits_at_k, 'precision@k': precisions_at_k, 'recall@k': recall_at_k, 'ndcg@k': ndcg, 'mrr@k': MRR, 'map@k': AveP_at_k}


    def logging_scores(self, scores):
        for k in scores['accuracy@k']:
            logging.info("Accuracy@{}: {:.2f}%".format(k, scores['accuracy@k'][k]*100))

        for k in scores['precision@k']:
            logging.info("Precision@{}: {:.2f}%".format(k, scores['precision@k'][k]*100))

        for k in scores['recall@k']:
            logging.info("Recall@{}: {:.2f}%".format(k, scores['recall@k'][k]*100))

        for k in scores['mrr@k']:
            logging.info("MRR@{}: {:.4f}".format(k, scores['mrr@k'][k]))

        for k in scores['ndcg@k']:
            logging.info("NDCG@{}: {:.4f}".format(k, scores['ndcg@k'][k]))

        for k in scores['map@k']:
            logging.info("MAP@{}: {:.4f}".format(k, scores['map@k'][k]))


    @staticmethod
    def avg_precisioncompute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  #+2 as we start our idx at 0
        return dcg


    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  #+2 as we start our idx at 0
        return dcg