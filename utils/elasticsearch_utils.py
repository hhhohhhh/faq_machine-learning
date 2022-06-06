#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/3/29 16:21

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/29 16:21   wangfc      1.0         None

"""
from datetime import datetime
from elasticsearch import Elasticsearch,helpers
import json

def build_elasticsearch_index(dataset,corpus,model,embedding_dim=312,chunk_size = 500):
    """
    dataset：  数据集名称
    embedding_dim： embedding 的维度

    """
    ids = list(corpus.keys())
    sentences = [corpus[id] for id in ids]

    es = Elasticsearch()
    # 建立索引：Index data, if the index does not exists
    if not es.indices.exists(index=dataset):
        try:
            es_index = {
                "mappings": {
                    "properties": {
                        "question": {"type": "text"},
                        "question_vector": {"type": "dense_vector", "dims": embedding_dim}
                    }
                }
            }
            es.indices.create(index=dataset, body=es_index, ignore=[400])

            print("Index data (you can stop it by pressing Ctrl+C once):")
            with tqdm.tqdm(total=len(ids)) as pbar:
                for start_idx in range(0, len(ids), chunk_size):
                    end_idx = start_idx + chunk_size
                    # 将句子进行编码
                    embeddings = model.encode(sentences[start_idx:end_idx], show_progress_bar=False)
                    bulk_data = []
                    for qid, question, embedding in zip(ids[start_idx:end_idx], sentences[start_idx:end_idx],
                                                        embeddings):
                        bulk_data.append({
                            "_index": dataset,
                            "_id": qid,
                            "_source": {
                                "question": question,
                                "question_vector": embedding
                            }
                        })
                    helpers.bulk(es, bulk_data)
                    pbar.update(chunk_size)
        except:
            print("During index an exception occured. Continue\n\n")




def basic_operations():
    # 默认host为localhost,port为9200.但也可以指定host与port
    es = Elasticsearch(hosts="10.20.32.187:9200")
    index = 'employer'
    # 使用create()方法创建一个名为 *** 的 Index
    result  = es.indices.create(index=index)
    # 如果创建成功，则会返回下面结果
    print(result)
    # {'acknowledged': True, 'shards_acknowledged': True, 'index': 'names'}

    doc_type = "test-type"
    id_num = "1"
    # Index 里面单条的记录称为 Document（文档）。许多条 Document 构成了一个 Index。
    # Document 使用 JSON 格式表示，下面是一个例子。
    data ={
      "user": "李四",
      "title": "工程师",
      "desc": "系统管理"
    }
    # 添加或更新数据
    # index 参数代表了索引名称，doc_type 代表了文档类型，body 则代表了文档具体内容，id 则是数据的唯一标识 ID。
    # index () 方法则不需要，如果不指定 id，会自动生成一个 id
    result = es.index(index=index,
             doc_type=doc_type,
             id=id_num,
             body=data)

    # 获取索引为my_index,文档类型为test_type的所有数据,result为一个字典类型
    result = es.search(index=index,doc_type=doc_type)


    #更新数据
    # 指定数据的 id 和内容，调用 update () 方法即可，但需要注意的是， Elasticsearch 对应的更新 API 对传递数据的格式是有要求的，更新时使用的具体代码如下：
    data ={
        "doc":{
          "user": "李四",
          "title": "工程师",
          "desc": "系统管理",
          "timestamp": datetime.now()
        }
    }
    result = es.update(index=index, doc_type=doc_type, body=data, id=1)
    print(result)
    # result 字段为 updated，即表示更新成功， _version字段，这代表更新后的版本号数，2 代表这是第二个版本，因为之前已经插入过一次数据，所以第一次插入的数据是版本 1。


    # 删除 Index
    result = es.indices.delete(index=index, ignore=[400, 404])
    print(result)


def create_index(hosts="10.20.32.187:9200",index='news',doc_type='politics'):
    # 默认host为localhost,port为9200.但也可以指定host与port
    es = Elasticsearch(hosts= hosts)
    # 使用create()方法创建一个名为 *** 的 Index
    es.indices.create(index=index,ignore=[400])
    # 如果创建成功，则会返回下面结果, {'acknowledged': True, 'shards_acknowledged': True, 'index': 'names'}
    # 新建一个索引并指定需要分词的字段
    # mapping 信息中指定了分词的字段，🌝其中将 title 字段的类型 type 指定为 text，
    # 并将分词器 analyzer 和搜索分词器 search_analyzer 设置为 ik_max_word ，即使用刚刚安装的中文分词插件，如果不指定，则默认使用英文分词器
    mapping = {
        'properties': {
            'title': {
                'type': 'text',
                'analyzer': 'ik_max_word',
                'search_analyzer': 'ik_max_word'
            }
        }
    }
    result = es.indices.put_mapping(index=index, doc_type=doc_type, body=mapping)
    print(result)

    return es


datas = [
    {
        'title': '设计灵魂离职，走下神坛的苹果设计将去向何方？',
        'url': 'https://www.tmtpost.com/4033397.html',
        'date': '2019-06-29 11:30'
    },
    {
        'title': '医生中的建筑设计师，凭什么挽救了上万人的生命？',
        'url': 'https://www.tmtpost.com/4034052.html',
        'date': '2019-06-29 11:10'
    },
    {
        'title': '中国网红二十年：从痞子蔡、芙蓉姐姐到李佳琦，流量与变现的博弈',
        'url': 'https://www.tmtpost.com/4034045.html',
        'date': '2019-06-29 11:03'
    },
    {
        'title': '网易云音乐、喜马拉雅等音频类应用被下架，或因违反相关规定',
        'url': 'https://www.tmtpost.com/nictation/4034040.html',
        'date': '2019-06-29 10:07'
    }
]

def dsl_search(es, index='news', doc_type='politics',title='网红 设计师' ):
    """
    Elasticsearch 支持的 DSL 语句来进行查询，使用 match 指定全文检索，检索的字段是 title
    """
    dsl = {
        'query': {
            'match': {
                'title': title
            }
        }
    }
    result = es.search(index=index, doc_type=doc_type, body=dsl)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def interactive_search(query=None,model=None,dataset='quora'):
    """
    #Interactive search queries
    交互式搜索
    """
    es = Elasticsearch()
    if query is None:
        query = input("Please enter a question: ")
    encode_start_time = time.time()
    question_embedding = model.encode(query)
    encode_end_time = time.time()

    # Lexical search
    bm25 = es.search(index=dataset, body={"query": {"match": {"question": query}}})

    # Sematic search
    sem_search = es.search(index=dataset, body={
          "query": {
            "script_score": {
              "query": {"match_all": {}},
              "script": {
                "source": "cosineSimilarity(params.queryVector, doc['question_vector']) + 1.0",
                "params": { "queryVector": question_embedding}
              }
            }
          }
        })

    print("Input question:", query)
    print("Computing the embedding took {:.3f} seconds, BM25 search took {:.3f} seconds, semantic search with ES took {:.3f} seconds".format(encode_end_time-encode_start_time, bm25['took']/1000, sem_search['took']/1000))

    print("BM25 results:")
    for hit in bm25['hits']['hits'][0:5]:
        print("\t{}".format(hit['_source']['question']))

    print("\nSemantic Search results:")
    for hit in sem_search['hits']['hits'][0:5]:
        print("\t{}".format(hit['_source']['question']))

    print("\n\n========\n")

if __name__ == '__main__':
    es = create_index()
    for data in datas:
        es.index(index='news', doc_type='politics', body=data)

    result = es.search(index='news', doc_type='politics')
    print(json.dumps(result, indent=4, ensure_ascii=False))


    result = dsl_search(es=es)




