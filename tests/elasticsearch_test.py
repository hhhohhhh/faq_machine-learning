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
from elasticsearch import Elasticsearch


def create_index(hosts="10.20.32.187:9200",index='news',doc_type='politics'):
    # 默认host为localhost,port为9200.但也可以指定host与port
    es = Elasticsearch(hosts=hosts)
    # 使用create()方法创建一个名为 *** 的 Index
    es.indices.create(index=index,ignore=[400])
    # 如果创建成功，则会返回下面结果, {'acknowledged': True, 'shards_acknowledged': True, 'index': 'names'}
    # 新建一个索引并指定需要分词的字段
    # mapping 信息中指定了分词的字段，🌝其中将 title 字段的类型 type 指定为 text，并将分词器 analyzer 和搜索分词器 search_analyzer 设置为 ik_max_word ，即使用刚刚安装的中文分词插件，如果不指定，则默认使用英文分词器
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

    return result


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