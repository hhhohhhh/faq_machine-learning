# elasticsearch 安装
## 配置好java环境变量
从官网下载 elasticsearch，解压，进入Elasticsearch目录
在linux环境下输入以下执行命令即可启动：
bin/elasticsearch

默认启动外网是无法访问的，需要对配置文件进行修改：
vi config/elasticsearch.yml
添加如下内容：
network.host: 0.0.0.0
http.port: 8200
(默认的9200)

关闭es再次启动：（通常来说会遇到如下错误，因为es对资源需求量较大，一般linux默认配置并没有达到es资源需求）
【报错1】max file descriptors [4096] for elasticsearch process likely too low, consider increasing to at least [65536]
需要修改系统文件/etc/security/limits.conf并添加（或修改）如下行
* soft nofile 65536
* hard nofile 131072
其中的*号代表该配置对所有用户适用

注意！修改完后再启动es会发现错误依旧存在，这时候应先退出登录（windows适用putty远程登录linux的用户应退出putty），再次登录，方可生效。
【报错2】max virtual memory areas vm.max_map_count [65530] likely too low, increase to at least [262144]
需要修改系统文件/etc/sysctl.conf并添加（或修改）如下行

vm.max_map_count=655360
保存退出后，需要运行如下命令使其生效：
sysctl -p

此时我们通过浏览器打开页面，就能访问ES了：
roc-3:8200

# Python API
在CentOS 7上安装了Python3.6，安装时使用下面的命令
pip install elasticsearch

```json
{
name: "t2ztM-f",
cluster_name: "docker-cluster",
cluster_uuid: "DTTrGi_UR12p8Vbc9MTNAQ",
version: {
number: "6.3.2",
build_flavor: "oss",
build_type: "tar",
build_hash: "053779d",
build_date: "2018-07-20T05:20:23.451332Z",
build_snapshot: false,
lucene_version: "7.3.1",
minimum_wire_compatibility_version: "5.6.0",
minimum_index_compatibility_version: "5.0.0"
},
tagline: "You Know, for Search"
}

```


如果我们配置了特殊的端口（例如本文中配置为8086，而不是默认的9200）应该如何访问呢？我们以python为例：
```python
from datetime import datetime
from elasticsearch import Elasticsearch

# 默认host为localhost,port为9200.但也可以指定host与port
es = Elasticsearch(hosts="10.20.32.187:9200")
index = 'employer'
# 使用create()方法创建一个名为 *** 的 Index
es.indices.create(index=index)
# 如果创建成功，则会返回下面结果
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
es.index(index=index,
         doc_type=doc_type,
         id=id_num,
         body=data)


# 获取索引为my_index,文档类型为test_type的所有数据,result为一个字典类型
result = es.search(index=index,doc_type=doc_type)

# 或者这样写:搜索id=1的文档
res = es.get(index=index, doc_type=doc_type, id=id_num)
for k in res.keys():
    print(k, res[k])


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

# 删除数据:删除数据调用 delete() 方法，指定需要删除的数据 id 
result = es.delete(index=index,doc_type=doc_type,id=id_num, ignore=[400, 404])

# 删除 Index
result = es.indices.delete(index=index, ignore=[400, 404])
print(result)

```

# 查询数据
```python
from elasticsearch import Elasticsearch
es = Elasticsearch()

# 新建一个索引并指定需要分词的字段
mapping = {
    'properties': {
        'title': {
            'type': 'text',
            'analyzer': 'ik_max_word',
            'search_analyzer': 'ik_max_word'
        }
    }
}
es.indices.delete(index='people', ignore=[400, 404])
es.indices.create(index='news', ignore=400)
result = es.indices.put_mapping(index='news', doc_type='politics', body=mapping)
print(result)



```
