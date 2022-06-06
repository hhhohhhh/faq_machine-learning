#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/10/27 16:44 


@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/27 16:44   wangfc      1.0         None
"""
from typing import Text


class MedicalDiagnosisAgent():

    # 1.意图识别输出 意图，找到对应 RESPONSE_TEMPLATE
    # 2. 当存在 对应的 slot的时候，生成对应的 cql_template
    # 3. 使用 cql 到指定的图数据进行搜索
    # 4. 将回答的结果放入 reply_template
    INTENT_TO_RESPONSE_TEMPLATE = {
        "定义": {
            "slot_list": ["Disease"],
            "slot_values": None,
            "cql_template": "MATCH(p:疾病) WHERE p.name='{Disease}' RETURN p.desc",
            "reply_template": "{Disease} 是这样的：\n",
            "ask_template": "您问的是 {Disease} 的定义吗？",
            "intent_strategy": "",
            "deny_response": "很抱歉没有理解你的意思呢~"
        },
        "病因": {
            "slot_list": ["Disease"],
            "slot_values": None,
            "cql_template": "MATCH(p:疾病) WHERE p.name='{Disease}' RETURN p.cause",
            "reply_template": "{Disease} 疾病的原因是：\n",
            "ask_template": "您问的是疾病 {Disease} 的原因吗？",
            "intent_strategy": "",
            "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
        },
        "预防": {
            "slot_list": ["Disease"],
            "slot_values": None,
            "cql_template": "MATCH(p:疾病) WHERE p.name='{Disease}' RETURN p.prevent",
            "reply_template": "关于 {Disease} 疾病您可以这样预防：\n",
            "ask_template": "请问您问的是疾病 {Disease} 的预防措施吗？",
            "intent_strategy": "",
            "deny_response": "额~似乎有点不理解你说的是啥呢~"
        },
        "临床表现(病症表现)": {
            "slot_list": ["Disease"],
            "slot_values": None,
            "cql_template": "MATCH(p:疾病)-[r:has_symptom]->(q:症状) WHERE p.name='{Disease}' RETURN q.name",
            "reply_template": "{Disease} 疾病的病症表现一般是这样的：\n",
            "ask_template": "您问的是疾病 {Disease} 的症状表现吗？",
            "intent_strategy": "",
            "deny_response": "人类的语言太难了！！"
        },
        "相关病症": {
            "slot_list": ["Disease"],
            "slot_values": None,
            "cql_template": "MATCH(p:疾病)-[r:acompany_with]->(q:疾病) WHERE p.name='{Disease}' RETURN q.name",
            "reply_template": "{Disease} 疾病的具有以下并发疾病：\n",
            "ask_template": "您问的是疾病 {Disease} 的并发疾病吗？",
            "intent_strategy": "",
            "deny_response": "人类的语言太难了！！~"
        },
        "治疗方法": {
            "slot_list": ["Disease"],
            "slot_values": None,
            "cql_template": ["MATCH(p:疾病) WHERE p.name='{Disease}' RETURN p.cure_way",
                             "MATCH(p:疾病)-[r:recommand_drug]->(q) WHERE p.name='{Disease}' RETURN q.name",
                             "MATCH(p:疾病)-[r:recommand_recipes]->(q) WHERE p.name='{Disease}' RETURN q.name"],
            "reply_template": "{Disease} 疾病的治疗方式、可用的药物、推荐菜肴有：\n",
            "ask_template": "您问的是疾病 {Disease} 的治疗方法吗？",
            "intent_strategy": "",
            "deny_response": "没有理解您说的意思哦~"
        },
        "所属科室": {
            "slot_list": ["Disease"],
            "slot_values": None,
            # cure_department 可能是多个的时候出现错误
            "cql_template": "MATCH(p:疾病)-[r:cure_department]->(q:科室) WHERE p.name='{Disease}' RETURN q.name",
            "reply_template": "得了 {Disease} 可以挂这个科室哦：\n",
            "ask_template": "您想问的是疾病 {Disease} 要挂什么科室吗？",
            "intent_strategy": "",
            "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
        },
        "传染性": {
            "slot_list": ["Disease"],
            "slot_values": None,
            "cql_template": "MATCH(p:疾病) WHERE p.name='{Disease}' RETURN p.easy_get",
            "reply_template": "{Disease} 较为容易感染这些人群：\n",
            "ask_template": "您想问的是疾病 {Disease} 会感染哪些人吗？",
            "intent_strategy": "",
            "deny_response": "没有理解您说的意思哦~"
        },
        "治愈率": {
            "slot_list": ["Disease"],
            "slot_values": None,
            "cql_template": "MATCH(p:疾病) WHERE p.name='{Disease}' RETURN p.cured_prob",
            "reply_template": "得了{Disease} 的治愈率为：",
            "ask_template": "您想问 {Disease} 的治愈率吗？",
            "intent_strategy": "",
            "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
        },
        "治疗时间": {
            "slot_list": ["Disease"],
            "slot_values": None,
            "cql_template": "MATCH(p:疾病) WHERE p.name='{Disease}' RETURN p.cure_lasttime",
            "reply_template": "疾病 {Disease} 的治疗周期为：",
            "ask_template": "您想问 {Disease} 的治疗周期吗？",
            "intent_strategy": "",
            "deny_response": "很抱歉没有理解你的意思呢~"
        },
        "化验/体检方案": {
            "slot_list": ["Disease"],
            "slot_values": None,
            "cql_template": "MATCH(p:疾病)-[r:need_check]->(q:检查) WHERE p.name='{Disease}' RETURN q.name",
            "reply_template": "得了 {Disease} 需要做以下检查：\n",
            "ask_template": "您是想问 {Disease} 要做什么检查吗？",
            "intent_strategy": "",
            "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
        },
        "禁忌": {
            "slot_list": ["Disease"],
            "slot_values": None,
            "cql_template": "MATCH(p:疾病)-[r:not_eat]->(q:食物) WHERE p.name='{Disease}' RETURN q.name",
            "reply_template": "得了 {Disease} 切记不要吃这些食物哦：\n",
            "ask_template": "您是想问 {Disease} 不可以吃的食物是什么吗？",
            "intent_strategy": "",
            "deny_response": "额~似乎有点不理解你说的是啥呢~~"
        },
        "unrecognized": {
            "slot_values": None,
            "replay_answer": "非常抱歉，我还不知道如何回答您，我正在努力学习中~",
        }
    }

    def __init__(self,tmp_model_dir, nlu_interpreter,graph):
        self.graph = graph
        # tmp_model_dir 为 rasa 模型加压后的临时路径，在中断服务的时候需要删除资源
        self.tmp_model_dir = tmp_model_dir
        self.nlu_interpreter = nlu_interpreter



    async def handle_message(self,message:Text):
        """
        处理 message,返回 response 的方法
        """
        # 使用 nlu_interpreter 进行解析
        result = await self.nlu_interpreter.parse(message)
        # 使用 nlu_interpreter 解析结果更新对应意图的 response_template
        response_template = self._update_response_template(result)
        # 使用 slot 信息和 cql template 进行图搜索，返回对应的 response，并且 reply template 拼接为 response
        response = self._get_response(response_template)
        return response

        # 根据意图强度来确认回复策略
        # conf = intent_rst.get("confidence")
        # if conf >= intent_threshold_config["accept"]:
        #     slot_info["intent_strategy"] = "accept"
        # elif conf >= intent_threshold_config["deny"]:
        #     slot_info["intent_strategy"] = "clarify"
        # else:
        #     slot_info["intent_strategy"] = "deny"

    def _update_response_template(self,result):
        intent = result.get('intent').get('name')
        confidence = result.get('confidence')
        # entities = [{'entity': 'disease', 'start': 0, 'end': 2, 'confidence_entity': 0.9995735287666321, 'value': '脑瘤', 'extractor': 'DIETClassifier'}],
        entities_info = result.get('entities')

        response_template = self.INTENT_TO_RESPONSE_TEMPLATE.get(intent).copy()
        # 对需要的槽位进行填槽
        slot_attributes = response_template.get("slot_list")
        slot_attribute_to_value_dict = {}
        for slot_attribute in slot_attributes:
            # 初始化 slot_attribute_to_value_dict 对应的 slot_name的值为 None
            slot_attribute_to_value_dict[slot_attribute] = None
            for entity_info in entities_info:
                # 当 slot_attribute 和 entity的属性 相同的时候,更新槽位的信息
                if slot_attribute.lower() == entity_info['entity']:
                    slot_attribute_to_value_dict[slot_attribute] = \
                        self._entity_link(entity_info['value'], entity_info['entity'])

        # 更新 response_template
        response_template["slot_values"] = slot_attribute_to_value_dict
        return response_template


    def _entity_link(self,mention, etype):
        """
        对于识别到的实体mention,如果其不是知识库中的标准称谓
        则对其进行实体链指，将其指向一个唯一实体（待实现）
        """
        return mention

    def _get_response(self,response_template):
        """
        根据语义槽获取答案回复
        """
        cql_template = response_template.get("cql_template")
        reply_template = response_template.get("reply_template")
        ask_template = response_template.get("ask_template")
        slot_values = response_template.get("slot_values")
        strategy = response_template.get("intent_strategy")

        # if strategy == "accept":
        cql = []
        if isinstance(cql_template, list):
            for cqlt in cql_template:
                cql.append(cqlt.format(**slot_values))
        else:
            cql = cql_template.format(**slot_values)
        # 进行数据库搜索
        answer = neo4j_searcher(graph=self.graph,cql_list=cql)
        if not answer or answer =='None':
            response_template["replay_answer"] = "唔~我装满知识的大脑此刻很贫瘠"
        else:
            pattern = reply_template.format(**slot_values)
            response_template["replay_answer"] = pattern + answer
        response = response_template["replay_answer"]
        # elif strategy == "clarify":
        #     # 澄清用户是否问该问题
        #     pattern = ask_template.format(**slot_values)
        #     slot_info["replay_answer"] = pattern
        #     # 得到肯定意图之后需要给用户回复的答案
        #     cql = []
        #     if isinstance(cql_template, list):
        #         for cqlt in cql_template:
        #             cql.append(cqlt.format(**slot_values))
        #     else:
        #         cql = cql_template.format(**slot_values)
        #     answer = neo4j_searcher(cql)
        #     if not answer:
        #         slot_info["replay_answer"] = "唔~我装满知识的大脑此刻很贫瘠"
        #     else:
        #         pattern = reply_template.format(**slot_values)
        #         slot_info["choice_answer"] = pattern + answer
        # elif strategy == "deny":
        #     slot_info["replay_answer"] = slot_info.get("deny_response")

        return response


def neo4j_searcher(graph,cql_list):
    ress = ""
    if isinstance(cql_list, list):
        for cql in cql_list:
            rst = []
            data = graph.run(cql).data()
            if not data:
                continue
            for d in data:
                d = list(d.values())
                if isinstance(d[0], list):
                    rst.extend(d[0])
                else:
                    rst.extend(d)

            data = "、".join([str(i) for i in rst])
            ress += data + "\n"
    else:
        data = graph.run(cql_list).data()
        if not data:
            return ress
        rst = []
        for d in data:
            d = list(d.values())
            if isinstance(d[0], list):
                rst.extend(d[0])
            else:
                rst.extend(d)

        data = "、".join([str(i) for i in rst])
        ress += data

    return ress
