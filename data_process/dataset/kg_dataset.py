# encoding:utf8
import os
import re
import json
import codecs
import threading
from pathlib import Path
from typing import List, Text, Set, Dict

from py2neo import Graph
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

from data_process.dataset.intent_classifier_dataset import IntentClassifierProcessor
from data_process.dataset.rasa_dataset import DiagnosisDataToRasaDataset
from utils.io import load_json

logger = logging.getLogger(__name__)


def print_data_info(data_path):
    triples = []
    i = 0
    with open(data_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            data = json.loads(line)
            print(json.dumps(data, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
            i += 1
            if i >= 5:
                break
    return triples


class MedicalDiagnosisDataProcessor(object):
    """
    创建 docker中的 neo4j 数据库
    docker run -it -d -p 7474:7474 -p 7687:7687 --name neo4j_1 neo4j:4.3.6-community
    http://10.20.33.3:7474
    初始的密码为neo4j
    输入结束之后，会进入到新密码设置页面

    """

    def __init__(self,
                 corpus: Text = 'corpus',
                 dataset_dir: Text = 'kbqa',
                 medical_kg_data_filename='medical.json',
                 medical_kg_data_subdir="medical_kg_data",
                 output_data_subdir='kbqa_diagnosis_20211025',
                 label_column='label_class',
                 text_column='text',
                 output_dir='output/kbqa_diagnosis_bot_01',
                 neo4j_database_config=dict(host="127.0.0.1", port=7687, user="neo4j", password="123456"),
                 if_creat_rasa_format_data=True
                 ):
        self.data_dir = os.path.join(corpus, dataset_dir)

        self.medical_kg_data_path = os.path.join(self.data_dir, medical_kg_data_filename)
        # print_data_info(self.medical_kg_data_path)
        self.medical_kg_data_dir = os.path.join(self.data_dir, medical_kg_data_subdir)
        self.output_data_dir = os.path.join(self.data_dir, output_data_subdir)
        self.neo4j_database_config = neo4j_database_config
        self.graph = Graph(**neo4j_database_config)

        # 8 类实体的类型 作为 图节点： 可以使用实体识别进行识别
        self.entity_types = ['diseases', 'symptoms', 'drugs', 'recipes', 'foods', 'checks', 'departments', 'producers']

        # 10 类疫病的属性： 可以使用意图模型或者关系抽取
        self.disease_attributes = ['name', 'desc', 'prevent', 'cause', 'easy_get', 'cure_department', 'cure_way',
                                   'cure_lasttime', 'symptom', 'cured_prob']

        # 11 类关系类型: 可以使用意图模型或者关系抽取
        self.relation_types = ['rels_symptom', 'rels_acompany', 'rels_category', 'rels_department', 'rels_noteat',
                               'rels_doeat', 'rels_recommandeat', 'rels_commonddrug', 'rels_recommanddrug',
                               'rels_check', 'rels_drug_producer']

        self.get_kg()

        # 读取意图数据
        indent_classifier_processor = IntentClassifierProcessor(dataset_dir=dataset_dir,
                                                                output_data_subdir=output_data_subdir,
                                                                label_column=label_column,
                                                                text_column=text_column)

        if if_creat_rasa_format_data:
            raw_data = indent_classifier_processor.new_version_data_dict['train']
            # 创建 rasa 格式的数据进行意图和 实体的标注，后期训练 rasa 的 diet 模型进行识别
            self.diagnosis_to_rasa_dataset = DiagnosisDataToRasaDataset(raw_data=raw_data, output_dir=output_dir,
                                                                   entity_attribute_to_value_dict=self.entity_attribute_to_value_dict,
                                                                   intent_column=label_column,
                                                                   question_column=text_column,
                                                                   use_label_as_response=True
                                                                   )

    def get_kg(self):
        medical_kg_data_filenames = self.get_medical_kg_data()
        if medical_kg_data_filenames is None:
            self.build_kg()

        self.entity_attribute_to_value_dict = self._get_entity_attribute_to_value_dict()

    def build_kg(self):
        # 共8类节点:
        # 使用实体识别模型对这 8 种实体进行识别
        self.diseases = set()  # 疾病
        self.symptoms = set()  # 症状
        self.drugs = set()  # 药品
        self.recipes = set()  # 菜谱
        self.foods = set()  # 食物
        self.checks = set()  # 检查
        self.departments = set()  # 科室
        self.producers = set()  # 药企
        self.disease_infos = []  # 疾病信息

        # 构建 11 类节点实体关系
        # 使用意图识别模型来识别意图，该意图 可以映射为 某种关系
        self.rels_symptom = set()  # 疾病症状关系
        self.rels_acompany = set()  # 疾病并发关系
        self.rels_category = set()  # 疾病与科室之间的关系
        self.rels_department = set()  # 科室－科室关系
        self.rels_noteat = set()  # 疾病－忌吃食物关系
        self.rels_doeat = set()  # 疾病－宜吃食物关系
        self.rels_recommandeat = set()  # 疾病－推荐吃食物关系
        self.rels_commonddrug = set()  # 疾病－通用药品关系
        self.rels_recommanddrug = set()  # 疾病－热门药品关系
        self.rels_check = set()  # 疾病－检查关系
        self.rels_drug_producer = set()  # 厂商－药物关系

        self.extract_triples(self.data_path)
        self.create_entitys()
        self.create_relations()

        self.set_diseases_attributes()
        self.export_entitys_relations()

    def extract_triples(self, data_path):
        print("从json文件中转换抽取三元组")
        with open(data_path, 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines(), ncols=80):
                data_json = json.loads(line)
                disease_dict = {}
                disease = data_json['name']
                # 获取疾病的属性
                disease_dict['name'] = disease
                disease_dict['desc'] = ''
                disease_dict['prevent'] = ''
                disease_dict['cause'] = ''
                disease_dict['easy_get'] = ''
                disease_dict['cure_department'] = ''
                disease_dict['cure_way'] = ''
                disease_dict['cure_lasttime'] = ''
                disease_dict['symptom'] = ''
                disease_dict['cured_prob'] = ''
                self.diseases.add(disease)

                if 'symptom' in data_json:
                    for symptom in data_json['symptom']:
                        self.symptoms.add(symptom)
                        self.rels_symptom.add((disease, 'has_symptom', symptom))

                if 'acompany' in data_json:
                    for acompany in data_json['acompany']:
                        self.rels_acompany.add((disease, 'acompany_with', acompany))
                        self.diseases.add(acompany)

                if 'desc' in data_json:
                    disease_dict['desc'] = data_json['desc']

                if 'prevent' in data_json:
                    disease_dict['prevent'] = data_json['prevent']

                if 'cause' in data_json:
                    disease_dict['cause'] = data_json['cause']

                if 'get_prob' in data_json:
                    disease_dict['get_prob'] = data_json['get_prob']

                if 'easy_get' in data_json:
                    disease_dict['easy_get'] = data_json['easy_get']

                if 'cure_department' in data_json:
                    cure_department = data_json['cure_department']
                    if len(cure_department) == 1:
                        self.rels_category.add((disease, 'cure_department', cure_department[0]))
                        self.departments.add(cure_department[0])
                    if len(cure_department) == 2:
                        big = cure_department[0]
                        small = cure_department[1]
                        self.rels_department.add((small, 'belongs_to', big))
                        self.rels_category.add((disease, 'cure_department', small))

                        self.departments.add(small)
                        self.departments.add(big)

                    disease_dict['cure_department'] = cure_department

                if 'cure_way' in data_json:
                    disease_dict['cure_way'] = data_json['cure_way']

                if 'cure_lasttime' in data_json:
                    disease_dict['cure_lasttime'] = data_json['cure_lasttime']

                if 'cured_prob' in data_json:
                    disease_dict['cured_prob'] = data_json['cured_prob']

                if 'common_drug' in data_json:
                    common_drug = data_json['common_drug']
                    for drug in common_drug:
                        self.drugs.add(drug)
                        self.rels_commonddrug.add((disease, 'has_common_drug', drug))

                if 'recommand_drug' in data_json:
                    recommand_drug = data_json['recommand_drug']
                    for drug in recommand_drug:
                        self.drugs.add(drug)
                        self.rels_recommanddrug.add((disease, 'recommand_drug', drug))

                if 'not_eat' in data_json:
                    not_eat = data_json['not_eat']
                    for _not in not_eat:
                        self.rels_noteat.add((disease, 'not_eat', _not))
                        self.foods.add(_not)

                    do_eat = data_json['do_eat']
                    for _do in do_eat:
                        self.rels_doeat.add((disease, 'do_eat', _do))
                        self.foods.add(_do)

                if 'recommand_eat' in data_json:
                    recommand_eat = data_json['recommand_eat']
                    for _recommand in recommand_eat:
                        self.recipes.add(_recommand)
                        self.rels_recommandeat.add((disease, 'recommand_recipes', _recommand))

                if 'check' in data_json:
                    check = data_json['check']
                    for _check in check:
                        self.checks.add(_check)
                        self.rels_check.add((disease, 'need_check', _check))

                if 'drug_detail' in data_json:
                    for det in data_json['drug_detail']:
                        det_spilt = det.split('(')
                        if len(det_spilt) == 2:
                            p, d = det_spilt
                            d = d.rstrip(')')
                            if p.find(d) > 0:
                                p = p.rstrip(d)
                            self.producers.add(p)
                            self.drugs.add(d)
                            self.rels_drug_producer.add((p, 'production', d))
                        else:
                            d = det_spilt[0]
                            self.drugs.add(d)

                self.disease_infos.append(disease_dict)

    def write_nodes(self, entitys: Set[Text], entity_type: Text):
        """
        节点(Nodes) :
        Cypher使用()来表示一个节点
        () #最简单的节点形式，表示一个任意无特征的节点，其实就是一个空节点
        (matrix) #如果想指向一个节点在其他地方，我们可以给节点添加一个变量名(matrix)。
                  变量名被限制单一的语句里，一个变量名在不同的语句中可能有不同的意义，或者没意义
        (:Movie) #添加标签名称， 标签可以将节点进行分组
        (matrix:Movie) #一个节点有一个标签名（Movie），并且把它分配到变量matrix上
                       #节点的标签(被定义在：后)，用于声明节点的类型或者角色。
                       # 注意节点可以有多个标签。标签被用于 限制搜索 pattern， 保留他们在匹配结构中，不使用标签在query里。
        (matrix:Movie {title: "The Matrix"}) #花括号里定义节点的属性，属性都是键值对,属性可以用来存储信息或者来条件匹配(查找）
        (matrix:Movie {title: "The Matrix", released: 1999}) #多个属性
        (matrix:Movie:Promoted) #多个标签

        创建一个节点：
        CREATE (matrix:Movie {tagline:"Welcome to the Real World",title:"The Matrix",released:"1999"})

        # 搜索节点
        MATCH (movie:Movie) RETURN node
        MATCH (movie:Movie {title:"The Matrix"}) RETURN movie

        1. 查询 label=疾病，name=肺气肿的 节点:
        MATCH (n:疾病{name:'肺气肿'}) return n

        """
        # 去除重复的值
        print("写入 {0} 实体共 {1} 个".format(entity_type, entitys.__len__()))
        for node in tqdm(entitys, ncols=80):
            cql = """MERGE(n:{label}{{name:'{entity_name}'}})""".format(
                label=entity_type, entity_name=node.replace("'", ""))
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)

    def write_edges(self, triples, head_type, tail_type):
        """
        查询 节点为 肺气肿 的关系 recommand_recipes 的所有节点的名称
        MATCH(p:疾病)-[r:recommand_recipes]->(q) WHERE p.name='肺气肿' RETURN q.name
        """
        print("写入关系共 {} 个".format(triples.__len__()))
        for head, relation, tail in tqdm(triples, ncols=80):
            cql = """MATCH(p:{head_type}),(q:{tail_type})
                    WHERE p.name='{head}' AND q.name='{tail}'
                    MERGE (p)-[r:{relation}]->(q)""".format(
                head_type=head_type, tail_type=tail_type, head=head.replace("'", ""),
                tail=tail.replace("'", ""), relation=relation)
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)

    def set_attributes(self, disease_infos, etype):
        """
        @time:  2021/10/22 14:47
        @author:wangfc
        @version:
        @description: 写入 疾病的所有属性

        @params:
        @return:
        """
        print("写入 {0} 实体的属性共 {1} 个".format(etype, disease_infos.__len__()))
        for e_dict in tqdm(disease_infos, ncols=80):
            name = e_dict['name']
            for k, v in e_dict.items():
                if k in ['cure_department', 'cure_way']:
                    cql = """MATCH (n:{label})
                        WHERE n.name='{name}'
                        set n.{k}={v}""".format(label=etype, name=name.replace("'", ""), k=k, v=v)
                else:
                    cql = """MATCH (n:{label})
                        WHERE n.name='{name}'
                        set n.{k}='{v}'""".format(label=etype, name=name.replace("'", ""), k=k,
                                                  v=v.replace("'", "").replace("\n", ""))
                try:
                    self.graph.run(cql)
                except Exception as e:
                    print(e)
                    print(cql)

    def create_entitys(self):
        """
        entity_types = '疾病','症状', '药品', '菜谱','菜谱','检查','科室', '药企'
        """
        self.write_nodes(self.drugs, '药品')
        self.write_nodes(self.recipes, '菜谱')
        self.write_nodes(self.foods, '菜谱')
        self.write_nodes(self.checks, '检查')
        self.write_nodes(self.departments, '科室')
        self.write_nodes(self.producers, '药企')
        self.write_nodes(self.diseases, '疾病')
        self.write_nodes(self.symptoms, '症状')

    def create_relations(self):
        self.write_edges(self.rels_department, '科室', '科室')
        self.write_edges(self.rels_noteat, '疾病', '食物')
        self.write_edges(self.rels_doeat, '疾病', '食物')
        self.write_edges(self.rels_recommandeat, '疾病', '菜谱')
        self.write_edges(self.rels_commonddrug, '疾病', '药品')
        self.write_edges(self.rels_recommanddrug, '疾病', '药品')
        self.write_edges(self.rels_check, '疾病', '检查')
        self.write_edges(self.rels_drug_producer, '药企', '药品')
        self.write_edges(self.rels_symptom, '疾病', '症状')
        self.write_edges(self.rels_acompany, '疾病', '疾病')
        self.write_edges(self.rels_category, '疾病', '科室')

    def set_diseases_attributes(self):
        # self.set_attributes(self.disease_infos,"疾病")
        t = threading.Thread(target=self.set_attributes, args=(self.disease_infos, "疾病"))
        t.setDaemon(False)
        t.start()

    def get_medical_kg_data(self):
        """
        读取保存的graph json数据
        """
        data_filenames = os.listdir(self.medical_kg_data_dir)
        for data_filename in data_filenames:
            path = os.path.join(self.medical_kg_data_dir, data_filename)
            data = load_json(json_path=path)
            attribute_name = os.path.splitext(data_filename)[0]
            self.__setattr__(attribute_name, data)
            logger.info(f"读取 {attribute_name} 共 {data.__len__()} from {path}")
        return data_filenames

    def export_data(self, data: Set, data_filename: Text, path: Path = None):
        path = os.path.join(self.medical_kg_data_dir, data_filename)
        data = list(data)
        if isinstance(list(data)[0], str):
            data = sorted([d.strip("...") for d in set(data)])
        with codecs.open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def export_entitys_relations(self):
        self.export_data(self.drugs, 'drugs.json')
        self.export_data(self.recipes, 'recipes.json')
        self.export_data(self.foods, 'foods.json')
        self.export_data(self.checks, 'checks.json')
        self.export_data(self.departments, 'departments.json')
        self.export_data(self.producers, 'producers.json')
        self.export_data(self.diseases, 'diseases.json')
        self.export_data(self.symptoms, 'symptoms.json')

        self.export_data(self.rels_department, 'rels_department.json')
        self.export_data(self.rels_noteat, 'rels_noteat.json')
        self.export_data(self.rels_doeat, 'rels_doeat.json')
        self.export_data(self.rels_recommandeat, 'rels_recommandeat.json')
        self.export_data(self.rels_commonddrug, 'rels_commonddrug.json')
        self.export_data(self.rels_recommanddrug, 'rels_recommanddrug.json')
        self.export_data(self.rels_check, 'rels_check.json')
        self.export_data(self.rels_drug_producer, 'rels_drug_producer.json')
        self.export_data(self.rels_symptom, 'rels_symptom.json')
        self.export_data(self.rels_acompany, 'rels_acompany.json')
        self.export_data(self.rels_category, 'rels_category.json')

    def _get_entity_attribute_to_value_dict(self) -> Dict[Text, List[Text]]:
        """
        获取 entity_attribute 对应的值的字典
        1) entity value 为空字符的情况
        2) entity_values 存在着交叉的情况
            disease vs symptom
        """
        def if_drop_entity_value(entity_value):
            if_drop=False
            if entity_value == "" or entity_value.__len__() == 1 or re.match(pattern=r'\d+', string=entity_value)\
                    or entity_value=='医生':
                if_drop =  True

            return if_drop

        entity_type_to_value_dict = {}
        for entity_type in self.entity_types:
            entity_values = self.__getattribute__(entity_type)
            # 过滤 entity_value 为空的情况
            entity_values_set = set()
            for entity_value in entity_values:
                if not isinstance(entity_value, str):
                    logger.error(f"entity_type={entity_type},value={entity_value} is not str type")
                entity_value = entity_value.strip()
                if if_drop_entity_value(entity_value):
                    logger.warning(f"entity_type={entity_type},value={entity_value} is 空字符或者长度为1的值")
                else:
                    entity_values_set.add(entity_value)
            # 去除 entity_attribute 复数形式s
            entity_type_to_value_dict.update({entity_type[:-1]: entity_values_set})

        intersection_dict = self._check_entity_values(entity_type_to_value_dict)
        entity_type_to_value_dict = self._adjust_entity_values(intersection_dict,entity_type_to_value_dict)
        return entity_type_to_value_dict

    def _check_entity_values(self, entity_type_to_value_dict: Dict[Text, Set[Text]]):
        intersection_dict = {}
        compare_entity_types = set(entity_type_to_value_dict.keys())
        for entity_type, entity_values in entity_type_to_value_dict.items():
            compare_entity_types = compare_entity_types.difference({entity_type})
            for another_entity_type in compare_entity_types:
                another_entity_values = entity_type_to_value_dict.get(another_entity_type)
                intersection_values = entity_values.intersection(another_entity_values)
                if intersection_values:
                    # values_str = '\n'.join(intersection_values)
                    logger.error(
                        f"{entity_type} 和 {another_entity_type} 存在交集个数共={intersection_values.__len__()}:\n{intersection_values}")
                    intersection_dict.update({(entity_type, another_entity_type): intersection_values})
        return intersection_dict

    def _adjust_entity_values(self, intersection_dict, entity_type_to_value_dict: Dict[Text, Set[Text]]):
        entity_type_to_pattern = dict(disease=r'.*[疹|症|病|炎|疮|癣|癌|近视|斜视|障碍|溃疡|畸形|痛经|结石|穿孔|水肿|脂肪肝|息肉'
                                              r'|梗阻|梗塞|头痛|斜颈|耳聋|息肉|静脉曲张'
                                              r'|骨裂|流产|囊肿|脓肿|瘤|脱位|骨折|贫血|中毒|黄疸|结核|高血压|痴呆|梗死|'
                                              r'性早熟|紫癜]$'
                                              r'|不孕不育|腹泻|冻疮|雀斑|痢疾|扁平苔癣|倒睫|鸡眼|中暑|遗精|遗尿'
                                              r'|风湿性血管炎|静脉曲张|偏头痛|脚气|口臭|失眠|扁平足|足外翻|拇外翻|膝内翻|睑外翻'
                                              r'肛裂|露阴癖|指节垫|信息成瘾|异性装扮癖|谵妄|慢性咳嗽'
                                              r'|甲状腺结节|蛔虫性肠梗阻|盲肠阿米巴肉芽肿'
                                              r'|心肌梗死|气胸|脓胸|乳糜胸|动脉粥样硬化|血胸|胎盘早剥|子痫'
                                              r'|遗传性肥胖|缺铁性贫血|蜘蛛痣|先天性无阴道|先天性无虹膜|新生儿呕血和便血|先天性X因子缺乏'
                                              r'|肠套叠|小?儿?便秘|小儿咳嗽|血管痣|隐睾|磨牙|鲜红斑痣'
                                              r'|腰椎间盘突出|坐骨神经痛|粉碎性骨折|偏瘫|截瘫'
                                              r'|脑积水|脑疝|重症肌无力|骨质疏松|急性胃扩张|巨舌'
                                              r'|尿潴留|一氧化碳中毒|瘫痪|晕厥|老年大便失禁|前列腺增生'
                                              r'|肺动静脉瘘|涎瘘|脑脊液鼻漏|病理性REM睡眠|神经性呕吐'
                                              r'|猝死|休克',
                                      symptom=r".*[阳性|征|痉挛|抽搐|萎缩|异常|耳鸣|内陷|损伤|损害|撕脱伤|脱离|困难|肥大|积液|感染|狭窄"
                                              r"|纤维化|衰竭|失调|失禁|过低|不足|闭锁"
                                              r"|减退|缺乏|过缓|拥挤|粘连|乳糜泻|危象|闭合不全|侧弯|功能不全|下垂|脱垂|稀疏"
                                              r"|瘙痒|营养不良|硬斑|出血|肿块|皲裂|出血|缺血|便血|失常|亢进|昏迷|震颤]$"
                                              r"|.+[杂|浊]音$|乳糜尿|肾性糖尿|月经量少|月经不调|口腔黏膜白斑"
                                              r"|腹胀|咳嗽|眩晕|肠粘连|脱发"
                                              r"|脱水|斑秃|早搏|乳头溢液|呃逆|肾阳虚|热衰竭|新生儿青紫|新生儿反应低下|新生儿低体温"
                                              r"|小儿遗尿|胎儿窘迫"
                                              r"|头皮血肿|痰饮|血瘀体质|气阴两虚"
                                              r"|β-氨基酸尿|蛋白尿|营养代谢缺乏|胎膜早破|夜惊|妊娠纹|鼻中隔穿孔|膀胱憩室"
                                              r"|偏执状态|延髓性麻痹|阴道横隔|尿磷|血尿|肺不张|",
                                      food="苹果|奶酪")

        # # check="(.(*)征)"
        # import re
        # string ='肝内胆管结石'
        # re.match(pattern = pattern,string=string)


        def match_and_adjust(entity_type, another_entity_type, intersection_value):
            entity_pattern = entity_type_to_pattern.get(entity_type)
            another_entity_pattern = entity_type_to_pattern.get(another_entity_type)
            if entity_pattern and re.match(pattern=entity_pattern, string=intersection_value):
                new_values = entity_type_to_value_dict.get(entity_type).union({intersection_value})
                entity_type_to_value_dict.update({entity_type:new_values})
                entity_type_to_value_dict.get(another_entity_type).difference_update({intersection_value})
            elif another_entity_pattern and re.match(pattern=another_entity_pattern, string=intersection_value):
                entity_type_to_value_dict.get(entity_type).difference_update({intersection_value})
                new_values = entity_type_to_value_dict.get(another_entity_type).union({intersection_value})
                entity_type_to_value_dict.update({another_entity_type:new_values})
            else:
                if (entity_type == 'disease' and another_entity_type == 'symptom') or \
                        (entity_type == 'disease' and another_entity_type == 'check') or \
                        (entity_type == 'disease' and another_entity_type == 'department') or \
                        (entity_type == 'symptom' and another_entity_type == 'check'):
                    logger.error(f"{intersection_value} 没有匹配到 {entity_type} 或者 {another_entity_type}")
                    entity_type_to_value_dict.get(entity_type).difference_update({intersection_value})
                    entity_type_to_value_dict.get(another_entity_type).difference_update({intersection_value})
                elif entity_type == 'drug' and another_entity_type == 'producer':
                    entity_type_to_value_dict.get(entity_type).union({intersection_value})
                    entity_type_to_value_dict.get(another_entity_type).difference_update({intersection_value})
                elif entity_type == 'recipe' and another_entity_type == 'food':
                    entity_type_to_value_dict.get(entity_type).difference_update({intersection_value})
                    entity_type_to_value_dict.get(another_entity_type).union({intersection_value})
                else:
                    logger.error(f"{intersection_value} 没有匹配到 {entity_type} 或者 {another_entity_type}")

        for (entity_type, another_entity_type), intersection_values in intersection_dict.items():
            for intersection_value in intersection_values:
                match_and_adjust(entity_type, another_entity_type, intersection_value)

        intersection_dict = self._check_entity_values(entity_type_to_value_dict)
        if intersection_dict:
            logger.error(f"仍旧存在交叉的entity value={intersection_dict}")
            raise ValueError

        return entity_type_to_value_dict




if __name__ == '__main__':
    path = "./graph_data/medical.json"
    # print_data_info(path)
    extractor = MedicalDiagnosisDataProcessor()
    extractor.extract_triples(path)
    # extractor.create_entitys()
    # extractor.create_relations()
    # extractor.set_diseases_attributes()
    extractor.export_entitys_relations()
