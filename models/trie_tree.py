#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/19 11:14 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/19 11:14   wangfc      1.0         None
"""
from copy import deepcopy
from typing import Text, List, Union, Dict, Tuple
from collections import defaultdict
from enum import Enum, unique
from collections import deque

# from data_process import text_preprocess
# from data_process.data_example import FAQSentenceExample
import logging
logger = logging.getLogger(__name__)

@unique
class SequenceType(Enum):
    word = 'word'
    sentence = 'sentence'

class Entity():
    def __init__(self,text:Text,start_position:int,end_position:int,entity_type:Text):
        """
        end_position: 不包含的关系
        """
        self.text = text
        self.start_position = start_position
        self.end_position = end_position
        self.entity_type = entity_type

    def __repr__(self):
        return f"entity={self.text},entity_type={self.entity_type},start_position={self.start_position}," \
               f"end_position={self.end_position}"


class TrieNode:
    def __init__(self, value='', length=0, sequence_type: SequenceType = SequenceType.sentence):
        # 标记记录 node的 sequence 类型
        # self.sequence_type = sequence_type

        # 记录该节点的所有子节点： 字典的形式
        # defaultdict 接受一个工厂函数作为参数,factory_function 可以是list、set、str等等，
        # 作用是当key不存在时，返回的是工厂函数的默认值，比如list对应[ ]，str对应的是空字符串，set对应set( )，int对应
        self.children = defaultdict(TrieNode)

        # 标记该节点是否为一个词的结束位置，用于分词
        self._is_word = False
        # 如果是 word的话，可以增加 word 的属性,可以用于 实体标注
        self._word_type = None

        # 标记该节点是否是一个句子的结束位置
        self._is_sentence = False
        # 标记 sentence的意图标签
        self._sentence_intent = None
        # 记录当前节点的字符值：
        self._value = value
        # 标记该节点到初始节点的长度
        self._length = length
        # 截止到该节点的 sequence
        self._sequence = ''

        self._node_index = None

    @property
    def is_word(self):
        return self._is_word

    @is_word.setter
    def is_word(self, value):
        if not isinstance(value, bool):
            raise ValueError("node.is_word must be a boolean!")
        self._is_word = value

    @property
    def word_type(self):
        return self._word_type

    @word_type.setter
    def word_type(self, value):
        if not isinstance(value, str):
            raise TypeError("node.word_type must be a str!")
        self._word_type = value

    @property
    def is_sentence(self):
        return self._is_sentence

    @is_sentence.setter
    def is_sentence(self, value):
        if not isinstance(value, bool):
            raise ValueError("node.is_sentence must be a boolean!")
        self._is_sentence = value

    @property
    def sentence_intent(self):
        return self._sentence_intent

    @sentence_intent.setter
    def sentence_intent(self, value):
        if not isinstance(value, str):
            raise ValueError("node.sentence_intent must be a str!")
        self._sentence_intent = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        if not isinstance(v, Text):
            raise ValueError(f"node.value must be a string")
        self._value = v

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        if not isinstance(value, int):
            raise ValueError(f"node.length must be a integer")
        self._length = value

    @property
    def sequence(self):
        return self._sequence

    @sequence.setter
    def sequence(self, value):
        if not isinstance(value, Text):
            raise ValueError(f"node.sequence must be a string")
        self._sequence = value

    @property
    def node_index(self):
        return self._node_index

    @node_index.setter
    def node_index(self, value):
        if not isinstance(value, int):
            raise ValueError(f"node.node_index must be a integer")
        self._node_index = value

    def __repr__(self):
        # if self.sequence_type==SequenceType.word:
        return f"node_index={self.node_index},value={self.value}," \
               f"is_word={self.is_word},word_type={self.word_type}," \
               f"length={self.length},sequence={self.sequence}," \
               f"is_sentence={self.is_sentence},intent={self.sentence_intent}"
        # elif self.sequence_type == SequenceType.sentence:
        #     return f"node={self.value},is_sequence={self.is_sequence},length={self.length},sequence={self.sequence}"


class TrieTree():
    """
    208. 实现 Trie (前缀树)
    实现一个 Trie (前缀树)，包含 insert, search, 和 startsWith 这三个操作。

    示例:
    Trie trie = new Trie();

    trie.insert("apple");
    trie.search("apple");   // 返回 true
    trie.search("app");     // 返回 false
    trie.startsWith("app"); // 返回 true
    trie.insert("app");
    trie.search("app");     // 返回 true
    说明:

    你可以假设所有的输入都是由小写字母 a-z 构成的。
    保证所有输入均为非空字符串

    来源：力扣（LeetCode）
    链接：https://leetcode-cn.com/problems/implement-trie-prefix-tree
    著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    """
    node_num = 0

    def __init__(self, sequence_type: SequenceType = SequenceType.word):
        """
        Initialize your data structure here.
        """
        self.sequence_type = sequence_type
        self.root = TrieNode()
        self.node_num += 1
        self.case_sensitivity = False

    def __repr__(self):
        return f"class:{self.__class__.__name__} has node_num={self.node_num}"

    def insert(self, sequence: Text,word_type=None) -> None:
        """
        1) Inserts a word into the trie: 进行分词
        2）插入 (word,word_type): 进行实体识别
        3）输入 句子

        """
        # if sequence =='牙疼':
        #     print(sequence)

        # 获取根节点
        current_node = self.root
        # if isinstance(sequence, FAQSentenceExample):
        #     sequence_text = sequence.text
        if isinstance(sequence, str):
            sequence_text = sequence
        else:
            raise ValueError(f"不支持插入的数据类型{type(sequence)}")
        # 对文本进行预处理
        # sequence_text = text_preprocess(sequence_text)
        # 遍历所有的字符
        for index, w in enumerate(sequence_text):
            # 判断当前节点的所有children 中是否有 w 的key， 必须使用 [] 来取值
            child_node = current_node.children.get(w)
            if child_node is None:
                # 创建新的子节点
                child_node = current_node.children[w]
                # 如果存在该 key,则获取对应的node，继续向下迭代
                # 如果不存在该 key，则会自动生成 该 key ---> node,因为是 defaultdict(factory_function) 类型，然后继续向下迭代
                # 增加 child_node value 属性：
                child_node.value = w
                # 记录 node的 length
                child_node.length = index + 1
                # 记录节点数
                self.node_num += 1

                child_node.node_index = self.node_num
            elif child_node is not None and index==sequence_text.__len__() \
                    and (child_node.is_word or child_node.is_sentence):
                logger.error(f"当前 sequence_text = {sequence_text} 的最后一个子节点已经存在并且已经标记是 word or sentence")
            # 更新当前及诶单
            current_node = child_node

        # 迭代完成的时候，标记该节点为一个词
        if self.sequence_type == SequenceType.word:
            current_node.is_word = True
            # 增加 child_node word_type 属性：
            current_node.word_type = word_type
        elif self.sequence_type == SequenceType.sentence:
            current_node.is_sentence = True
        # 记录 句子的意图标签
        # if isinstance(sequence, FAQSentenceExample):
        #     current_node.sentence_intent = sequence.intent

    # def search(self, word: str) -> bool:
    #     """
    #     Returns if the word is in the trie.
    #     """
    #     # 获取 root 节点为当前节点
    #     current_node = self.root
    #     # 对word 的每个字符做迭代判断
    #     # 如果直到迭代完成的时候的那个节点的word为true,则说明该 word 在这个 tire 树中
    #     for w in word:
    #         # 如果 w 不在 当前节点的 children当中
    #         if w not in current_node.children:
    #             return False
    #     # 判断最后的节点的word
    #     # 迭代完成的时候，标记该节点为一个词
    #     if current_node.is_word == True:
    #         return True
    #     else:
    #         return False

    def get_nodes_start_with_prefix(self, prefix: Text) -> List[TrieNode]:
        """
        返回 prefix 开头的节点
        """
        node_ls = []
        current_node = self.root
        for w in prefix:
            child_node = current_node.children.get(w)
            if child_node is None:
                break
            else:
                node_ls.append(child_node)
                current_node = child_node

        return node_ls

    def get_sequences_start_with_prefix(self, prefix: Text,
                                        length_lower_threshold=None, length_upper_threshold=None,
                                        sequence_type: SequenceType = SequenceType.sentence) -> List[Text]:
        """
        @time:  2021/7/19 11:41
        @author:wangfc
        @version:
        @description: 搜索所有以 prefix 开头的 sequence

        @params:
        prefix: 前缀
        length_lower_threshold： 如果存在，大于最小的阈值
        length_upper_threshold: 如果存在，小于最大的阈值
        sequence_type： 序列类型

        @return:
        """
        root_node = deepcopy(self.root)
        searched_sequences = []
        current_node = root_node
        for w in prefix:
            if w not in current_node.children:
                return None
            else:
                current_node = current_node.children.get(w)

        queue = deque()
        # 对节点的 sequence 属性赋值
        current_node.sequence = prefix
        queue.append(current_node)

        def is_node_in_length_range(current_node: TrieNode, length_lower_threshold=None, length_upper_threshold=None):
            is_in_range = False
            if (length_lower_threshold and length_upper_threshold) and \
                    (current_node.length > length_lower_threshold) and (current_node.length < length_upper_threshold):
                is_in_range = True
            elif (length_lower_threshold and not length_upper_threshold) and (
                    current_node.length > length_lower_threshold):
                is_in_range = True
            elif (not length_lower_threshold and length_upper_threshold) and (
                    current_node.length < length_upper_threshold):
                is_in_range = True
            elif (not length_lower_threshold) and (not length_upper_threshold):
                is_in_range = True
            return is_in_range

        # 直到 队列中 的 节点为 0
        while queue.__len__() > 0:
            current_node = queue.popleft()
            sequence = current_node.sequence

            if is_node_in_length_range(current_node, length_lower_threshold, length_upper_threshold):
                # 判断是否是否是  sequence的结束的节点
                if sequence_type == SequenceType.word and current_node.is_word:
                    searched_sequences.append(sequence)
                # elif sequence_type == SequenceType.sentence and current_node.is_sentence:
                #     # 当存在意图的时候，输出 text + intent
                #     if current_node.sentence_intent:
                #         sentence_with_intent = FAQSentenceExample(text=sequence, intent=current_node.sentence_intent)
                #         searched_sequences.append(sentence_with_intent)
                #     else:
                #         searched_sequences.append(sequence)
                # elif current_node.is_word or current_node.is_sentence:
                #     if current_node.sentence_intent:
                #         sentence_with_intent = FAQSentenceExample(text=sequence, intent=current_node.sentence_intent)
                #         searched_sequences.append(sentence_with_intent)
                #     else:
                #         searched_sequences.append(sequence)

            # 遍历满足条件的当前节点的子节点
            for key, child_node in current_node.children.items():
                # 更新子节点的  sequence 属性
                child_node_sequence = sequence + key
                child_node.sequence = child_node_sequence
                # 对于 node 的选择，只使用 length_upper_threshold，因为如果使用 length_lower_threshold,可能过早的把潜在的节点剔除了
                if is_node_in_length_range(current_node, length_upper_threshold=length_upper_threshold):
                    # 将子节点加入到queue
                    queue.append(child_node)
        return searched_sequences

    def entity_labeling(self,sequence:Text,max_entity_labeling_mode=True)->List[Entity]:
        """
        对句子进行实体识别
        匹配最长的entity
        """
        entity_ls:List[Entity] = []

        if not isinstance(sequence,str):
            raise ValueError(f"输入的必须是字符串格式")
        # 开始的搜索位置
        offset = 0
        while offset< sequence.__len__():
            # logger.debug(f"offset={offset}/{sequence.__len__()} of {sequence}")
            current_node = self.root
            last_node = None
            for index, char in enumerate(sequence[offset:]):
                # 获取当前字符的 node
                current_node = current_node.children.get(char)
                # 当 child_node 非空，在匹配最长的entity模式:
                # 继续更新当前节点 current_node

                if current_node is not None:
                    if (offset + index + 1== sequence.__len__() or not max_entity_labeling_mode):
                        start_position = offset
                        end_position = offset + index + 1
                        if current_node.is_word:
                            #  当前 current_node 节点是否是 word 并且 最后一个字符 or 非匹配最长的entity模式:
                            # 记录 实体信息:

                            if current_node.word_type is None:
                                raise ValueError(f"child_node 是 word节点，但是没有 word_type 属性")
                            entity_ls.append(Entity(sequence[offset:end_position],start_position,end_position,
                                                    current_node.word_type))
                        # 更新 offset
                        offset += end_position
                        break
                    # 更新 last_node
                    last_node = current_node

                else:
                    if index>0 :
                        start_position = offset
                        end_position = offset + index
                        if last_node.is_word:
                            # 当前字符不存在对应的节点，中断
                            # 当使用最长匹配的时候，当 child_node 为空 & 当前节点 is_word 的时候，标注
                            if last_node.word_type is None:
                                raise ValueError(f"current_node是 word节点，但是没有 word_type 属性")
                            # assert sequence[offset:end_position] ==
                            entity_ls.append(Entity(sequence[offset:end_position], start_position, end_position,
                                                    last_node.word_type))
                            # logger.debug(f"新增实体　{Entity}")
                        offset= end_position
                    else:
                        # 当 index==0的时候，既last_node 为None ,说明这个首个字符就没有匹配上，offset 应该在当前移动1一个位置
                        offset = offset + index +1
                    break
        if entity_ls:
            entity_ls = sorted(entity_ls, key=lambda entity: entity.start_position)
        return entity_ls







def test_edit_distance():
    import pandas as pd
    from models.edit_distance import computer_edit_distance

    # # 读取金融专业数据
    # financial_terms_path = os.path.join('data', 'finance', 'financial_term.yml')
    # financial_terms_ls =  read_financial_terms(financial_terms_path=financial_terms_path)
    #
    # # 更新 financial_term_trietree
    # financial_term_trietree = TrieTree(sequence_type=SequenceType.word)
    # for term in financial_terms_ls:
    #     financial_term_trietree.insert(term)
    # # 查询
    # financial_term_trietree.search(prefix='限',sequence_type=SequenceType.word,length_lower_threshold=4,
    #                                          length_upper_threshold=7)

    #
    path = './data/domain_classification/gyzq_sms_0628_none_financial_term_data_dropdups.xlsx'
    faq_log_data_df = pd.read_excel(path)
    faq_log_data_df = faq_log_data_df[faq_log_data_df.CHANNEL_ID == 2].copy()
    head_texts = faq_log_data_df.loc[faq_log_data_df.index[:100], 'TEXT']

    financial_term_trietree = TrieTree(sequence_type=SequenceType.word)
    text_set = set()
    for index in head_texts.index:
        text = head_texts.loc[index]
        striped_and_clean_text = "".join(text.strip().split())
        text_set.add(striped_and_clean_text)
        financial_term_trietree.insert(striped_and_clean_text)

    text_ls = sorted(text_set)
    sentence = '账号忘记了怎么办'
    prefix = sentence[:3]
    length_lower_threshold = len(sentence) - 5
    length_upper_threshold = len(sentence) + 5
    searched_sentences = financial_term_trietree.search(prefix=prefix,
                                                        length_lower_threshold=length_lower_threshold,
                                                        length_upper_threshold=length_upper_threshold)

    computer_edit_distance(sentence, searched_sentences)


if __name__ == '__main__':
    pass

