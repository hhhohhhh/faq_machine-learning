#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:

维特比算法（Viterbi algorithm）是在一个用途非常广的算法
viterbi 维特比算法解决的是 篱笆型的图 （ lattice ）的最短路径问题：
图的节点按列组织，每列的节点数量可以不一样，每一列的节点只能和相邻列的节点相连，不能跨列相连，节点之间有着不同的距离

Numpy实现：

tensorflow实现:


CRF:




@time: 2022/3/18 16:33

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/18 16:33   wangfc      1.0         None
"""
import math
from typing import List, Dict, Tuple, Iterable

import numpy as np
from numpy import mat
from tensorflow_addons.text.crf import viterbi_decode
import tensorflow as tf
from tensorflow_addons.utils.types import TensorLike


class Node():
    def __init__(self, value: int,
                 position: Tuple[int, int] = None,
                 shortest_path: int = None,
                 last_node: "Node" = None, ):
        # 当前节点的值
        self.value = value
        # 当前节点position为一个数组：第一个元素表示 step,第2个元素表示位置
        self.position = position
        self.shortest_path = shortest_path
        self.last_node = last_node

    def __repr__(self):
        return f"node position={self.position},value={self.value},shortest_path={self.shortest_path}"


class MinimalPathOfTriangle():
    """
    给定一个三角形 triangle ，找出自顶向下的最小路径和。
    每一步只能移动到下一行中相邻的结点上。
    相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。
    也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。

    示例 1：
    输入：triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
    输出：11
    解释：如下面简图所示：
       2
      3 4
     6 5 7
    4 1 8 3
    自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。
    示例 2：

    输入：triangle = [[-10]]
    输出：-10
     

    提示：
    1 <= triangle.length <= 200
    triangle[0].length == 1
    triangle[i].length == triangle[i - 1].length + 1
    -104 <= triangle[i][j] <= 104
     
    进阶：
    你可以只使用 O(n) 的额外空间（n 为三角形的总行数）来解决这个问题吗？

    来源：力扣（LeetCode）
    链接：https://leetcode-cn.com/problems/triangle
    著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    """

    def __init__(self, triangle: List[List[int]]):
        self.triangle = triangle
        self.steps = triangle.__len__()

    def find_minimal_path(self):
        """
        到达当前节点的最短路径 = min(上个步骤的节点的最短路径 + 两个节点之间的路径)

        计算复杂度 ：steps * node_num * (计算最小距离的复杂度 node_num) *  = O(T*n^2)
        穷举法的时间复杂度: O(n^T)
        实现方式：
        1. 迭代方法
        2. 使用 节点并距离最小距离
        2. 是否可以使用 heap 数据结构来设计 viterbi 算法呢？
        heap中保存的元素为一个元组：
            key : 当前节点的最短距离
            value : 上一个节点的值（或者位置）

        :return:
        """
        # 初始节点
        step = -1
        # node_position为一个数组：第一个元素表示 step,第2个元素表示位置
        position = (step, 0)
        start_node = Node(value='START', position=-1, shortest_path=0, )
        # 记录 step_to_nodes的字典
        position_to_node_dict = {position: start_node}

        # 增加最有一个step设置最后的结束节点，其距离值为0，其之前的上一步节点均可以达到该节点
        triangle = self.triangle + [[0]]

        for step in range(triangle.__len__()):
            # 对当前step 所有节点
            nodes_in_step = triangle[step]
            # 计算所有当前step的节点的最小距离
            for index, current_node_value in enumerate(nodes_in_step):
                # 计算当前节点的最小距离= 上个step 的节点最小句子 + 两个节点之间的距离
                current_node_position = (step, index)
                shortest_path, shortest_path_node = self.find_minimal_last_node(current_node_position,
                                                                                current_node_value,
                                                                                position_to_node_dict)
                # 如果是最后的节点，我们将其值表示为 end,方便路径的打印
                if step == triangle.__len__() - 1:
                    current_node_value = 'END'
                current_node = Node(value=current_node_value, position=current_node_position,
                                    shortest_path=shortest_path, last_node=shortest_path_node)
                position_to_node_dict.update({current_node_position: current_node})

        # 获取最有的距离和路径
        end_node_position = (triangle.__len__() - 1, 0)
        end_node = position_to_node_dict[end_node_position]

        minimal_path = end_node.shortest_path
        # 倒推回来获取最有的路径
        minimal_path_nodes = [end_node]
        current_node = end_node
        while current_node.last_node:
            current_node = current_node.last_node
            minimal_path_nodes.append(current_node)

        minimal_path_nodes = minimal_path_nodes[::-1]
        path = "->".join([str(node.value) for node in minimal_path_nodes])
        return minimal_path, path

    def find_minimal_last_node(self, current_node_position=None,
                               current_node_value=None,
                               position_to_node_dict: Dict[int, Node] = None) \
            -> [int, Node]:
        """
        计算当前节点的最短路径，返回最短路径和前一步的节点
        :param current_node_position:
        :param current_node_value:
        :param position_to_node_dict:
        :return:
        """
        # 获取上一步的所有节点
        # last_step = step-1
        # last_step_nodes = step_to_nodes[last_step]
        last_step_nodes = self.find_last_step_nodes(current_node_position, position_to_node_dict)
        # 计算上一步所有节点到当前节点的距离的最小值
        shortest_path = math.inf
        shortest_path_node = None
        for last_node in last_step_nodes:
            last_node_to_current_path = last_node.shortest_path + current_node_value
            if last_node_to_current_path < shortest_path:
                shortest_path = last_node_to_current_path
                shortest_path_node = last_node
        return shortest_path, shortest_path_node

    def find_last_step_nodes(self, current_node_position: Tuple[int, int], position_to_node_dict: Dict[int, Node]) \
            -> List[Node]:
        """
        获取上一步的节点，需要根据需求来限制上一步节点的位置
        :param current_node_position:
        :param position_to_node_dict:
        :return:
        """
        current_step = current_node_position[0]
        last_step = current_node_position[0] - 1
        if current_step == self.steps:
            # 最后一个结束节点
            last_indexes = list(range(self.triangle[-1].__len__()))
        else:
            # 更加定义获取该节点可以选择的上个step的所有节点
            last_indexes = [current_node_position[1], current_node_position[1] - 1]
            # index 必须大于 -1 和小于 last_step的index数量
            last_indexes = [index for index in last_indexes if
                            index > -1 and index < self.triangle[last_step].__len__()]
        last_positions = [(last_step, last_index) for last_index in last_indexes]
        return [position_to_node_dict[position] for position in last_positions]


class HMM():
    """

    https://zhuanlan.zhihu.com/p/85454896

    HMM:
    将词序列看做是观测到的现象，而词性、标签等信息看做是隐含状态，那么就可以通过维特比算法求解其隐含状态序列，
    而这也是 HMM 在分词，词性标注，命名实体识别中的应用。

    对于 HMM 而言，其中一个重要的任务就是要找出最有可能产生其观测序列的隐含序列。一般来说，HMM问题可由下面五个元素描述
        观测序列（observations: O）：实际观测到的现象序列
        隐含状态（states）：所有的可能的隐含状态
        初始概率（ start_probability: PI）：每个隐含状态的初始概率
        转移概率（ transition_probability: A）：从一个隐含状态转移到另一个隐含状态的概率
        发射概率（emission_probability: B）：某种隐含状态产生某种观测现象的概率,

    HMM 两个假设: 1) 齐次马尔科夫链假设: 即任意时刻的隐藏状态只依赖于它前一个隐藏状态，与其他时刻的状态无关（当然，初始时刻的状态由初始概率 PI决定）
                    P(h_i |h_1,h_2,....,h_i-1,o_1,o_2,...,o_i) = P(h_i|h_i-1)
                2）观测独立性假设: 即任意时刻的观察状态只仅仅依赖于当前时刻的隐藏状态
                    P(o_i |h_1,h_2,....,h_i,o_1,o_2,...,o_i-1 ) = P(o_i|h_i)

    HMM模型一共有三个经典的问题需要解决：
    1. 评估观察序列 条件概率：即给定模型λ=(A,B,Π)和观测序列O={o1,o2,...oT}，计算在模型λ下观测序列O出现的概率P(O|λ)。
       这个问题的求解需要用到前向后向算法，我们在这个系列的第二篇会详细讲解。
    2. 参数学习问题： 即给定观测序列O={o1,o2,...oT}，估计模型λ=(A,B,Π)的参数，使该模型下观测序列的条件概率P(O|λ)最大。
        这个问题的求解需要用到基于 EM算法的鲍姆-韦尔奇算法，
    3. 预测问题，也称为解码问题：即给定模型λ=(A,B,Π)和观测序列O={o1,o2,...oT}，求给定观测序列条件下，最可能出现的对应的状态序列，
        这个问题的求解需要用到基于动态规划的维特比算法，

    """

    def __init__(self, hidden_states:List[int], observation_states:List[int],
                 initial_probabilities=None,
                 transition_probabilities=None,
                 emission_probabilities=None):
        # 隐藏状态，比如 词性
        self.hidden_states = hidden_states
        self.hidden_state_size = hidden_states.__len__()
        # 观测状态
        self.observation_states = observation_states
        self.observation_states_size = observation_states.__len__()
        # 初始概率 = ( hidden_state_size,1)
        self.initial_probabilities = np.array(initial_probabilities)
        # 转移概率矩阵 = [hidden_state_size,hidden_state_size]，
        # 其中 a(i,j) = P(h_t = j |h_t-1=i) 表示从 last_hidden_state = i ->  current_hidden_state=j 的转移概率
        self.transition_probabilities = np.array(transition_probabilities)
        # 发射概率矩阵 = [hidden_state_size,observation_state_size]
        # e(i,j) = P(o_t = j |h_t= i ) 表示从 current_hidden_state = i ->  current_observation_state=j 的发射概率
        # e[:,j] 表示  所有隐藏状态 到 观察状态 j的 发射概率
        self.emission_probabilities = np.array(emission_probabilities)

    def _get_data_with_distribute(self, dist):
        # 根据给定的概率分布随机返回数据（索引）
        r = np.random.rand()
        for i, p in enumerate(dist):
            if r < p:
                return i
            r -= p

    def generate(self, steps: int):
        '''
        根据给定的参数 steps 生成观测序列
        steps: 指定要生成数据的数量
        '''
        observation_states = []

        # 根据初始状态概率分布Π生成 第一个 隐藏状态 i_1
        y = self._get_data_with_distribute(dist=self.initial_probabilities.squeeze())

        for step in range(steps):
            # 根据 hidden_state 生成 observation_state distribution
            observation_states_distribution = self.emission_probabilities[y]
            # 根据 observation_state distribution 生成 observation_state o_t
            x = self._get_data_with_distribute(dist=observation_states_distribution)
            observation_states.append(x)

            # 获取下一步的 hidden_state
            # 根据转移概率矩阵获取 next step hidden_state distribution
            hidden_states_distribution = self.transition_probabilities[y]
            # 根据该 distribution 生成 hidden_state
            y = self._get_data_with_distribute(dist = hidden_states_distribution)

        return observation_states


    def evaluate_forward(self, observations):
        '''
        根据给定的参数 评估 观察序列 条件概率 P(O| (A,B,Pi))
        X: 观测数据

        https://www.cnblogs.com/pinard/p/6955871.html
        alpha_t_i =P(o_1,o_2,...,o_t,i_t= q_i|lambda)

        动态规划：
        1. 初始值
        2. 迭代状态

        '''
        steps = observations.__len__()
        # 定义 alpha, axis=0 表示时间轴，axis=1 表示不同的隐藏状态
        alpha = np.empty(shape=(steps, self.hidden_state_size),dtype=np.float32)

        # 根据前向概率的定义 初始化 alpha[0]
        # (h,1) * (h,1) = (h,1)  -> (1,h)
        alpha[0] = np.transpose(self.initial_probabilities * self.emission_probabilities[:,observations[0]])
        for step in range(1,steps):
            # 根据 前向算法的迭代公式
            # (1,h) * (h,h) = (h * h) : 使用 broadcasting
            # np.sum()      -> (h,1)  : 对所有 last step 隐藏状态求和
            # (h,1) * (h,1) = (h,1) -> (1,h) :
            alpha_step = np.sum(alpha[step-1,:] * self.transition_probabilities.T,axis=-1,keepdims=True) * \
                         self.emission_probabilities[:,observations[step]]
            alpha[step] = alpha_step.T
         # 对最后的step 的所有的隐藏状态累加
        probability = np.sum(alpha[-1])
        return probability

    def evaluate_backward(self,observations):
        """
        beta_t_i =P(o_1,o_2,...,)
        """
        pass











    def fit(self,observations):
        '''
        根据给定观测序列反推参数
        '''



    def viterbi_decode(self, observations: Iterable):
        """
        已知 初始概率，转移概率和发射概率 和观测状态， 使用 隐身马尔科夫模型预测最大概率路径：
        转换为 数学目标: 求条件概率最大值
            max P( (h_1,h_2,...,h_t)|(o_1,O_2,...,o_t))
            as P(AB|C) = P(A|BC) * P(B|C)
               = P(h_t| h_1,h_2,...,h_t-1,o_1,O_2,...,o_t) * P(h_1,h_2,...,h_t-1|o_1,O_2,...,o_t)
            as
               = P(h_t|h_t-1,o_t) * P(h_1,h_2,...,h_t-1|o_1,O_2,...,o_t)
               ？？？

        使用 HMM 模型简化：
            P(o_i|h_1,h_2,...,h_t) = P(o_i| h_i)

        使用维特比算法解码最大概率的路径（最优路径）: 在走的所有路径中，找到一条概率最大的路径

        如何计算当前节点（隐藏状态）的概率最大值 ？
            P(h_i = q_j) = max( P(h_i-1)* P(h_i|h_i-1) * P(o_i |h_i-1) )
        节点 ： 每个隐藏状态作为 节点
            P(hidden_state|last_state) -> 转移概率
            P(hidden_state|start_state) -> 初始概率
            P(observation_state|hidden_state) -> 发射概率

        :return:
        """
        # 使得 observations = (steps,),axis=0轴的每个数值表示每个 step的 observation
        observations = np.array(observations)

        # 计算每个step 的所有节点的最大概率路径：
        # 分为 step=0 和 step=2,...,T 时刻两个部分

        # 记录所有的 max_indexes 索引(当前节点对应最大概率的上个节点的索引)
        max_indexes_ls = []

        # 当step=0 的时候 :
        # 每个节点（隐藏状态）-> 观察状态 o的 最大概率 = 初始概率 * 发射概率 = pi(h_0= i) * e(o|i)
        # self.emission_probabilities[:,observations[0]] 表示每个隐藏状态 -> 观察状态 o 的向量
        # max_probability_in_step.shape = (hidden_state_size,1)
        max_probability_between_hidden_states = self.initial_probabilities * self.emission_probabilities[:,
                                                                             observations[0]]
        # max_probability_in_step.shape = (1, hidden_state_size)
        max_probability_between_hidden_states = max_probability_between_hidden_states.T

        # 每个 step 中的到达每个隐藏状态节点的最大概率 = max(上一节点最大概率 * 转移概率 * 发射概率)
        for step in range(1, observations.__len__()):
            # t-1 时刻隐藏状态 i —> t 时刻隐藏状态 j condition on
            # P(h_t = j | h_t-1=i,o) = P(h_t-1=i) * a(j|i) * e( o|j)

            # 表示上一步的所有隐藏状态的最大概率
            # transition_probabilities      = (hidden_state_size,hidden_state_size)
            # 每一行表示 每个 last_hidden_state_size 的 转移概率
            # emission_probabilities[:,observations[step]].T = (1, hidden_state_size)
            # 表示每个隐藏状态到o的发射概率

            # # method1:
            # 计算每个节点的所有路径的概率:
            # probability_in_step = (last_hidden_state_size,hidden_state_size)
            # probability_in_step[i,j] 表示从上个 隐藏状态节点 i —>  隐藏状态节点 j 的概率
            # probability_in_step = max_probability_between_hidden_states * self.transition_probabilities \
            #                       * self.emission_probabilities[:,observations[step]].T
            # # 沿着 last_hidden_state_size 轴(axis=0)计算所有节点的最大概率  和 上一个节点的索引
            # # max_indexes = (hidden_state_size, )
            # max_indexes_a = np.argmax(probability_in_step,axis=0)
            # # max_probability_in_step =(1,hidden_state_size)
            # max_probability_in_step_a = np.take_along_axis(probability_in_step,max_indexes_a[np.newaxis], axis=0)

            # method2:
            # probability_between_hidden_states= (h,h)
            probability_between_hidden_states = max_probability_between_hidden_states * self.transition_probabilities
            # max_probability_between_hidden_states = (h,)
            max_probability_between_hidden_states = np.max(probability_between_hidden_states, axis=0, keepdims=True)
            # max_indexes
            max_indexes = np.argmax(probability_between_hidden_states, axis=0)
            #
            max_probability_in_step = max_probability_between_hidden_states * self.emission_probabilities[:,
                                                                              observations[step]].T
            # np.allclose(max_probability_in_step_a,max_probability_in_step)

            # 叠加每个step的 max_indexes
            max_indexes_ls.append(max_indexes)

        # max_indexes_array 的 axis=0 表示每个 step
        max_indexes_array = np.array(max_indexes_ls)

        # 对最后step的 max_indexes 求最大值概率
        max_index = np.argmax(max_indexes)
        max_probability = max_probability_in_step.squeeze()[max_index]
        # 根据 max_index 反向做最优路径回溯 获取 path
        best_path_ls = [max_index]
        for max_indexes in max_indexes_array[::-1]:
            max_index = max_indexes[max_index]
            best_path_ls.append(max_index)
        best_path_ls = best_path_ls[::-1]
        best_path = "->".join([str(x) for x in best_path_ls])
        print(f"best_path best_path={best_path}, max_probability={max_probability}")
        return max_probability, best_path_ls


def viterbi_decode(score: TensorLike, transition_params: TensorLike) -> tf.Tensor:
    """
    tensorflow的文档，说明 viterbi_decode 和 crf_decode 实现了相同功能，前者是numpy的实现，后者是 tensor 的实现

    Decode the highest scoring sequence of tags outside of TensorFlow.

    This should only be used at test time.

    Args:
      score: A [seq_len, num_tags] matrix of unary potentials.
            score[i,j] 表示每个 step=i,输出隐藏状态 j的概率
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
            转移概率矩阵
    Returns:
      viterbi: A [seq_len] list of integers containing the highest scoring tag indices.
      viterbi_score: A float containing the score for the Viterbi sequence.

    """
    # 隐藏状态数量 hidden_state_size = num_tags
    # trellis 表示每个 step 所有隐藏节点的最大概率 (seq_len,hidden_state_size)
    trellis = np.zeros_like(score)
    # backpointers 表示每个step 所有隐藏状态节点的最大概率的路径的索引 (seq_len,hidden_state_size)
    backpointers = np.zeros_like(score, dtype=np.int32)
    # 初始的 所有隐藏节点的最大概率 (hidden_state_size,),相当于  初始概率 * 转移概率
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        # 上个 step 的 所有隐藏节点的最大概率 + 转移概率矩阵 （概率转换为log 后可以将乘法转换为加法）
        # (hidden_state_size,1) + (hidden_state_size,hidden_state_size) = (hidden_state_size,hidden_state_size)
        # v[i,j] 表示上一个节点i -> 当前节点j的最大概率
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        # 当前step的概率 + 发射概率
        # (hidden_size,) + (hidden_size,) = (hidden_size,)
        trellis[t] = score[t] + np.max(v, 0)
        #
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score


if __name__ == '__main__':
    triangle = [[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]
    triangle = [[-10]]
    minimal_path_of_triangle = MinimalPathOfTriangle(triangle=triangle)
    minimal_path_of_triangle.find_minimal_path()

    hidden_state_size = 3
    observation_state_size = 2
    hidden_states = list(range(hidden_state_size))
    observation_states = list(range(observation_state_size))

    transition_probabilities = [[0.5, 0.2, 0.3],
                                [0.3, 0.5, 0.2],
                                [0.2, 0.3, 0.5]]
    #
    emission_probabilities = [[0.5, 0.5],
                              [0.4, 0.6],
                              [0.7, 0.3]]

    initial_probabilities = [[0.2],
                             [0.4],
                             [0.4]]
    # 观察序列 [红,白, 红]
    observations = [[0],
                    [1],
                    [0]]
    hmm = HMM(hidden_states=hidden_states,
              observation_states=observation_states,
              initial_probabilities=initial_probabilities,
              transition_probabilities=transition_probabilities,
              emission_probabilities=emission_probabilities)
    # max_probability, best_path_ls = hmm.viterbi_decode(observations=observations)

    observations_output = hmm.generate(steps=3)

    probability = hmm.evaluate_forward(observations=observations)
