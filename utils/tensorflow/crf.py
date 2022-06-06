#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/2/9 17:28 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/9 17:28   wangfc      1.0         None
"""

import tensorflow as tf
from tensorflow import TensorShape

from tensorflow_addons.utils.types import TensorLike
from typeguard import typechecked
from typing import Tuple, Any, List, Union


# original code taken from
# https://github.com/tensorflow/addons/blob/master/tensorflow_addons/text/crf.py
# (modified to our neeeds)
from tensorflow_addons.text.crf import viterbi_decode


class CrfDecodeForwardRnnCell(tf.keras.layers.AbstractRNNCell):
    """Computes the forward decoding in a linear-chain CRF.

    See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.

    """

    @typechecked
    def __init__(self, transition_params: TensorLike, **kwargs: Any) -> None:
        """Initialize the CrfDecodeForwardRnnCell.

        Args:
          transition_params: A [num_tags, num_tags] matrix of binary
            potentials. This matrix is expanded into a
            [1, num_tags, num_tags] in preparation for the broadcast
            summation occurring within the cell.
        """
        super().__init__(**kwargs)
        self._transition_params = tf.expand_dims(transition_params, 0)
        self._num_tags = transition_params.shape[0]

    @property
    def state_size(self) -> int:
        return self._num_tags

    @property
    def output_size(self) -> int:
        """Returns count of tags."""
        return self._num_tags

    def build(self, input_shape: Union[TensorShape, List[TensorShape]]) -> None:
        """Creates the variables of the layer."""
        super().build(input_shape)

    def call(
        self, inputs: TensorLike, state: TensorLike
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Build the CrfDecodeForwardRnnCell.

        Args:
          inputs: A [batch_size, num_tags] matrix of unary potentials.
          state: A [batch_size, num_tags] matrix containing the previous step's
                score values.

        Returns:
          output: A [batch_size, num_tags * 2] matrix of backpointers and scores.
          new_state: A [batch_size, num_tags] matrix of new score values.
        """
        state = tf.expand_dims(state[0], 2)
        transition_scores = state + self._transition_params
        new_state = inputs + tf.reduce_max(transition_scores, [1])

        backpointers = tf.argmax(transition_scores, 1)
        backpointers = tf.cast(backpointers, tf.float32)

        # apply softmax to transition_scores to get scores in range from 0 to 1
        scores = tf.reduce_max(tf.nn.softmax(transition_scores, axis=1), [1])

        # In the RNN implementation only the first value that is returned from a cell
        # is kept throughout the RNN, so that you will have the values from each time
        # step in the final output. As we need the backpointers as well as the scores
        # for each time step, we concatenate them.
        return tf.concat([backpointers, scores], axis=1), new_state


def crf_decode_forward(
    inputs: TensorLike,
    state: TensorLike,
    transition_params: TensorLike,
    sequence_lengths: TensorLike,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Computes forward decoding in a linear-chain CRF.

    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous step's
            score values.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
      sequence_lengths: A [batch_size] vector of true sequence lengths.

    Returns:
      output: A [batch_size, num_tags * 2] matrix of backpointers and scores.
      new_state: A [batch_size, num_tags] matrix of new score values.
    """
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
    mask = tf.sequence_mask(sequence_lengths, tf.shape(inputs)[1])
    # 使用转移矩阵的参数初始化 CrfDecodeForwardRnnCell
    crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params)
    # 生成 RNN层
    crf_fwd_layer = tf.keras.layers.RNN(
        crf_fwd_cell, return_sequences=True, return_state=True
    )
    # 使用 RNN层进行迭代 和 参数更新
    return crf_fwd_layer(inputs, state, mask=mask)


def crf_decode_backward(
    backpointers: TensorLike, scores: TensorLike, state: TensorLike
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Computes backward decoding in a linear-chain CRF.

    Args:
      backpointers: A [batch_size, num_tags] matrix of backpointer of next step (in time order).
                    应该是 [batch_size, sequence_length, num_tags]
      scores: A [batch_size, num_tags] matrix of scores of next step (in time order).
              应该是 [batch_size, sequence_length, num_tags]
      state: A [batch_size, 1] matrix of tag index of next step.

    Returns:
      new_tags: A [batch_size, num_tags] tensor containing the new tag indices.
      new_scores: A [batch_size, num_tags] tensor containing the new score values.
    """
    # batch 中每个 time_step 的 tag_index
    backpointers = tf.transpose(backpointers, [1, 0, 2])
    scores = tf.transpose(scores, [1, 0, 2])

    def _scan_fn(_state: TensorLike, _inputs: TensorLike) -> tf.Tensor:
        """
        _state:  initializer or output ,shape = (batch_size,1)
        _inputs: elems[i] = backpointers[i],shape = (batch_size,num_tag)
        """
        _state = tf.cast(tf.squeeze(_state, axis=[1]), dtype=tf.int32)
        # idxs: (batch_size,2), 维度1为 batch_index,维度2 tag_index
        idxs = tf.stack([tf.range(tf.shape(_inputs)[0]), _state], axis=1)
        # 获取每个 example 对应的 tag score: shape =(batch_size,1)
        return tf.expand_dims(tf.gather_nd(_inputs, idxs), axis=-1)

    # 进行迭代，使用 state 作为 initializer，经过 _scan_fn(state, elems[i]) 不断迭代生成 state
    output_tags = tf.scan(_scan_fn, elems=backpointers, initializer=state)
    # the dtype of the input parameters of tf.scan need to match
    # convert state to float32 to match the type of scores
    state = tf.cast(state, dtype=tf.float32)
    output_scores = tf.scan(_scan_fn, scores, state)

    return tf.transpose(output_tags, [1, 0, 2]), tf.transpose(output_scores, [1, 0, 2])


def crf_decode(
    potentials: TensorLike, transition_params: TensorLike, sequence_length: TensorLike
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Decode the highest scoring sequence of tags.

    CRF 的解码模块，得到最佳解码序列和对应的得分。
    there are two ways to do decoding.
    One is using crf_decode to do decoding in Tensorflow ,
    the other one is using viterbi_decode in Numpy.

    我们应该了解 viterbi_decode （tensorflow_addons.text.viterbi_decode），再来了解 crf_decode

    https://github.com/tensorflow/addons/issues/2088
    Add scores for decoded tags to crf_decode

    Args:
      potentials: A [batch_size, max_seq_len, num_tags] tensor of unary potentials.
                发射矩阵 (unary scores (logits for every token)):表示 每个 example 的每个token位置的 tag的概率分布情况
                生成方式可以是： input = [batch_size, max_seq_len, hidden_size] * Dense_layer(units = num_tags) -> logits
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
                概率转移矩阵,表示 由一个状态转移到另一个状态的得分, 一开始是一个随即初始化的矩阵, 随着网络进行训练。

      sequence_length: A [batch_size] vector of true sequence lengths.

    Returns:
      decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.  Contains the highest scoring tag indices.
      decode_scores: A [batch_size, max_seq_len] matrix, containing the score of `decode_tags`.
                    相对于 tensorflow 源码增加的输出
      best_score: A [batch_size] vector, containing the best score of `decode_tags`.
    """
    sequence_length = tf.cast(sequence_length, dtype=tf.int32)

    # If max_seq_len is 1, we skip the algorithm and simply return the
    # argmax tag and the max activation.
    def _single_seq_fn() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        decode_tags = tf.cast(tf.argmax(potentials, axis=2), dtype=tf.int32)
        decode_scores = tf.reduce_max(tf.nn.softmax(potentials, axis=2), axis=2)
        best_score = tf.reshape(tf.reduce_max(potentials, axis=2), shape=[-1])
        return decode_tags, decode_scores, best_score

    def _multi_seq_fn() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Computes forward decoding. Get last score and backpointers.
        # 获取初始的状态 initial_state = [batch_size,1,tag_num]
        initial_state = tf.slice(potentials, [0, 0, 0], [-1, 1, -1])
        initial_state = tf.squeeze(initial_state, axis=[1])
        # 输入状态：第二个开始
        inputs = tf.slice(potentials, [0, 1, 0], [-1, -1, -1])

        sequence_length_less_one = tf.maximum(
            tf.constant(0, dtype=tf.int32), sequence_length - 1
        )
        # crf的前向传播
        output, last_score = crf_decode_forward(
            inputs, initial_state, transition_params, sequence_length_less_one
        )

        # output is a matrix of size [batch-size, max-seq-length, num-tags * 2]
        # split the matrix on axis 2 to get the backpointers and scores, which are
        # both of size [batch-size, max-seq-length, num-tags]
        backpointers, scores = tf.split(output, 2, axis=2)

        backpointers = tf.cast(backpointers, dtype=tf.int32)
        backpointers = tf.reverse_sequence(
            backpointers, sequence_length_less_one, seq_axis=1
        )

        scores = tf.reverse_sequence(scores, sequence_length_less_one, seq_axis=1)

        initial_state = tf.cast(tf.argmax(last_score, axis=1), dtype=tf.int32)
        initial_state = tf.expand_dims(initial_state, axis=-1)

        initial_score = tf.reduce_max(tf.nn.softmax(last_score, axis=1), axis=[1])
        initial_score = tf.expand_dims(initial_score, axis=-1)
        # crf 后向传播
        decode_tags, decode_scores = crf_decode_backward(
            backpointers, scores, initial_state
        )

        decode_tags = tf.squeeze(decode_tags, axis=[2])
        decode_tags = tf.concat([initial_state, decode_tags], axis=1)
        decode_tags = tf.reverse_sequence(decode_tags, sequence_length, seq_axis=1)

        decode_scores = tf.squeeze(decode_scores, axis=[2])
        decode_scores = tf.concat([initial_score, decode_scores], axis=1)
        decode_scores = tf.reverse_sequence(decode_scores, sequence_length, seq_axis=1)

        best_score = tf.reduce_max(last_score, axis=1)

        return decode_tags, decode_scores, best_score

    if potentials.shape[1] is not None:
        # shape is statically know, so we just execute
        # the appropriate code path
        if potentials.shape[1] == 1:
            return _single_seq_fn()

        return _multi_seq_fn()

    return tf.cond(tf.equal(tf.shape(potentials)[1], 1), _single_seq_fn, _multi_seq_fn)
