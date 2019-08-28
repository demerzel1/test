# GMM attention mechanism
# Alex Graves. Generating Sequences With Recurrent Neural Networks (2013)

import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from functools import partial

_zero_state_tensors = rnn_cell_impl._zero_state_tensors


class GmmAttention(attention_wrapper.AttentionMechanism):
    def __init__(self,
                 memory,
                 num_mixtures=16,
                 memory_sequence_length=None,
                 check_inner_dims_defined=True,
                 score_mask_value=None,
                 name='GmmAttention'):

        self.dtype = memory.dtype
        self.num_mixtures = num_mixtures
        self.query_layer = tf.layers.Dense(
            3 * num_mixtures, name='gmm_query_layer', use_bias=True, dtype=self.dtype)

        with tf.name_scope(name, 'GmmAttentionMechanismInit'):
            if score_mask_value is None:
                score_mask_value = 0.
            self._maybe_mask_score = partial(
                attention_wrapper._maybe_mask_score,
                memory_sequence_length=memory_sequence_length,
                score_mask_value=score_mask_value)
            self._value = attention_wrapper._prepare_memory(
                memory, memory_sequence_length, check_inner_dims_defined)
            self._batch_size = (
                self._value.shape[0].value or tf.shape(self._value)[0])
            self._alignments_size = (
                    self._value.shape[1].value or tf.shape(self._value)[1])

    @property
    def values(self):
        return self._value

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def alignments_size(self):
        return self._alignments_size

    @property
    def state_size(self):
        return self.num_mixtures

    def initial_alignments(self, batch_size, dtype):
        max_time = self._alignments_size
        return _zero_state_tensors(max_time, batch_size, dtype)

    def initial_state(self, batch_size, dtype):
        state_size_ = self.state_size
        return _zero_state_tensors(state_size_, batch_size, dtype)

    def __call__(self, query, state):
        with tf.variable_scope("GmmAttention"):
            previous_kappa = state
            
            params = self.query_layer(query)
            alpha_hat, beta_hat, kappa_hat = tf.split(params, num_or_size_splits=3, axis=1)

            # [batch_size, num_mixtures, 1]
            alpha = tf.expand_dims(tf.exp(alpha_hat), axis=2)
            # softmax makes the alpha value more stable.
            # alpha = tf.expand_dims(tf.nn.softmax(alpha_hat, axis=1), axis=2)
            beta = tf.expand_dims(tf.exp(beta_hat), axis=2)
            kappa = tf.expand_dims(previous_kappa + tf.exp(kappa_hat), axis=2)

            # [1, 1, max_input_steps]
            mu = tf.reshape(tf.cast(tf.range(self.alignments_size), dtype=tf.float32),
                            shape=[1, 1, self.alignments_size])

            # [batch_size, max_input_steps]
            phi = tf.reduce_sum(alpha * tf.exp(-beta * (kappa - mu) ** 2.), axis=1)

        alignments = self._maybe_mask_score(phi)
        state = tf.squeeze(kappa, axis=2)

        return alignments, state