from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import math


@tf.keras.utils.register_keras_serializable(package="kerasbert")
class MultiHeadAttention(tf.keras.layers.Layer):
    """MultiHeadAttention layer.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `query`, `key,` `value` are the same, then
    this is self-attention. Each timestep in `query` attends to the
    corresponding sequence in `key`, and returns a fixed-width vector.

    This layer first projects `query`, `key` and `value`. These are
    (effectively) a list of tensors of length `num_attention_heads`, where the
    corresponding shapes are [batch_size, <query_sequence_length>, key_dim],
    [batch_size, <key_sequence_length>, key_dim],
    [batch_size, <key_sequence_length>, value_dim].

    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor.

    Finally, the result tensor with the last dimension as value_dim can take an
    linear projection and return.

    Examples:

    Performs 1D cross-attention over two sequence inputs with an attention mask.
    Returns the additional attention weights over heads.

    >>> layer = MultiHeadAttention(num_heads=2, key_dim=2)
    >>> target = tf.keras.Input(shape=[8, 16])
    >>> source = tf.keras.Input(shape=[4, 16])
    >>> output_tensor, weights = layer([source, target], return_attention_scores=True)
    >>> print(output_tensor.shape)
    (None, 8, 16)
    >>> print(weights.shape)
    (None, 2, 8, 4)

    Performs 2D self-attention over a 5D input tensor.

    >>> layer = MultiHeadAttention(num_heads=2, key_dim=2)
    >>> input_tensor = tf.keras.Input(shape=[5, 3, 4, 16])
    >>> output_tensor = layer([input_tensor, input_tensor])
    >>> print(output_tensor.shape)
    (None, 5, 3, 4, 16)

    Arguments:
        num_heads: Number of attention heads.
        key_dim: Size of each attention head for query and key.
        value_dim: Size of each attention head for value.
        kernel_initializer: Initializer for dense layer kernels.
        bias_initializer: Initializer for dense layer biases.
        kernel_regularizer: Regularizer for dense layer kernels.
        bias_regularizer: Regularizer for dense layer biases.
        activity_regularizer: Regularizer for dense layer activity.
        kernel_constraint: Constraint for dense layer kernels.
        bias_constraint: Constraint for dense layer kernels.

    Call arguments:
        inputs: a list = [source, target], where, `source` is a `Tensor` of shape `[B, S, E]`,
            and `target` is a `Tensor` of shape `[B, T, E]`.
        attention_mask: a boolean mask of shape `[B, T, S]`, that prevents attention
            to certain positions.
        return_attention_scores: A boolean to indicate whether the output should
            be attention output if True, or (attention_output, attention_scores) if
            False. Defaults to False.

    Returns:
        attention_output: The result of the computation, of shape [B, T, E],
            where `T` is for target sequence shapes and `E` is the target input last
            dimension if `output_dim` is `None`. Otherwise, `output_dim`.
        attention_scores: [Optional] multi-head attention coeffients of shape [B, N, T, S]
    """

    def __init__(self,
                 num_heads,
                 key_dim,
                 value_dim=None,
                 output_dim=None,
                 activation=None,
                 initializer="glorot_uniform",
                 dropout_prob=0.1,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim
        self._output_dim = output_dim
        self._activation = activation
        self._initializer = tf.keras.initializers.get(initializer)
        self._dropout_prob = dropout_prob

    def get_config(self):
        config = {
                "num_heads": self._num_heads,
                "key_dim": self._key_dim,
                "value_dim": self._value_dim,
                "output_dim": self._output_dim,
                "activation": self._activation,
                "initializer": tf.keras.initializers.serialize(self._initializer),
                "dropout_prob": self._dropout_prob,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        """Builds layers and variables.
        """
        common_kwargs = dict(
                activation=self._activation,
                kernel_initializer = self._initializer)
        key_units = self._num_heads * self._key_dim
        value_units = self._num_heads * self._value_dim
        self._query_dense = tf.keras.layers.Dense(key_units,
                name="query",
                **common_kwargs)
        self._key_dense = tf.keras.layers.Dense(key_units,
                name="key",
                **common_kwargs)
        self._value_dense = tf.keras.layers.Dense(value_units,
                name="value",
                **common_kwargs)
        self._score_dropout = tf.keras.layers.Dropout(rate=self._dropout_prob)

        if self._output_dim:
            output_units = self._output_dim
        else:
            output_units = input_shape[1][-1]
        self._output_layer = tf.keras.layers.Dense(output_units,
                name="attention_output",
                **common_kwargs)

        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, attention_mask=None, return_attention_scores=False, training=None):
        """
        Args:
            inputs[0]: source, A Tensor with shape [B, S, dim]
            inputs[1]: target, A Tensor with shape [B, T, dim]
        Returns:
          attention_output: Multi-headed outputs of attention computation.
          attention_scores: Multi-headed attention weights.
        """
        self._validate_call_args(inputs=inputs, mask=attention_mask)
        source = inputs[0]
        target = inputs[1]

        # N = num_attention_heads
        # H = key_dim
        # V = value_dim
        T = target.shape[1]
        query = self._query_dense(target)
        query = tf.reshape(query, [-1, T, self._num_heads, self._key_dim])
        #`query` = [B, N, T, H]
        query = tf.transpose(query, [0, 2, 1, 3])

        S = source.shape[1]
        key = self._key_dense(source)
        key = tf.reshape(key, [-1, S, self._num_heads, self._key_dim])
        #`key` = [B, N, S, H]
        key = tf.transpose(key, [0, 2, 1, 3])

        value = self._value_dense(source)
        value = tf.reshape(value, [-1, S, self._num_heads, self._value_dim])
        #`value` = [B, N, S, V]
        value = tf.transpose(value, [0, 2, 1, 3])

        #`attention_scores` = [B, N, T, S]
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                        1.0 / math.sqrt(float(self._key_dim)))

        if attention_mask is not None:
            #`attention_mask` = [B, 1, T, S]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
            attention_scores += adder
        attention_scores = tf.nn.softmax(attention_scores)
        attention_scores = self._score_dropout(attention_scores, training=training)

        #`attention_output` = [B, N, T, V]
        attention_output = tf.matmul(attention_scores, value)
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        #`attention_output` = [B, T, N*V]
        attention_output = tf.reshape(attention_output, [-1, T, self._num_heads * self._value_dim])

        attention_output = self._output_layer(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

    def _validate_call_args(self, inputs, mask):
        """Validates arguments of the call method."""
        class_name = self.__class__.__name__
        if not isinstance(inputs, list):
          raise ValueError(
              '{} layer must be called on a list of inputs, namely [source, target] '
              ''.format(class_name))
        if len(inputs) != 2:
          raise ValueError(
              '{} layer accepts inputs list of length 2 '
              'namely [source, target]. '
              'Given length: {}'.format(class_name, len(inputs)))

