# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras-based TransformerEncoder block layer."""

import tensorflow as tf
from .multi_head_attention import MultiHeadAttention
from .activation_utils import get_activation, serialize_activation

@tf.keras.utils.register_keras_serializable(package="kerasbert")
class TransformerEncoderBlock(tf.keras.layers.Layer):
  """TransformerEncoderBlock layer.

  This layer implements the Transformer Encoder from
  "Attention Is All You Need". (https://arxiv.org/abs/1706.03762),
  which combines a `MultiHeadAttention` layer with a
  two-layer feedforward network.

  References:
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    [BERT: Pre-training of Deep Bidirectional Transformers for Language
     Understanding](https://arxiv.org/abs/1810.04805)
  """

  def __init__(self,
               num_attention_heads=12,
               intermediate_dim=3072,
               intermediate_activation="gelu",
               kernel_initializer="glorot_uniform",
               norm_epsilon=1e-12,
               output_dropout=0.0,
               attention_dropout=0.0,
               **kwargs):
    """Initializes `TransformerEncoderBlock`.

    Arguments:
      num_attention_heads: Number of attention heads.
      intermediate_dim: The output dimension of the first Dense layer in a two-layer
        feedforward network.
      intermediate_activation: The activation for the first Dense layer in a two-layer
        feedforward network.
      kernel_initializer: Initializer for dense layer kernels.
      norm_epsilon: Epsilon value to initialize normalization layers.
      output_dropout: Dropout probability for the post-attention and output
        dropout.
      attention_dropout: Dropout probability for within the attention layer.
      **kwargs: keyword arguments/
    """
    super(TransformerEncoderBlock, self).__init__(**kwargs)

    self._num_heads = num_attention_heads
    self._intermediate_dim = intermediate_dim
    self._intermediate_activation = get_activation(intermediate_activation)
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._norm_epsilon = norm_epsilon
    self._output_dropout_prob = output_dropout
    self._attention_score_dropout_prob = attention_dropout

  def build(self, input_shape):
    input_tensor = input_shape[0] if len(input_shape) == 2 else input_shape
    input_tensor_shape = tf.TensorShape(input_tensor)
    if len(input_tensor_shape.as_list()) != 3:
      raise ValueError("TransformerEncoderBlock expects a three-dimensional "
                       "input of shape [batch, sequence, width].")
    batch_size, sequence_length, hidden_size = input_tensor_shape

    if len(input_shape) == 2:
      mask_tensor_shape = tf.TensorShape(input_shape[1])
      expected_mask_tensor_shape = tf.TensorShape(
          [batch_size, sequence_length, sequence_length])
      if not expected_mask_tensor_shape.is_compatible_with(mask_tensor_shape):
        raise ValueError("When passing a mask tensor to "
                         "TransformerEncoderBlock, the mask tensor must be of "
                         "shape [batch, sequence_length, sequence_length] "
                         "(here %s). Got a mask tensor of shape %s." %
                         (expected_mask_tensor_shape, mask_tensor_shape))
    if hidden_size % self._num_heads != 0:
      raise ValueError(
          "The input size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (hidden_size, self._num_heads))
    self._attention_head_size = int(hidden_size // self._num_heads)
    self._attention_layer = MultiHeadAttention(
            num_heads=self._num_heads,
            key_dim=self._attention_head_size,
            initializer=self._kernel_initializer,
            dropout_prob=self._attention_score_dropout_prob,
            name="self_attention")
    self._attention_output_dropout = tf.keras.layers.Dropout(rate=self._output_dropout_prob)
    self._attention_layer_norm = tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype=tf.float32)
    self._intermediate_layer = tf.keras.layers.Dense(
            self._intermediate_dim,
            activation=self._intermediate_activation,
            kernel_initializer=self._kernel_initializer,
            name="intermediate")
    self._output_layer = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=self._kernel_initializer,
            name="output")
    self._output_dropout = tf.keras.layers.Dropout(rate=self._output_dropout_prob)
    # Use float32 in layernorm for numeric stability.
    self._output_layer_norm = tf.keras.layers.LayerNormalization(
            name="output_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype=tf.float32)

    super(TransformerEncoderBlock, self).build(input_shape)

  def get_config(self):
    config = {
        "num_attention_heads":
            self._num_heads,
        "intermediate_dim":
            self._intermediate_dim,
        "intermediate_activation":
            serialize_activation(self._intermediate_activation),
        "output_dropout":
            self._output_dropout_prob,
        "attention_dropout":
            self._attention_score_dropout_prob,
        "kernel_initializer":
            tf.keras.initializers.serialize(self._kernel_initializer),
        "norm_epsilon":
            self._norm_epsilon,
    }
    base_config = super(TransformerEncoderBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
    if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
      input_tensor, attention_mask = inputs
    else:
      input_tensor, attention_mask = (inputs, None)

    attention_output, attention_weights = self._attention_layer(
        [input_tensor, input_tensor], attention_mask=attention_mask, return_attention_scores=True)
    attention_output = self._attention_output_dropout(attention_output, training=training)
    attention_output = self._attention_layer_norm(input_tensor + attention_output)
    intermediate_output = self._intermediate_layer(attention_output)
    layer_output = self._output_layer(intermediate_output)
    layer_output = self._output_dropout(layer_output, training=training)

    # During mixed precision training, layer norm output is always fp32 for now.
    # Casts fp32 for the subsequent add.
    layer_output = tf.cast(layer_output, tf.float32)
    return self._output_layer_norm(layer_output + attention_output)
