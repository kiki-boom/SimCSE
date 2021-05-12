import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="kerasbert")
class OnDeviceEmbedding(tf.keras.layers.Layer):
  """Performs an embedding lookup suitable for accelerator devices.

  This layer uses either tf.gather or tf.one_hot to translate integer indices to
  float embeddings.

  Arguments:
    vocab_size: Number of elements in the vocabulary.
    embedding_width: Output size of the embedding layer.
    initializer: The initializer to use for the embedding weights. Defaults to
      "glorot_uniform".
    use_one_hot: Whether to use tf.one_hot over tf.gather for the embedding
      lookup. Defaults to False (that is, using tf.gather). Setting this option
      to True may improve performance, especially on small vocabulary sizes, but
      will generally require more memory.
    scale_factor: Whether to scale the output embeddings. Defaults to None (that
      is, not to scale). Setting this option to a float will let values in
      output embeddings multiplied by scale_factor.
  """

  def __init__(self,
         vocab_size,
         embedding_width,
         initializer="glorot_uniform",
         use_one_hot=False,
         scale_factor=None,
         **kwargs):
    super(OnDeviceEmbedding, self).__init__(**kwargs)
    self._vocab_size = vocab_size
    self._embedding_width = embedding_width
    self._initializer = initializer
    self._use_one_hot = use_one_hot
    self._scale_factor = scale_factor

  def get_config(self):
    config = {
        "vocab_size": self._vocab_size,
        "embedding_width": self._embedding_width,
        "initializer": self._initializer,
        "use_one_hot": self._use_one_hot,
        "scale_factor": self._scale_factor,
    }
    base_config = super(OnDeviceEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    self.embeddings = self.add_weight(
        "embeddings",
        shape=[self._vocab_size, self._embedding_width],
        initializer=self._initializer,
        dtype=tf.float32
    )
    super(OnDeviceEmbedding, self).build(input_shape)

  def call(self, inputs):
    flat_inputs = tf.reshape(inputs, [-1])
    if self._use_one_hot:
      one_hot_data = tf.one_hot(
          flat_inputs, depth=self._vocab_size, dtype=self.embeddings.dtype)
      embeddings = tf.matmul(one_hot_data, self.embeddings)
    else:
      embeddings = tf.gather(self.embeddings, flat_inputs)
    embeddings = tf.reshape(
        embeddings,
        tf.concat([tf.shape(inputs), [self._embedding_width]], axis=0))
    embeddings.set_shape(inputs.shape.as_list() + [self._embedding_width])
    if self._scale_factor:
      embeddings *= self._scale_factor
    return embeddings

