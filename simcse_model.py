import tensorflow as tf
from kerasbert import bert_encoder_block


@tf.keras.utils.register_keras_serializable(package="simcse")
class SimCSEPool(tf.keras.models.Model):
    def __init__(self, encoder_config, **kwargs):
        super(SimCSEPool, self).__init__(**kwargs)
        self.encoder = bert_encoder_block.BertEncoderBlock(**encoder_config)

    def call(self, inputs):
        _, pooler_output = self.encoder(inputs)
        return pooler_output


@tf.keras.utils.register_keras_serializable(package="simcse")
class SimCSELastAvg(tf.keras.models.Model):
    def __init__(self, encoder_config, **kwargs):
        super(SimCSELastAvg, self).__init__(**kwargs)
        self.encoder = bert_encoder_block.BertEncoderBlock(**encoder_config)
        self.pool_layer = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs):
        encoder_outputs, _ = self.encoder(inputs)
        output = self.pool_layer(encoder_outputs[-1])
        return output


@tf.keras.utils.register_keras_serializable(package="simcse")
class SimCSEFirstLastAvg(tf.keras.models.Model):
    def __init__(self, encoder_config, **kwargs):
        super(SimCSELastAvg, self).__init__(**kwargs)
        self.encoder = bert_encoder_block.BertEncoderBlock(**encoder_config)
        self.pool_layer = tf.keras.layers.GlobalAveragePooling1D()
        self.output_layer = tf.keras.layers.Average()

    def call(self, inputs):
        encoder_outputs, _ = self.encoder(inputs)
        outputs = [
            self.pool_layer(encoder_outputs[0]),
            self.pool_layer(encoder_outputs[-1]),
        ]
        output = self.output_layer(outputs)
        return output


@tf.keras.utils.register_keras_serializable(package="simcse")
class SimCSECLS(tf.keras.models.Model):
    def __init__(self, encoder_config, **kwargs):
        super(SimCSELastAvg, self).__init__(**kwargs)
        self.encoder = bert_encoder_block.BertEncoderBlock(**encoder_config)

    def call(self, inputs):
        encoder_outputs, _ = self.encoder(inputs)
        output = tf.keras.layers.Lambda(lambda x: x[:, 0])(encoder_outputs[-1])
        return output


@tf.keras.utils.register_keras_serializable(package="simcse")
class SimLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(SimLoss, self).__init__(**kwargs)
        
    def call(self, y_true, y_pred):
        # 构造标签
        idxs = tf.range(0, tf.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        y_true = tf.equal(idxs_1, idxs_2)
        y_true = tf.cast(y_true, tf.float32)
        # 计算相似度
        y_pred = tf.math.l2_normalize(y_pred, axis=1)
        similarities = tf.matmul(y_pred, y_pred, transpose_b=True)
        similarities = similarities - tf.eye(tf.shape(y_pred)[0]) * 1e12
        similarities = similarities * 20
        loss = tf.keras.losses.categorical_crossentropy(y_true, similarities, from_logits=True)
        return tf.reduce_mean(loss)
    
    def get_config(self):
        return super(SimLoss, self).get_config()
