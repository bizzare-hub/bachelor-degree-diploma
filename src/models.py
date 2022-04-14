import tensorflow as tf
from tensorflow.keras.layers import (GlobalAveragePooling1D, GlobalMaxPool1D,
                                     Dense, Activation)
from transformers import TFAutoModel


def RobertaModel(input_shape=(128, 2), embedding_length=1, pool_type='cls',
                 head=False, activation=None):
    """
    params:
        pool_type: type of merging final semantic information outputted by encoder
            'cls': take classification token embedding;
            'average': averaging the sequences embeddings;
            'max': max pooling (or smth) the sequences embeddings
    """
    x = inputs = tf.keras.Input(input_shape, dtype=tf.int32)

    input_ids, attention_mask = tf.unstack(x, axis=-1)

    encoder = TFAutoModel.from_pretrained('roberta-base').layers[0]

    outputs = encoder([input_ids, attention_mask])

    if pool_type == 'cls':
        x = outputs[1]
    elif pool_type == 'average':
        x = GlobalAveragePooling1D(name='Aggregation')(outputs[0])
    elif pool_type == 'max':
        x = GlobalMaxPool1D(name='Aggregation')(outputs[0])
    else:
        raise NotImplementedError()

    x = Dense(embedding_length, name='embedding')(x)

    if head:
        x = Activation(activation, name='logits')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
