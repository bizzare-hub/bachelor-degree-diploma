from typing import Union, List, Tuple
import tensorflow as tf
from tensorflow.keras.layers import (GlobalAveragePooling1D, GlobalMaxPool1D,
                                     Dense, Activation)
from transformers import TFAutoModel


def _initialize_hugginface_model(
    backbone_name: str = 'roberta-base',
    input_shape: Union[List[int], Tuple[int]] = (256, 2),
    pool_type: str = 'cls',
    n_classes: int = 6,
    activation: str = None
):
    x = inputs = tf.keras.Input(input_shape, dtype=tf.int32)

    input_ids, attention_mask = tf.unstack(x, axis=-1)

    encoder = TFAutoModel.from_pretrained(backbone_name).layers[0]

    outputs = encoder([input_ids, attention_mask])

    if pool_type == 'cls':
        x = outputs[1]
    elif pool_type == 'average':
        x = GlobalAveragePooling1D(name='Aggregation')(outputs[0])
    elif pool_type == 'max':
        x = GlobalMaxPool1D(name='Aggregation')(outputs[0])
    else:
        raise NotImplementedError()

    x = Dense(n_classes, name='class_head')(x)

    if activation is not None:
        x = Activation(activation, name='class_head_act')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def RobertaBaseModel(
    input_shape=(256, 2),
    pool_type='average',
    n_classes=6,
    activation=None
):
    config = {
        'backbone_name': 'roberta-base',
        'input_shape': input_shape,
        'pool_type': pool_type,
        'n_classes': n_classes,
        'activation': activation
    }

    model = _initialize_hugginface_model(**config)

    return model


def RobertaLargeModel(
    input_shape=(256, 2),
    pool_type='average',
    n_classes=6,
    activation=None
):
    config = {
        'backbone_name': 'roberta-large',
        'input_shape': input_shape,
        'pool_type': pool_type,
        'n_classes': n_classes,
        'activation': activation
    }

    model = _initialize_hugginface_model(**config)

    return model


def BertBaseModel(
    input_shape=(256, 2),
    pool_type='average',
    n_classes=6,
    activation=None
):
    config = {
        'backbone_name': 'bert-base-cased',
        'input_shape': input_shape,
        'pool_type': pool_type,
        'n_classes': n_classes,
        'activation': activation
    }

    model = _initialize_hugginface_model(**config)

    return model


def BertLargeModel(
    input_shape=(256, 2),
    pool_type='average',
    n_classes=6,
    activation=None
):
    config = {
        'backbone_name': 'bert-large-cased',
        'input_shape': input_shape,
        'pool_type': pool_type,
        'n_classes': n_classes,
        'activation': activation
    }

    model = _initialize_hugginface_model(**config)

    return model
