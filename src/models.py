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

    input_ids, attention_mask, token_type_ids = tf.unstack(x, axis=-1)
    model_inp = [input_ids, attention_mask, token_type_ids]

    encoder = TFAutoModel.from_pretrained(backbone_name).layers[0]

    outputs = encoder(model_inp)

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


def _initialize_hugginface_pair_model(
    backbone_name: str = 'roberta-base',
    input_shape: Union[List[int], Tuple[int]] = (256, 2),
    pool_type: str = 'cls',
    n_classes: int = 6,
    activation: str = None,
):
    x_lhs = inputs_lhs = tf.keras.Input(input_shape, dtype=tf.int32)
    x_rhs = inputs_rhs = tf.keras.Input(input_shape, dtype=tf.int32)

    input_ids_lhs, attn_mask_lhs, token_type_lhs = tf.unstack(x_lhs, axis=-1)
    input_ids_rhs, attn_mash_rhs, token_type_rhs = tf.unstack(x_rhs, axis=-1)

    encoder = TFAutoModel.from_pretrained(backbone_name).layers[0]

    output_lhs, output_rhs = encoder([input_ids_lhs, attn_mask_lhs, token_type_lhs]),\
                             encoder([input_ids_rhs, attn_mash_rhs, token_type_rhs])

    if pool_type == 'cls':
        x_lhs, x_rhs = output_lhs[1], output_rhs[1]
    elif pool_type == 'average':
        agg_layer = GlobalAveragePooling1D(name='Aggregation')
        x_lhs = agg_layer(output_lhs[0])
        x_rhs = agg_layer(output_rhs[0])
    elif pool_type == 'max':
        agg_layer = GlobalMaxPool1D(name='Aggregation')
        x_lhs = agg_layer(output_lhs[0])
        x_rhs = agg_layer(output_rhs[0])
    else:
        raise NotImplementedError()

    head = Dense(n_classes, name='class_head')

    x_lhs, x_rhs = head(x_lhs), head(x_rhs)

    if activation is not None:
        act = Activation(activation, name='class_head_act')
        x_lhs, x_rhs = act(x_lhs), act(x_rhs)

    output = tf.concat([x_lhs, x_rhs], axis=-1)

    return tf.keras.Model(inputs=[inputs_lhs, inputs_rhs], outputs=output)


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
