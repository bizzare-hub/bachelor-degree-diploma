from functools import partial
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.losses import LossFunctionWrapper


@tf.function
def sigmoid_cross_entropy_with_logits(inputs, focal=False):
    labels, logits = inputs

    if focal:
        return tfa.losses.sigmoid_focal_crossentropy(
            labels, logits, from_logits=True)
    else:
        return tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)


@tf.function
def multiclass_sigmoid_cross_entropy_with_logits(labels, logits, focal=False):
    """
    labels [B, N_CLASSES]: B multi-hot vectors
    logits [B, N_CLASSES]: B float vectors representing logits
    """
    labels = tf.cast(labels, tf.float32)

    labels = tf.transpose(labels)
    logits = tf.transpose(logits)

    loss_fn = partial(sigmoid_cross_entropy_with_logits, focal=focal)

    losses = tf.map_fn(
        loss_fn,
        (labels, logits),
        fn_output_signature=tf.float32
    )

    return tf.reduce_mean(losses)


class MulticlassCrossentropy(LossFunctionWrapper):
    def __init__(self, focal=False, **kwargs):
        super(MulticlassCrossentropy, self).__init__(
            multiclass_sigmoid_cross_entropy_with_logits,
            focal=focal,
            reduction=tf.keras.losses.Reduction.NONE,
            name="MulticlassCrossentropy",
            **kwargs
        )
