import tensorflow as tf

from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow_addons.losses import metric_learning


def _dim_1_pairwise_distance(arr):
    """
    Copy of 1-dim L2 distance?

    params:
    arr: [batch_size, 1]
    """

    return tf.math.subtract(
        arr, tf.transpose(arr))


def dynamic_margin_loss(scores, preds,
                        min_distance=0.05,
                        distance_metric="L2"):
    """
    DynamicMarginRankedLoss on scalars.

    params:
        scores: [batch_size, ?]
        preds: [batch_size, 1]
    """
    sshape = tf.shape(scores)
    scores = tf.reshape(scores, [sshape[0], 1])

    scores_dists = tf.math.subtract(
        scores, tf.transpose(scores))

    if distance_metric == 'L2':
        pred_dists = metric_learning.pairwise_distance(
            preds, squared=False)
    elif distance_metric == 'squared-L2':
        pred_dists = metric_learning.pairwise_distance(
            preds, squared=True)
    else:
        pred_dists = distance_metric(preds)

    # that way we also remove duplicates (ij : scores_i - scores_j < 0)
    mask = tf.math.greater(scores_dists, min_distance)
    mask = tf.cast(mask, tf.float32)

    mask_sum = tf.reduce_sum(mask)

    distances = tf.math.add(
        -1.0 * pred_dists * mask,
        scores_dists * mask
    )
    distances = tf.maximum(distances, 0.0)

    loss = tf.reduce_sum(distances) / mask_sum

    return loss


class DynamicMarginLoss(LossFunctionWrapper):
    def __init__(
        self,
        min_distance,
        distance_metric
    ):
        super().__init__(
            dynamic_margin_loss,
            min_distance=min_distance,
            distance_metric=distance_metric,
            reduction=tf.keras.losses.Reduction.NONE,
            name='DynamicMarginLoss'
        )
