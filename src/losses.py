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


@tf.function
def score_sigmoid_binary_cross_entropy_with_logits(scores, logits,
                                                   min_distance=0.2):
    """
    Paired BCE with scores
    scores [batch_size, 2]: scores for input pairs
    logits [batch_size, 1]: logits
    min_distance [1]: considering  a pair if
                      |score1 - score2| > min_distance
    """
    score_lhs, score_rhs = tf.unstack(scores, axis=-1)

    mask = tf.math.greater(
        tf.math.abs(score_lhs - score_rhs), min_distance)
    mask = tf.cast(mask, tf.float32)
    mask_sum = tf.reduce_sum(mask) + 1e-5

    labels = tf.math.greater_equal(score_lhs, score_rhs)
    labels = tf.expand_dims(tf.cast(labels, tf.float32), axis=1)

    losses = tf.squeeze(
        tf.nn.sigmoid_cross_entropy_with_logits(labels, logits))
    losses = losses * mask

    loss = tf.math.divide(
        tf.reduce_sum(losses), mask_sum)

    return loss


def margin_ranking_loss(labels, preds, margin=0.5):
    """
    torch.nn.MarginRankingLoss implementation

    preds [[B, 1], [B, 1]]: lhs and rhs scores
    """
    labels = tf.cast(labels, preds.dtype)

    score_lhs, score_rhs = tf.unstack(preds, axis=-1)

    losses = tf.maximum(
        -1. * labels * (score_lhs - score_rhs) + margin, 0.0)

    return tf.reduce_mean(losses)


def dynamic_margin_loss(scores, preds,
                        min_distance=0.05,
                        max_distance=2.0,
                        distance_metric="L2"):
    """
    DynamicMarginRankingLoss on scalars.

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
    mask_min = tf.math.greater(scores_dists, min_distance)
    mask_max = tf.math.greater(scores_dists, max_distance)
    mask = tf.cast(
        tf.logical_or(mask_min, mask_max),
        tf.float32
    )

    mask_sum = tf.reduce_sum(mask)

    distances = tf.math.add(
        -1.0 * pred_dists * mask,
        scores_dists * mask
    )
    distances = tf.maximum(distances, 0.0)

    loss = tf.reduce_sum(distances) / mask_sum

    return loss


class ScoreBinaryCrossentropy(LossFunctionWrapper):
    def __init__(
        self,
        min_distance
    ):
        super().__init__(
            score_sigmoid_binary_cross_entropy_with_logits,
            min_distance=min_distance,
            reduction=tf.keras.losses.Reduction.NONE,
            name="ScoreBinaryCrossEntropy"
        )


class MarginRankingLoss(LossFunctionWrapper):
    def __init__(
        self,
        margin
    ):
        super().__init__(
            margin_ranking_loss,
            margin=margin,
            reduction=tf.keras.losses.Reduction.NONE,
            name="MarginRankingLoss"
        )


class DynamicMarginLoss(LossFunctionWrapper):
    def __init__(
        self,
        min_distance,
        max_distance,
        distance_metric
    ):
        super().__init__(
            dynamic_margin_loss,
            min_distance=min_distance,
            max_distance=max_distance,
            distance_metric=distance_metric,
            reduction=tf.keras.losses.Reduction.NONE,
            name="DynamicMarginLoss"
        )
