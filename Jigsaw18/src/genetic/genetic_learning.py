from functools import partial
import tensorflow as tf
import numpy as np

from .geneticalgorithm import geneticalgorithm as ga
from ..data_utils import extract_probs


def genetic_loss(weights: np.ndarray,
                 less_toxic_probs: np.ndarray,
                 more_toxic_probs: np.ndarray):
    """
    less_toxic_preds [N, 6]: probabilities of 6 classes of toxicity
                             for each of the less toxic comments
    more_toxic_preds [N, 6]: -||- for each of the more toxic comments
    weights [6]: optimized weights, one for each class
    """
    less_toxic_scores = np.sum(
        less_toxic_probs * weights[np.newaxis],
        axis=-1)
    more_toxic_scores = np.sum(
        more_toxic_probs * weights[np.newaxis],
        axis=-1)

    return 1.0 - np.mean(less_toxic_scores < more_toxic_scores)


def train_genetic(loss_fn, varbound, dim=6, var_type='int'):
    model = ga(function=loss_fn, dimension=dim,
               variable_type=var_type, variable_boundaries=varbound)

    model.run()
    best_weights = model.output_dict["variable"]

    return best_weights


def run_genetic(less_toxic_data: tf.data.Dataset, more_toxic_data: tf.data.Dataset,
                model: tf.keras.Model, genetic_cfg: dict, n_samples: int):
    """
    Genetic algorithm training
    """
    less_toxic_probs = extract_probs(less_toxic_data, model, n_samples)
    more_toxic_probs = extract_probs(more_toxic_data, model, n_samples)

    loss_fn = partial(genetic_loss,
                      less_toxic_probs=less_toxic_probs,
                      more_toxic_probs=more_toxic_probs)

    best_weights = train_genetic(loss_fn, **genetic_cfg)
    best_score = loss_fn(best_weights)

    return best_weights, best_score
