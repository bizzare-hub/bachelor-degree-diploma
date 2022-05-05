import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf

from ..data_utils import extract_probs


def train_linear(data: tf.data.Dataset,
                 scores: np.ndarray,
                 model: tf.keras.Model,
                 n_samples: int):
    """
    Train linear model on ruddit scores using
    Jigsaw18 probs
    """
    data_probs = extract_probs(data, model, n_samples)

    assert(data_probs.shape[0] == scores.shape[0]), "x and y don't match!"

    x_train, x_test, y_train, y_test = train_test_split(
        data_probs, scores, test_size=0.1)

    ridge = Ridge(alpha=0.0, positive=True)
    ridge.fit(x_train, y_train)

    train_preds = ridge.predict(x_train)
    test_preds = ridge.predict(x_test)

    print(f"MSE on train: {mean_squared_error(y_train, train_preds)}\n"
          f"MSE on test: {mean_squared_error(y_test, test_preds)}")

    weights = ridge.coef_

    return weights


def evaluate(less_toxic_data: tf.data.Dataset, more_toxic_data: tf.data.Dataset,
             model: tf.keras.Model, n_samples: int, weights: np.ndarray):
    """
    Evaluate results on Jigsaw2022 validation data
    """
    less_toxic_probs = extract_probs(less_toxic_data, model, n_samples)
    more_toxic_probs = extract_probs(more_toxic_data, model, n_samples)

    less_toxic_scores = np.sum(
        less_toxic_probs * weights[np.newaxis],
        axis=-1)
    more_toxic_scores = np.sum(
        more_toxic_probs * weights[np.newaxis],
        axis=-1)

    return np.mean(less_toxic_scores < more_toxic_scores)
