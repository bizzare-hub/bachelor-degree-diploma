import numpy as np
import pandas as pd
import tensorflow as tf


AUTOTUNE = tf.data.experimental.AUTOTUNE


class Jigsaw18Dataset:
    def __init__(self,
                 data_path: list,
                 label_columns: list = None,
                 batch_size: int = 32,
                 n_epochs: int = None,
                 shuffle: bool = False,
                 shuffle_buffer_size: int = None):
        """
        This is a dataset for multiclass classification
        """
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size

        data_df = [pd.read_csv(dp) for dp in data_path]
        data_df = pd.concat(data_df, axis=0)
        data, labels = self._extract_data(data_df, label_columns)

        if self.shuffle and self.shuffle_buffer_size is None:
            self.shuffle_buffer_size = data_df.shape[0]

        self.n_samples = data_df.shape[0]

        dataset = tf.data.Dataset.from_tensor_slices(data)

        dataset = self._preprocess_dataset(dataset)
        if labels is not None:
            labels = tf.data.Dataset.from_tensor_slices(labels)
            dataset = tf.data.Dataset.zip((dataset, labels))
        dataset = self._shuffle_dataset(dataset)
        self.dataset = dataset.batch(self.batch_size).prefetch(AUTOTUNE)

        if self.n_epochs is not None:
            self.dataset = self.dataset.repeat(self.n_epochs)

        self.n_steps = (self.n_samples // self.batch_size) + 1

    def _shuffle_dataset(self, data: tf.data.Dataset):
        if self.shuffle:
            data = data.shuffle(self.shuffle_buffer_size)

        return data

    @staticmethod
    def _extract_data(df: pd.DataFrame, label_columns: list):
        df.loc[:, 'input_ids'] = df.input_ids.apply(lambda x: list(map(int, x[1:-1].split(', '))))
        df.loc[:, 'attention_mask'] = df.attention_mask.apply(lambda x: list(map(int, x[1:-1].split(', '))))

        input_ids = np.stack(df.input_ids.values).astype(np.int32)
        attn_mask = np.stack(df.attention_mask.values).astype(np.int32)

        if label_columns is not None:
            labels = df[label_columns].values.astype(np.float32)
        else:
            labels = None

        return (input_ids, attn_mask), labels

    @staticmethod
    def _preprocess_dataset(data: tf.data.Dataset):
        def preprocess(input_ids, attn_mask):
            return tf.stack([input_ids, attn_mask], axis=-1)

        data = data.map(preprocess, num_parallel_calls=AUTOTUNE)

        return data
