import numpy as np
import pandas as pd
import tensorflow as tf


AUTOTUNE = tf.data.experimental.AUTOTUNE


class DummyJigsawDataset:
    def __init__(self,
                 data_path: list,
                 batch_size: int = 32,
                 n_epochs: int = None,
                 shuffle: bool = False,
                 shuffle_buffer_size: int = None):
        """
        This is a dataset for initial experiments on UnintendedBias dataset.
        The target is just toxicity score located in [0; 1] range.
        """
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size

        data_df = [pd.read_csv(dp)[['input_ids', 'attention_mask', 'label']] for dp in data_path]
        data_df = pd.concat(data_df, axis=0)
        data, scores = self._extract_data(data_df)

        if self.shuffle and self.shuffle_buffer_size is None:
            self.shuffle_buffer_size = data_df.shape[0]

        self.n_samples = data_df.shape[0]

        dataset = tf.data.Dataset.from_tensor_slices(data)
        scores = tf.data.Dataset.from_tensor_slices(scores)

        dataset = self._preprocess_dataset(dataset)
        dataset = tf.data.Dataset.zip((dataset, scores))
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
    def _extract_data(df: pd.DataFrame):
        df.loc[:, 'input_ids'] = df.input_ids.apply(lambda x: list(map(int, x[1:-1].split(', '))))
        df.loc[:, 'attention_mask'] = df.attention_mask.apply(lambda x: list(map(int, x[1:-1].split(', '))))

        input_ids = np.stack(df.input_ids.values).astype(np.int32)
        attn_mask = np.stack(df.attention_mask.values).astype(np.int32)

        scores = df.label.values.astype(np.float32)

        return (input_ids, attn_mask), scores

    @staticmethod
    def _preprocess_dataset(data: tf.data.Dataset):
        def preprocess(input_ids, attn_mask):
            return tf.stack([input_ids, attn_mask], axis=-1)

        data = data.map(preprocess, num_parallel_calls=AUTOTUNE)

        return data


class JigsawDatasetPaired(DummyJigsawDataset):
    def __init__(self,
                 data_path: list,
                 max_length: int = 256,
                 pad_token: int = 1,
                 split_token: int = 2,
                 batch_size: int = 32,
                 n_epochs: int = None,
                 shuffle: bool = True,
                 shuffle_buffer_size: int = None):
        super().__init__(
            data_path=data_path,
            batch_size=batch_size,
            n_epochs=None,
            shuffle=False,
            shuffle_buffer_size=None
        )

        self.n_epochs = n_epochs
        self.max_length = max_length
        self.pad_token = pad_token
        self.split_token = split_token
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size

        self.dataset = self.dataset.unbatch()

        left_dataset = self._shuffle_dataset(self.dataset)
        right_dataset = self._shuffle_dataset(self.dataset)

        # prepare (lhs, [SEP], rhs), (score1 - score2)
        dataset_input = self._preprocess_input(left_dataset, right_dataset)
        dataset_score = self._preprocess_score(left_dataset, right_dataset)

        self.dataset_input = dataset_input
        self.dataset_score = dataset_score

        dataset = tf.data.Dataset.zip((dataset_input, dataset_score))

        dataset = self._shuffle_dataset(dataset)
        self.dataset = dataset.batch(self.batch_size).prefetch(AUTOTUNE)

        if self.n_epochs is not None:
            self.dataset = self.dataset.repeat(self.n_epochs)

    def _preprocess_input(self, data_lhs: tf.data.Dataset, data_rhs: tf.data.Dataset):

        input_lhs = data_lhs.map(lambda x, y: x, num_parallel_calls=AUTOTUNE)
        input_rhs = data_rhs.map(lambda x, y: x, num_parallel_calls=AUTOTUNE)
        input_data = tf.data.Dataset.zip((input_lhs, input_rhs))

        def process(lhs, rhs):
            lhs, _ = tf.unstack(lhs, axis=-1)
            rhs, _ = tf.unstack(rhs, axis=-1)

            if lhs[-1] != self.pad_token:
                end_lhs = tf.shape(lhs)[0]
            else:
                end_lhs = tf.where(lhs == self.pad_token)[0][0]
                end_lhs = tf.cast(end_lhs, tf.int32)

            if rhs[-1] != self.pad_token:
                end_rhs = tf.shape(rhs)[0]
            else:
                end_rhs = tf.where(rhs == self.pad_token)[0][0]
                end_rhs = tf.cast(end_rhs, tf.int32)

            lhs = lhs[:end_lhs]
            rhs = rhs[:end_rhs]

            input_ids = tf.concat([lhs, rhs], axis=0)

            shape = tf.shape(input_ids)[0]
            pad_shape = [[0, self.max_length - shape]]

            input_ids = tf.pad(
                input_ids, paddings=pad_shape, constant_values=self.pad_token)
            attn_mask = tf.concat(
                [tf.ones(shape, dtype=tf.int32),
                 tf.zeros(self.max_length - shape, dtype=tf.int32)],
                axis=0
            )

            sample = tf.stack([input_ids, attn_mask], axis=-1)
            sample = tf.reshape(sample, shape=[self.max_length, 2])

            return sample

        input_data = input_data.map(process, num_parallel_calls=AUTOTUNE)

        return input_data

    @staticmethod
    def _preprocess_score(data_lhs: tf.data.Dataset, data_rhs: tf.data.Dataset):

        score_lhs = data_lhs.map(lambda x, y: y, num_parallel_calls=AUTOTUNE)
        score_rhs = data_rhs.map(lambda x, y: y, num_parallel_calls=AUTOTUNE)
        score_data = tf.data.Dataset.zip((score_lhs, score_rhs))

        def preprocess(lhs, rhs): return tf.math.abs(lhs - rhs)

        score_data = score_data.map(preprocess, num_parallel_calls=AUTOTUNE)

        return score_data


class JigsawDataset:
    def __init__(self,
                 data_path,
                 group_size: int = None,
                 batch_size: int = 32,
                 n_epochs: int = None,
                 shuffle: bool = False,
                 shuffle_buffer_size: int = None):
        """
        This is a dataset with better shuffling strategy on UnintendedBias dataset.
        The target is just toxicity score located in [0; 1] range.

        Input data needs to be sorted by scores.
        """
        self.group_size = group_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size

        data_df = pd.read_csv(data_path)
        data, scores = self._extract_data(data_df)

        if self.shuffle and self.shuffle_buffer_size is None:
            self.shuffle_buffer_size = data_df.shape[0]

        self.n_samples = data_df.shape[0]

        dataset = tf.data.Dataset.from_tensor_slices(data)
        scores = tf.data.Dataset.from_tensor_slices(scores)

        dataset = self._preprocess_dataset(dataset)
        dataset = tf.data.Dataset.zip((dataset, scores))

        if self.shuffle:
            dataset = self._shuffle_dataset(dataset)
        self.dataset = dataset.batch(self.batch_size).prefetch(AUTOTUNE)

        if self.n_epochs is not None:
            self.dataset = self.dataset.repeat(self.n_epochs)

        self.n_steps = (self.n_samples // self.batch_size) + 1

    def _shuffle_dataset(self, data: tf.data.Dataset):
        def _shuffle_in_group(sample, score):
            pm_idxes = tf.random.shuffle(tf.range(self.group_size))

            sample = tf.gather(sample, pm_idxes)
            score = tf.gather(score, pm_idxes)

            return tuple((sample, score))

        data = data.batch(self.group_size).map(_shuffle_in_group, num_parallel_calls=AUTOTUNE).unbatch()\
            .batch(4).shuffle(self.shuffle_buffer_size // 4).unbatch()

        return data

    @staticmethod
    def _extract_data(df: pd.DataFrame):
        df.loc[:, 'input_ids'] = df.input_ids.apply(lambda x: list(map(int, x[1:-1].split(', '))))
        df.loc[:, 'attention_mask'] = df.attention_mask.apply(lambda x: list(map(int, x[1:-1].split(', '))))

        input_ids = np.stack(df.input_ids.values).astype(np.int32)
        attn_mask = np.stack(df.attention_mask.values).astype(np.int32)

        scores = df.label.values.astype(np.float32)

        return (input_ids, attn_mask), scores

    @staticmethod
    def _preprocess_dataset(data: tf.data.Dataset):
        def preprocess(input_ids, attn_mask):
            return tf.stack([input_ids, attn_mask], axis=-1)

        data = data.map(preprocess, num_parallel_calls=AUTOTUNE)

        return data
