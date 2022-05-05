import copy
import numpy as np
import pandas as pd
import tensorflow as tf

from .data_utils import encode_df


AUTOTUNE = tf.data.experimental.AUTOTUNE


class Jigsaw22Dataset:
    def __init__(self,
                 data_path: list,
                 label_columns: list = None,
                 extract_online: bool = False,
                 extract_cfg: dict = None,
                 batch_size: int = 32,
                 n_epochs: int = None,
                 shuffle: bool = False,
                 shuffle_buffer_size: int = None):
        """
        Dataset for simple training

        extract_online == True ~ tokenize texts at creation
                       == False ~ texts are already tokenized and
                                  'input_ids' & 'attention_mask' exist.

        if extract_online == True 'cfg' should be provided:
            {'tokenizer', 'max_length', 'col_name'}
        """
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size

        data_df = [pd.read_csv(dp) for dp in data_path]
        data_df = pd.concat(data_df, axis=0)
        if extract_online:
            data = self._extract_data_online(data_df.copy(), extract_cfg)
        else:
            data = self._extract_data(data_df.copy())
        labels = self._extract_labels(data_df.copy(), label_columns)

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
    def _extract_data(df: pd.DataFrame):
        df.loc[:, 'input_ids'] = df.input_ids.apply(lambda x: list(map(int, x[1:-1].split(', '))))
        df.loc[:, 'attention_mask'] = df.attention_mask.apply(lambda x: list(map(int, x[1:-1].split(', '))))

        input_ids = np.stack(df.input_ids.values).astype(np.int32)
        attn_mask = np.stack(df.attention_mask.values).astype(np.int32)

        return input_ids, attn_mask

    # @staticmethod
    # def _extract_data_online(df: pd.DataFrame, extract_cfg: dict):
    #     tokenizer = extract_cfg.pop('tokenizer')
    #     max_length = extract_cfg.pop('max_length')
    #     col_name = extract_cfg.pop('col_name')
    #
    #     inputs = encode_df(
    #         df.copy(), tokenizer, col_name=col_name, max_length=max_length
    #     )[['input_ids', 'attention_mask']].values
    #
    #     input_ids, attn_mask = inputs[:, 0], inputs[:, 1]
    #
    #     input_ids = np.stack(input_ids).astype(np.int32)
    #     attn_mask = np.stack(attn_mask).astype(np.int32)
    #
    #     return input_ids, attn_mask

    @staticmethod
    def _extract_data_online(df: pd.DataFrame, extract_cfg: dict):
        tokenizer = extract_cfg.pop('tokenizer')
        max_length = extract_cfg.pop('max_length')
        col_name = extract_cfg.pop('col_name')

        inputs = encode_df(
            df.copy(), tokenizer, col_name=col_name, max_length=max_length
        )[['input_ids', 'attention_mask', 'token_type_ids']].values

        input_ids, attn_mask, token_type = inputs[:, 0], inputs[:, 1], inputs[:, 2]

        input_ids = np.stack(input_ids).astype(np.int32)
        attn_mask = np.stack(attn_mask).astype(np.int32)
        token_type = np.stack(token_type).astype(np.int32)

        return input_ids, attn_mask, token_type

    def _extract_labels(self, df: pd.DataFrame, label_columns: list):
        if label_columns is not None:
            labels = df[label_columns].values.astype(np.float32)
        else:
            labels = None

        return labels

    # @staticmethod
    # def _preprocess_dataset(data: tf.data.Dataset):
    #     def preprocess(input_ids, attn_mask):
    #         return tf.stack([input_ids, attn_mask], axis=-1)
    #
    #     data = data.map(preprocess, num_parallel_calls=AUTOTUNE)
    #
    #     return data

    @staticmethod
    def _preprocess_dataset(data: tf.data.Dataset):
        def preprocess(input_ids, attn_mask, token_type):
            return tf.stack([input_ids, attn_mask, token_type], axis=-1)

        data = data.map(preprocess, num_parallel_calls=AUTOTUNE)

        return data


class Jigsaw22DatasetPaired(Jigsaw22Dataset):
    def __init__(self,
                 data_path: list,
                 tokenizer,
                 paired: bool = False,
                 max_length: int = 256,
                 batch_size: int = 32,
                 n_epochs: int = None,
                 shuffle: bool = True,
                 shuffle_buffer_size: int = None):
        """
        Paired dataset for validation data of Jigsaw2022
        Assuming 'less_toxic' and 'more_toxic' columns exist
        """
        self.paired = paired
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size

        data_df = [pd.read_csv(dp) for dp in data_path]
        data_df = pd.concat(data_df, axis=0)

        data_df = self._clean_text(data_df.copy(), ['less_toxic', 'more_toxic'])
        dataset = self._prepare_data(
            data_df.copy(), tokenizer, max_length)

        if self.shuffle and self.shuffle_buffer_size is None:
            self.shuffle_buffer_size = data_df.shape[0]

        self.n_samples = data_df.shape[0]

        dataset = self._shuffle_dataset(dataset)
        self.dataset = dataset.batch(self.batch_size).prefetch(AUTOTUNE)

        if self.n_epochs is not None:
            self.dataset = self.dataset.repeat(self.n_epochs)

        self.n_steps = (self.n_samples // self.batch_size) + 1

    def _prepare_data(self, df: pd.DataFrame, tokenizer, max_length):
        def _swap(df: pd.DataFrame, lhs_col: str, rhs_col: str, idxes):
            """Straightforward swap"""
            tmp_df = df.copy()

            df.loc[idxes, lhs_col] = df.loc[idxes, rhs_col]
            df.loc[idxes, rhs_col] = tmp_df.loc[idxes, lhs_col]

            return df

        # swap_idxes = np.random.choice(
        #     np.arange(df.shape[0]), size=df.shape[0] // 2, replace=False)
        # df = _swap(df.copy(), 'less_toxic', 'more_toxic', swap_idxes)

        if self.paired:
            data = self._extract_data_online(
                df.copy(),
                {'tokenizer': tokenizer, 'max_length': max_length, 'col_name': ['less_toxic', 'more_toxic']})

            data = tf.data.Dataset.from_tensor_slices(data)
            data = self._preprocess_dataset(data)

            # 1 - first more toxic, 0 - no
            labels = df.label.values.astype(np.int32)
            labels = tf.data.Dataset.from_tensor_slices(labels)
        else:
            lhs_data = self._extract_data_online(
                df.copy(),
                {'tokenizer': tokenizer, 'max_length': max_length, 'col_name': 'less_toxic'})
            rhs_data = self._extract_data_online(
                df.copy(),
                {'tokenizer': tokenizer, 'max_length': max_length, 'col_name': 'more_toxic'})

            lhs_data = tf.data.Dataset.from_tensor_slices(lhs_data)
            rhs_data = tf.data.Dataset.from_tensor_slices(rhs_data)

            lhs_data = self._preprocess_dataset(lhs_data)
            rhs_data = self._preprocess_dataset(rhs_data)

            data = tf.data.Dataset.zip((lhs_data, rhs_data))

            # 1 - x1 > x2; -1 - x1 < x2
            labels = -1 * np.ones(df.shape[0], dtype=np.int32)
            # labels[swap_idxes] = 1
            labels = tf.data.Dataset.from_tensor_slices(labels)

        dataset = tf.data.Dataset.zip((data, labels))

        return dataset

    @staticmethod
    def _clean_text(df: pd.DataFrame, col_names: list):
        for col_n in col_names:
            df.loc[:, col_n] = df[col_n].str.replace('https?://\S+|www\.\S+', ' social medium ')
            df.loc[:, col_n] = df[col_n].str.replace('\s+', ' ')
            df.loc[:, col_n] = df[col_n].str.strip()

        df.drop_duplicates(subset=col_names, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
