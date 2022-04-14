import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tqdm import tqdm


def clean(df: pd.DataFrame, col_name: str = 'text', min_len: int = 15):
    df.loc[:, col_name] = df[col_name].str.replace('https?://\S+|www\.\S+', ' social medium ')
    df.loc[:, col_name] = df[col_name].str.replace('\s+', ' ')
    df.loc[:, col_name] = df[col_name].str.strip()
    df = df[df[col_name].str.len() > min_len]

    df = df.drop_duplicates(subset=col_name, keep=False)

    df.reset_index(drop=True, inplace=True)

    return df


def encode_multilabel(df: pd.DataFrame, columns):
    """
    Dummy inverse transform from
    MultiLabelBinarizer in sklearn
    """
    multilabel_list = []
    labels = []
    label_gen = iter(range(1000))

    for idx, values in tqdm(enumerate(df[columns].values)):
        values = list(values)
        if values not in multilabel_list:
            multilabel_list.append(values)
            labels.append(int(next(label_gen)))

        index = multilabel_list.index(values)
        df.loc[idx, 'label'] = labels[index]

    return df


def get_folds(df, split_by='label', n_folds=5, shuffle=True):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle)

    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df[split_by])):
        df.loc[val_idx, 'fold'] = fold
    df['fold'] = df['fold'].astype(np.uint8)

    return df


def encode_df(df: pd.DataFrame, tokenizer,
              col_name='text', max_length=128,
              drop=True):
    input_ids_list, attn_mask_list = [], []

    n_total = df.shape[0]
    for idx, text in tqdm(enumerate(df[col_name].values), total=n_total):
        output = tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length'
        )

        input_ids, attn_mask = output['input_ids'], output['attention_mask']

        input_ids_list.append(input_ids)
        attn_mask_list.append(attn_mask)

    n_cols = df.shape[1]

    df.insert(n_cols, 'input_ids', input_ids_list)
    df.insert(n_cols + 1, 'attention_mask', attn_mask_list)

    if drop:
        df.drop(col_name, 1, inplace=True)

    return df


def extract_probs(data: tf.data.Dataset, model: tf.keras.Model, n_samples: int):
    data_probs = []

    for samples in tqdm(data, total=n_samples):
        logits = model(samples, training=False)
        probs = tf.math.sigmoid(logits).numpy()

        data_probs.extend(probs)

    data_probs = np.stack(data_probs)

    return data_probs
