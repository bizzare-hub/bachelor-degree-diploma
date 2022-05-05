import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def clean(df: pd.DataFrame, col_name: str = 'text', min_len: int = 15):
    df.loc[:, col_name] = df[col_name].str.replace('https?://\S+|www\.\S+', ' social medium ')
    df.loc[:, col_name] = df[col_name].str.replace('\s+', ' ')
    df.loc[:, col_name] = df[col_name].str.strip()
    df = df[df[col_name].str.len() > min_len]

    df = df.drop_duplicates(subset=col_name, keep=False)

    df.reset_index(drop=True, inplace=True)

    return df


def rank_comments(df: pd.DataFrame, aggregate_cols: list, weights: list,
                  out_col='label'):
    """Weighted sum of diff. toxicity labels"""

    df.loc[:, out_col] = (weights * df[aggregate_cols]).sum(axis=1)

    return df


def get_groups(df: pd.DataFrame, n_groups: int = 10, col_name: str = 'target'):
    """Group col. 'col_name' values"""
    q_orders = [(1. / n_groups) * n for n in range(n_groups)]
    quantiles = df[col_name].quantile(q_orders).values
    quantiles = sorted(list(set(quantiles)))

    print(f'Effective number of groups: {len(quantiles)}')
    print(f'Quantiles: {quantiles}')

    for i, q in enumerate(quantiles):
        df.loc[df[col_name] >= q, f'{col_name}_group'] = i

    return df


def get_folds(df, split_by='label', n_folds=5, shuffle=True):
    """
    Get stratified folds

    !: Use it only if 'labels' have a comparably small number of values
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle)

    split_vals = df[split_by].unique()
    for idx, sv in enumerate(split_vals):
        df.loc[df[split_by] == sv, f'encoded_split_col'] = idx

    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df['encoded_split_col'])):
        df.loc[val_idx, 'fold'] = fold
    df['fold'] = df['fold'].astype(np.uint8)

    df = df.drop('encoded_split_col', 1)

    return df


def encode_df(df: pd.DataFrame, tokenizer,
              col_name='text', max_length=128,
              drop=True):
    """Supports both types of inputs: (paired and singular)"""
    input_ids_list, attn_mask_list = [], []
    token_type_list = []  # actually used only for paired

    for idx, text in enumerate(df[col_name].values):
        if isinstance(text, str):
            text = [text]

        output = tokenizer.encode_plus(
            *text,
            truncation=True,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length'
        )

        input_ids, attn_mask = output['input_ids'], output['attention_mask']
        token_type = output['token_type_ids']

        input_ids_list.append(input_ids)
        attn_mask_list.append(attn_mask)
        token_type_list.append(token_type)

    n_cols = df.shape[1]

    df.insert(n_cols, 'input_ids', input_ids_list)
    df.insert(n_cols + 1, 'attention_mask', attn_mask_list)
    df.insert(n_cols + 2, 'token_type_ids', token_type_list)

    if drop:
        df.drop(col_name, 1, inplace=True)

    return df


