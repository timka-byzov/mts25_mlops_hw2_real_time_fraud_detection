import json
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import add_distance_features, add_time_features


CATEGORICAL_COLUMNS = ['gender', 'merch', 'cat_id', 'one_city', 'us_state', 'jobs']
TIME_COLUMNS = ['hour', 'year', 'month', 'day_of_month', 'day_of_week']
CONTINUOUS_COLUMNS = ['amount', 'population_city', 'distance']
DROPPED_COLUMNS = ['name_1', 'name_2', 'street', 'post_code']
INPUT_COLUMNS = [
    'transaction_time', 'merch', 'cat_id', 'amount', 'name_1', 'name_2',
    'gender', 'street', 'one_city', 'us_state', 'post_code', 'lat', 'lon',
    'population_city', 'jobs', 'merchant_lat', 'merchant_lon',
]


def category_key(value):
    if pd.isna(value):
        return '__missing__'
    if isinstance(value, (float, np.floating)) and value.is_integer():
        return str(int(value))
    return str(value)


def build_preprocessing_state(train):
    category_mappings = {}
    for column in CATEGORICAL_COLUMNS:
        mapping = train[[column, column + '_cat']].drop_duplicates()
        category_mappings[column] = {
            category_key(value): encoded
            for value, encoded in mapping.itertuples(index=False, name=None)
        }
    mean_encodings = {}
    for column in [name + '_cat' for name in CATEGORICAL_COLUMNS] + TIME_COLUMNS:
        means = train.groupby(column)['target'].mean()
        mean_encodings[column] = {category_key(key): float(value) for key, value in means.items()}
    return {
        'version': 1,
        'category_mappings': category_mappings,
        'mean_encodings': mean_encodings,
        'continuous_means': {column: float(train[column].mean()) for column in CONTINUOUS_COLUMNS},
    }


class TransactionPreprocessor:
    def __init__(self, state):
        if state['version'] != 1:
            raise ValueError('Unsupported preprocessing artifact version')
        self.state = state

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def transform(self, transactions):
        missing_columns = set(INPUT_COLUMNS) - set(transactions.columns)
        if missing_columns:
            raise ValueError('Missing transaction fields: ' + ', '.join(sorted(missing_columns)))
        frame = transactions[INPUT_COLUMNS].copy().reset_index(drop=True)
        frame = frame.drop(columns=DROPPED_COLUMNS)
        for column in CATEGORICAL_COLUMNS:
            frame[column + '_cat'] = frame[column].map(category_key).map(
                self.state['category_mappings'][column]
            ).fillna('cat_NAN')
            frame = frame.drop(columns=column)
        frame = add_time_features(frame)
        for column, encoding in self.state['mean_encodings'].items():
            frame[column] = frame[column].fillna('cat_NAN')
            frame[column + '_mean_enc'] = frame[column].map(category_key).map(encoding)
        frame = add_distance_features(frame)
        for column in CONTINUOUS_COLUMNS:
            values = pd.to_numeric(frame[column], errors='raise').fillna(
                self.state['continuous_means'][column]
            )
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'{column} must be finite and nonnegative')
            frame[column + '_log'] = np.log(values + 1)
            frame = frame.drop(columns=column)
        return frame
