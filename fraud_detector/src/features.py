import numpy as np
import pandas as pd

from preprocessor import INPUT_COLUMNS, category_key


CATEGORICAL_FEATURES = ['gender', 'cat_id', 'merch', 'us_state', 'jobs']


def prepare_features(transactions):
    """Готовит одинаковые признаки для обучения и скоринга, не хранит статистики."""
    missing = set(INPUT_COLUMNS) - set(transactions.columns)
    if missing:
        raise ValueError('Missing transaction fields: ' + ', '.join(sorted(missing)))
    frame = transactions.reset_index(drop=True)
    features = pd.DataFrame(index=frame.index)
    for column in ['amount', 'population_city']:
        values = pd.to_numeric(frame[column], errors='raise').fillna(0)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'{column} must be finite and nonnegative')
        features['amount_log' if column == 'amount' else 'population_log'] = np.log1p(values)

    coordinates = frame[['lat', 'lon', 'merchant_lat', 'merchant_lon']].apply(
        pd.to_numeric, errors='raise'
    ).fillna(0)
    if not np.isfinite(coordinates.to_numpy()).all():
        raise ValueError('Coordinates must be finite')
    if (coordinates[['lat', 'merchant_lat']].abs() > 90).any().any() or (
        coordinates[['lon', 'merchant_lon']].abs() > 180
    ).any().any():
        raise ValueError('Coordinates are out of range')
    lat, lon, merchant_lat, merchant_lon = np.radians(coordinates.to_numpy()).T
    distance = np.sin((merchant_lat - lat) / 2) ** 2
    distance += np.cos(lat) * np.cos(merchant_lat) * np.sin((merchant_lon - lon) / 2) ** 2
    features['distance_km'] = 6371 * 2 * np.arcsin(np.sqrt(np.clip(distance, 0, 1)))

    dates = pd.to_datetime(frame['transaction_time'], utc=True)
    features['hour'] = dates.dt.hour.fillna(0).astype(int)
    features['weekday'] = dates.dt.dayofweek.fillna(0).astype(int)
    features['month'] = dates.dt.month.fillna(0).astype(int)
    features['is_weekend'] = (features['weekday'] >= 5).astype(int)
    for column in CATEGORICAL_FEATURES:
        features[column] = frame[column].map(category_key)
    return features
