import pandas as pd
import logging
import os
from pathlib import Path
from catboost import CatBoostClassifier
from preprocessor import category_key

# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model
model = CatBoostClassifier(thread_count=2)
model.load_model(Path(__file__).resolve().parents[1] / 'models/my_catboost.cbm')

# Define optimal threshold
model_th = float(os.getenv('FRAUD_THRESHOLD', '0.98'))
if not 0 <= model_th <= 1:
    raise ValueError('FRAUD_THRESHOLD must be between 0 and 1')
logger.info('Pretrained model imported successfully...')


def make_pred(dt, source_info="kafka"):

    dt = dt.copy()

    # Меняем формат категориальных фичей на string перед скорингом
    expected_categorical = ['hour',
                            'year',
                            'month',
                            'day_of_month',
                            'day_of_week',
                            'gender_cat',
                            'merch_cat',
                            'cat_id_cat',
                            'one_city_cat',
                            'us_state_cat',
                            'jobs_cat']
    for col in expected_categorical:
        if col in dt.columns:
            dt[col] = dt[col].map(category_key)

    # Calculate score
    scores = model.predict_proba(dt[model.feature_names_], thread_count=2)[:, 1]
    submission = pd.DataFrame({
        'score':  scores,
        'fraud_flag': (scores > model_th) * 1
    })
    logger.info(f'Prediction complete for data from {source_info}')

    # Return proba for positive class
    return submission