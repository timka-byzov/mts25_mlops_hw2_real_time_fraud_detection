import pandas as pd
import logging
import os
import json
from pathlib import Path
from catboost import CatBoostClassifier
from features import prepare_features
from preprocessor import TransactionPreprocessor, category_key

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


class ReloadingScorer:
    def __init__(self, model_dir=None):
        root = Path(__file__).resolve().parents[1]
        self.model_dir = Path(model_dir or os.getenv('MODEL_DIR', root.parent / 'runtime/models'))
        self.preprocessor = TransactionPreprocessor.load(root / 'models/preprocessing.json')
        self.manifest_text = None
        self.model = None
        self.version = 'baseline'

    def predict(self, transactions):
        try:
            manifest_text = (self.model_dir / 'current.json').read_text(encoding='utf-8')
        except FileNotFoundError:
            result = make_pred(self.preprocessor.transform(transactions))
            result['model_version'] = 'baseline'
            return result

        if manifest_text != self.manifest_text:
            manifest = json.loads(manifest_text)
            if manifest['preprocessing'] != 'v2':
                raise ValueError('Unsupported model preprocessing version')
            updated_model = CatBoostClassifier(thread_count=2)
            updated_model.load_model(self.model_dir / manifest['model_file'])
            version = str(manifest['version'])
            self.model = updated_model
            self.version = version
            self.manifest_text = manifest_text
            logger.info('Loaded model %s', version)

        features = prepare_features(transactions)
        scores = self.model.predict_proba(features[self.model.feature_names_], thread_count=2)[:, 1]
        return pd.DataFrame({
            'score': scores,
            'fraud_flag': (scores > model_th).astype(int),
            'model_version': self.version,
        })
