import json
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import pytest

from features import CATEGORICAL_FEATURES, prepare_features
from preprocessor import TransactionPreprocessor
from scorer import ReloadingScorer, make_pred


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def transactions():
    return pd.read_csv(ROOT / 'interface/examples/transactions.csv').head(40)


def publish_model(directory, transactions, version, invert=False):
    features = prepare_features(transactions)
    labels = (np.arange(len(features)) % 2) ^ invert
    model = CatBoostClassifier(
        iterations=12, depth=2, random_seed=42, thread_count=2,
        verbose=False, allow_writing_files=False,
    )
    model.fit(features, labels, cat_features=CATEGORICAL_FEATURES)
    model.save_model(directory / f'{version}.cbm')
    temporary = directory / 'current.tmp'
    temporary.write_text(json.dumps({
        'version': version, 'model_file': f'{version}.cbm',
        'training_month': version, 'preprocessing': 'v2',
    }))
    temporary.replace(directory / 'current.json')


def test_baseline_is_used_until_first_publication(tmp_path, transactions):
    scorer = ReloadingScorer(tmp_path)
    preprocessor = TransactionPreprocessor.load(ROOT / 'fraud_detector/models/preprocessing.json')
    expected = make_pred(preprocessor.transform(transactions))
    actual = scorer.predict(transactions)
    pd.testing.assert_frame_equal(actual[['score', 'fraud_flag']], expected)
    assert actual['model_version'].eq('baseline').all()


def test_existing_scorer_reloads_published_model(tmp_path, transactions):
    scorer = ReloadingScorer(tmp_path)
    publish_model(tmp_path, transactions, '2020-03')
    first = scorer.predict(transactions)
    publish_model(tmp_path, transactions, '2020-04', invert=True)
    second = scorer.predict(transactions)
    assert first['model_version'].eq('2020-03').all()
    assert second['model_version'].eq('2020-04').all()
    assert not np.allclose(first['score'], second['score'])
    assert second['score'].between(0, 1).all()


def test_broken_publication_does_not_silently_use_baseline(tmp_path, transactions):
    (tmp_path / 'current.json').write_text('{')
    with pytest.raises(ValueError):
        ReloadingScorer(tmp_path).predict(transactions)


def test_features_use_transaction_time_and_ignore_target(transactions):
    transactions = transactions.head(1).copy()
    transactions['transaction_time'] = '2020-04-05 21:30:00'
    transactions['target'] = 1
    features = prepare_features(transactions)
    assert features[['hour', 'weekday', 'month', 'is_weekend']].iloc[0].tolist() == [21, 6, 4, 1]
    transactions['target'] = 0
    pd.testing.assert_frame_equal(prepare_features(transactions), features)


def test_new_preprocessing_requires_transaction_fields():
    with pytest.raises(ValueError, match='Missing transaction fields'):
        prepare_features(pd.DataFrame([{'amount': 10}]))
