from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preprocessing import load_train_data, run_preproc
from preprocessor import TransactionPreprocessor, build_preprocessing_state
from scorer import make_pred, model_th

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def preprocessing_pair(tmp_path):
    training = pd.read_csv(ROOT / 'interface/examples/transactions.csv')
    training['target'] = np.arange(len(training)) % 2
    train_path = tmp_path / 'train.csv'
    training.to_csv(train_path, index=False)
    train = load_train_data(train_path)
    return train, TransactionPreprocessor(build_preprocessing_state(train))


@pytest.mark.parametrize('missing_case', ['continuous', 'unknown_category'])
def test_saved_statistics_preserve_reference_features(preprocessing_pair, missing_case):
    train, preprocessor = preprocessing_pair
    transactions = pd.read_csv(ROOT / 'interface/examples/transactions.csv').head(12).copy()
    transactions.loc[2, 'amount'] = np.nan
    transactions.loc[3, 'population_city'] = np.nan
    if missing_case == 'unknown_category':
        transactions.loc[0, 'jobs'] = 'previously_unseen_job'
    expected = run_preproc(train, transactions.copy())
    actual = preprocessor.transform(transactions)
    pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)


def test_prediction_is_independent_of_input_column_order():
    preprocessor = TransactionPreprocessor.load(ROOT / 'fraud_detector/models/preprocessing.json')
    transactions = pd.read_csv(ROOT / 'interface/examples/transactions.csv').head(5)
    first = make_pred(preprocessor.transform(transactions))
    shuffled = transactions[transactions.columns[::-1]].copy()
    shuffled['irrelevant_field'] = 123
    second = make_pred(preprocessor.transform(shuffled))
    pd.testing.assert_frame_equal(first, second)
    assert first['score'].between(0, 1).all()
    assert first['fraud_flag'].tolist() == (first['score'] > model_th).astype(int).tolist()


def test_missing_required_field_is_rejected():
    preprocessor = TransactionPreprocessor.load(ROOT / 'fraud_detector/models/preprocessing.json')
    with pytest.raises(ValueError, match='Missing transaction fields'):
        preprocessor.transform(pd.DataFrame([{'amount': 10}]))


def test_missing_time_preserves_predictions_for_other_rows():
    preprocessor = TransactionPreprocessor.load(ROOT / 'fraud_detector/models/preprocessing.json')
    transactions = pd.read_csv(ROOT / 'interface/examples/transactions.csv').head(5)
    expected = make_pred(preprocessor.transform(transactions.iloc[1:]))
    transactions.loc[0, 'transaction_time'] = None
    actual = make_pred(preprocessor.transform(transactions))
    assert actual['score'].between(0, 1).all()
    np.testing.assert_allclose(actual['score'].iloc[1:], expected['score'], rtol=0, atol=1e-12)
