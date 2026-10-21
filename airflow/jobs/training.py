import json
import logging
import os
import shutil

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from features import CATEGORICAL_FEATURES, prepare_features
from jobs.data import ensure_data
from jobs.storage import connect, model_directory, read_manifest, run_directory, write_json

logger = logging.getLogger(__name__)


def prepare_dataset(run_id):
    """Сохраняет временной train/test split. Месяц фиксируется на run_id и не меняется при повторе."""
    ensure_data()
    directory = run_directory(run_id)
    selection_path = directory / 'selection.json'
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
    else:
        window_months = int(os.getenv('WINDOW_MONTHS', '3'))
        if window_months < 2:
            raise ValueError('WINDOW_MONTHS must be at least 2')
        with connect() as connection:
            dates = connection.execute(
                "SELECT DISTINCT date_trunc('month', transaction_time) "
                'FROM training_transactions ORDER BY 1'
            ).fetchall()
        months = [pd.Timestamp(row[0]) for row in dates]
        manifest = read_manifest()
        if manifest:
            target = pd.Timestamp(manifest['training_month']) + pd.DateOffset(months=1)
            if target > months[-1]:
                from airflow.exceptions import AirflowSkipException
                raise AirflowSkipException('No new month in training data')
        else:
            target = months[-1]
        start = target - pd.DateOffset(months=window_months - 1)
        expected = set(pd.date_range(start, target, freq='MS'))
        if not expected.issubset(set(months)):
            raise ValueError('Not enough consecutive months for the training window')
        version = target.strftime('%Y-%m') + '-' + directory.name
        selection = {
            'training_month': target.strftime('%Y-%m'),
            'start_month': start.strftime('%Y-%m'),
            'version': version,
            'model_file': version + '.cbm',
        }
        write_json(selection_path, selection)
    start = pd.Timestamp(selection['start_month'])
    target = pd.Timestamp(selection['training_month'])
    end = target + pd.DateOffset(months=1)
    with connect() as connection:
        records = connection.execute(
            'SELECT payload, target FROM training_transactions '
            'WHERE transaction_time >= %s AND transaction_time < %s ORDER BY transaction_time, id',
            (start.to_pydatetime(), end.to_pydatetime()),
        ).fetchall()
    frame = pd.DataFrame([dict(payload, target=label) for payload, label in records])
    if frame.empty:
        raise ValueError('Training window is empty')
    timestamps = pd.to_datetime(frame['transaction_time'])
    train = frame.loc[timestamps < target].reset_index(drop=True)
    test = frame.loc[timestamps >= target].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError('Both train and test months must contain rows')
    train.to_pickle(directory / 'train.pkl')
    test.to_pickle(directory / 'test.pkl')
    logger.info('Window %s..%s: train=%s test=%s', selection['start_month'],
                selection['training_month'], len(train), len(test))
    return selection


def train_model(run_id):
    """Читает сохранённый split и обучает CPU-модель. Повтор пишет только файл этого запуска."""
    directory = run_directory(run_id)
    train = pd.read_pickle(directory / 'train.pkl')
    test = pd.read_pickle(directory / 'test.pkl')
    if train['target'].nunique() != 2:
        raise ValueError('Training sample must contain both target classes')
    model = CatBoostClassifier(
        iterations=50, depth=5, learning_rate=0.1, loss_function='Logloss',
        thread_count=2, random_seed=42, verbose=False, allow_writing_files=False,
    )
    model.fit(prepare_features(train), train['target'], cat_features=CATEGORICAL_FEATURES)
    scores = model.predict_proba(prepare_features(test), thread_count=2)[:, 1]
    clipped = np.clip(scores, 1e-12, 1 - 1e-12)
    labels = test['target'].to_numpy()
    metrics = {
        'train_rows': len(train), 'test_rows': len(test),
        'test_accuracy': float(((scores >= 0.5) == labels).mean()),
        'test_logloss': float(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean()),
    }
    model.save_model(str(directory / 'model.cbm'))
    write_json(directory / 'metrics.json', metrics)
    logger.info('Test metrics: %s', metrics)
    return metrics


def publish_model(run_id):
    """Публикует готовый файл и атомарно меняет current.json. Повтор не сдвигает месяц ещё раз."""
    directory = run_directory(run_id)
    selection = json.loads((directory / 'selection.json').read_text())
    previous = read_manifest()
    if previous and previous['training_month'] >= selection['training_month']:
        return previous['version']
    models = model_directory()
    model_path = models / selection['model_file']
    temporary = model_path.with_suffix('.cbm.tmp')
    shutil.copyfile(directory / 'model.cbm', temporary)
    temporary.replace(model_path)
    metrics = json.loads((directory / 'metrics.json').read_text())
    manifest = dict(selection, preprocessing='v2', **metrics)
    write_json(models / 'current.json', manifest)
    logger.info('Published model %s', selection['version'])
    return selection['version']
