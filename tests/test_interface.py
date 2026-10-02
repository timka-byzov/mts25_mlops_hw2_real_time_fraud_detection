from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import results

ROOT = Path(__file__).resolve().parents[1]


def test_empty_results_are_displayed(monkeypatch):
    monkeypatch.setattr(results, 'fetch_results', lambda: {
        'fraudulent': pd.DataFrame(columns=['transaction_id', 'score', 'fraud_flag', 'created_at']),
        'recent_scores': pd.DataFrame(columns=['transaction_id', 'score']),
        'total': 0,
    })
    app = AppTest.from_file(str(ROOT / 'interface/app.py')).run()
    next(button for button in app.button if button.label == 'Посмотреть результаты').click().run()
    assert not app.exception
    assert app.metric[0].value == '0'
    assert len(app.info) == 2


def test_results_table_and_histogram_are_displayed(monkeypatch):
    monkeypatch.setattr(results, 'fetch_results', lambda: {
        'fraudulent': pd.DataFrame([['fraud-1', 0.999, 1, '2026-10-02']],
                                  columns=['transaction_id', 'score', 'fraud_flag', 'created_at']),
        'recent_scores': pd.DataFrame([['tx-1', 0.2], ['fraud-1', 0.999]],
                                     columns=['transaction_id', 'score']),
        'total': 2,
    })
    app = AppTest.from_file(str(ROOT / 'interface/app.py')).run()
    next(button for button in app.button if button.label == 'Посмотреть результаты').click().run()
    assert not app.exception
    assert app.dataframe[0].value['transaction_id'].tolist() == ['fraud-1']
    assert len(app.get('plotly_chart')) == 1
    assert '2 транзакций' in app.caption[0].value
