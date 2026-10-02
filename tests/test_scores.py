import pytest

from score_writer.app import validate_score


@pytest.mark.parametrize('payload', [
    None,
    [],
    {'transaction_id': 'x', 'score': 0.5},
    {'transaction_id': 'x', 'score': float('nan'), 'fraud_flag': 1},
    {'transaction_id': 'x', 'score': float('inf'), 'fraud_flag': 1},
    {'transaction_id': 'x', 'score': 1.1, 'fraud_flag': 1},
    {'transaction_id': 'x', 'score': True, 'fraud_flag': 1},
    {'transaction_id': 'x', 'score': 0.5, 'fraud_flag': 2},
    {'transaction_id': 'x', 'score': 0.5, 'fraud_flag': True},
    {'transaction_id': '', 'score': 0.5, 'fraud_flag': 0},
])
def test_invalid_score_is_rejected(payload):
    with pytest.raises(ValueError):
        validate_score(payload)


@pytest.mark.parametrize('score, flag', [(0, 0), (1, 1), (0.8, 0)])
def test_valid_score(score, flag):
    assert validate_score({'transaction_id': 'x', 'score': score, 'fraud_flag': flag}) == ('x', score, flag)
