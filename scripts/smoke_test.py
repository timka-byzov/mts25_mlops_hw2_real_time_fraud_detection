import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import uuid

from confluent_kafka import Consumer, Producer
import numpy as np
import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'fraud_detector/src'))
sys.path.insert(0, str(ROOT / 'interface'))

from preprocessor import TransactionPreprocessor
from scorer import make_pred
from results import fetch_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--brokers', default='127.0.0.1:9095')
    parser.add_argument('--postgres-host', default='127.0.0.1')
    parser.add_argument('--postgres-port', default='5433')
    parser.add_argument('--timeout', type=int, default=120)
    args = parser.parse_args()
    os.environ['POSTGRES_HOST'] = args.postgres_host
    os.environ['POSTGRES_PORT'] = args.postgres_port
    prefix = 'smoke-' + uuid.uuid4().hex
    observer = Consumer({
        'bootstrap.servers': args.brokers, 'group.id': prefix,
        'auto.offset.reset': 'earliest', 'enable.auto.commit': False,
    })
    producer = Producer({'bootstrap.servers': args.brokers, 'enable.idempotence': True})
    observer.subscribe(['scores', 'errors'])
    deadline = time.monotonic() + args.timeout
    while not observer.assignment():
        observer.poll(0.5)
        if time.monotonic() > deadline:
            raise TimeoutError('Kafka observer could not acquire partitions')

    transactions = pd.read_csv(ROOT / 'interface/examples/transactions.csv').head(120)
    predictions = make_pred(TransactionPreprocessor.load(
        ROOT / 'fraud_detector/models/preprocessing.json'
    ).transform(transactions))
    identifiers = [prefix + '-' + str(index) for index in range(len(transactions))]
    expected = {
        identifier: (float(row.score), int(row.fraud_flag))
        for identifier, row in zip(identifiers, predictions.itertuples(index=False))
    }
    records = transactions.astype(object).where(pd.notna(transactions), None).to_dict(orient='records')
    for identifier, record in zip(identifiers, records):
        producer.produce('transactions', key=identifier,
                         value=json.dumps({'transaction_id': identifier, 'data': record}, allow_nan=False))
    producer.produce('transactions', key=identifiers[0], value=json.dumps({
        'transaction_id': identifiers[0], 'data': records[0],
    }, allow_nan=False))
    producer.produce('transactions', key=prefix, value=json.dumps({'transaction_id': prefix, 'data': {}}))
    producer.produce('scores', key=prefix, value=json.dumps({
        'transaction_id': prefix, 'score': 2, 'fraud_flag': 1,
    }))
    assert producer.flush(30) == 0
    received = Counter()
    rejected_stages = set()
    with psycopg.connect(
        host=args.postgres_host, port=args.postgres_port,
        dbname=os.getenv('POSTGRES_DB', 'fraud'), user=os.getenv('POSTGRES_USER', 'fraud'),
        password=os.getenv('POSTGRES_PASSWORD', 'fraud'), autocommit=True,
    ) as connection:
        while time.monotonic() < deadline:
            message = observer.poll(0.25)
            if message is not None:
                if message.error():
                    raise RuntimeError(str(message.error()))
                record = json.loads(message.value())
                if message.topic() == 'scores' and record.get('transaction_id') in expected:
                    assert set(record) == {'transaction_id', 'score', 'fraud_flag'}
                    score, flag = expected[record['transaction_id']]
                    np.testing.assert_allclose(record['score'], score, rtol=0, atol=1e-12)
                    assert record['fraud_flag'] == flag
                    received[record['transaction_id']] += 1
                elif message.topic() == 'errors' and message.key() == prefix.encode():
                    rejected_stages.add(record['stage'])
            stored = connection.execute(
                'SELECT transaction_id, score, fraud_flag FROM transaction_scores '
                'WHERE transaction_id = ANY(%s)', (identifiers,),
            ).fetchall()
            if len(stored) == len(expected) and received[identifiers[0]] >= 2 and set(received) == set(expected) and rejected_stages == {'scoring', 'storage'}:
                break
        else:
            raise TimeoutError(f'Pipeline incomplete: stored={len(stored)}, scores={len(received)}, rejected={rejected_stages}')
        assert len(stored) == len(expected)
        for identifier, score, flag in stored:
            np.testing.assert_allclose(score, expected[identifier][0], rtol=0, atol=1e-12)
            assert flag == expected[identifier][1]
        all_rows = connection.execute(
            'SELECT transaction_id, score, fraud_flag FROM transaction_scores ORDER BY id DESC'
        ).fetchall()
        displayed = fetch_results()
        assert displayed['recent_scores']['transaction_id'].tolist() == [row[0] for row in all_rows[:100]]
        assert displayed['fraudulent']['transaction_id'].tolist() == [row[0] for row in all_rows if row[2] == 1][:10]
        assert displayed['total'] == len(all_rows)
        invalid_count = connection.execute(
            'SELECT COUNT(*) FROM transaction_scores WHERE transaction_id = %s', (prefix,),
        ).fetchone()[0]
        assert invalid_count == 0
    observer.close()
    print(json.dumps({
        'status': 'passed', 'transactions': len(expected),
        'fraudulent': sum(flag for _, flag in expected.values()),
        'duplicate_delivery_checked': True, 'invalid_messages_checked': sorted(rejected_stages),
        'histogram_records': len(displayed['recent_scores']),
        'fraud_table_records': len(displayed['fraudulent']),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
