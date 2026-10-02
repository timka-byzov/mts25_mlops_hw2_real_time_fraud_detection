import json
import logging
import os
from pathlib import Path
import signal
import sys

import pandas as pd
from confluent_kafka import Consumer, Producer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from preprocessor import TransactionPreprocessor
from scorer import make_pred

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        brokers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.consumer = Consumer({
            'bootstrap.servers': brokers,
            'group.id': 'ml-scorer',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,
        })
        self.producer = Producer({
            'bootstrap.servers': brokers,
            'enable.idempotence': True,
            'delivery.timeout.ms': 30000,
        })
        self.transactions_topic = os.getenv('KAFKA_TRANSACTIONS_TOPIC', 'transactions')
        self.scores_topic = os.getenv('KAFKA_SCORING_TOPIC', 'scores')
        self.errors_topic = os.getenv('KAFKA_ERRORS_TOPIC', 'errors')
        self.preprocessor = TransactionPreprocessor.load(ROOT / 'models/preprocessing.json')
        self.running = True
        self.heartbeat = Path('/app/logs/heartbeat')
        self.consumer.subscribe([self.transactions_topic])

    def stop(self, *_):
        self.running = False

    def publish(self, topic, record, transaction_id):
        delivery_errors = []

        def delivery_report(error, _):
            if error is not None:
                delivery_errors.append(str(error))

        self.producer.produce(
            topic, key=transaction_id, value=json.dumps(record, allow_nan=False),
            on_delivery=delivery_report,
        )
        pending = self.producer.flush(35)
        if pending or delivery_errors:
            raise RuntimeError(f'Kafka delivery failed: {delivery_errors}, pending={pending}')

    def score_message(self, message):
        payload = json.loads(message.value())
        transaction_id = payload['transaction_id']
        if not isinstance(transaction_id, str) or not transaction_id or len(transaction_id) > 256:
            raise ValueError('transaction_id must be a nonempty string of at most 256 characters')
        if not isinstance(payload['data'], dict):
            raise ValueError('Transaction data must be an object')
        features = self.preprocessor.transform(pd.DataFrame([payload['data']]))
        prediction = make_pred(features, 'kafka_stream').iloc[0]
        return {
            'transaction_id': transaction_id,
            'score': float(prediction['score']),
            'fraud_flag': int(prediction['fraud_flag']),
        }

    def process_messages(self):
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        try:
            while self.running:
                self.heartbeat.touch()
                message = self.consumer.poll(1)
                if message is None:
                    continue
                if message.error():
                    raise RuntimeError(str(message.error()))
                try:
                    result = self.score_message(message)
                except (ValueError, KeyError, TypeError, OverflowError) as error:
                    logger.warning('Invalid transaction at %s:%s: %s', message.partition(), message.offset(), error)
                    self.publish(self.errors_topic, {
                        'stage': 'scoring', 'error': str(error),
                        'source_partition': message.partition(), 'source_offset': message.offset(),
                    }, message.key())
                else:
                    self.publish(self.scores_topic, result, result['transaction_id'])
                    logger.info('Scored transaction %s', result['transaction_id'])
                self.consumer.commit(message=message, asynchronous=False)
        finally:
            self.producer.flush(35)
            self.consumer.close()


if __name__ == '__main__':
    ProcessingService().process_messages()
