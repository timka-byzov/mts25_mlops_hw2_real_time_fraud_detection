import json
import logging
import math
import os
from pathlib import Path
import signal

from confluent_kafka import Consumer, Producer
import psycopg

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def validate_score(payload):
    if not isinstance(payload, dict) or set(payload) != {'transaction_id', 'score', 'fraud_flag'}:
        raise ValueError('Score message must contain transaction_id, score and fraud_flag')
    transaction_id = payload['transaction_id']
    score = payload['score']
    fraud_flag = payload['fraud_flag']
    if not isinstance(transaction_id, str) or not transaction_id or len(transaction_id) > 256:
        raise ValueError('Invalid transaction_id')
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('score must be a finite probability')
    if type(fraud_flag) is not int or fraud_flag not in (0, 1):
        raise ValueError('fraud_flag must be 0 or 1')
    return transaction_id, float(score), fraud_flag


class ScoreWriter:
    def __init__(self):
        brokers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.consumer = Consumer({
            'bootstrap.servers': brokers, 'group.id': 'score-writer',
            'auto.offset.reset': 'earliest', 'enable.auto.commit': False,
        })
        self.producer = Producer({
            'bootstrap.servers': brokers, 'enable.idempotence': True,
            'delivery.timeout.ms': 30000,
        })
        self.connection = psycopg.connect(
            host=os.getenv('POSTGRES_HOST', 'postgres'),
            port=os.getenv('POSTGRES_PORT', '5432'),
            dbname=os.getenv('POSTGRES_DB', 'fraud'),
            user=os.getenv('POSTGRES_USER', 'fraud'),
            password=os.getenv('POSTGRES_PASSWORD', 'fraud'),
            connect_timeout=10,
        )
        self.consumer.subscribe([os.getenv('KAFKA_SCORING_TOPIC', 'scores')])
        self.errors_topic = os.getenv('KAFKA_ERRORS_TOPIC', 'errors')
        self.running = True
        self.heartbeat = Path('/app/logs/heartbeat')

    def stop(self, *_):
        self.running = False

    def save(self, record):
        values = validate_score(record)
        with self.connection.transaction():
            self.connection.execute(
                'INSERT INTO transaction_scores (transaction_id, score, fraud_flag) '
                'VALUES (%s, %s, %s) ON CONFLICT (transaction_id) DO NOTHING', values,
            )

    def reject(self, message, error):
        delivery_errors = []

        def delivery_report(error, _):
            if error is not None:
                delivery_errors.append(str(error))

        self.producer.produce(self.errors_topic, key=message.key(), value=json.dumps({
            'stage': 'storage', 'error': str(error),
            'source_partition': message.partition(), 'source_offset': message.offset(),
        }), on_delivery=delivery_report)
        if self.producer.flush(35) or delivery_errors:
            raise RuntimeError(f'Cannot deliver rejected message: {delivery_errors}')

    def run(self):
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
                    self.save(json.loads(message.value()))
                except (ValueError, KeyError, TypeError) as error:
                    logger.warning('Invalid score: %s', error)
                    self.reject(message, error)
                self.consumer.commit(message=message, asynchronous=False)
        finally:
            self.consumer.close()
            self.connection.close()


if __name__ == '__main__':
    ScoreWriter().run()
