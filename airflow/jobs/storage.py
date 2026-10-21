import hashlib
import json
import os
from pathlib import Path

import psycopg


def connect():
    return psycopg.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        dbname=os.getenv('POSTGRES_DB', 'fraud'),
        user=os.getenv('POSTGRES_USER', 'fraud'),
        password=os.getenv('POSTGRES_PASSWORD', 'fraud'),
    )


def create_tables(connection):
    connection.execute('''
        CREATE TABLE IF NOT EXISTS training_transactions (
            id BIGSERIAL PRIMARY KEY,
            transaction_time TIMESTAMP NOT NULL,
            target SMALLINT NOT NULL CHECK (target IN (0, 1)),
            payload JSONB NOT NULL
        )
    ''')
    connection.execute('''
        CREATE INDEX IF NOT EXISTS training_transactions_time
        ON training_transactions (transaction_time)
    ''')
    connection.execute('''
        CREATE TABLE IF NOT EXISTS training_data_updates (
            run_id TEXT PRIMARY KEY,
            month DATE NOT NULL
        )
    ''')


def run_directory(run_id):
    name = hashlib.sha256(run_id.encode()).hexdigest()[:12]
    directory = Path(os.getenv('AIRFLOW_WORK_DIR', '/opt/airflow/work')) / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def model_directory():
    directory = Path(os.getenv('MODEL_DIR', '/opt/models'))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_manifest():
    path = model_directory() / 'current.json'
    return json.loads(path.read_text()) if path.exists() else None


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)
