import os

import pandas as pd
import psycopg


def fetch_results():
    with psycopg.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        dbname=os.getenv('POSTGRES_DB', 'fraud'),
        user=os.getenv('POSTGRES_USER', 'fraud'),
        password=os.getenv('POSTGRES_PASSWORD', 'fraud'),
        connect_timeout=5,
    ) as connection:
        connection.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        fraudulent = connection.execute(
            'SELECT transaction_id, score, fraud_flag, created_at '
            'FROM transaction_scores WHERE fraud_flag = 1 ORDER BY id DESC LIMIT 10'
        ).fetchall()
        recent_scores = connection.execute(
            'SELECT transaction_id, score FROM transaction_scores ORDER BY id DESC LIMIT 100'
        ).fetchall()
        total = connection.execute('SELECT COUNT(*) FROM transaction_scores').fetchone()[0]
    return {
        'fraudulent': pd.DataFrame(fraudulent, columns=['transaction_id', 'score', 'fraud_flag', 'created_at']),
        'recent_scores': pd.DataFrame(recent_scores, columns=['transaction_id', 'score']),
        'total': total,
    }
