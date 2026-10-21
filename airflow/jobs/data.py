import os

import pandas as pd
from psycopg.types.json import Jsonb

from jobs.storage import connect, create_tables


def insert_rows(connection, frame):
    records = frame.astype(object).where(pd.notna(frame), None).to_dict('records')
    rows = []
    for record in records:
        target = int(record.pop('target'))
        timestamp = pd.Timestamp(record['transaction_time'])
        record['transaction_time'] = timestamp.isoformat()
        rows.append((timestamp.to_pydatetime(), target, Jsonb(record)))
    with connection.cursor() as cursor:
        cursor.executemany(
            'INSERT INTO training_transactions (transaction_time, target, payload) '
            'VALUES (%s, %s, %s)', rows,
        )


def seed_data(connection):
    """Читает train.csv и заполняет пустую витрину. Непустую таблицу не меняет."""
    if connection.execute('SELECT 1 FROM training_transactions LIMIT 1').fetchone():
        return
    batch_size = int(os.getenv('BATCH_SIZE', '5000'))
    if batch_size < 1:
        raise ValueError('BATCH_SIZE must be positive')
    samples = {}
    source = os.getenv('TRAIN_CSV', '/opt/airflow/data/train.csv')
    for chunk in pd.read_csv(source, chunksize=50000):
        months = pd.to_datetime(chunk['transaction_time']).dt.to_period('M')
        for month, frame in chunk.groupby(months):
            previous = samples.get(month)
            size = len(previous) if previous is not None else 0
            if size < batch_size:
                added = frame.head(batch_size - size)
                samples[month] = pd.concat([previous, added]) if size else added.copy()
    if not samples:
        raise ValueError('Training CSV is empty')
    history_months = int(os.getenv('HISTORY_MONTHS', '36'))
    if history_months < 2:
        raise ValueError('HISTORY_MONTHS must be at least 2')
    cutoff = max(samples) - (history_months - 1)
    for month in sorted(month for month in samples if month >= cutoff):
        insert_rows(connection, samples[month])


def ensure_data():
    """Создаёт витрину и начальную выборку в одной транзакции; повтор безопасен."""
    with connect() as connection:
        connection.execute('SELECT pg_advisory_xact_lock(7303)')
        create_tables(connection)
        seed_data(connection)


def append_month(run_id):
    """Дописывает месяц из последнего батча и удаляет старую историю. run_id защищает от повтора."""
    batch_size = int(os.getenv('BATCH_SIZE', '5000'))
    history_months = int(os.getenv('HISTORY_MONTHS', '36'))
    if batch_size < 1 or history_months < 2:
        raise ValueError('BATCH_SIZE must be positive and HISTORY_MONTHS at least 2')
    with connect() as connection:
        connection.execute('SELECT pg_advisory_xact_lock(7303)')
        create_tables(connection)
        seed_data(connection)
        previous = connection.execute(
            'SELECT month FROM training_data_updates WHERE run_id = %s', (run_id,),
        ).fetchone()
        if previous:
            return previous[0].strftime('%Y-%m')
        latest = pd.Timestamp(connection.execute(
            "SELECT date_trunc('month', MAX(transaction_time)) FROM training_transactions"
        ).fetchone()[0])
        next_month = latest + pd.DateOffset(months=1)
        records = connection.execute(
            'SELECT payload, target FROM training_transactions '
            'WHERE transaction_time >= %s AND transaction_time < %s '
            'ORDER BY id LIMIT %s',
            (latest.to_pydatetime(), next_month.to_pydatetime(), batch_size),
        ).fetchall()
        frame = pd.DataFrame([dict(payload, target=target) for payload, target in records])
        frame['transaction_time'] = pd.to_datetime(frame['transaction_time']) + pd.DateOffset(months=1)
        insert_rows(connection, frame)
        cutoff = next_month - pd.DateOffset(months=history_months - 1)
        connection.execute(
            'DELETE FROM training_transactions WHERE transaction_time < %s',
            (cutoff.to_pydatetime(),),
        )
        connection.execute(
            'INSERT INTO training_data_updates (run_id, month) VALUES (%s, %s)',
            (run_id, next_month.date()),
        )
    return next_month.strftime('%Y-%m')
