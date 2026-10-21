from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.python import PythonOperator

from jobs.data import append_month


with DAG(
    'add_training_month',
    start_date=datetime(2024, 1, 1),
    schedule=os.getenv('MONTH_SCHEDULE', '*/10 * * * *'),
    catchup=False,
    max_active_runs=1,
    default_args={'retries': 1, 'retry_delay': timedelta(seconds=30)},
    tags=['fraud'],
) as dag:
    append = PythonOperator(
        task_id='append_month', python_callable=append_month,
        op_kwargs={'run_id': '{{ run_id }}'},
    )
