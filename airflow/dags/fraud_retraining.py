from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.python import PythonOperator

from jobs.training import prepare_dataset, publish_model, train_model


with DAG(
    'fraud_retraining',
    start_date=datetime(2024, 1, 1),
    schedule=os.getenv('RETRAIN_SCHEDULE', '*/5 * * * *'),
    catchup=False,
    max_active_runs=1,
    default_args={'retries': 1, 'retry_delay': timedelta(seconds=30)},
    tags=['fraud'],
) as dag:
    prepare = PythonOperator(
        task_id='prepare_dataset', python_callable=prepare_dataset,
        op_kwargs={'run_id': '{{ run_id }}'},
    )
    train = PythonOperator(
        task_id='train_model', python_callable=train_model,
        op_kwargs={'run_id': '{{ run_id }}'},
    )
    publish = PythonOperator(
        task_id='publish_model', python_callable=publish_model,
        op_kwargs={'run_id': '{{ run_id }}'},
    )
    prepare >> train >> publish
