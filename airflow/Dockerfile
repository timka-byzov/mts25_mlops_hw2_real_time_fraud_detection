FROM apache/airflow:2.10.5-python3.12

USER root
RUN mkdir -p /opt/models && chown airflow:root /opt/models

USER airflow
COPY airflow/requirements.txt /requirements.txt
RUN pip install --no-cache-dir "apache-airflow==2.10.5" -r /requirements.txt
