import streamlit as st
import pandas as pd
from kafka import KafkaProducer
import json
import time
import os
import uuid
from pathlib import Path

import plotly.express as px
from results import fetch_results

# Конфигурация Kafka
KAFKA_CONFIG = {
    "bootstrap_servers": os.getenv("KAFKA_BROKERS", "kafka:9092"),
    "topic": os.getenv("KAFKA_TOPIC", "transactions")
}

def load_file(uploaded_file):
    """Загрузка CSV файла в DataFrame"""
    try:
        frame = pd.read_csv(uploaded_file)
        required = set(pd.read_csv(Path(__file__).parent / 'examples/transactions.csv', nrows=0).columns)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError('Отсутствуют поля: ' + ', '.join(sorted(missing)))
        if frame.empty:
            raise ValueError('CSV не содержит транзакций')
        return frame
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {str(e)}")
        return None

def send_to_kafka(df, topic, bootstrap_servers):
    """Отправка данных в Kafka с уникальным ID транзакции"""
    producer = None
    try:
        df = df.copy().reset_index(drop=True)
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, allow_nan=False).encode("utf-8"),
            security_protocol="PLAINTEXT",
            acks="all",
            retries=3
        )
        
        # Генерация уникальных ID для всех транзакций
        df['transaction_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        
        progress_bar = st.progress(0)
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            # Отправляем данные вместе с ID
            producer.send(
                topic, 
                value={
                    "transaction_id": row['transaction_id'],
                    "data": row.drop('transaction_id').astype(object).where(pd.notna(row.drop('transaction_id')), None).to_dict()
                }
            ).get(timeout=30)
            progress_bar.progress((idx + 1) / total_rows)
            
        producer.flush()
     
        return True
    except Exception as e:
        st.error(f"Ошибка отправки данных: {str(e)}")
        return False
    finally:
        if producer is not None:
            producer.close(timeout=5)

# Инициализация состояния
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

# Интерфейс
st.title("📤 Отправка данных в Kafka")

# Блок загрузки файлов
uploaded_file = st.file_uploader(
    "Загрузите CSV файл с транзакциями",
    type=["csv"]
)

if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
    # Добавляем файл в состояние
    st.session_state.uploaded_files[uploaded_file.name] = {
        "status": "Загружен",
        "df": load_file(uploaded_file)
    }
    if st.session_state.uploaded_files[uploaded_file.name]["df"] is not None:
        st.success(f"Файл {uploaded_file.name} успешно загружен!")

# Список загруженных файлов
if st.session_state.uploaded_files:
    st.subheader("🗂 Список загруженных файлов")
    
    for file_name, file_data in st.session_state.uploaded_files.items():
        cols = st.columns([4, 2, 2])
        
        with cols[0]:
            st.markdown(f"**Файл:** `{file_name}`")
            st.markdown(f"**Статус:** `{file_data['status']}`")
        
        with cols[2]:
            if st.button(f"Отправить {file_name}", key=f"send_{file_name}"):
                if file_data["df"] is not None:
                    with st.spinner("Отправка..."):
                        success = send_to_kafka(
                            file_data["df"],
                            KAFKA_CONFIG["topic"],
                            KAFKA_CONFIG["bootstrap_servers"]
                        )
                        if success:
                            st.session_state.uploaded_files[file_name]["status"] = "Отправлен"
                            st.rerun()
                else:
                    st.error("Файл не содержит данных")

st.divider()
example_path = Path(__file__).parent / 'examples/transactions.csv'
st.download_button('Скачать пример CSV', example_path.read_bytes(), 'transactions.csv', 'text/csv')
if st.button('Отправить пример'):
    if send_to_kafka(pd.read_csv(example_path), KAFKA_CONFIG['topic'], KAFKA_CONFIG['bootstrap_servers']):
        st.success('Пример отправлен в Kafka')

st.divider()
st.header('Результаты скоринга')
if st.button('Посмотреть результаты'):
    try:
        st.session_state['scoring_results'] = fetch_results()
    except Exception as error:
        st.error(f'Не удалось получить результаты: {error}')

if 'scoring_results' in st.session_state:
    results = st.session_state['scoring_results']
    st.metric('Транзакций в базе', results['total'])
    st.subheader('10 последних фродовых транзакций')
    if results['fraudulent'].empty:
        st.info('Фродовые транзакции пока не найдены')
    else:
        st.dataframe(results['fraudulent'], hide_index=True, use_container_width=True)
    st.subheader('Распределение скоров последних 100 транзакций')
    if results['recent_scores'].empty:
        st.info('Результатов скоринга пока нет')
    else:
        figure = px.histogram(
            results['recent_scores'], x='score', nbins=20,
            range_x=[0, 1], labels={'score': 'Скор модели'},
        )
        figure.update_layout(yaxis_title='Количество транзакций', bargap=0.05)
        st.plotly_chart(figure, use_container_width=True)
        st.caption(f"В распределении {len(results['recent_scores'])} транзакций")
