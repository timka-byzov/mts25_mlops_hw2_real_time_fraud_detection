# Real-Time Fraud Detection System

DISCLAIMER

Сервис подготовлен в демонстрационных целях для студентов курса МТС ШАД 2025 в рамках занятий по MLOps. Датасеты предоставлены в рамках соревнования https://www.kaggle.com/competitions/teta-ml-1-2025

Система для обнаружения мошеннических транзакций в реальном времени с использованием ML-модели и Kafka для потоковой обработки данных.

## 🏗️ Архитектура

Компоненты системы:
1. **`interface`** (Streamlit UI):
   
   Создан для удобной симуляции потоковых данных с транзакциями. Реальный продукт использовал бы прямой поток данных из других систем.
    - Имитирует отправку транзакций в Kafka через CSV-файлы.
    - Генерирует уникальные ID для транзакций.
    - Загружает транзакции отдельными сообщениями формата JSON в топик kafka `transactions`.
    

2. **`fraud_detector`** (ML Service):
   - Загружает предобученную модель CatBoost (`my_catboost.cbm`).
   - Выполняет препроцессинг данных:
     - Извлечение временных признаков
     - Гео-расстояния
     - Кодирование категориальных переменных
   - Производит скоринг с порогом 0.98.
   - Выгружает результат скоринга в топик kafka `scores`.
   - Применяет сохранённые статистики препроцессинга; обучение и загрузка train.csv при запуске не нужны.

3. **Kafka Infrastructure**:
   - Zookeeper + Kafka брокер
   - `kafka-setup`: автоматически создает топики `transactions` и `scores`
   - Kafka UI: веб-интерфейс для мониторинга сообщений (порт 8080)

4. **`score_writer`**:
   - Читает `scores` и сохраняет три поля результата в PostgreSQL.
   - Повторный `transaction_id` не создаёт дубликат.

5. **PostgreSQL**:
   - Таблица `transaction_scores` создаётся автоматически.
   - Данные сохраняются в Docker volume.

## 🚀 Быстрый старт

### Требования
- Docker 20.10+
- Docker Compose 2.0+

### Запуск
```bash
git clone https://github.com/timka-byzov/mts25_mlops_hw2_real_time_fraud_detection.git mlops-fraud-detection
cd mlops-fraud-detection

# Сборка и запуск всех сервисов
docker compose up --build
```
На Mac с установленным Podman вместо `docker compose` используйте `podman compose`.

После запуска:
- **Streamlit UI**: http://localhost:8501
- **Kafka UI**: http://localhost:8080
- **Логи сервисов**: 
  ```bash
  docker compose logs <service_name>  # Например: fraud_detector, kafka, interface
  ```

## 🛠️ Использование

### 1. Загрузка данных:

 - Загрузите CSV через интерфейс Streamlit. Для тестирования работы проекта используется файл формата `test.csv` из соревнования https://www.kaggle.com/competitions/teta-ml-1-2025
 - Пример структуры данных:
    ```csv
    transaction_time,amount,lat,lon,merchant_lat,merchant_lon,gender,...
    2023-01-01 12:30:00,150.50,40.7128,-74.0060,40.7580,-73.9855,M,...
    ```
 - Для первых тестов рекомендуется загружать небольшой семпл данных (до 100 транзакций) за раз, чтобы исполнение кода не заняло много времени.

### 2. Мониторинг:
 - **Kafka UI**: Просматривайте сообщения в топиках transactions и scores
 - **Логи обработки**: `docker compose logs fraud_detector score_writer`

### 3. Результаты:

 - Для проверки без скачивания датасета нажмите «Отправить пример» или загрузите `interface/examples/transactions.csv`.
 - В разделе «Результаты скоринга» нажмите «Посмотреть результаты»: появятся 10 последних фродовых записей и гистограмма скоров последних 100 транзакций.
 - Для обновления данных повторно нажмите кнопку. При пустой базе интерфейс показывает соответствующее сообщение.


 - Скоринговые оценки пишутся в топик scores в формате:
    ```json
    {
    "score": 0.995, 
    "fraud_flag": 1, 
    "transaction_id": "d6b0f7a0-8e1a-4a3c-9b2d-5c8f9d1e2f3a"
    }
    ```
## Структура проекта
```
.
├── fraud_detector/
│   ├── src/preprocessing.py
│   ├── src/preprocessor.py
│   ├── src/scorer.py
│   ├── app/app.py
│   └── Dockerfile
├── interface/
│   └── app.py              # Streamlit UI
├── score_writer/
├── postgres/init.sql
├── docker-compose.yml
└── README.md
```

## Настройки Kafka
```yml
Топики:
- transactions (входные данные)
- scores (результаты скоринга)

Репликация: 1 (для разработки)
Партиции: 3
```

*Примечание:* 

Для полной функциональности убедитесь, что:
1. Модель `my_catboost.cbm` размещена в `fraud_detector/models/`
2. Артефакт `preprocessing.json` размещён в `fraud_detector/models/` (включён в репозиторий)
3. Порты 8080, 8501, 9095 и 5433 свободны на хосте


Для автоматической проверки работающего проекта:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
python scripts/smoke_test.py
```

Для остановки с сохранением данных: `docker compose down`.
Для изменения порога задайте `FRAUD_THRESHOLD` перед запуском (по умолчанию `0.98`).

Повторная подготовка статистик из своего `train.csv`, если потребуется:

```bash
python scripts/prepare_preprocessing.py --train /path/to/train.csv
```
