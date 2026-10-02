CREATE TABLE IF NOT EXISTS transaction_scores (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(256) NOT NULL UNIQUE,
    score DOUBLE PRECISION NOT NULL CHECK (score >= 0 AND score <= 1),
    fraud_flag SMALLINT NOT NULL CHECK (fraud_flag IN (0, 1)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS transaction_scores_fraud_recent
    ON transaction_scores (id DESC) WHERE fraud_flag = 1;
