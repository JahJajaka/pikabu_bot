CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    conv_text varchar(1000),
    chat_id INTEGER,
    updated_at TIMESTAMP
);