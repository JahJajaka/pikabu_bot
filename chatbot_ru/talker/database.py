from os import getenv
import psycopg2
import logging
import datetime

class Database:

    def db_connector(func):
        def with_connection_(*args,**kwargs):
            cnn = psycopg2.connect(host=getenv('DB_HOST'), port = getenv('DB_PORT'), database=getenv('DB_NAME'), user=getenv('DB_USER'), password=getenv('DB_PW'))
            try:
                rv = func(cnn, *args,**kwargs)
            except Exception:
                cnn.rollback()
                logging.error("Database connection error")
                raise
            else:
                cnn.commit()
            finally:
                cnn.close()
            return rv
        return with_connection_

    @db_connector
    def create_new_chat_if_needed(cnn, chat_id: int):
        cur = cnn.cursor()
        cur.execute(f"""SELECT chat_id FROM conversations WHERE chat_id={chat_id} ORDER BY id ASC""")
        if not cur.fetchone():
            cur.execute(f"""INSERT INTO conversations (chat_id, conv_text, updated_at) VALUES (%s,%s,%s)""", (chat_id,"", datetime.datetime.utcnow()))

    @db_connector
    def get_messages(cnn, chat_id: int):
        cur = cnn.cursor()
        cur.execute(f"""SELECT conv_text FROM conversations WHERE chat_id={chat_id} ORDER BY id ASC""")
        query_result = [r[0] for r in cur.fetchall()][0]
        if query_result:
            return query_result



    @db_connector
    def update_conversation(cnn, chat_id: int, answer: str):
        cur = cnn.cursor()
        cur.execute(f""" UPDATE conversations SET conv_text = %s, updated_at = %s WHERE chat_id = %s""", (answer, datetime.datetime.utcnow(), chat_id))      