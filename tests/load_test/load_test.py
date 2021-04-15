from typing import List
import pandas as pd
from random import randint

FILE_PATH = '/locust/load_test/test_text.txt'

def build_messages(file_path: str):
    result = []
    with open(file_path, 'r')  as text:
        full_text = text.read().split()
    file_end = len(full_text)
    readed = 0
    while readed < file_end:
        read_message = randint(1, 20)
        readed += read_message
        if readed > file_end:
            break
        message = full_text[readed-read_message:readed]
        result.append(' '.join(message))
    return result

def build_dataframe(number_of_chats: int):
    column_names = [100000+chat for chat in range(number_of_chats)]
    messages_df = pd.DataFrame(columns=column_names)
    for chat_id in column_names:
        messages_df[chat_id]=pd.Series(build_messages(FILE_PATH))
    messages_df.fillna("просто текст", inplace=True)
    messages_df.to_csv('results.csv')
    return messages_df





