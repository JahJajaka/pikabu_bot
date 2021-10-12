
import logging
import time
import numpy as np
import threading
from queue import Empty, Queue
from inference_models.nlp import NLP, ModelLoader
import Log
from fastapi import FastAPI, Depends
from timeit import default_timer as timer
import toml
from db import crud, models, schemas, database
from sqlalchemy.orm import Session
models.Base.metadata.create_all(bind=database.engine)
inf_config = toml.load('config.toml')
#inf_config = toml.load('/workspace/talker/config.toml')
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = Log.get_logger()

BATCH_SIZE = 3
BATCH_TIMEOUT = 0.5
CHECK_INTERVAL = 0.01


model = ModelLoader().load_model(NLP(), inf_config['inference_model'] )
app = FastAPI()
requests_queue = Queue()
logger.info(f"Application started")




def get_db():
    db_ = database.SessionLocal()
    try:
        yield db_
    finally:
        db_.close()

def cutoff_from_start(message: str, history: str) -> str:
    turns = [f'|0|{item}' for item in history.split('|0|')[1:]]
    max_result_length = 11+len(message)+inf_config['num_tokens_to_produce']
    cutoff_symbols = 0
    for item in turns:
        cutoff_symbols += len(item)
        if cutoff_symbols >= max_result_length:
            break
    return history[cutoff_symbols:]

@app.post("/message")
def answer(conv: schemas.Conversation, db_: Session = Depends(get_db)):
    logger.info(f'New message received from {conv.chat_id}')
    if len(conv.conv_text) > 200:
        conv.conv_text = conv.conv_text[:200]
    try:
        if not crud.get_chat(db=db_, chat_id=conv.chat_id):
            crud.create_chat(db=db_, conv=conv)
        history = crud.get_chat(db=db_, chat_id=conv.chat_id).conv_text
        if history and len(history) > 500:
            history = cutoff_from_start(conv.conv_text, history)
        request = {'input': conv.conv_text, 'history': history, 'time': time.time()}
        requests_queue.put(request)
        while 'output' not in request:
            time.sleep(CHECK_INTERVAL)
        crud.update_conversation(db=db_, chat_id=conv.chat_id, conv_text=request['new_history'])  
        return {'text': request['output']} 
    except Exception as e:
        logger.error(f"Error while processing message. Error msg: {e}")
        logger.exception(e) 

def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (
            len(requests_batch) > BATCH_SIZE or
            (len(requests_batch) > 0 and time.time() - requests_batch[0]['time'] > BATCH_TIMEOUT)):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
                if len(requests_batch) >= BATCH_SIZE:
                    break
            except Empty:
                continue
        logger.info(f'Batch size: {len(requests_batch)}')
        batch_inputs = [request['input'] for request in requests_batch]
        batch_history = [request['history'] for request in requests_batch]
        start = timer()
        batch_outputs, batch_new_history = model.get_answer_ru(batch_inputs, batch_history)
        end = timer()
        logger.info(f'Get answer from model: {end - start}')
        for request, output, new_history in zip(requests_batch, batch_outputs, batch_new_history):
            request['output'] = output
            request['new_history'] = new_history

threading.Thread(target=handle_requests_by_batch).start()