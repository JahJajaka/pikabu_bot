
import re
from os import path
import logging
from inference_models.nlp import NLP, ModelLoader
import Log
from fastapi import FastAPI, Depends
#from timeit import default_timer as timer
import toml
from db import crud, models, schemas, database
from sqlalchemy.orm import Session
models.Base.metadata.create_all(bind=database.engine)
config_file_path = path.join(path.dirname(path.abspath(__file__)), 'config.toml')
inf_config = toml.load(config_file_path)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = Log.get_logger()

model = ModelLoader().load_model(NLP(), inf_config['inference_model'] )
app = FastAPI()
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
    cutoff_from_start = 0
    for item in turns:
        cutoff_from_start += len(item)
        if cutoff_from_start >= max_result_length:
            break                               
    return history[cutoff_from_start:]


@app.post("/message")
async def answer(conv: schemas.Conversation, db_: Session = Depends(get_db)):
    logger.info(f'New message received from {conv.chat_id}')
    if len(conv.conv_text) > 200:
        conv.conv_text = conv.conv_text[:200]
    try:
        if not crud.get_chat(db=db_, chat_id=conv.chat_id):
            crud.create_chat(db=db_, conv=conv)
        history = crud.get_chat(db=db_, chat_id=conv.chat_id).conv_text
        if history and len(history) > 500:
            history = cutoff_from_start(conv.conv_text, history)
        #start = timer()
        answer, new_history = model.get_answer_ru(conv.conv_text, history)
        #end = timer()
        #logger.info(f'Get answer from model: {end - start}')
        crud.update_conversation(db=db_, chat_id=conv.chat_id, conv_text=new_history)  
        return {"text": answer if answer else "мне нечем ответить"}
    except Exception as e:
        logger.error(f"Error while processing message. Error msg: {e}")
        logger.exception(e)    