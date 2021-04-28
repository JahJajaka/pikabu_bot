
import logging
from inference_models.nlp import NLP, ModelLoader
import Log
from fastapi import FastAPI, Depends
#from timeit import default_timer as timer
import toml
from db import crud, models, schemas, database
from sqlalchemy.orm import Session
models.Base.metadata.create_all(bind=database.engine)
inf_config = toml.load('config.toml')
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


@app.post("/message")
async def answer(conv: schemas.Conversation, db_: Session = Depends(get_db)):
    logger.info(f'New message received from {conv.chat_id}')
    try:
        if not crud.get_chat(db=db_, chat_id=conv.chat_id):
            crud.create_chat(db=db_, conv=conv)
        history = crud.get_chat(db=db_, chat_id=conv.chat_id).conv_text
        if history and len(history) > 500:
            crud.update_conversation(db=db_, chat_id=conv.chat_id, conv_text=None) 
            return {"text": "Так о чем это мы? Давай начнем сначала. Напиши что-нибудь."}
        #start = timer()
        answer, new_history = model.get_answer_ru(conv.conv_text, history)
        #end = timer()
        #logger.info(f'Get answer from model: {end - start}')
        crud.update_conversation(db=db_, chat_id=conv.chat_id, conv_text=new_history)  
        return {"text": answer if answer else "мне нечем ответить"}
    except Exception as e:
        logger.error(f"Error while processing message. Error msg: {e}")
        logger.exception(e)    