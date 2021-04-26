
import logging
from inference_models.nlp import NLP, ModelLoader
from database import Database
import Log
import json
from fastapi import FastAPI, Request
#from timeit import default_timer as timer
import toml
inf_config = toml.load('config.toml')
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = Log.get_logger()

model = ModelLoader().load_model(NLP(), inf_config['inference_model'] )
app = FastAPI()
logger.info(f"Application started")
@app.post("/message")
async def answer(request: Request):
    request_data = json.loads(await request.body())
    chat_id = request_data['chat_id']
    message = request_data['message']
    logger.info(f'New message received from {chat_id}')
    try:
        Database.create_new_chat_if_needed(chat_id)
        history = Database.get_messages(chat_id)
        if history and len(history) > 500:
            Database.update_conversation(chat_id, None) 
            return {"text": "Так о чем это мы? Давай начнем сначала. Напиши что-нибудь."}
        #start = timer()
        answer, new_history = model.get_answer_ru(message, history)
        #end = timer()
        #logger.info(f'Get answer from model: {end - start}')
        Database.update_conversation(chat_id, new_history)  
        return {"text": answer if answer else "мне нечем ответить"}
    except Exception as e:
        logger.error(f"Error while processing message. Error msg: {e}")
        logger.exception(e)    