
from os import getenv
import logging
from aiogram import Bot, Dispatcher, executor, types
import Log
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = Log.get_logger()

# Initialize bot and dispatcher
bot = Bot(token=getenv('BOT_API_TOKEN'))
dp = Dispatcher(bot)

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends `/start` or `/help` command
    """
    await message.reply("привет!\nЯ русскоговорящий бот тренированный на Pikabu.\nПочитай вот тут, если хочешь узнать подробности:\nhttps://habr.com/en/company/icl_services/blog/548244/")


@dp.channel_post_handler()
@dp.message_handler()
async def answer(message: types.Message):
    headers = {"Accept": "application/json",
                'Content-Type': 'application/json'}
    payload = {
    "message": f"{message.text}",
    "chat_id": f"{message.chat.id}"       
    }
    response = requests.post(f'{getenv("TALKER_HOST")}:{getenv("TALKER_PORT")}{getenv("TALKER_ENDPOINT")}', headers=headers,data=json.dumps(payload)).json()                   
    await message.answer(response['text']) 

if __name__ == '__main__':
    logger.info('Bot started successfuly')
    try:
        executor.start_polling(dp, skip_updates=True)
    except Exception as e:
        logger.error(f"Error while getting updates. Error msg: {e}")
        logger.exception(e)
