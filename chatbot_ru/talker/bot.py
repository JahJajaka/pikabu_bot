
from os import getenv
import logging
from chat import NLP
from aiogram import Bot, Dispatcher, executor, types
from database import Database
import Log

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = Log.get_logger()

# Initialize bot and dispatcher
bot = Bot(token=getenv('BOT_API_TOKEN'))
dp = Dispatcher(bot)
nlp = NLP()


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends `/start` or `/help` command
    """
    await message.reply("привет!\nЯ русскоговорящий бот тренированный на Pikabu.\nПочитай вот тут, если хочешь узнать подробности:\nhttps://habr.com/en/company/icl_services/blog/548244/")


@dp.channel_post_handler()
@dp.message_handler()
async def answer(message: types.Message):
    logger.info(f'New message received from {message.chat.id}')
    Database.create_new_chat_if_needed(message.chat.id)
    history = Database.get_messages(message.chat.id)
    if history and len(history) > 500:
        Database.update_conversation(message.chat.id, None) 
        await message.answer("Так о чем это мы? Давай начнем сначала. Напиши что-нибудь.")
        return
    answer, new_history = nlp.get_answer_ru(message.text, history)
    Database.update_conversation(message.chat.id, new_history)  
    await message.answer(answer)


if __name__ == '__main__':
    logger.info('Bot started successfuly')
    executor.start_polling(dp, skip_updates=True)