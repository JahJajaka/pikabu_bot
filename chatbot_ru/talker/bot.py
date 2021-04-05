
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
    await message.reply("Hi!\nI'm Bender test")

@dp.message_handler(commands=['set_webhook'])
async def send_welcome(message: types.Message):
    """
    setup webhook to send new messages to api
    """
    await bot.set_webhook('http://69.6.22.117:8000/chat')
    await message.reply("new webhook is setup")

@dp.message_handler(commands=['delete_webhook'])
async def send_welcome(message: types.Message):
    """
    setup webhook to send new messages to api
    """
    await bot.delete_webhook()
    await message.reply("new webhook is deleted")

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