from sqlalchemy.orm import Session
import datetime
from . import models, schemas


def create_chat(db: Session, conv: schemas.Conversation):
    db_chat = models.Conversation(chat_id = conv.chat_id)
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return db_chat

def get_chat(db: Session, chat_id: int):
    return db.query(models.Conversation).filter(models.Conversation.chat_id == chat_id).first()

def update_conversation(db: Session, chat_id: int, conv_text: str):
    db_chat = get_chat(db=db, chat_id=chat_id)
    db_chat.conv_text = conv_text
    db_chat.updated_at = datetime.datetime.utcnow()
    #setattr(user, 'no_of_logins', user.no_of_logins+1)
    db.commit()
