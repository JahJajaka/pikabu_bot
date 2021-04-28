
from sqlalchemy import  Column, Integer, String, BigInteger, TIMESTAMP
import datetime
from .database import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    conv_text = Column(String(1000))
    chat_id = Column(BigInteger, unique=True)
    updated_at = Column(TIMESTAMP, nullable=False, default=datetime.datetime.utcnow())