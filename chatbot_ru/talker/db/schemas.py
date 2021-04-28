from datetime import datetime
from pydantic import BaseModel



class Conversation(BaseModel):
    conv_text: str
    chat_id: int