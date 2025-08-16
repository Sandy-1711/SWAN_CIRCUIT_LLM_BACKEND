from utils.llm import generate_chat_name
from database import SessionLocal
from sqlalchemy.orm import Session
from fastapi import Depends
from crud import create_chat, create_message


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_new_chat(prompt, user_id, db: Session):
    """Stream naming stages and create the chat once name is done."""
    chat_name = ""

    for chunk in generate_chat_name(prompt):
        yield chunk
        if chunk.get("stage") == "naming_done":
            chat_name = chunk["name"]

    created_chat = create_chat(db, user_id, chat_name)
    yield {
        "status": "saved",
        "chat_id": created_chat.id,
        "chat_name": created_chat.chat_name,
    }