from sqlalchemy.orm import Session, joinedload
from model import User, Message, Chat
from utils.auth import hash_password

from model import RoleEnum
from typing import Optional


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, user_data):
    db_user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        username=user_data.username,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def update_user(db: Session, user_id: int, user_data):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        return None

    if user_data.email:
        db_user.email = user_data.email
    if user_data.username:
        db_user.username = user_data.username
    if user_data.password:
        db_user.hashed_password = hash_password(user_data.password)

    db.commit()
    db.refresh(db_user)
    return db_user


def delete_user(db: Session, user_id: int):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        return None

    db.delete(db_user)
    db.commit()
    return db_user


def create_chat(db: Session, user_id: int, chat_name: str):
    db_chat = Chat(user_id=user_id, chat_name=chat_name)
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return db_chat


def create_message(
    db: Session,
    user_id: int,
    chat_id: int,
    message: str,
    role: RoleEnum,
    model_id: Optional[str] = None,
    prompt: Optional[str] = None,
    code: Optional[str] = None,
    output: Optional[dict] = None,
):
    db_message = Message(
        model_id=model_id,
        user_id=user_id,
        chat_id=chat_id,
        prompt=prompt,
        code=code,
        output=output,
        message=message,
        role=role,
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message


def get_chat_by_id(db: Session, chat_id: int, user_id: int):
    return (
        db.query(Chat)
        .filter(Chat.id == chat_id, Chat.user_id == user_id)
        .options(joinedload(Chat.messages))
        .first()
    )


def update_message(
    db: Session,
    message_id: int,
    prompt: str,
    code: str,
    output: dict,
):
    db_message = db.query(Message).filter(Message.id == message_id).first()
    if not db_message:
        return None

    db_message.code = code
    db_message.output = output
    db_message.prompt = prompt
    db_message.updated = True
    db.commit()
    db.refresh(db_message)
    return db_message


def get_all_chats_by_userid(db: Session, user_id: int):
    return (
        db.query(Chat)
        .filter(Chat.user_id == user_id)
        .options(joinedload(Chat.messages))
        .order_by(Chat.created_at.desc())
        .all()
    )
