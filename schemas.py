from pydantic import BaseModel, EmailStr
from typing import Literal


class InferenceData(BaseModel):
    prompt: str
    model_id: Literal["baseline", "chained", "graph"] | None = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr
    username: str
    access_token: str | None = None
    token_type: str | None = None

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    username: str
    email: str
    id: int
    token_type: str


class TokenData(BaseModel):
    email: str | None = None


class UserUpdate(BaseModel):
    email: EmailStr | None = None
    username: str | None = None
    password: str | None = None

    class Config:
        from_attributes = True


class NewChat(BaseModel):
    prompt: str


class CurrentChat(BaseModel):
    prompt: str
    chat_id: int


class FeedbackChunk(BaseModel):
    prompt: str
    code: str
    output: dict
    msgId: int

