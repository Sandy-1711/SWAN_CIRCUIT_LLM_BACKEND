from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session
from database import SessionLocal
from crud import get_user_by_email
from utils.auth import decode_token
from schemas import TokenData

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = None

    if "access_token" in request.cookies:
        token = request.cookies["access_token"]

    # Optional fallback to Authorization header
    if not token and "authorization" in request.headers:
        auth_header = request.headers["authorization"]
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = decode_token(token)
    if payload is None or "sub" not in payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = get_user_by_email(db, email=payload["sub"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user
