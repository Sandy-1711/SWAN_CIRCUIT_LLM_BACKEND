from fastapi import APIRouter, HTTPException, status

# import services.auth_service as auth_service
from schemas import UserOut, UserCreate, Token, UserLogin, TokenData
from database import SessionLocal
from sqlalchemy.orm import Session
from fastapi import Depends, Response
from crud import get_user_by_email, create_user
from utils.auth import verify_password, decode_token, create_access_token
from dependencies import get_current_user

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(response: Response, user: UserCreate, db: Session = Depends(get_db)):
    if get_user_by_email(db, user.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )
    new_user = create_user(db, user)
    token = create_access_token(data={"sub": user.email})
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=False,  # Set to True in production (HTTPS)
        samesite="None",
        max_age=1800,  # 30 minutes
        domain="localhost",  # ✅ Add this explicitly
    )
    return UserOut(
        id=new_user.id,
        email=new_user.email,
        username=new_user.username,
        access_token=token,
        token_type="bearer",
    )


@router.post("/login", response_model=Token)
async def login(response: Response, user: UserLogin, db: Session = Depends(get_db)):
    user_fetched = get_user_by_email(db, user.email)
    if not user_fetched or not verify_password(
        user.password, user_fetched.hashed_password
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email or password"
        )

    token = create_access_token(data={"sub": user.email})
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=False,  # Set to True in production (HTTPS)
        samesite="None",
        max_age=1800,  # 30 minutes
        domain="localhost",  # ✅ Add this explicitly
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "email": user_fetched.email,
        "username": user_fetched.username,
        "id": user_fetched.id,
    }


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"message": "Logged out successfully"}


@router.get("/auth")
async def auth_endpoint():
    return {"message": "Authentication endpoint"}


@router.get("/health")
async def health_endpoint():
    return {"message": "Health check endpoint"}
