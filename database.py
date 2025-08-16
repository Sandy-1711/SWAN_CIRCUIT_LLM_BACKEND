from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import text
import os
from dotenv import load_dotenv

load_dotenv()
SQLALCHEMY_DATABASE_URL = os.getenv("DB_URL")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
# def drop_all_tables():
#     with engine.connect() as conn:
#         conn.execute(text("DROP SCHEMA public CASCADE"))
#         conn.execute(text("CREATE SCHEMA public"))
#         conn.commit()

# drop_all_tables()
# print("dropped tables")