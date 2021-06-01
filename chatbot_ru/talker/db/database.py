from os import getenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

db_user = getenv('DB_USER')
db_pass = getenv('DB_PW')
db_name = getenv('DB_NAME')
db_port = getenv('DB_PORT')
db_host = getenv('DB_HOST')
SQLALCHEMY_DATABASE_URL = f'postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'
engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()