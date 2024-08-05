# initialize_db.py
from sqlalchemy.orm import Session
from app.database import SessionLocal, engine
from app import models

def init_db():
    models.Base.metadata.create_all(bind=engine)
    print("Database tables created.")

if __name__ == "__main__":
    init_db()
