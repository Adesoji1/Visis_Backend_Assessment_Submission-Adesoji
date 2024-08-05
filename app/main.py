from fastapi import FastAPI
from app.api.endpoints import books, summaries
from app.database import engine, Base

app = FastAPI()

# Create the database tables
Base.metadata.create_all(bind=engine)

app.include_router(books.router, prefix="/books", tags=["books"])
app.include_router(summaries.router, prefix="/summaries", tags=["summaries"])


