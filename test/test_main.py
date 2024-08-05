from fastapi.testclient import TestClient
from app.main import app
from app.database import Base, engine, SessionLocal
from sqlalchemy.orm import sessionmaker
import pytest

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

client = TestClient(app)

@pytest.fixture(scope="module")
def test_db():
    Base.metadata.create_all(bind=engine)
    yield TestingSessionLocal()
    Base.metadata.drop_all(bind=engine)

def test_create_and_read_book(test_db):
    # Create a new book
    response = client.post("/books/", json={
        "title": "Test Book",
        "author": "Test Author",
        "publisher": "Test Publisher",
        "description": "Test Description",
        "pages": [
            {"page_number": 1, "content": "Page 1 content", "image_url": "url/to/page1/image.png"},
            {"page_number": 2, "content": "Page 2 content", "image_url": "url/to/page2/image.png"}
        ]
    })
    assert response.status_code == 200
    book_id = response.json()["id"]

    # Read the book
    response = client.get(f"/books/{book_id}")
    assert response.status_code == 200
    assert response.json()["title"] == "Test Book"

def test_generate_summary(test_db):
    # Create a new book
    response = client.post("/books/", json={
        "title": "Test Book",
        "author": "Test Author",
        "publisher": "Test Publisher",
        "description": "Test Description",
        "pages": [
            {"page_number": 1, "content": "Page 1 content", "image_url": "url/to/page1/image.png"},
            {"page_number": 2, "content": "Page 2 content", "image_url": "url/to/page2/image.png"}
        ]
    })
    assert response.status_code == 200
    book_id = response.json()["id"]

    # Generate summary for the book
    response = client.post(f"/summaries/generate?book_id={book_id}")
    assert response.status_code == 200
    assert "summary" in response.json()
