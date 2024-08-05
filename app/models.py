# from sqlalchemy import Column, Integer, String, Text, ForeignKey
# from sqlalchemy.orm import relationship
# from app.database import Base

# class Book(Base):
#     __tablename__ = "books"
#     id = Column(Integer, primary_key=True, index=True)
#     title = Column(String, index=True)
#     author = Column(String)
#     publisher = Column(String)
#     description = Column(Text)
#     pages = relationship("BookPage", back_populates="book")

# class BookPage(Base):
#     __tablename__ = "book_pages"
#     id = Column(Integer, primary_key=True, index=True)
#     page_number = Column(Integer)
#     content = Column(Text)
#     image_url = Column(String, nullable=True)
#     book_id = Column(Integer, ForeignKey("books.id"))
#     book = relationship("Book", back_populates="pages")


from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from app.database import Base

class Book(Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    author = Column(String, index=True)
    publisher = Column(String, index=True)
    description = Column(String, index=True)
    
    pages = relationship("BookPage", back_populates="book")

class BookPage(Base):
    __tablename__ = "book_pages"

    id = Column(Integer, primary_key=True, index=True)
    page_number = Column(Integer, index=True)
    content = Column(String, index=True)
    image_url = Column(String, index=True, nullable=True)
    book_id = Column(Integer, ForeignKey("books.id"))

    book = relationship("Book", back_populates="pages")
