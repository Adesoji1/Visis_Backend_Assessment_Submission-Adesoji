# from sqlalchemy.orm import Session
# from app import models, schemas

# def get_book(db: Session, book_id: int):
#     return db.query(models.Book).filter(models.Book.id == book_id).first()

# def create_book(db: Session, book: schemas.BookCreate):
#     db_book = models.Book(
#         title=book.title,
#         author=book.author,
#         publisher=book.publisher,
#         description=book.description,
#     )
#     db.add(db_book)
#     db.commit()
#     db.refresh(db_book)
#     for page in book.pages:
#         db_page = models.BookPage(
#             page_number=page.page_number,
#             content=page.content,
#             image_url=page.image_url,
#             book_id=db_book.id,
#         )
#         db.add(db_page)
#     db.commit()
#     return db_book


# def get_page(db: Session, book_id: int, page_number: int):
#     return db.query(models.BookPage).filter(
#         models.BookPage.book_id == book_id,
#         models.BookPage.page_number == page_number
#     ).first()

# from sqlalchemy.orm import Session
# from app import models, schemas

# def get_book(db: Session, book_id: int):
#     return db.query(models.Book).filter(models.Book.id == book_id).first()

# def create_book(db: Session, book: schemas.BookCreate):
#     db_book = models.Book(
#         title=book.title,
#         author=book.author,
#         publisher=book.publisher,
#         description=book.description,
#     )
#     db.add(db_book)
#     db.commit()
#     db.refresh(db_book)
#     print(f"Created book with ID: {db_book.id}")
#     for page in book.pages:
#         db_page = models.BookPage(
#             page_number=page.page_number,
#             content=page.content,
#             image_url=page.image_url,
#             book_id=db_book.id,
#         )
#         db.add(db_page)
#         print(f"Added page number {page.page_number} with book_id {db_book.id}")
#     db.commit()
#     return db_book

# def get_page(db: Session, book_id: int, page_number: int):
#     print(f"Fetching page {page_number} for book_id {book_id}")
#     page = db.query(models.BookPage).filter(
#         models.BookPage.book_id == book_id,
#         models.BookPage.page_number == page_number
#     ).first()
#     if page:
#         print(f"Found page: {page.page_number}, Content: {page.content}")
#     else:
#         print("Page not found")
#     return page

from sqlalchemy.orm import Session
from app import models, schemas

def get_book(db: Session, book_id: int):
    return db.query(models.Book).filter(models.Book.id == book_id).first()

def create_book(db: Session, book: schemas.BookCreate):
    db_book = models.Book(
        title=book.title,
        author=book.author,
        publisher=book.publisher,
        description=book.description,
    )
    db.add(db_book)
    db.commit()
    db.refresh(db_book)
    print(f"Created book with ID: {db_book.id}")
    for page in book.pages:
        db_page = models.BookPage(
            page_number=page.page_number,
            content=page.content,
            image_url=page.image_url,
            book_id=db_book.id,
        )
        db.add(db_page)
        print(f"Added page number {page.page_number} with book_id {db_book.id}")
    db.commit()
    return db_book

def update_book(db: Session, book_id: int, book_update: schemas.BookCreate):
    db_book = get_book(db, book_id)
    if db_book is None:
        return None
    
    db_book.title = book_update.title
    db_book.author = book_update.author
    db_book.publisher = book_update.publisher
    db_book.description = book_update.description
    
    # Clear existing pages
    db.query(models.BookPage).filter(models.BookPage.book_id == book_id).delete()
    
    # Add new pages
    for page in book_update.pages:
        db_page = models.BookPage(
            page_number=page.page_number,
            content=page.content,
            image_url=page.image_url,
            book_id=db_book.id,
        )
        db.add(db_page)
        print(f"Added page number {page.page_number} with book_id {db_book.id}")
    
    db.commit()
    return db_book

def delete_book(db: Session, book_id: int):
    db_book = get_book(db, book_id)
    if db_book:
        db.query(models.BookPage).filter(models.BookPage.book_id == book_id).delete()
        db.delete(db_book)
        db.commit()
    return db_book

def get_page(db: Session, book_id: int, page_number: int):
    print(f"Fetching page {page_number} for book_id {book_id}")
    page = db.query(models.BookPage).filter(
        models.BookPage.book_id == book_id,
        models.BookPage.page_number == page_number
    ).first()
    if page:
        print(f"Found page: {page.page_number}, Content: {page.content}")
    else:
        print("Page not found")
    return page
