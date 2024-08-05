# from typing import List
# from fastapi import APIRouter, Depends, HTTPException
# from sqlalchemy.orm import Session
# from app import crud, schemas, models
# from app.database import get_db  # Ensure this line is cor

# router = APIRouter()

# @router.post("/", response_model=schemas.Book)
# def create_book(book: schemas.BookCreate, db: Session = Depends(get_db)):
#     return crud.create_book(db=db, book=book)

# @router.get("/{book_id}", response_model=schemas.Book)
# def read_book(book_id: int, db: Session = Depends(get_db)):
#     db_book = crud.get_book(db, book_id=book_id)
#     if db_book is None:
#         raise HTTPException(status_code=404, detail="Book not found")
#     return db_book



# # @router.get("/{book_id}/pages/{page_number}", response_model=schemas.BookPage)
# # def read_book_page(book_id: int, page_number: int, db: Session = Depends(get_db)):
# #     db_page = db.query(models.BookPage).filter(
# #         models.BookPage.book_id == book_id,
# #         models.BookPage.page_number == page_number
# #     ).first()
# #     if db_page is None:
# #         raise HTTPException(status_code=404, detail="Page not found")
# #     return db_page

# # @router.get("/", response_model=List[schemas.Book])
# # def read_books(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
# #     books = db.query(models.Book).offset(skip).limit(limit).all()
# #     return books




# @router.get("/{book_id}/pages/{page_number}", response_model=schemas.BookPage)
# def read_book_page(book_id: int, page_number: int, db: Session = Depends(get_db)):
#     db_page = crud.get_page(db, book_id=book_id, page_number=page_number)
#     if db_page is None:
#         raise HTTPException(status_code=404, detail="Page not found")
#     return db_page


# from fastapi import APIRouter, Depends, HTTPException
# from sqlalchemy.orm import Session
# from typing import List
# from app import crud, schemas, models
# from app.database import get_db

# router = APIRouter()

# @router.post("/", response_model=schemas.Book)
# def create_book(book: schemas.BookCreate, db: Session = Depends(get_db)):
#     return crud.create_book(db=db, book=book)

# @router.get("/{book_id}", response_model=schemas.Book)
# def read_book(book_id: int, db: Session = Depends(get_db)):
#     db_book = crud.get_book(db, book_id=book_id)
#     if db_book is None:
#         raise HTTPException(status_code=404, detail="Book not found")
#     return db_book

# @router.get("/{book_id}/pages/{page_number}", response_model=schemas.BookPage)
# def read_book_page(book_id: int, page_number: int, db: Session = Depends(get_db)):
#     db_page = crud.get_page(db, book_id=book_id, page_number=page_number)
#     if db_page is None:
#         raise HTTPException(status_code=404, detail="Page not found")
#     return db_page

# @router.put("/{book_id}", response_model=schemas.Book)
# def update_book(book_id: int, book: schemas.BookCreate, db: Session = Depends(get_db)):
#     db_book = crud.update_book(db=db, book_id=book_id, book_update=book)
#     if db_book is None:
#         raise HTTPException(status_code=404, detail="Book not found")
#     return db_book

# @router.delete("/{book_id}", response_model=schemas.Book)
# def delete_book(book_id: int, db: Session = Depends(get_db)):
#     db_book = crud.delete_book(db=db, book_id=book_id)
#     if db_book is None:
#         raise HTTPException(status_code=404, detail="Book not found")
#     return db_book


from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app import crud, schemas, models
from app.database import get_db

router = APIRouter()

@router.post("/", response_model=schemas.Book, status_code=status.HTTP_201_CREATED)
def create_book(book: schemas.BookCreate, db: Session = Depends(get_db)):
    try:
        return crud.create_book(db=db, book=book)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/{book_id}", response_model=schemas.Book)
def read_book(book_id: int, db: Session = Depends(get_db)):
    db_book = crud.get_book(db, book_id=book_id)
    if db_book is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    return db_book

@router.get("/{book_id}/pages/{page_number}", response_model=schemas.BookPage)
def read_book_page(book_id: int, page_number: int, db: Session = Depends(get_db)):
    db_page = crud.get_page(db, book_id=book_id, page_number=page_number)
    if db_page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found")
    return db_page

@router.put("/{book_id}", response_model=schemas.Book)
def update_book(book_id: int, book: schemas.BookCreate, db: Session = Depends(get_db)):
    db_book = crud.update_book(db=db, book_id=book_id, book_update=book)
    if db_book is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    return db_book

@router.delete("/{book_id}", response_model=schemas.Book)
def delete_book(book_id: int, db: Session = Depends(get_db)):
    db_book = crud.delete_book(db=db, book_id=book_id)
    if db_book is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    return db_book

