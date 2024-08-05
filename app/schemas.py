# from pydantic import BaseModel
# from typing import List, Optional

# class BookPage(BaseModel):
#     page_number: int
#     content: str
#     image_url: Optional[str] = None

# class BookBase(BaseModel):
#     title: str
#     author: str
#     publisher: str
#     description: str
#     pages: List[BookPage]

# class BookCreate(BookBase):
#     pass

# class Book(BookBase):
#     id: int

#     class Config:
#         from_attributes = True

from typing import List, Optional
from pydantic import BaseModel, Field

class BookPageBase(BaseModel):
    page_number: int = Field(..., gt=0, description="Page number must be greater than 0")
    content: str = Field(..., min_length=1, description="Content must not be empty")
    image_url: Optional[str] = Field(None, description="Optional image URL")

class BookPageCreate(BookPageBase):
    pass

class BookPage(BookPageBase):
    id: int

    class Config:
        from_attributes = True

class BookBase(BaseModel):
    title: str = Field(..., min_length=1, description="Title must not be empty")
    author: str = Field(..., min_length=1, description="Author must not be empty")
    publisher: str = Field(..., min_length=1, description="Publisher must not be empty")
    description: str = Field(..., min_length=1, description="Description must not be empty")

class BookCreate(BookBase):
    pages: List[BookPageCreate] = Field(..., min_items=1, description="A book must have at least one page")

class Book(BookBase):
    id: int
    pages: List[BookPage] = []

    class Config:
        from_attributes = True
