# # if the content field is  directly on the Book model 
# from fastapi import APIRouter, Depends, HTTPException
# from sqlalchemy.orm import Session
# from pydantic import BaseModel
# from app.database import get_db  # Ensure this line is correct
# from app import crud, schemas

# router = APIRouter()

# class BookText(BaseModel):
#     text: str

# @router.post("/generate")
# def generate_summary(book_id: int, db: Session = Depends(get_db)):
#     db_book = crud.get_book(db, book_id=book_id)
#     if db_book is None:
#         raise HTTPException(status_code=404, detail="Book not found")
    
#     summary = summarize_text(db_book.content)
#     return {"summary": summary}

# def summarize_text(text: str) -> str:
#     # Placeholder for summary generation logic
#     return "This is a summary of the book."



# from fastapi import APIRouter, Depends, HTTPException
# from sqlalchemy.orm import Session
# from pydantic import BaseModel
# from app.database import get_db
# from app import crud, schemas

# router = APIRouter()

# class BookText(BaseModel):
#     book_id: int

# @router.post("/generate")
# def generate_summary(book_text: BookText, db: Session = Depends(get_db)):
#     db_book = crud.get_book(db, book_id=book_text.book_id)
#     if db_book is None:
#         raise HTTPException(status_code=404, detail="Book not found")

#     # Aggregate content from all pages
#     full_text = " ".join(page.content for page in db_book.pages)

#     summary = summarize_text(full_text)
#     return {"summary": summary}

# def summarize_text(text: str) -> str:
#     # Placeholder for summary generation logic
#     return "This is a summary of the book."


# from fastapi import APIRouter, Depends, HTTPException, Query
# from sqlalchemy.orm import Session
# from app.database import get_db
# from app import crud, schemas

# router = APIRouter()

# @router.post("/generate")
# def generate_summary(book_id: int = Query(..., description="The ID of the book to summarize"), db: Session = Depends(get_db)):
#     db_book = crud.get_book(db, book_id=book_id)
#     if db_book is None:
#         raise HTTPException(status_code=404, detail="Book not found")

#     # Aggregate content from all pages
#     full_text = " ".join(page.content for page in db_book.pages)

#     summary = summarize_text(full_text)
#     return {"summary": summary}

# def summarize_text(text: str) -> str:
#     # Placeholder for summary generation logic
#     return "This is a summary of the book."

# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from typing import List
# from app.summarizer import summarize_text

# router = APIRouter()

# class BookText(BaseModel):
#     text: str

# @router.post("/generate")
# def generate_summary(book_text: BookText):
#     if not book_text.text:
#         raise HTTPException(status_code=400, detail="Text is required")
    
#     try:
#         summary = summarize_text(book_text.text)
#         return {"summary": summary}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.summarizer import summarize_text
from gtts import gTTS
from io import BytesIO
from starlette.responses import StreamingResponse

router = APIRouter()

class BookText(BaseModel):
    text: str

@router.post("/generate")
def generate_summary(book_text: BookText):
    if not book_text.text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        summary = summarize_text(book_text.text)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate/audio")
async def generate_audio_summary(book_text: BookText):
    if not book_text.text:
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        summary = summarize_text(book_text.text)
        tts = gTTS(text=summary, lang='en')
        audio_buffer = BytesIO()
        tts.save(audio_buffer)
        audio_buffer.seek(0)

        return StreamingResponse(audio_buffer, media_type="audio/mpeg", headers={"Content-Disposition": "inline; filename=summary.mp3"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


