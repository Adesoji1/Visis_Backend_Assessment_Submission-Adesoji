# import os
# import re
# import spacy
# from langdetect import detect
# from sqlalchemy.orm import Session
# from app.database import SessionLocal, engine
# from app import models
# import logging
# from tqdm import tqdm

# # Ensure the tables are created
# models.Base.metadata.create_all(bind=engine)

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Load the SpaCy model and set the max length
# nlp = spacy.load("en_core_web_sm")
# nlp.max_length = 1500000  # Increase the max length further

# def preprocess_text(text):
#     # Split the text into smaller chunks
#     chunk_size = 1000000  # Adjust chunk size as needed
#     chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
#     cleaned_text = ""
#     for chunk in chunks:
#         # Use SpaCy to process each chunk
#         doc = nlp(chunk)
#         tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
#         cleaned_text += ' '.join(tokens) + ' '
    
#     return cleaned_text.strip()

# def extract_metadata(text):
#     # Extract metadata such as author, editor, and language
#     author = "Unknown"
#     editor = None
#     language = "Unknown"
    
#     author_match = re.search(r'Author:\s*(.*)', text)
#     if author_match:
#         author = author_match.group(1).strip()
    
#     editor_match = re.search(r'Editor:\s*(.*)', text)
#     if editor_match:
#         editor = editor_match.group(1).strip()
    
#     try:
#         language = detect(text)
#     except:
#         language = "Unknown"
    
#     return author, editor, language

# def process_txts(txt_dir):
#     books = []
#     txt_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]
    
#     for txt_filename in tqdm(txt_files, desc="Processing text files"):
#         txt_path = os.path.join(txt_dir, txt_filename)
#         with open(txt_path, 'r', encoding='utf-8') as file:
#             text = file.read()
#         cleaned_text = preprocess_text(text)
#         author, editor, language = extract_metadata(text)
#         books.append({
#             "title": txt_filename.replace(".txt", ""),
#             "author": author,
#             "editor": editor,
#             "publisher": "Unknown",  # Set default or extract from text if available
#             "description": "No description available",  # Set default or extract from text if available
#             "language": language,
#             "text": cleaned_text,
#             "pages": []  # Placeholder for pages
#         })
#         logging.info(f"Processed file: {txt_filename}")
    
#     return books

# def save_books_to_db(books):
#     db: Session = SessionLocal()
#     for book in tqdm(books, desc="Saving books to database"):
#         db_book = models.Book(
#             title=book["title"],
#             author=book["author"],
#             editor=book["editor"],
#             publisher=book["publisher"],
#             description=book["description"],
#             language=book["language"],
#             text=book["text"]
#         )
#         db.add(db_book)
#         db.commit()
#         db.refresh(db_book)
#         for page in book["pages"]:
#             db_page = models.BookPage(
#                 page_number=page["page_number"],
#                 content=page["content"],
#                 image_url=page["image_url"],
#                 book_id=db_book.id
#             )
#             db.add(db_page)
#         db.commit()
#         logging.info(f"Inserted book into database: {book['title']}")
#     db.close()

# # Example usage
# if __name__ == "__main__":
#     txt_dir = "bookpdf"
#     logging.info("Starting the text processing...")
#     books = process_txts(txt_dir)
#     logging.info("Finished processing text files. Now saving to database...")
#     save_books_to_db(books)
#     logging.info("All books have been saved to the database.")





# import os
# import re
# import spacy
# from langdetect import detect
# from sqlalchemy.orm import Session
# from app.database import SessionLocal, engine
# from app import models
# import logging
# from tqdm import tqdm

# # Ensure the tables are created
# models.Base.metadata.create_all(bind=engine)

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Load the SpaCy model and set the max length
# nlp = spacy.load("en_core_web_sm")
# nlp.max_length = 1500000  # Increase the max length further

# def preprocess_text(text):
#     # Split the text into smaller chunks
#     chunk_size = 1000000  # Adjust chunk size as needed
#     chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
#     cleaned_text = ""
#     for chunk in chunks:
#         # Use SpaCy to process each chunk
#         doc = nlp(chunk)
#         tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
#         cleaned_text += ' '.join(tokens) + ' '
    
#     return cleaned_text.strip()

# def extract_metadata(text):
#     # Extract metadata such as author, editor, and language
#     author = "Unknown"
#     editor = None
#     language = "Unknown"
    
#     author_match = re.search(r'Author:\s*(.*)', text)
#     if author_match:
#         author = author_match.group(1).strip()
    
#     editor_match = re.search(r'Editor:\s*(.*)', text)
#     if editor_match:
#         editor = editor_match.group(1).strip()
    
#     try:
#         language = detect(text)
#     except:
#         language = "Unknown"
    
#     return author, editor, language

# def process_txts(txt_dir):
#     books = []
#     txt_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]
    
#     for txt_filename in tqdm(txt_files, desc="Processing text files"):
#         txt_path = os.path.join(txt_dir, txt_filename)
#         with open(txt_path, 'r', encoding='utf-8') as file:
#             text = file.read()
#         cleaned_text = preprocess_text(text)
#         author, editor, language = extract_metadata(text)
#         books.append({
#             "title": txt_filename.replace(".txt", ""),
#             "author": author,
#             "editor": editor,
#             "publisher": "Unknown",  # Set default or extract from text if available
#             "description": "No description available",  # Set default or extract from text if available
#             "language": language,
#             "text": cleaned_text,
#             "pages": []  # Placeholder for pages
#         })
#         logging.info(f"Processed file: {txt_filename}")
    
#     return books

# def save_books_to_db(books):
#     db: Session = SessionLocal()
#     for book in tqdm(books, desc="Saving books to database"):
#         db_book = models.Book(
#             title=book["title"],
#             author=book["author"],
#             editor=book["editor"],
#             publisher=book["publisher"],
#             description=book["description"],
#             language=book["language"],
#             text=book["text"]
#         )
#         db.add(db_book)
#         db.commit()
#         db.refresh(db_book)
#         for page in book["pages"]:
#             db_page = models.BookPage(
#                 page_number=page["page_number"],
#                 content=page["content"],
#                 image_url=page["image_url"],
#                 book_id=db_book.id
#             )
#             db.add(db_page)
#         db.commit()
#         logging.info(f"Inserted book into database: {book['title']}")
#     db.close()

# # Example usage
# if __name__ == "__main__":
#     txt_dir = "bookpdf"
#     logging.info("Starting the text processing...")
#     books = process_txts(txt_dir)
#     logging.info("Finished processing text files. Now saving to database...")
#     save_books_to_db(books)
#     logging.info("All books have been saved to the database.")


import os
import re
import spacy
from langdetect import detect
from sqlalchemy.orm import Session
from app.database import SessionLocal, engine
from app import models
import logging
from tqdm import tqdm

# Ensure the tables are created
models.Base.metadata.create_all(bind=engine)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the SpaCy model and set the max length
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 1500000  # Increase the max length further

def preprocess_text(text):
    # Split the text into smaller chunks
    chunk_size = 1000000  # Adjust chunk size as needed
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    cleaned_text = ""
    for chunk in chunks:
        # Use SpaCy to process each chunk
        doc = nlp(chunk)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        cleaned_text += ' '.join(tokens) + ' '
    
    return cleaned_text.strip()

def extract_metadata(text):
    # Extract metadata such as author, editor, and language
    author = "Unknown"
    editor = None
    language = "Unknown"
    
    author_match = re.search(r'Author:\s*(.*)', text)
    if author_match:
        author = author_match.group(1).strip()
    
    editor_match = re.search(r'Editor:\s*(.*)', text)
    if editor_match:
        editor = editor_match.group(1).strip()
    
    try:
        language = detect(text)
    except:
        language = "Unknown"
    
    return author, editor, language

def process_txts(txt_dir):
    books = []
    txt_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]
    
    for txt_filename in tqdm(txt_files, desc="Processing text files"):
        txt_path = os.path.join(txt_dir, txt_filename)
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        cleaned_text = preprocess_text(text)
        author, editor, language = extract_metadata(text)
        books.append({
            "title": txt_filename.replace(".txt", ""),
            "author": author,
            "editor": editor,
            "publisher": "Unknown",  # Set default or extract from text if available
            "description": "No description available",  # Set default or extract from text if available
            "language": language,
            "text": cleaned_text,
            "pages": []  # Placeholder for pages
        })
        logging.info(f"Processed file: {txt_filename}")
    
    return books

def save_books_to_db(books):
    db: Session = SessionLocal()
    for book in tqdm(books, desc="Saving books to database"):
        # Check if the book already exists in the database
        existing_book = db.query(models.Book).filter(models.Book.title == book["title"]).first()
        if existing_book:
            logging.info(f"Book already exists in database, skipping: {book['title']}")
            continue
        
        db_book = models.Book(
            title=book["title"],
            author=book["author"],
            editor=book["editor"],
            publisher=book["publisher"],
            description=book["description"],
            language=book["language"],
            text=book["text"]
        )
        db.add(db_book)
        db.commit()
        db.refresh(db_book)
        for page in book["pages"]:
            db_page = models.BookPage(
                page_number=page["page_number"],
                content=page["content"],
                image_url=page["image_url"],
                book_id=db_book.id
            )
            db.add(db_page)
        db.commit()
        logging.info(f"Inserted book into database: {book['title']}")
    db.close()

# Example usage
if __name__ == "__main__":
    txt_dir = "bookpdf"
    logging.info("Starting the text processing...")
    books = process_txts(txt_dir)
    logging.info("Finished processing text files. Now saving to database...")
    save_books_to_db(books)
    logging.info("All books have been saved to the database.")
