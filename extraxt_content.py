# import fitz  # PyMuPDF
# import os
# from sqlalchemy.orm import Session
# from app.database import SessionLocal
# from app import crud, schemas

# # Define paths
# pdf_path = "/home/adesoji/Downloads/DO-YOU-WONDER-ABOUT-RAIN-SNOW-SLEET-AND-HAIL-Free-Childrens-Book-By-Monkey-Pen.pdf"
# output_dir = "extracted_content"
# os.makedirs(output_dir, exist_ok=True)

# def extract_text_and_images_from_pdf(pdf_path, output_dir):
#     doc = fitz.open(pdf_path)
#     book_pages = []

#     for page_num in range(len(doc)):
#         page = doc[page_num]
#         text = page.get_text()

#         # Extract images
#         image_list = page.get_images(full=True)
#         image_files = []

#         for img_index, img in enumerate(image_list):
#             xref = img[0]
#             base_image = doc.extract_image(xref)
#             image_bytes = base_image["image"]

#             image_ext = base_image["ext"]
#             image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
#             image_filepath = os.path.join(output_dir, image_filename)
            
#             with open(image_filepath, "wb") as img_file:
#                 img_file.write(image_bytes)
            
#             image_files.append(image_filepath)

#         book_pages.append({
#             "page_number": page_num + 1,
#             "content": text,
#             "image_urls": image_files
#         })

#     return book_pages

# # Extract content
# book_pages = extract_text_and_images_from_pdf(pdf_path, output_dir)

# # Define book metadata
# book_data = schemas.BookCreate(
#     title="DO YOU WONDER ABOUT RAIN, SNOW, SLEET, AND HAIL",
#     author="T. Albert",
#     publisher="Monkey Pen Ltd",
#     description="A children's book explaining the water cycle and different forms of precipitation.",
#     pages=[]
# )

# # Add extracted pages to book data
# for page in book_pages:
#     book_page = schemas.BookPage(
#         page_number=page["page_number"],
#         content=page["content"],
#         image_url=page["image_urls"][0] if page["image_urls"] else None
#     )
#     book_data.pages.append(book_page)

# # Save to database
# db: Session = SessionLocal()
# crud.create_book(db=db, book=book_data)



import fitz  # PyMuPDF
import os
from sqlalchemy.orm import Session
from app.database import SessionLocal, engine, Base
from app import crud, schemas

# Create the database tables
Base.metadata.create_all(bind=engine)

# Define paths
pdf_path = "bookpdf/DO-YOU-WONDER-ABOUT-RAIN-SNOW-SLEET-AND-HAIL-Free-Childrens-Book-By-Monkey-Pen.pdf"
output_dir = "extracted_content"
os.makedirs(output_dir, exist_ok=True)

def extract_text_and_images_from_pdf(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    book_pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Extract images
        image_list = page.get_images(full=True)
        image_files = []

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image_ext = base_image["ext"]
            image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
            image_filepath = os.path.join(output_dir, image_filename)
            
            with open(image_filepath, "wb") as img_file:
                img_file.write(image_bytes)
            
            image_files.append(image_filepath)

        book_pages.append({
            "page_number": page_num + 1,
            "content": text,
            "image_urls": image_files
        })

    return book_pages

# Extract content
book_pages = extract_text_and_images_from_pdf(pdf_path, output_dir)

# Define book metadata
book_data = schemas.BookCreate(
    title="DO YOU WONDER ABOUT RAIN, SNOW, SLEET, AND HAIL",
    author="T. Albert",
    publisher="Monkey Pen Ltd",
    description="A children's book explaining the water cycle and different forms of precipitation.",
    pages=[]
)

# Add extracted pages to book data
for page in book_pages:
    book_page = schemas.BookPage(
        page_number=page["page_number"],
        content=page["content"],
        image_url=page["image_urls"][0] if page["image_urls"] else None
    )
    book_data.pages.append(book_page)

# Save to database
db: Session = SessionLocal()
crud.create_book(db=db, book=book_data)
