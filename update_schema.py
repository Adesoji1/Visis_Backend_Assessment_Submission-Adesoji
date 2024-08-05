from sqlalchemy import create_engine, MetaData, Table, text

# Create an engine and get the metadata
engine = create_engine("sqlite:///test.db")
metadata = MetaData()

# Reflect the books table
books = Table('books', metadata, autoload_with=engine)

# Add missing columns if they do not exist
with engine.connect() as connection:
    if 'author' not in books.c:
        connection.execute(text('ALTER TABLE books ADD COLUMN author TEXT'))
    if 'editor' not in books.c:
        connection.execute(text('ALTER TABLE books ADD COLUMN editor TEXT'))
    if 'publisher' not in books.c:
        connection.execute(text('ALTER TABLE books ADD COLUMN publisher TEXT'))
    if 'description' not in books.c:
        connection.execute(text('ALTER TABLE books ADD COLUMN description TEXT'))
    if 'language' not in books.c:
        connection.execute(text('ALTER TABLE books ADD COLUMN language TEXT'))

print("Database schema updated.")
