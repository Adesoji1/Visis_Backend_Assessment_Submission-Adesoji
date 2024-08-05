import os
import requests
from bs4 import BeautifulSoup

# Base URL for constructing full download links
base_url = "https://www.gutenberg.org"

# URL of the page to scrape
url = "https://www.gutenberg.org/browse/authors/c"

# Path to save the downloaded plain text files
save_path = "/home/adesoji/Downloads/visis-backend-assessment-Adesoji/bookpdf"

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Send a GET request to the URL
response = requests.get(url)
response.raise_for_status()  # Check that the request was successful

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find all 'a' tags with href attribute that matches the book links
book_links = soup.select('li.pgdbetext a[href*="/ebooks/"]')

# Loop through each book link to find and download "Plain Text UTF-8" file
for book_link in book_links:
    book_url = base_url + book_link['href']

    # Request the book page
    book_response = requests.get(book_url)
    book_response.raise_for_status()

    # Parse the book page
    book_soup = BeautifulSoup(book_response.content, 'html.parser')

    # Find the "Plain Text UTF-8" download link
    download_link = book_soup.select_one('a[href*=".txt.utf-8"]')
    if download_link:
        file_url = base_url + download_link['href']
        
        # Get the file name from the URL and change the extension to .txt
        file_name = os.path.basename(file_url).replace('.txt.utf-8', '.txt')
        save_file_path = os.path.join(save_path, file_name)

        # Download the file
        file_response = requests.get(file_url)
        file_response.raise_for_status()

        # Save the file to the specified path
        with open(save_file_path, 'wb') as file:
            file.write(file_response.content)

        print(f"Downloaded and saved: {save_file_path}")

print("All files have been downloaded and saved successfully.")


# # #######Loop through links with retry mechanism
# import os
# import requests
# from bs4 import BeautifulSoup
# from tqdm import tqdm
# from tenacity import retry, wait_fixed, stop_after_attempt

# # Base URL for constructing full download links
# base_url = "https://www.gutenberg.org"

# # List of URLs to scrape from authors A to F
# author_urls = [
#     f"https://www.gutenberg.org/browse/authors/{letter}" for letter in "acdef"
# ]

# # Path to save the downloaded plain text files
# save_path = "/home/adesoji/Downloads/visis-backend-assessment-Adesoji/bookpdf"

# # Create the directory if it doesn't exist
# os.makedirs(save_path, exist_ok=True)

# @retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
# def download_file(file_url, save_file_path):
#     response = requests.get(file_url)
#     response.raise_for_status()
#     with open(save_file_path, 'wb') as file:
#         file.write(response.content)

# for url in author_urls:
#     # Send a GET request to the URL
#     response = requests.get(url)
#     response.raise_for_status()  # Check that the request was successful

#     # Parse the HTML content using BeautifulSoup
#     soup = BeautifulSoup(response.content, 'html.parser')

#     # Find all 'a' tags with href attribute that matches the book links
#     book_links = soup.select('li.pgdbetext a[href*="/ebooks/"]')

#     # Loop through each book link to find and download "Plain Text UTF-8" file
#     for book_link in tqdm(book_links, desc=f"Processing {url}"):
#         book_url = base_url + book_link['href']

#         # Request the book page
#         book_response = requests.get(book_url)
#         book_response.raise_for_status()

#         # Parse the book page
#         book_soup = BeautifulSoup(book_response.content, 'html.parser')

#         # Find the "Plain Text UTF-8" download link
#         download_link = book_soup.select_one('a[href*=".txt.utf-8"]')
#         if download_link:
#             file_url = base_url + download_link['href']
            
#             # Get the file name from the URL and change the extension to .txt
#             file_name = os.path.basename(file_url).replace('.txt.utf-8', '.txt')
#             save_file_path = os.path.join(save_path, file_name)

#             # Download the file with retry mechanism
#             try:
#                 download_file(file_url, save_file_path)
#                 print(f"Downloaded and saved: {save_file_path}")
#             except Exception as e:
#                 print(f"Failed to download {file_url}: {e}")

# print("All files have been downloaded and saved successfully.")


