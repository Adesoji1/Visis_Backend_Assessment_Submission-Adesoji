# import torch
# from transformers import pipeline

# # Initialize the summarization pipeline with GPU support if available
# device = 0 if torch.cuda.is_available() else -1
# summarizer = pipeline("summarization", device=device)

# def summarize_text(text: str, max_length: int = 130, min_length: int = 30) -> str:
#     """
#     Summarizes the given text.
    
#     :param text: The text to summarize.
#     :param max_length: The maximum length of the summary.
#     :param min_length: The minimum length of the summary.
#     :return: The generated summary.
#     """
#     summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
#     return summary[0]['summary_text']


# import torch
# from transformers import pipeline

# # Initialize the summarization pipeline with GPU support if available
# device = 0 if torch.cuda.is_available() else -1
# summarizer = pipeline("summarization", device=device)

# def summarize_text(text: str, max_length: int = 130, min_length: int = 30) -> str:
#     """
#     Summarizes the given text.
    
#     :param text: The text to summarize.
#     :param max_length: The maximum length of the summary.
#     :param min_length: The minimum length of the summary.
#     :return: The generated summary.
#     """
#     # Adjust max_length based on the length of the input text
#     input_length = len(text.split())
#     adjusted_max_length = min(max_length, input_length // 2)

#     summary = summarizer(text, max_length=adjusted_max_length, min_length=min_length, do_sample=False)
#     return summary[0]['summary_text']


# import torch
# from transformers import pipeline

# # Initialize the summarization pipeline with GPU support if available
# device = 0 if torch.cuda.is_available() else -1
# summarizer = pipeline("summarization", device=device)

# def summarize_text(text: str, max_length: int = 130, min_length: int = 30) -> str:
#     """
#     Summarizes the given text.
    
#     :param text: The text to summarize.
#     :param max_length: The maximum length of the summary.
#     :param min_length: The minimum length of the summary.
#     :return: The generated summary.
#     """
#     # Adjust max_length based on the length of the input text
#     input_length = len(text.split())
#     adjusted_max_length = min(max_length, input_length // 2)
#     adjusted_min_length = min(min_length, adjusted_max_length // 2)
    
#     summary = summarizer(text, max_length=adjusted_max_length, min_length=adjusted_min_length, do_sample=False)
#     return summary[0]['summary_text']


import torch
from transformers import pipeline

# Model name and revision to use
# model_name = "sshleifer/distilbart-xsum-12-1"
# revision="e85cfe1"
model_name = "sshleifer/distilbart-cnn-12-6"
revision = "a4f8f3e"


# Initialize the summarization pipeline with GPU support if available
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model=model_name, revision=revision, device=device)

def summarize_text(text: str, max_length: int = 130, min_length: int = 30) -> str:
    """
    Summarizes the given text.
    
    :param text: The text to summarize.
    :param max_length: The maximum length of the summary.
    :param min_length: The minimum length of the summary.
    :return: The generated summary.
    """
    # Adjust max_length based on the length of the input text
    input_length = len(text.split())
    adjusted_max_length = min(max_length, input_length // 2)
    adjusted_min_length = min(min_length, adjusted_max_length // 2)
    
    summary = summarizer(text, max_length=adjusted_max_length, min_length=adjusted_min_length, do_sample=False)
    return summary[0]['summary_text']



