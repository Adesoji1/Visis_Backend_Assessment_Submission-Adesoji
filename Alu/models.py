from llama_cpp import Llama, LlamaCache
import pdf

cache = LlamaCache()
rope_scaling = {
    "type": "dynamic",
    "factor_base": 1000.0
}


llm = Llama.from_pretrained(
    repo_id="Adesoji7/llama3.1-8B-4Bit-InstructionTuned-OIG",
    n_gpu_layers=30,# Number of layers to load on the GPU. Adjust based on your GPU memory.
    n_ctx=2028,
    filename="unsloth.Q4_K_M.gguf", 
    verbose=False,
    rope_scaling=rope_scaling,
    cache=cache
)


def generate_book_Points(book_description):
    prompt = f"""
    Analyze the book description and extract the 8 most crucial points in a clear, bullet-point format:

    Focus on:
    *   Title: The title of the book.
    *   Publisher: Who published the book.
    *   Main theme(s)
    *   Key characters or figures
    *   Central conflict or problem
    *   Setting (time period and/or location)
    *   Major plot points or events (without spoilers)
    *   Primary takeaways or lessons
    *   Target audience or genre
    *   Unique elements or writing style

    Example:
    Book Description: "In a dystopian future, Katniss Everdeen volunteers to take her younger sister's place in a deadly competition where teenagers fight to the death. This bestselling trilogy explores themes of survival, rebellion, and love."

    Summary:
    1. Title: The Hunger Games
    2. Publisher: Scholastic Press
    3. Setting: Dystopian future
    3. Main Character: Katniss Everdeen
    4. Central Conflict: Deadly competition for survival
    5. Themes: Survival, rebellion, love
    6. Major Plot Point: Katniss volunteers for her sister
    7. Target Audience: Young Adult, Dystopian Fiction
    8. Unique Element: Intense action and suspense
    9. Key Takeaway: Resilience and hope in the face of oppression

    Book Description:
    {book_description}

    Summary:
    """

    output = llm(prompt, max_tokens=400, stop=None)

    relevant_text = output['choices'][0]['text'].split("\n")

    return relevant_text


# book_description = "Published in 1999, 'Don't Make Me Think' by Steve Krug revolutionized the field of web usability. This timeless guide advocates for intuitive web design, emphasizing that users should be able to navigate and interact with websites effortlessly, without having to think deeply about how things work. Krug provides practical advice and real-world examples to help designers create user-friendly websites that prioritize clarity and ease of use."
# summary = generate_book_Points(book_description)
# print(summary)


def generate_book_Summary(book_description):
    prompt = f"""
    Summarize the following book description in a concise and informative paragraph:

    

    Book Description:
    {book_description}

    Summary:
    """

    output = llm(prompt, max_tokens=333, stop=None)
    summary = output['choices'][0]['text'].strip()
    # return output

    return summary


# book_description = "Published in 1999, 'Don't Make Me Think' by Steve Krug revolutionized the field of web usability. This timeless guide advocates for intuitive web design, emphasizing that users should be able to navigate and interact with websites effortlessly, without having to think deeply about how things work. Krug provides practical advice and real-world examples to help designers create user-friendly websites that prioritize clarity and ease of use."
# summary = generate_book_Summary(book_description)
# print(summary)
