import asyncio
import nest_asyncio
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv

# Apply nest_asyncio to enable async in Jupyter/IPython environments
nest_asyncio.apply()

load_dotenv('./../.env')

urls = [
    'https://economictimes.indiatimes.com/markets/stocks/news',
    'https://www.livemint.com/latest-news',
    'https://www.livemint.com/latest-news/page-2',
    'https://www.livemint.com/latest-news/page-3',
    'https://www.moneycontrol.com/'
]
loader = WebBaseLoader(urls)

pages = []
for doc in loader.lazy_load():
    pages.append(doc)

# print(pages[0].page_content[:100])
# print(pages[0].metadata)

def format_docs(docs):
    return "\n\n".join([x.page_content for x in docs])

context = format_docs(pages)

# print(context)

import re

def text_clean(text):
    text = re.sub(r'\n\n+', '\n\n', text)
    text = re.sub(r'\t+', '\t', text)
    text = re.sub(r'\s+', ' ', text)
    return text

context = text_clean(context)

# print(context)

from scripts import llm

# response = llm.ask_llm(context, "What is todays news?")
# print(response)

# response = llm.ask_llm(context, "Extract stock market news from the given text.")
# print(response)

# response = llm.ask_llm(context[:10_000], "Extract stock market news from the given text.")
# print(response)

def chunk_text(text, chunk_size, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

chunks = chunk_text(context, 10_000)
# print(chunks)

question = "Extract stock market news from the given text."

chunk_summary = []
for chunk in chunks:
    response = llm.ask_llm(chunk, question)
    chunk_summary.append(response)

# for chunk in chunk_summary:
#     print(chunk)
#     print("\n\n")
#     break

summary = "\n\n".join(chunk_summary)
print(summary)

# question = "Write a detailed report in Markdown from the given context."
question = """Write a detailed market news report in markdown format. Think carefully then write the report."""
response = llm.ask_llm(summary, question)

import os

os.makedirs("data", exist_ok=True)

with open("data/report.md", "w") as f:
    f.write(response)

with open("data/summary.md", "w") as f:
    f.write(summary)