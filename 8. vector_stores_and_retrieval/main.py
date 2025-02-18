from langchain_community.document_loaders import PyPDFDirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
import os

from dotenv import load_dotenv
import os

from ollama import embeddings

load_dotenv("./../.env")

# pdfs = []
# for root, dirs, files in os.walk("rag-dataset"):
#     # print(root, dirs, files)
#     for file in files:
#         if file.endswith(".pdf"):
#             pdfs.append(os.path.join(root, file))

# docs = []
# for pdf in pdfs:
#     loader = PyMuPDFLoader(pdf)
#     temp_docs = loader.load()
#     docs.extend(temp_docs)


directory_path = "rag-dataset/"
loader = PyPDFDirectoryLoader(directory_path)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
)
chunks = text_splitter.split_documents(docs)

# print(len(chunks))

## checking tokens
# import tiktoken

# encoding = tiktoken.encoding_for_model("gpt-4o-mini")
# print(len(encoding.encode(chunks[0].page_content)), len(
#     encoding.encode(chunks[1].page_content)
# ), len(encoding.encode(docs[1].page_content)))

embeddings = OllamaEmbeddings(
    model="nomic-embed-text", 
    base_url="http://localhost:11434"
)

vector_store = Chroma(
    # collection_name="health_supplements",
    embedding_function=embeddings,
    persist_directory="./health_supplements",
)

## putting chunks(docs) in vector store
vector_store.add_documents(chunks)

## retrieval
question = "how to gain muscle mass?"
docs = vector_store.similarity_search(question, k=2)
print(docs)
