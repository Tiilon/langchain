import warnings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.vectorstores import Chroma
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.prompts import ChatPromptTemplate

import os

from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text", base_url="http://localhost:11434"
)

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./health_supplements",
)

### Retrieval
retriever = vector_store.as_retriever(search_kwargs={"k": 5},
                          search_type="similarity")

# docs = retriever.invoke("how to gain muscle mass?")
# print(docs)

prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
    Question: {question} 
    Context: {context} 
    Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt)

llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")

def format_docs(docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# question = "how to lose weight?"
# output = rag_chain.invoke(question)
# print(output)


question = "how to gain muscle mass?"
response = rag_chain.invoke(question)
print(response)
