from langchain_community.document_loaders import PyMuPDFLoader
import os

from dotenv import load_dotenv
load_dotenv('./../.env')

import os

pdfs = []
for root, dirs, files in os.walk("rag-dataset"):
    # print(root, dirs, files)
    for file in files:
        if file.endswith(".pdf"):
            pdfs.append(os.path.join(root, file))


docs = []
for pdf in pdfs:
    loader = PyMuPDFLoader(pdf)
    temp_docs = loader.load()
    docs.extend(temp_docs)

# print(len(docs))

# combine all docs into one document
def format_docs(docs):
    return "\n\n".join([x.page_content for x in docs])


context = format_docs(docs)


## Question Answering using LLM
from langchain_ollama import ChatOllama
from langchain_core.prompts import (SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate)
from langchain_core.output_parsers import StrOutputParser

base_url = "http://localhost:11434"
model = 'llama3.2:1b'

llm = ChatOllama(
    base_url=base_url,
    model = model,
)

system = SystemMessagePromptTemplate.from_template("""You are helpful AI assistant who works as document summarizer. 
                                                   You must not hallucinate or provide any false information.""")

prompt = """Summarize the given context in {words}.
            ### Context:
            {context}

            ### Summary:"""

prompt = HumanMessagePromptTemplate.from_template(prompt)

messages = [system, prompt]
template = ChatPromptTemplate(messages)

summary_chain = template | llm | StrOutputParser()

response = summary_chain.invoke({'context': context, 'words': 200})
print(response)