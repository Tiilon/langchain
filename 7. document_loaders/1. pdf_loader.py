from langchain_community.document_loaders import PyMuPDFLoader
import os

from dotenv import load_dotenv
load_dotenv('./../.env')


## Loading one pdf
# loader = PyMuPDFLoader(
#     "7. document_loaders/rag-dataset/health supplements/1. dietary supplements - for whom.pdf"
# )

# loaded_docs = loader.load()

# # get metadata
# print(loaded_docs[0].metadata)
# print(loaded_docs[0].page_content)

################################################################

## Loading multiple pdfs
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


## checking the length of the context in tokens
'''
Using tiktoken to check the length of the context in tokens
Make sure the token is always less than the accepted limit by the model
'''
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o-mini")
# print(len(encoding.encode("congratulations")))
# print(len(encoding.encode("rweuoe")))

# print(len(encoding.encode(context)))

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

system = SystemMessagePromptTemplate.from_template("""You are helpful AI assistant who answer user question based on the provided context. 
                                                    Do not answer in more than {words} words""")

prompt = """Answer user question based on the provided context ONLY! If you do not know the answer, just say "I don't know".
            ### Context:
            {context}

            ### Question:
            {question}

            ### Answer:
        """

prompt = HumanMessagePromptTemplate.from_template(prompt)

messages = [system, prompt]
template = ChatPromptTemplate(messages)

# print(template)

# template_result = template.invoke({'context': context, 'question': "How to gain muscle mass?", 'words': 50})
# print(template_result)

qna_chain = template | llm | StrOutputParser()
# response = qna_chain.invoke({'context': context, 'question': "How to gain muscle mass?", 'words': 50})
# print(response)

response = qna_chain.invoke({'context': context, 'question': "How to reduce the weight?", 'words': 50})
print(response)