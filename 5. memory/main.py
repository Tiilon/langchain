from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from sqlalchemy import create_engine
from langchain_core.output_parsers import StrOutputParser
load_dotenv('./../.env')

base_url = "http://localhost:11434"
model = 'llama3.2'

llm = ChatOllama(
    base_url=base_url,
    model = model,
)

system = SystemMessagePromptTemplate.from_template("You are helpful assistant.")
human = HumanMessagePromptTemplate.from_template("{input}")

messages = [system, MessagesPlaceholder(variable_name='history'), human]

prompt = ChatPromptTemplate(messages=messages)

chain = prompt | llm | StrOutputParser()

def get_session_history(session_id):
    engine = create_engine("sqlite:///chat_history.db")
    return SQLChatMessageHistory(
        session_id=session_id, 
        connection=engine  # Use 'connection' instead of 'connection_string'
    )


runnable_with_history = RunnableWithMessageHistory(chain, get_session_history, 
                                                   input_messages_key='input', 
                                                   history_messages_key='history')

def chat_with_llm(session_id, input):
    output = runnable_with_history.invoke(
        {'input': input},
        config={'configurable': {'session_id': session_id}}
    )

    return output

user_id = "kgptalkie"
about = "My name is Laxmi Kant. I work for KGP Talkie."

response1 = chat_with_llm(user_id, about)
print(response1)
print('########################################')
response2 = chat_with_llm(user_id, "what is my name?")
print(response2)