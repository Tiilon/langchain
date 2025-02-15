from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from sqlalchemy import create_engine
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv('./../.env')

base_url = "http://localhost:11434"
model = 'llama3.2:1b'

llm = ChatOllama(
    base_url=base_url,
    model = model,
)

system = SystemMessage(content="You are helpful assistant.")
human = HumanMessagePromptTemplate.from_template("{input}")
messages = [system, MessagesPlaceholder(variable_name='history'), human]
prompt = ChatPromptTemplate(messages=messages)

chain = prompt | llm | StrOutputParser()

def get_session_history(user_session):
    engine = create_engine('sqlite:///chat_message_history.db')
    history = SQLChatMessageHistory(session_id=user_session, connection=engine)
    return history

runnable_with_history = RunnableWithMessageHistory(chain,get_session_history, input_messages_key='input', history_messages_key='history')

## for non streaming response
def chat_with_llm(session_id, input):
    output = runnable_with_history.invoke({'input': input}, config={'configurable': {'session_id': session_id}})
    return output

## for streaming response
# def chat_with_llm(session_id, input):
#     for chunk in runnable_with_history.stream(
#         {'input': input}, 
#         config={'configurable': {'session_id': session_id}}
#     ): 
#         print(chunk, end='', flush=True)
#     print()  # New line after streaming


user_session = "sparrow"

print(chat_with_llm(user_session, 'what is my profession?'))