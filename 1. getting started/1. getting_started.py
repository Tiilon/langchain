from dotenv import load_dotenv
from langchain_ollama import ChatOllama
load_dotenv('./../.env')

base_url = "http://localhost:11434"
model = 'llama3.2:1b'

llm = ChatOllama(
    base_url=base_url,
    model = model,
    temperature = 0.8,
    num_predict = 256
)

# response = llm.invoke('can you be a chatbot?')
# print(response)

response = ""
for chunk in llm.stream('can you be a chatbot?'):
    response += chunk.content

print(response)
