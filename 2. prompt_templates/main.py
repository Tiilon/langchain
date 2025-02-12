from dotenv import load_dotenv
from langchain_ollama import ChatOllama
load_dotenv('./../.env')

base_url = "http://localhost:11434"
model = 'llama3.2:1b'

llm = ChatOllama(
    base_url=base_url,
    model = model,
)

question = "tell me about earth in 3 points?"
# response = llm.invoke(question)
# print(response.content)


### LANGCHAIN MESSAGES ###
'''
System message allows you to make llms assume roles or act as characters.
Human message is what the user inputs.
AI message is what the llm generates.
'''
from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

system = SystemMessage(content="You are a phd teacher. You answer in short sentences.")
question = HumanMessage(content="tell me about earth in 3 points?")
messages = [system, question]
# response = llm.invoke(messages)
# print(response.content)

### LANGCHAIN PROMPT TEMPLATES ###
'''
Prompt Templates allow you to format the messages sent to the llm.You are able to parse variables to the messages for flexibility
'''
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    ChatPromptTemplate
)

system = SystemMessagePromptTemplate.from_template("You are a {school} teacher. You answer in short sentences.")
question = HumanMessagePromptTemplate.from_template("tell me about {topic} in {points} points?")

## .format replaces {variable} with the values
# system.format(school='elemetary')
# question.format(topics='sun', points=5)

messages = [system, question]
template = ChatPromptTemplate(messages)


## This is how it is flexible. You can change the values of the variables.
#option 1
final_question_1 = template.invoke({
    "school": "elementary",
    "topic": "sun",
    "points": 3
})

#option 2
final_question_2 = template.invoke({
    "school": "phd",
    "topic": "stars",
    "points": 5
})

response = llm.invoke(final_question_1)
response2 = llm.invoke(final_question_2)
print(response.content)
print(response2.content)