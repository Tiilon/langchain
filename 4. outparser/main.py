from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    PromptTemplate
    )

load_dotenv('./../.env')

base_url = "http://localhost:11434"
model = 'llama3.2'

llm = ChatOllama(
    base_url=base_url,
    model = model,
)

## `Pydantinc` Output Parser
from typing import  Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class Joke(BaseModel):
    """Joke to tell user"""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")
    rating: Optional[int] = Field(description="The rating of the joke is from 1 to 10", default=None)

parser = PydanticOutputParser(pydantic_object=Joke)
# instruction = parser.get_format_instructions()
# print(instruction)

## option 1
prompt = PromptTemplate(
    template='''
    Answer the user query with a joke. Here is your formatting instruction.
    {format_instruction}

    Query: {query}
    Answer:''',
    input_variables=['query'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = prompt | llm

output = chain.invoke({'query': 'Tell me a joke about the cat'})
# print(output)

## option 2 - Parsing with `.with_structured_output()` method
prompt = PromptTemplate(
    template='''
    Answer the user query with a joke. 

    Query: {query}
    Answer:''',
    input_variables=['query'],
)

structured_llm = llm.with_structured_output(Joke)
chain = prompt | structured_llm

output = chain.invoke({'query': 'Tell me a joke about the cat'})
# print(output)

#########################################################
## `JSON` Output Parser
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser(pydantic_object=Joke)
structured_llm = llm.with_structured_output(Joke)
prompt = PromptTemplate(
    template='''
    Answer the user query with a joke. 

    Query: {query}
    Answer:''',
    input_variables=['query'],
)

chain = prompt | structured_llm
output = chain.invoke({'query': 'Tell me a joke about the cat'})
# print(output)

###############################################################
## CSV Output Parser
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

# print(parser.get_format_instructions())
format_instruction = parser.get_format_instructions()

prompt = PromptTemplate(
    template='''
    Answer the user query with a list of values. Here is your formatting instruction.
    {format_instruction}

    Query: {query}
    Answer:''',
    input_variables=['query'],
    partial_variables={'format_instruction': format_instruction}
)   

chain = prompt | llm | parser

output = chain.invoke({'query': 'generate my website seo keywords. I have content about the NLP and LLM.'})
# print(output)

###############################################################
## Datatime Output Parser
from langchain.output_parsers import DatetimeOutputParser
parser = DatetimeOutputParser()

format_instruction = parser.get_format_instructions()
print(format_instruction)

prompt = PromptTemplate(
    template='''
    Answer the user query with a datetime. Here is your formatting instruction.
    {format_instruction}

    Query: {query}
    Answer:''',
    input_variables=['query'],
    partial_variables={'format_instruction': format_instruction}
)
chain = prompt | llm | parser
output = chain.invoke({'query': 'when the America got discovered?'})
print(output)