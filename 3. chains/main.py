''' 
 - Runnables can be considered as tasks
 - Prompt templates are runnables
 - Chains are sequences of runnables
 - '|' separates runnables in a chain
'''

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
    )
from langchain_core.output_parsers import StrOutputParser

load_dotenv('./../.env')

base_url = "http://localhost:11434"
model = 'llama3.2:1b'

llm = ChatOllama(
    base_url=base_url,
    model = model,
)

system = SystemMessagePromptTemplate.from_template('You are {school} teacher. You answer in short sentences.')

question = HumanMessagePromptTemplate.from_template('tell me about the {topics} in {points} points')


messages = [system, question]
template = ChatPromptTemplate(messages)

## This is without chain
# question = template.invoke({'school': 'primary', 'topics': 'solar system', 'points': 5})
# response = llm.invoke(question)
# print(response.content)    

## This is with chain
# chain = template | llm
# response = chain.invoke({'school': 'primary', 'topics': 'solar system', 'points': 5})
# print(response.content)

###########################################################################################
## chain with output parser
'''
incase the llm is not the last runnable in the chain, you can use StrOutputParser to parse the output
this will return a string of only the content of the response without the other metadata
'''
# chain = template | llm | StrOutputParser()
# response = chain.invoke({'school': 'primary', 'topics': 'solar system', 'points': 5})
# print(response)

## Chaining Runnables (Chain Multiple Runnables)
chain = template | llm | StrOutputParser()
response = chain.invoke({'school': 'primary', 'topics': 'solar system', 'points': 5})

analysis_prompt = ChatPromptTemplate.from_template('''analyze the following text: {response}
                                                   You need tell me that how difficult it is to understand.
                                                   Answer in one sentence only.
                                                   ''')

# fact_check_chain = analysis_prompt | llm | StrOutputParser()
# output = fact_check_chain.invoke({'response': response})
# print(output)


## chaining multiple chains
composed_chain = {"response": chain} | analysis_prompt | llm | StrOutputParser()
output = composed_chain.invoke({'school': 'primary', 'topics': 'solar system', 'points': 5})
# print(output)

###########################################################################################
## Parallel LCEL Chain
'''
Output for the parallel chain is a dictionary where the keys are the names of the runnables and the values are the outputs of the runnables.
You cannot use StrOutputParser() here.
'''

# chain 1
system = SystemMessagePromptTemplate.from_template('You are {school} teacher. You answer in short sentences.')

question = HumanMessagePromptTemplate.from_template('tell me about the {topics} in {points} points')


messages = [system, question]
template = ChatPromptTemplate(messages)
fact_chain = template | llm | StrOutputParser()

# output = fact_chain.invoke({'school': 'primary', 'topics': 'solar system', 'points': 2})
# print(output)

# chain 2
question = HumanMessagePromptTemplate.from_template('write a poem on {topics} in {sentences} lines')


messages = [system, question]
template = ChatPromptTemplate(messages)
poem_chain = template | llm | StrOutputParser()

# output = poem_chain.invoke({'school': 'primary', 'topics': 'solar system', 'sentences': 2})
# print(output)

from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(fact = fact_chain, poem = poem_chain)

output = chain.invoke({'school': 'primary', 'topics': 'solar system', 'points': 2, 'sentences': 2})
# print(output['fact'])
# print('\n\n')
# print(output['poem'])

###########################################################################################
## Chain Router
prompt = """Given the user review below, classify it as either being about `Positive` or `Negative`.
            Do not respond with more than one word.

            Review: {review}
            Classification:"""

template = ChatPromptTemplate.from_template(prompt)

chain = template | llm | StrOutputParser()

# review = "Thank you so much for providing such a great plateform for learning. I am really happy with the service."
review = "I am not happy with the service. It is not good."
# output = chain.invoke({'review': review})
# print(output)

positive_prompt = """
                You are expert in writing reply for positive reviews.
                You need to encourage the user to share their experience on social media.
                Review: {review}
                Answer:"""

positive_template = ChatPromptTemplate.from_template(positive_prompt)
positive_chain = positive_template | llm | StrOutputParser()

negative_prompt = """
                You are expert in writing reply for negative reviews.
                You need first to apologize for the inconvenience caused to the user.
                You need to encourage the user to share their concern on following Email:'udemy@kgptalkie.com'.
                Review: {review}
                Answer:"""


negative_template = ChatPromptTemplate.from_template(negative_prompt)
negative_chain = negative_template | llm | StrOutputParser()

def rout(info):
    if 'positive' in info['sentiment'].lower():
        return positive_chain
    else:
        return negative_chain

# print(rout({'sentiment': 'negetive'}))
'''
RunnableLambda is like passing output of one runnable as input to another runnable.
'''
from langchain_core.runnables import RunnableLambda
full_chain = {"sentiment": chain, 'review': lambda x: x['review']} | RunnableLambda(rout)
# print(full_chain.invoke({'review': review}))

###########################################################################################
## Make Custom Chain Runnables with RunnablePassthrough and RunnableLambda
'''
RunnablePassthrough just passes the output of one runnable as input to another runnable.
RunnableLambda is like passing output of one runnable as input to another runnable but it allows you to do something with the output.
'''
from langchain_core.runnables import RunnablePassthrough

def char_counts(text):
    return len(text)

def word_counts(text):
    return len(text.split())

prompt = ChatPromptTemplate.from_template("Explain these inputs in 5 sentences: {input1} and {input2}")

chain = prompt | llm | StrOutputParser() | {'char_counts': RunnableLambda(char_counts), 
                                            'word_counts': RunnableLambda(word_counts), 
                                            'output': RunnablePassthrough()}

output = chain.invoke({'input1': 'Earth is planet', 'input2': 'Sun is star'})

print(output)

###########################################################################################
## Custom Chain using `@chain` decorator

from langchain_core.runnables import chain
@chain
def custom_chain(params):
    return {
        'fact': fact_chain.invoke(params),
        'poem': poem_chain.invoke(params),
    }


params = {'school': 'primary', 'topics': 'solar system', 'points': 2, 'sentences': 2}
output = custom_chain.invoke(params)
print(output['fact'])
print('\n\n')
print(output['poem'])