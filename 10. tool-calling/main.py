import warnings
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

import os

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")


### USING BUILT IN TOOLS
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

response = search.invoke("What is today's stock market news?")
# print(response)
######################################################
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


question = "what is the capital of France?"
question = "What is LLM?"


# print(wikipedia.invoke(question))
###########################################################
@tool
def wikipedia_search(query):
    """
    Search wikipedia for general information.

    Args:
    query: The search query
    """

    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    response = wikipedia.invoke(query)
    return response

from langchain_community.tools.pubmed.tool import PubmedQueryRun

## USING CUSTOM TOOLS
@tool
def pubmed_search(query):
    """
    Search pubmed for medical and life sciences queries.

    Args:
    query: The search query
    """

    search = PubmedQueryRun()
    response = search.invoke(query)
    return response

from langchain_community.tools import TavilySearchResults
@tool
def tavily_search(query):
    """
    Search the web for realtime and latest information.
    for examples, news, stock market, weather updates etc.

    Args:
    query: The search query
    """

    search = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
    )
    response = search.invoke(query)
    return response


@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two integer numbers together

    Args:
    a: First Integer
    b: Second Integer
    """
    return int(a) * int(b)


@tool
def add(a: int, b: int) -> int:
    """
    Add two integer numbers together

    Args:
    a: First Integer
    b: Second Integer
    """
    return int(a) + int(b)


# print(add.name, add.description, add.args, add.args_schema.schema())

# print(add.invoke({"a": 1, "b": 2}))
# print(multiply.invoke({"a": 1, "b": 2}))


tools = [wikipedia_search, pubmed_search, tavily_search, multiply]
list_of_tools = {tool.name: tool for tool in tools}

llm_with_tools = llm.bind_tools(tools)
# query = "What is the latest news"
query = "What is today's stock market news?"
# query = "What is LLM?"
# query = "How to treat lung cancer?"
# query = "what is 2 * 3?"
# query = "What is medicine for lung cancer?"
response = llm_with_tools.invoke(query)

from langchain_core.messages import HumanMessage

messages = [HumanMessage(query)]

ai_msg = llm_with_tools.invoke(messages)

messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:

    name = tool_call["name"].lower()

    selected_tool = list_of_tools[name]

    tool_msg = selected_tool.invoke(tool_call)

    messages.append(tool_msg)

print(messages)

response = llm_with_tools.invoke(messages)
print(response.content)
