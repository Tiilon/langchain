import warnings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.tools import TavilySearchResults

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text", base_url="http://localhost:11434"
)

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./health_supplements",
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5},
                          search_type="similarity")


@tool
def health_supplements(query: str) -> str:
    """Search for information about Health Supplements.
    For any questions about Health and Gym Supplements, you must use this tool!,

    Args:
        query: The search query.
    """
    response = retriever.invoke(query)
    return response


@tool
def search(query: str) -> str:
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

## Creating Agent
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

prompt = hub.pull("hwchase17/openai-functions-agent")

# print(prompt.messages)

tools = [search, health_supplements]
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# question = "What is the best supplement for muscle gain?"
# question = "what's weather in New York?"
# question = "What are the side effects of taking too much vitamin D?"
question = "what is the capital of France?"
response = agent_executor.invoke({"input": question})
print(response['output'])