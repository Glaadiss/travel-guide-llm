import operator
import os
from typing import Annotated, Sequence, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


# Define the state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# Initialize tools
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)


def retrieve_with_metadata(query: str) -> str:
    docs = docsearch.similarity_search(query)
    results = []
    for doc in docs:
        results.append(f"Content: {doc.page_content}\nPage: {int(doc.metadata['page'])}\n")
    return "\n".join(results)


retriever_tool = Tool(
    name=INDEX_NAME,
    func=retrieve_with_metadata,
    description="Useful for answering questions about travel in Vietnam. Returns content and metadata.",
)

search_tool = Tool(
    name="Search",
    func=DuckDuckGoSearchRun().run,
    description="Useful for searching the internet for current information.",
)

tools = [retriever_tool, search_tool]


# Initialize LLM and tools
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with answering questions about travel in Vietnam. Returns content and metadata."
)


def reasoner(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Create and compile the graph
builder = StateGraph(MessagesState)

builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", tools_condition)
builder.add_edge("tools", "reasoner")

app = builder.compile()

if __name__ == "__main__":
    question = "What are the best things to do Hanoi and are there any events happening at the moment?"
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    print(result)
