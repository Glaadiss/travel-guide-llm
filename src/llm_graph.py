import operator
import os
from typing import Annotated, Sequence, TypedDict

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
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


search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
)


tools = [retriever_tool, search_tool]


# Initialize LLM and tools
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(
    content="""
    You are a helpful assistant tasked with answering questions about travel in Vietnam. 
    Return your answer in markdown format. 
    Return sources in a separate section (including page numbers for retriever tool). 
    If information is not available in the retriever tool, try to find it in the tavily_search_results_json.
    """
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


def run_llm(prompt: str, chat_history: list):
    # Convert chat history to BaseMessage objects
    messages = [
        (
            HumanMessage(content=msg["content"])
            if msg["role"] == "user"
            else SystemMessage(content=msg["content"]) if msg["role"] == "system" else AIMessage(content=msg["content"])
        )
        for msg in chat_history
    ]

    # Add the new prompt as a HumanMessage
    messages.append(HumanMessage(content=prompt))

    # Invoke the app with the messages
    result = app.invoke({"messages": messages})

    # Extract the last message from the result
    last_message = result["messages"][-1]

    return {"output": last_message.content, "intermediate_steps": result.get("intermediate_steps", [])}


if __name__ == "__main__":
    question = "What are the best things to do Hanoi and are there any events happening at the moment?"
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    print(result)
