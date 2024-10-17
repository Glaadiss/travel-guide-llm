import os
from typing import Any, Dict, List

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from prompts import REACT_PROMPT

INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)

    def retrieve_with_metadata(query: str) -> str:
        docs = docsearch.similarity_search(query)
        results = []
        for doc in docs:
            results.append(f"Content: {doc.page_content}\nPage: {int(doc.metadata['page'])}\n")
        return "\n".join(results)

    # Create custom retriever tool
    retriever_tool = Tool(
        name="retriever tool",
        func=retrieve_with_metadata,
        description="Useful for answering questions about travel in Vietnam. Returns content and metadata.",
    )

    search_tool = TavilySearchResults(
        name="search tool",
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
    )

    # Create list of tools
    tools = [retriever_tool, search_tool]

    react_prompt_template = PromptTemplate(
        template=REACT_PROMPT, input_variables=["input", "agent_scratchpad", "chat_history", "tools", "tool_names"]
    )
    agent = create_react_agent(chat, tools, react_prompt_template)

    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, handle_parsing_errors=True
    )

    # Format chat history for the prompt
    formatted_chat_history = "\n".join([f"{turn['role']}: {turn['content']}" for turn in chat_history])

    # Run the agent
    result = agent_executor.invoke(
        {
            "input": query,
            "chat_history": formatted_chat_history,
            "tools": tools,
            "tool_names": ", ".join([tool.name for tool in tools]),
        }
    )

    # Post-process the result to ensure sources are included
    final_answer = result["output"]
    if "intermediate_steps" in result:
        sources = set()
        for step in result["intermediate_steps"]:
            if "Observation" in step[1]:
                if step[0].tool == INDEX_NAME and "Page:" in step[1]:
                    page = step[1].split("Page:")[1].strip().split("\n")[0]
                    sources.add(f"Page {page}")
                elif step[0].tool == "Search" and "http" in step[1]:
                    url = step[1].split("http")[1].split()[0]
                    sources.add(f"https{url}")

        if sources:
            source_text = "\n\nSources:\n" + "\n".join(sources)
            final_answer += source_text

    return {"output": final_answer, "intermediate_steps": result.get("intermediate_steps", [])}
