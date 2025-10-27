import os
from typing import List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage

from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate, PromptTemplate, FewShotPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
# from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
import requests
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain_community.query_constructors.chroma import ChromaTranslator
import pprint
from typing_extensions import Annotated

from pydantic import BaseModel
from datetime import datetime
import re

from langchain_community.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser

from data.table_union_plane import get_short_term_memory, get_long_term_memory, get_all_tasks

load_dotenv()
os.getenv("GOOGLE_API_KEY")
os.getenv("LANGSMITH_API_KEY")
X_API_KEY = os.getenv("X_API_KEY")

llm = ChatOpenAI(
  api_key=os.getenv("OPENROUTER_API_KEY"),
  base_url=os.getenv("OPENROUTER_BASE_URL"),
  model="google/gemini-2.0-flash-001",
)
model = llm

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

def build_metadata_field_info() -> List[AttributeInfo]:
    return [
        AttributeInfo(
            name="project name",
            description="The name of the project that the work item belongs to",
            type="string",
        ),
        AttributeInfo(
            name="priority",
            description="The priority of the work item. One of ['urgent', 'high', 'medium', 'low', 'none'], from highest to lowest",
            type="string",
        ),
        AttributeInfo(
            name="state",
            description="The state of the work item. It can take one of the following values:'Todo', 'In Progress', 'Done', 'Backlog', 'Cancelled'",
            type="string",
        ),
        AttributeInfo(
            name="target date",
            description="The due date of the task, in the format YYYY-MM-DD. If the task does not have a due date, the value is None",
            type="float",
        ),
    ]


def build_vector_store(docs: List[Document]) -> Chroma:
    return Chroma.from_documents(docs, embeddings)

short_term_memory = []
short_term_memory.append(get_short_term_memory())
long_term_memory = []
long_term_memory.append(get_long_term_memory())

short_term_memory_vector_store = build_vector_store(short_term_memory)
long_term_memory_vector_store = build_vector_store(long_term_memory)

def build_self_query_retriever(
    llm, vector_store: Chroma, examples: List = None
) -> SelfQueryRetriever:
    metadata_field_info = build_metadata_field_info()

    document_content_description = "Information of a work item."

    examples = [
        ("Find all tasks of project Alpha",
         {"query": "get all tasks", "filter": 'eq("project name", "Alpha")'}),
        ("Find all tasks with priority high",
         {"query": "get all tasks", "filter": 'eq("priority", "high")'}),
        ("Find all tasks with priority low and state is Todo",
         {"query": "get all tasks", "filter": 'and(eq("priority", "low"), eq("state", "Todo")'}),
        ("Tasks with target date before 2025-10-09",
        {"query": "get all tasks","filter": f'lt("start date", {datetime(2025, 10, 9).timestamp()})'}),
        ("Tasks not in state Todo",
        {"query": "get all tasks","filter": 'ne("state", "Todo")'}),
    ]

    if examples:
        # Trường hợp priority_query: có examples
        prompt = get_query_constructor_prompt(
            document_content_description, metadata_field_info, examples=examples
        )
        # print(prompt.format(query="dummy question"))
        output_parser = StructuredQueryOutputParser.from_components()
        query_constructor = prompt | llm | output_parser
        
        # print(query_constructor.invoke({"query": "priority is urgent AND state is Backlog"}))

        # pprint.pprint(query_constructor)

        return SelfQueryRetriever(
            query_constructor=query_constructor,
            vectorstore=vector_store,
            structured_query_translator=ChromaTranslator(),
            verbose=True,
            search_type="similarity_score_threshold",
            search_kwargs={"k": 40, 'score_threshold': 0.1},
        )
    else:
        # Trường hợp store_search: dùng from_llm
        return SelfQueryRetriever.from_llm(
            llm,
            vector_store,
            document_content_description,
            metadata_field_info,
            # search_type="similarity_score_threshold",
            # search_kwargs={"score_threshold": 0.5},
        )

tasks = get_all_tasks()

vector_store = build_vector_store(tasks)

def short_term_memory_search(query: str) -> Dict[str, Any]:
    """
    Search for conversation in short term memory related to query and return results.
    """
    return short_term_memory_vector_store.similarity_search(query, k=5)

def long_term_memory_search(query: str) -> Dict[str, Any]:
    """
    Search for conversation in long term memory related to query and return results.

    """
    return long_term_memory_vector_store.similarity_search(query, k=5)

def store_search(query: str) -> Dict[str, Any]:
    # , filter: Optional[Dict[str, str]] = None
    """
    Search for tasks related to query and return tasks.

    Parameters:
        query (str): User query string.

    Returns:
        Dict[str, Any]: Related tasks.
    """
    print(f"Query: {query}")
    

    retriever = build_self_query_retriever(llm, vector_store)
    

    result = retriever.invoke(query)
    # return compressed_docs
    return result

def process_date_query(query: str) -> str:
    """
    Chuyển date trong query thành timestamp description
    để LLM hiểu và generate filter đúng
    """
    # Tìm date pattern YYYY-MM-DD trong query
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    match = re.search(date_pattern, query)
    
    if match:
        date_str = match.group()
        timestamp = datetime.strptime(date_str, "%Y-%m-%d").timestamp()
        # Thêm thông tin timestamp vào query
        query += f" (timestamp: {timestamp})"
    
    return query

tools = [store_search, process_date_query, short_term_memory_search, long_term_memory_search]

llm_with_tools = model.bind_tools(tools)

today = datetime.today().strftime('%Y-%m-%d')


system = f"""
You are a helpful assistant specialized at getting tasks information. 

A task contains information including name, priority, state, comments, start date, target date (due date), completed at, description and assignees. 
A task is the highest priority task if all other tasks have lower priority. 
A missed deadline task is a task that has state Done and its target date (due date) is before completed at (date) OR 
a task that has state 'In Progress' and its target date is before {today}. 

If the user ask for information related to past conversation, find related information in {short_term_memory} and {long_term_memory}.

Use information from provided tool to generate answers to the USER INPUT. Only provide answer with info related to the USER INPUT

"""

#Carefully consider the query to create a tool call. Priority levels from highest to lowest are: urgent, high, medium, low, none. 
"""Use the following format to find your answers: 
Question: The input query you must answer. 
Thought: You should always think about what to do, do not use any tool if it is not needed. 
IMPORTANT: All dates in query need to be converted to timestamp using process_date_query before calling store_search. Don't use store_search with queries you already used.
Action: Make the tool call to find the answer. 
Observation: The RELEVANT INFORMATION to the inquiry. The number of result you receive doesn't matter.
Repeat Question-thought-action-output until you are sure of the correct response. 

IMPORTANT: Only repeat after the previous chains of thought finishes. Don't create parallel chains of thought.
"""

few_shot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        # *examples,
        ("human", "{query}"),
    ]
)


chain = {"query": RunnablePassthrough()} | few_shot_prompt | llm_with_tools



class State(MessagesState):

    """Main graph state."""

    messages: Annotated[list[AnyMessage], add_messages]
    """The messages in the conversation."""


# node definitions
def assistant(state: State):
    print(chain.invoke(state["messages"]))
    return {"messages": [chain.invoke(state["messages"])]}


# build the graph
task_graph = StateGraph(State)

task_graph.add_node("assistant", assistant)
task_graph.add_node("tools", ToolNode(tools))

# task_graph.add_node("get_task",get_task)

task_graph.add_edge(START, "assistant")
task_graph.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
task_graph.add_edge("tools", "assistant")

memory = InMemorySaver()

compiled_graph = task_graph.compile(
    )



def stream_graph_updates(user_input: str):
    config = {"configurable": {"thread_id": "1"}}
    for event in compiled_graph.stream({"messages": [{"role": "user", "content": user_input}]}, config,
        stream_mode="values",
    ):
        for value in event.values():
            print("Assistant:")
            print(value[-1].content)

    # for chunk in compiled_graph.stream(
    #     {"messages": [{"role": "user", "content": user_input}]},
    #     stream_mode="updates"
    # ):
    #     print(chunk)
    #     print("\n")


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break


