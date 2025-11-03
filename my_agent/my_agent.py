import os
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate, PromptTemplate, FewShotPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
# from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from dotenv import load_dotenv

import pprint
from typing_extensions import Annotated

from datetime import datetime

from data.table_union_plane import get_short_term_memory, get_long_term_memory
from tools.tools import store_search, process_date_query

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

short_term_memory = get_short_term_memory()
long_term_memory = get_long_term_memory()

tools = [store_search, process_date_query]

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


