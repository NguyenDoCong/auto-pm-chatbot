import os
from typing import List, TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class TaskState(TypedDict):
    # Query
    query: Optional[str]
    # Result
    result: Optional[List[object]]
    messages: Annotated[list[AnyMessage], add_messages]

# mock data
tasks = [
    {
        "id": 1,
        "name": "Task 1",
        "description": "This is a sample task for demonstration purposes.",
        "status": "incomplete"
    },
    {
        "id": 2,
        "name": "Task 2",   
        "description": "This is another sample task for demonstration purposes.",
        "status": "incomplete"
    }
]

#tool definition
def get_all_tasks():
    """Get all tasks."""
    # for t in task:
    #     print(f"Task Name: {t['name']}")
    #     print(f"Description: {t['description']}")
    return {"result":tasks}

def get_task(name):
    """Get task by name."""
    for t in tasks:
        if t["name"] == name:
            return {"result": t}
    return {"error": "Task not found"}

def update_status(name):
    """Update task status to complete by task name."""
    for t in tasks:
        if t["name"] == name:
            t["status"] = "complete"
            return {"result": t}
    return {"error": "Task not found"}


tools = [get_all_tasks, get_task, update_status]

# assign model and tools to the agent
os.environ["GOOGLE_API_KEY"] = "AIzaSyBm8MgIuEWPIgIG3wt0CQcq9uk5JLNcqMo"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)

llm_with_tools = llm.bind_tools(tools)

# node definitions
def assistant(state: MessagesState):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])],
    }

# build the graph
task_graph = StateGraph(MessagesState)

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

compiled_graph = task_graph.compile()

# compiled_graph.invoke({})

