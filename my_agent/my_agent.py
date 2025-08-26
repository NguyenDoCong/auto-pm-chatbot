import os
from typing import List, TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama

class State(MessagesState):
    summary: str

# mock data
tasks = [
    {
        "id": 1,
        "name": "Task 1",
        "description": "This is a sample task for demonstration purposes.",
        "daily_report": [],
        "daily_report_summary": "",
        "status": "incomplete"
    },
    {
        "id": 2,
        "name": "Task 2",   
        "description": "This is another sample task for demonstration purposes.",
        "daily_report": [],    
        "daily_report_summary": "",            
        "status": "incomplete"
    }
]

# Initialize the model
# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model = ChatOllama(model="llama3.2")

# assign model and tools to the agent
# os.environ["GOOGLE_API_KEY"] = "AIzaSyBaRw3LMUUZrMUvuEVZswaxcwoQRMgl8tE"

#tool definition
def get_all_tasks():
    """Get all tasks."""
    return {"result":tasks}

def get_task(name):
    """Get task by name."""
    for t in tasks:
        if t["name"] == name:
            return {"result": t}
    return {"error": "Task not found"}

def add_report(id, report):
    """Add report by task id."""
    for t in tasks:
        if t["id"] == id:
            t["daily_report"].append(report)
            return {"result": t}
    return {"error": "Task not found"}

def update_status(id):
    """Update status by task id."""
    for t in tasks:
        if t["id"] == id:
            t["status"] = "complete"
            return {"result": t}
    return {"error": "Task not found"}

def summarize_reports(id: Optional[int] = None):
    """Summarize reports by task id."""
    task = None
    for t in tasks:
        if t["id"] == id:
            task = t

    if not task:
        return {"error": "Task not found"}
    
    daily_report = None
    daily_report = task["daily_report"]
    if not daily_report:
        return {"error": "No reports to summarize"}
            
    summary_message = f"Create a short summary of all reports of the task {task["name"]} by using the following reports: {daily_report}"

    response = model.invoke(summary_message)

    task["daily_report_summary"] = response.content

    return {"summary": response.content}

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tools = [get_all_tasks, get_task, add_report, update_status, summarize_reports, add, multiply]

llm_with_tools = model.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with getting tasks and changing task status.")

# node definitions
def assistant(state: State):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

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

memory = MemorySaver()

compiled_graph = task_graph.compile()

# compiled_graph.invoke({})

