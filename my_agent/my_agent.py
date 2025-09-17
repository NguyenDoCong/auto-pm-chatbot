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
import requests
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")

class State(MessagesState):
    summary: str

# Initialize the model
model = ChatOllama(model="llama3.2")
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embeddings = OllamaEmbeddings(model="all-minilm")

# assign model and tools to the agent

#tool definition
def get_all_tasks():
    """Get all tasks."""
    url = "https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues/"

    headers = {"x-api-key": "plane_api_b343a356f3d1480ab568697a162150dd"}

    response = requests.get(url, headers=headers)
    responses = response.json()["results"]
    results = []
    for response in responses:
        content = f'This is a work item that has id {response["id"]}, name {response["name"]}, described as {response["description_stripped"]}'
        assignees = ' '.join(response["assignees"])
        metadata = {"state": response["state"], "priority": response["priority"], "assignees": assignees}
        document = Document(page_content=content, metadata=metadata)
        results.append(document)

    return results

def doc_to_text(docs: List[Document]) -> str:
    """Convert list of Document to text."""
    texts = [doc.model_dump_json() for doc in docs]
    # return "\n".join(texts)
    return texts

def embed_all_tasks():
    """Embed all tasks."""
    tasks = get_all_tasks()["result"]
    texts = doc_to_text(tasks)
    embeddings = embeddings_model.embed_documents(texts)
    return embeddings

def add_to_vector_store():
    """Add all tasks to vector store."""
    tasks = get_all_tasks()
    texts = doc_to_text(tasks)
    vector_store = Chroma.from_documents(tasks, embeddings)


    

def get_task_ids_by_assignee_id(assignee_id: str):
    """Get task ids by assignee id."""
    url = "https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues/?fields=id,name,assignees,description_stripped"
    headers = {"x-api-key": "plane_api_b343a356f3d1480ab568697a162150dd"}
    response = requests.get(url, headers=headers)
    plane_results = response.json()["results"]
    results = []
    assignee_id='507817ec-d907-461e-a86a-be44fde519d3'
    for result in plane_results:
        if assignee_id in result["assignees"]:
            results.append({"name":result["name"],"description":result["description_stripped"]})

    return {"result":results}

def get_task_ids_by_assignee_id_and_priority(assignee_id: str):
    """Get task ids by assignee id with highest priority."""
    url = "https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues/?fields=id,name,assignees,description_stripped"
    headers = {"x-api-key": "plane_api_b343a356f3d1480ab568697a162150dd"}
    response = requests.get(url, headers=headers)
    plane_results = response.json()["results"]
    results = []
    assignee_id='507817ec-d907-461e-a86a-be44fde519d3'
    for result in plane_results:
        if assignee_id in result["assignees"]:
            results.append({"name":result["name"],"description":result["description_stripped"]})

    return {"result":results}    

def get_task(name: str):
    """Get task by name."""
    tasks = get_all_tasks()["result"]
    for t in tasks:
        if name.lower() in t["name"].lower():
            return {"result": t}
    return {"error": "Task not found"}

def add_report(name, report):
    """Add report by task name."""
    task = get_task(name)["result"]
    if not task:
        return {"error": "Task not found"}
    id = task["id"]
    # for t in tasks:
    #     if t["id"] == id:
    #         t["daily_report"].append(report)
    #         return {"result": t}

    url = f"https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues/{id}/comments/"

    payload = { "comment_html": report }
    headers = {
        "x-api-key": "plane_api_29c429f9822146aba6782cba5d3c1a4a",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    # print(response.json())        
    return {"result": response.json()}

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

def get_all_comments():
    """Get all comments."""
    url = "https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues/"
    headers = {"x-api-key": "plane_api_b343a356f3d1480ab568697a162150dd"}
    response = requests.get(url, headers=headers)
    responses = response.json()["results"]
    result = []
    for response in responses:
        pass


    return {"result":result}

def store_search(query: str):
    """
    Receive a query, search in store and return a response.
    The store contains tasks with metadata: state, priority, assignees.
    """
    tasks = get_all_tasks()

    vector_store = Chroma.from_documents(tasks, embeddings)
    # vector_store = FAISS.from_documents(tasks, embeddings)

    # uuids = [str(uuid4()) for _ in range(len(tasks))]
    # for id in uuids:
    #     print(id)

    # vector_store.add_documents(documents=tasks)

    metadata_field_info = [
        AttributeInfo(
            name="priority",
            description="The priority of the work item. One of ['urgent', 'high', 'medium', 'low', 'none'], from highest to lowest",
            type="string",
        ),
        AttributeInfo(
            name="state",
            description="The state of the work item",
            type="string",
        ),
        AttributeInfo(
            name="assignees",
            description="The list of assignees of the work item",
            type="string",
        ),
        AttributeInfo(
            name="created_by",
            description="The creator of the work item",
            type="string",
        )
    ]
    document_content_description = "Brief summary of a work item"

    llm = ChatOllama(model="llama3.2")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    retriever = SelfQueryRetriever.from_llm(
        llm,
        vector_store,
        document_content_description,
        metadata_field_info,
    )

    # retriever = vector_store.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={"k": 1},
    #     # filter={"priority": "urgent"},
    # )

    result = retriever.invoke(query)
    return {"result": result}


tools = [store_search]

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

