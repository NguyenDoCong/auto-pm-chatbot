import os
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
# from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
import requests
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_community.vectorstores import FAISS
from uuid import uuid4
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
from langgraph.types import CachePolicy

from IPython.display import Image, display
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

load_dotenv()
os.getenv("GOOGLE_API_KEY")
os.getenv("LANGSMITH_API_KEY")
X_API_KEY = os.getenv("X_API_KEY")
# Initialize the model
model = ChatOllama(model="llama3.1")
llm = ChatOllama(model="llama3.1")

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

embeddings = OllamaEmbeddings(model="all-minilm")

# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def get_data_from_api(url: str, table_name: str = None):
    headers = {"x-api-key": "plane_api_b343a356f3d1480ab568697a162150dd"}

    response = requests.get(url, headers=headers)
    if table_name:
        return response.json()

    return response.json()["results"]


def unite_tables():
    assignees = get_data_from_api(
        "https://api.plane.so/api/v1/workspaces/congnguyendo/members/", "members"
    )
    issues = get_data_from_api(
        "https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues"
    )
    states = get_data_from_api(
        "https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/states/"
    )

    # pprint.pprint(assignees)
    # print("Issues:", issues)

    # Tạo index theo id (id là string)
    assignee_index = {a["id"]: a["display_name"] for a in assignees}
    # pprint.pprint(assignee_index)
    state_index = {s["id"]: s["name"] for s in states}

    result = []
    for issue in issues:
        comments = get_data_from_api(f'https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues/{str(issue["id"])}/comments/')
        if not comments:
            comment_texts = []
        else:
            comment_texts = [c["comment_stripped"] for c in comments]
        enriched = {
            "id": issue["id"],
            "name": issue["name"],
            "priority": issue["priority"],
            "start_date": issue["start_date"],
            "target_date": issue["target_date"],
            "completed_at": issue["completed_at"],
            "description_stripped": issue["description_stripped"],
            "assignees": [assignee_index[aid] for aid in issue["assignees"]],
            "state": state_index.get(issue["state"]),
            "comments": comment_texts,
            }
        result.append(enriched)

    return result


# tool definition
def get_all_tasks():
    """Get all tasks."""

    responses = unite_tables()
    results = []
    for response in responses:
        assignees = " and ".join(response["assignees"])
        comments  = " and ".join(response["comments"])
        content = f"This is a work item that has name: {response['name']}, assignees: {assignees}, \
            'comments': {comments}, 'start date': {response['start_date']}, 'target_date': {response['target_date']}, \
            'completed at': {response['completed_at']} , 'description': {response['description_stripped']}"
        metadata = {"priority": str(response['priority']), "state": str(response['state'])}
        document = Document(page_content=content, metadata=metadata)
        results.append(document)

    # pprint.pprint(results)
    return results


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

    payload = {"comment_html": report}
    headers = {
        "x-api-key": "plane_api_29c429f9822146aba6782cba5d3c1a4a",
        "Content-Type": "application/json",
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

    summary_message = f"Create a short summary of all reports of the task {task['name']} by using the following reports: {daily_report}"

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

    return {"result": result}


# Danh sách priority theo thứ tự
PRIORITY_ORDER = ["urgent", "high", "medium", "low", "none"]


def build_metadata_field_info() -> List[AttributeInfo]:
    return [
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
        # AttributeInfo(
        #     name="assignees",
        #     description="The list of assignees of the work item",
        #     type="list[string]",
        # ),
        # AttributeInfo(
        #     name="created_by",
        #     description="The creator of the work item",
        #     type="string",
        # ),
    ]


def build_vector_store() -> Chroma:
    tasks = get_all_tasks()
    return Chroma.from_documents(tasks, embeddings)


def build_self_query_retriever(
    llm, vector_store: Chroma, examples: List = None
) -> SelfQueryRetriever:
    metadata_field_info = build_metadata_field_info()
    # metadata_field_info = []

    document_content_description = "Information of a work item."

    # examples = [
    #     ("Find all tasks with priority high",
    #      {"query": "get all tasks", "filter": 'eq("priority", "high")'}),
    #     ("Find all tasks with priority low and state is Todo",
    #      {"query": "get all tasks", "filter": 'and(eq("priority", "low"), eq("state", "Todo")'}),
    # ]

    if examples:
        # Trường hợp priority_query: có examples
        prompt = get_query_constructor_prompt(
            document_content_description, metadata_field_info, examples=examples
        )
        # print(prompt.format(query="dummy question"))
        output_parser = StructuredQueryOutputParser.from_components()
        query_constructor = prompt | llm | output_parser
        
        print(query_constructor.invoke({"query": "highest priority tasks with state Todo"}))

        return SelfQueryRetriever(
            query_constructor=query_constructor,
            vectorstore=vector_store,
            structured_query_translator=ChromaTranslator(),
            # search_type="similarity_score_threshold",
            # search_kwargs={"score_threshold": 0.5},
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

# vector_store = build_vector_store()

def store_search(query: str, filter: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Search in vector store using query.
    """
    tasks= get_all_tasks()
    # This text splitter is used to create the child documents
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=0)
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="full_documents", embedding_function=embeddings
    )
    # The storage layer for the parent documents
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )
    retriever.add_documents(tasks, ids=None)
    # retriever = build_self_query_retriever(llm, vector_store)
    # retriever = vector_store.as_retriever(
    #     search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3}
    # )

    result = retriever.invoke(query)
    return result


def priority_query(start_priority: str = "urgent") -> Dict[str, Any]:
    """
    Find tasks with the highest priority available.
    """
    vector_store = build_vector_store()

    retriever = build_self_query_retriever(llm, vector_store)

    start_idx = PRIORITY_ORDER.index(start_priority)
    for level in PRIORITY_ORDER[start_idx:]:
        query = f"filter tasks with priority {level}"
        docs: List[Document] = retriever.invoke(query)
        if docs:
            return {
                "priority_matched": level,
                "issues": [d.page_content for d in docs],
                "metadata": [d.metadata for d in docs],
            }

    return {"priority_matched": None, "issues": [], "metadata": []}


tools = [store_search]

llm_with_tools = model.bind_tools(tools)

# System message
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with getting tasks information for a user. Use the tool provided to you to query all task related information, \
        including name, priority, state, comments, start date, target date (due date), completed at, description and assignees. "
    "If the user query is about finding tasks with the highest priority, search for tasks with priority from highest to lowest.\
        Priority level from highest to lowest is: urgent, high, medium, low, none. "
    "If the tool response is none, try to query a lower priority level. If you don't find the response that has the answer you're looking for, \
        try to remember the task with highest priority at each response. Select the task that has the highest priority based on that memory and give the answer. "
    "If the user query is related to starting time of a task, look for the time mentioned in the start date key of the task. "   
    "If the user query is related to due date of a task, create a query consisted of the word 'target_date' and the date mentioned in the user query. " 
    "If the user query is related to finishing time of a task, look for the time mentioned in the completed at key of the task. "
    "If a task has completed at time, it means that the task is completed. "
    "If the user query is about the state of a task, look for the state key of the task. "
)

class State(MessagesState):
    # summary: str
    # user_query: str
    # related_issues: Optional[List[Dict[str, Any]]]
    # priority_matched: Optional[str]
    """Main graph state."""

    messages: Annotated[list[AnyMessage], add_messages]
    """The messages in the conversation."""


__all__ = [
    "State",
]

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

memory = InMemorySaver()

compiled_graph = task_graph.compile()

# try:
#     display(Image(compiled_graph.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass
# compiled_graph.invoke({})
# print(compiled_graph.get_graph().draw_mermaid())

def stream_graph_updates(user_input: str):
    # config = {"configurable": {"thread_id": "1"}}
    # for event in compiled_graph.stream({"messages": [{"role": "user", "content": user_input}]}, config,
    #     stream_mode="values",
    # ):
    #     for value in event.values():
    #         print("Assistant:", value[-1].content)
    for chunk in compiled_graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="updates"
    ):
        print(chunk)
        print("\n")

# while True:
#     try:
#         user_input = input("User: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("Goodbye!")
#             break
#         stream_graph_updates(user_input)
#     except:
#         # fallback if input() is not available
#         user_input = "What do you know about LangGraph?"
#         print("User: " + user_input)
#         stream_graph_updates(user_input)
#         break

# if __name__ == "__main__":
# # #     # result = store_search("list all tasks assigned to tuoithodudoi.phungquan")
# # #     # result = store_search("tasks with highest priority")

    
# # #     result = store_search("NOT target_date:None AND target_date:*")

# # #     pprint.pprint(result["result"])
# # #     # get_all_tasks()

#     # pprint.pprint(get_all_tasks())

#     # pprint.pp(store_search("tasks with low priority"))

#     build_self_query_retriever(llm, vector_store)