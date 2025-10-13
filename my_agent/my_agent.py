import os
from typing import List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough

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

from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import LongContextReorder
from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
from pydantic import BaseModel
from datetime import datetime
import re

load_dotenv()
os.getenv("GOOGLE_API_KEY")
os.getenv("LANGSMITH_API_KEY")
X_API_KEY = os.getenv("X_API_KEY")
# Initialize the model
model = ChatOllama(model="llama3.1")
llm = ChatOllama(model="llama3.1")

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
class Search(BaseModel):
    query: str
    priority: Optional[Literal["high", "medium", "low", "urgent"]] = None
    state: Optional[Literal["Backlog", "Todo", "Tn Progress", "Done", "Cancelled"]] = None

def construct_comparisons(query: Search):
    comparisons = []
    if query.priority is not None:
        comparisons.append(
            Comparison(
                comparator=Comparator.EQ,
                attribute="priority",
                value=query.priority,
            )
        )
    if query.state is not None:
        comparisons.append(
            Comparison(
                comparator=Comparator.EQ,
                attribute="state",
                value=query.state,
            )
        )
    return comparisons    

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
        try:
            date_obj = datetime.strptime(response['start_date'], "%Y-%m-%d")
            start_date = date_obj.timestamp()
        except Exception as e:
            # print(e)
            start_date = None
        try:
            date_obj = datetime.strptime(response['target_date'], "%Y-%m-%d")
            target_date = date_obj.timestamp()
        except Exception as e:
            # print(e)
            target_date = None
        try:
            date_obj = datetime.strptime(response['completed_at'], "%Y-%m-%d")
            completed_at = date_obj.timestamp()
        except Exception as e:
            # print(e)
            completed_at = None
        content = f"This is a work item that has name: {response['name']}, assignees: {assignees}, \
            'comments': {comments}, 'start date': {response['start_date']}, 'target date': {response['target_date']}, \
            'completed at': {response['completed_at']}, 'description': {response['description_stripped']}, \
            'priority': {response['priority']}, 'state': {response['state']}."
        metadata = {"priority": str(response['priority']), "state": str(response['state']), \
                    "start date": start_date, \
                    "target date": target_date, \
                    "completed at": completed_at}
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
        AttributeInfo(
            name="target date",
            description="The due date of the task, in the format YYYY-MM-DD. If the task does not have a due date, the value is None",
            type="float",
        ),
    ]


def build_vector_store() -> Chroma:
    tasks = get_all_tasks()
    return Chroma.from_documents(tasks, embeddings)

import dateutil.parser


def build_self_query_retriever(
    llm, vector_store: Chroma, examples: List = None
) -> SelfQueryRetriever:
    metadata_field_info = build_metadata_field_info()
    # metadata_field_info = []

    document_content_description = "Information of a work item."

    # date_obj = datetime.strptime(response['target_date'], "%Y-%m-%d")
    # target_date = date_obj.timestamp()
    
    # print(f"Target date ordinal: {target_date}")

    examples = [
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

vector_store = build_vector_store()

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

def store_search(query: str) -> Dict[str, Any]:
    # , filter: Optional[Dict[str, str]] = None
    """
    Search in vector store using query and filter if provided.
    """
    print(f"Query: {query}, Filter: {filter}")
    # tasks= get_all_tasks()
    # ###### child retriever
    # # This text splitter is used to create the child documents
    # child_splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=1)
    # # The vectorstore to use to index the child chunks
    # vectorstore = Chroma(
    #     collection_name="full_documents", embedding_function=embeddings
    # )
    # # The storage layer for the parent documents
    # store = InMemoryStore()
    # retriever = ParentDocumentRetriever(
    #     vectorstore=vectorstore,
    #     docstore=store,
    #     child_splitter=child_splitter,
    #     search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1}
    # )
    # retriever.add_documents(tasks, ids=None)

    # query_date = "2026-01-15"
    # query_timestamp = datetime.strptime(query_date, "%Y-%m-%d").timestamp()

    # # Tạo retriever với metadata filter
    # retriever = vector_store.as_retriever(
    #     search_kwargs={
    #         "filter": {"target date": {"$lt": query_timestamp}},
    #         "k": 10
    #     }
    # )

    # print(f"test: {retriever.invoke("tasks")}")

    retriever = build_self_query_retriever(llm, vector_store)
    # retriever = vector_store.as_retriever(
    #     search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1}
    # )

    ##### query analysis filter
    # comparisons = construct_comparisons(query)
    # _filter = Operation(operator=Operator.AND, arguments=comparisons)
    # chrom_filter = ChromaTranslator().visit_operation(_filter)

    ##### compressor
    # splitter = CharacterTextSplitter(chunk_size=30, chunk_overlap=1, separator=", ")
    # redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    # relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.1)
    # pipeline_compressor = DocumentCompressorPipeline(
    #     transformers=[splitter, redundant_filter]
    # )

    # Kiểm tra từng bước
    # tasks = get_all_tasks()
    # split_docs = splitter.split_documents(tasks)
    # print("Documents after splitting:")
    # pprint.pprint(split_docs)

    # filtered_docs = redundant_filter.transform_documents(split_docs)
    # print("Documents after redundant filter:")
    # pprint.pprint(filtered_docs)

    # final_docs = relevant_filter.transform_documents(filtered_docs)
    # print("Documents after relevant filter:")
    # pprint.pprint(final_docs)

    # retriever = Chroma.from_documents(tasks, embeddings).as_retriever()

    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=pipeline_compressor, base_retriever=retriever
    # )

    # ##### embeddings filter
    # embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.1)
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=embeddings_filter, base_retriever=retriever
    # )

    

    # compressed_docs = compression_retriever.invoke(query)
    # reordering = LongContextReorder()
    # reordered_docs = reordering.transform_documents(compressed_docs)

    # docs = retriever.get_relevant_documents(query=query.query, filter=chrom_filter)


    # pretty_print_docs(compressed_docs)

    # print(list(store.yield_keys()))

    # sub_docs = vectorstore.similarity_search(query)
    # print(sub_docs[0].page_content)

    result = retriever.invoke(query)
    # return compressed_docs
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

class Search(BaseModel):
    query: str
    state: Optional[str]
    priority: Optional[str]

def generate_query_from_user_input(user_input: str) -> str:
    """_summary_
    Generate a structured query from user input.
    Args:
        user_input (str): _description_

    Returns:
        str: _description_
    """
    system = """You are an expert at converting user questions into database queries. \
    You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
    Given a question, return a list of database queries optimized to retrieve the most relevant results.

    If there are acronyms or words you are not familiar with, do not try to rephrase them."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("examples", optional=True),
            ("human", "{question}"),
        ]
    )
    structured_llm = llm.with_structured_output(Search)
    query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm
    result = query_analyzer.invoke({"question": user_input})
    print(result)
    return result

def construct_comparisons(query: Search):
    comparisons = []
    if query.state is not None:
        comparisons.append(
            Comparison(
                comparator=Comparator.GT,
                attribute="state",
                value=query.state,
            )
        )
    if query.priority is not None:
        comparisons.append(
            Comparison(
                comparator=Comparator.EQ,
                attribute="priority",
                value=query.priority,
            )
        )
    return comparisons

def generate_filtered_query(search: Search) -> str:
    """_summary_

    Args:
        user_input (str): _description_

    Returns:
        str: _description_
    """
    comparisons = construct_comparisons(search)
    _filter = Operation(operator=Operator.AND, arguments=comparisons)
    result = ChromaTranslator().visit_operation(_filter)
    print(result)

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


tools = [store_search, process_date_query]

llm_with_tools = model.bind_tools(tools)


examples = [
    HumanMessage(
        "Missed deadline tasks", name="example_user"
    ),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {"name": "store_search", "args": {"query":"state Done"}, "id": "1"}
        ],
    ),
    ToolMessage("Document(id='db2f40ac-8216-49f9-a501-e018cdda8259', metadata={'state': 'Done', 'target date': 1758240000.0, 'start date': 1758240000.0, 'priority': 'none'}, page_content='This is a work item that has name: 28-6, assignees: nguyen.do.cong and tuoithodudoi.phungquan', 'comments': , 'start date': 2025-09-19, 'target date': 2025-09-19, 'completed at': 2025-10-13T03:40:04.150141Z, 'description': , 'priority': none, 'state': Done.')", tool_call_id="1"),
    AIMessage(
        "I have already searched for tasks in 'Done' and 'In Progress' states. Now I need to analyze the results to identify the missed deadline tasks.",
        name="example_assistant",
    ),
    AIMessage(
        "Here are the missed deadline tasks:\
        1. Task with name 28-6:\
        - State: Done\
        - Target date: 2025-09-19 (timestamp: 1758240000.0)\
        - Completed at: 2025-10-13T03:40:04.150141Z\
        Since the target date (2025-09-19) is before the completed at date (2025-10-13), this is a missed deadline task.\
        2. Task with name test-assign:\
        - State: In Progress\
        - Target date: 2025-09-21 (timestamp: 1758412800.0)\
        Since the target date (2025-09-21) is before 2025-10-13, this is a missed deadline task.",
        name="example_assistant",
    ),
]

today = datetime.today().strftime('%Y-%m-%d')

# System message
sys_msg = SystemMessage(
    content=f"You are a helpful assistant specialized at getting tasks information for users. Use the tool provided to you to get all task related information, \
        including name, priority, state, comments, start date, target date (due date), completed at, description and assignees, \
        Priority level from highest to lowest is: urgent, high, medium, low, none. A task is the highest priority task if all other tasks have lower priority. \
        A missed deadline task is a task that has state Done and its target date (due date) is before completed at (date) OR \
        a task that has state 'In Progress' and its target date is before {today}. \
        Use information from provided tool to generate answers to the USER INPUT. \
        Use the following format to find your answers: \
        Question: The input query you must answer. \
        Thought: Carefully consider the query to create a tool call. \
        The store_search cannot search for ambiguity queries, so make sure your queries are simple enough, \
        even if that means you have to make some round of calling tools with some sets of simplified queries. \
        IMPORTANT: All dates in query need to be converted to timestamp using process_date_query before calling to store_search\
        Action: Make the tool call to find the answer. \
        Observation: The RELEVANT INFORMATION to the inquiry. \
        Repeat Question-thought-action-output until you are sure of the correct response. \
        IMPORTANT: Only repeat after the previous chains of thought finishes. Don't create parallel chains of thought. "
)

system = f"You are a helpful assistant specialized at getting tasks information for users. Use the tool provided to you to get all task related information, \
        including name, priority, state, comments, start date, target date (due date), completed at, description and assignees, \
        Priority level from highest to lowest is: urgent, high, medium, low, none. A task is the highest priority task if all other tasks have lower priority. \
        A missed deadline task is a task that has state Done and its target date (due date) is before completed at (date) OR \
        a task that has state 'In Progress' and its target date is before {today}. \
        Use information from provided tool to generate answers to the USER INPUT. Only provide answer with info related to the USER INPUT\
        Use the following format to find your answers: \
        Question: The input query you must answer. \
        Thought: Carefully consider the query to create a tool call. \
        The store_search cannot search for ambiguity queries, so make sure your queries are simple enough, \
        even if that means you have to make some round of calling tools with some sets of simplified queries. \
        IMPORTANT: All dates in query need to be converted to timestamp using process_date_query before calling to store_search\
        Action: Make the tool call to find the answer. \
        Observation: The RELEVANT INFORMATION to the inquiry. \
        Repeat Question-thought-action-output until you are sure of the correct response. \
        IMPORTANT: Only repeat after the previous chains of thought finishes. Don't create parallel chains of thought. "

few_shot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        *examples,
        ("human", "{query}"),
    ]
)

chain = {"query": RunnablePassthrough()} | few_shot_prompt | llm_with_tools

#The store_search tool cannot retrieve ambiguity queries, so you are recommended to use simple queries. \
 #       Break abstract queries into multiple simple queries, then make a plan of calling tools with those simple queries if needed.\

# "If the user query is about finding tasks with the highest priority, search for tasks with priority from highest to lowest.\
#         Priority level from highest to lowest is: urgent, high, medium, low, none. "
#     "If the tool response is none, try to query a lower priority level. If you don't find the response that has the answer you're looking for, \
#         try to remember the task with highest priority at each response. Select the task that has the highest priority based on that memory and give the answer. "
#     "If the user query is related to starting time of a task, look for the time mentioned in the start date key of the task. "   
#     "If the user query is related to due date of a task, create a query consisted of the word 'target_date' and the date mentioned in the user query. " 
#     "If the user query is related to finishing time of a task, look for the time mentioned in the completed at key of the task. "
#     "If a task has completed at time, it means that the task is completed. "
#     "If the user query is about the state of a task, look for the state key of the task. "

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

compiled_graph = task_graph.compile()

# try:
#     display(Image(compiled_graph.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass
# compiled_graph.invoke({})
# print(compiled_graph.get_graph().draw_mermaid())

def stream_graph_updates(user_input: str):
    config = {"configurable": {"thread_id": "1"}}
    for event in compiled_graph.stream({"messages": [{"role": "user", "content": user_input}]}, config,
        stream_mode="values",
    ):
        for value in event.values():
            print("Assistant:", value[-1].content)
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

# if __name__ == "__main__":
# # #     # result = store_search("list all tasks assigned to tuoithodudoi.phungquan")
# # #     # result = store_search("tasks with highest priority")

    
# # #     result = store_search("NOT target_date:None AND target_date:*")

# # #     pprint.pprint(result["result"])
# # #     # get_all_tasks()

#     # pprint.pprint(get_all_tasks())

#     # pprint.pp(store_search("medium priority tasks with state Backlog"))

#     # build_self_query_retriever(llm, vector_store)

#     # tmp = generate_query_from_user_input("tasks with target date before 2025-10-09")
#     # generate_filtered_query(tmp)
#     pprint.pp(store_search("tasks"))
