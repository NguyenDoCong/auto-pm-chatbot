import requests
from typing import List
from langchain.schema import Document
import pprint
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from uuid import uuid4

# from langchain_core.runnables import chain
import faiss
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional
from pydantic import BaseModel, Field
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain_community.query_constructors.chroma import ChromaTranslator
from typing import Any, Dict

load_dotenv()
os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embeddings = OllamaEmbeddings(model="all-minilm")

class Item(BaseModel):
    """Information about an Item."""

    # ^ Doc-string for the entity State.
    # This doc-string is sent to the LLM as the description of the schema State,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    state: Optional[str] = Field(default=None, description="The value of the state")
    priority: Optional[str] = Field(default=None, description="The value of priority")


# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return 'none' for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)

def get_data_from_api(url: str, table_name: str = None):
    headers = {"x-api-key": "plane_api_b343a356f3d1480ab568697a162150dd"}

    response = requests.get(url, headers=headers)
    if table_name:
        return response.json()
        
    return response.json()["results"]

def unite_tables():
    assignees = get_data_from_api('https://api.plane.so/api/v1/workspaces/congnguyendo/members/','members')
    issues = get_data_from_api('https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues')
    states = get_data_from_api('https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/states')

    # pprint.pprint(assignees)
    # print("Issues:", issues)

    # Tạo index theo id (id là string)
    assignee_index = {a["id"]: a["display_name"] for a in assignees}
    state_index = {s["id"]: s["group"] for s in states}

    # pprint.pprint(assignee_index)

    result = []
    for issue in issues:
        enriched = {
            "id": issue["id"],
            "name": issue["name"],
            "priority": issue["priority"],
            "assignees": [assignee_index[aid] for aid in issue["assignees"]],
            "state": state_index.get(issue["state"])            
        }
        result.append(enriched)

    return result

def get_all_tasks():
    """Get all tasks."""
    responses = unite_tables()

    results = []
    for response in responses:
        content = f"This is a work item that has name {response['name']}"
        assignees = " ".join(response["assignees"])
        metadata = {"priority": response["priority"], "assignees": assignees}

        document = Document(page_content=content, metadata=metadata)
        # content = (
        #     f'Name: {response["name"]} | '
        #     f'id: {response["id"]} | '
        #     f'assignees: {response["assignees"]} | '
        #     f'description: {response["description_stripped"]} | '
        #     f'priority: {response["priority"]}'
        # )
        # document = Document(page_content=content)

        results.append(document)

    return results


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
        ),
    ]


def build_vector_store() -> Chroma:
    tasks = get_all_tasks()
    return Chroma.from_documents(tasks, embeddings)

def build_self_query_retriever(
    llm, vector_store: Chroma, examples: List = None
) -> SelfQueryRetriever:
    metadata_field_info = build_metadata_field_info()
    document_content_description = "Brief summary of a work item"

    if examples:
        # Trường hợp priority_query: có examples
        prompt = get_query_constructor_prompt(
            document_content_description, metadata_field_info, examples=examples
        )
        output_parser = StructuredQueryOutputParser.from_components()
        query_constructor = prompt | llm | output_parser

        return SelfQueryRetriever(
            query_constructor=query_constructor,
            vectorstore=vector_store,
            structured_query_translator=ChromaTranslator(),
        )
    else:
        # Trường hợp store_search: dùng from_llm
        return SelfQueryRetriever.from_llm(
            llm,
            vector_store,
            document_content_description,
            metadata_field_info,
            search_kwargs={"k": 10},
        )


def store_search(query: str) -> Dict[str, Any]:
    """
    Search in vector store using query + metadata filter.
    """
    vector_store = build_vector_store()
    retriever = build_self_query_retriever(model, vector_store)

    result = retriever.invoke(query)
    return {"result": result}

if __name__ == "__main__":
    result = store_search("List all started tasks")
    pprint.pprint(result['result'])