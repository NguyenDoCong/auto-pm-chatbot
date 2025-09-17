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

load_dotenv()
os.getenv("GOOGLE_API_KEY")


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


def get_all_tasks():
    """Get all tasks."""
    url = "https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues/"

    headers = {"x-api-key": "plane_api_b343a356f3d1480ab568697a162150dd"}

    response = requests.get(url, headers=headers)
    responses = response.json()["results"]
    results = []
    for response in responses:
        content = f"This is a work item that has name {response['name']}"
        assignees = " ".join(response["assignees"])
        metadata = {
            "state": response["state"],
            "priority": response["priority"].lower(),  # Chuyển về chữ thường
            "assignees": assignees,
        }
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


def doc_to_text(docs: List[Document]) -> str:
    """Convert list of Document to text."""
    texts = [doc.model_dump_json() for doc in docs]
    # return "\n".join(texts)
    return texts


def add_to_vector_store():
    """Add all tasks to vector store."""
    embeddings_model = OllamaEmbeddings(model="llama3.2")
    tasks = get_all_tasks()["result"]
    texts = doc_to_text(tasks)
    # vector_store = FAISS(
    #     embedding_function=embeddings_model,
    #     docstore=InMemoryDocstore(),
    #     index_to_docstore_id={},
    # )
    vector_store = FAISS.from_documents(tasks, embeddings_model)

    uuids = [str(uuid4()) for _ in range(len(texts))]

    vector_store.add_documents(documents=tasks, ids=uuids)


def generate_filter_from_query(query: str) -> dict:
    """Generate filter key-value pairs from a natural language query."""
    llm = ChatOllama(model="llama3.2")
    structured_llm = llm.with_structured_output(schema=Item)
    # template = f"""
    # Given the following query: "{query}", extract key-value pairs for filtering tasks.
    # The keys should match the metadata fields: ['state'].
    # Respond in JSON format only.
    # """
    # prompt = ChatPromptTemplate.from_template(template)
    prompt = prompt_template.invoke({"text": query})
    response = structured_llm.invoke(prompt)
    # chain = prompt | llm

    # response = chain.invoke(query)
    return response.model_dump()


if __name__ == "__main__":
    embeddings = OllamaEmbeddings(model="all-minilm")
    tasks = get_all_tasks()

    vector_store = Chroma.from_documents(tasks, embeddings)

    metadata_field_info = [
        AttributeInfo(
            name="priority",
            description="The priority of the work item. One of ['none', 'urgent', 'high', 'medium', 'low']",
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
    ]
    document_content_description = "Brief summary of a work item"

    llm = ChatOllama(model="llama3.2")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    retriever = SelfQueryRetriever.from_llm(
        llm,
        vector_store,
        document_content_description,
        metadata_field_info,
        enable_limit=False,
    )

    # retriever = vector_store.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={"k": 10},
    # )
    # pprint.pprint(vector_store.get())

    # result = retriever.invoke("show all data")
    # pprint.pprint(result)

    # prompt = get_query_constructor_prompt(
    #     document_content_description,
    #     metadata_field_info,
    # )
    # output_parser = StructuredQueryOutputParser.from_components()
    # query_constructor = prompt | llm | output_parser

    # retriever = SelfQueryRetriever(
    #     query_constructor=query_constructor,
    #     vectorstore=vector_store,
    #     structured_query_translator=ChromaTranslator(),
    # )

    # print(prompt.format(query="What is the most urgent task?"))

    # print(query_constructor.invoke(
    #     {
    #         "query": "list all tasks"
    #     }
    # ))

    # pprint.pprint(vector_store.get())

    # Thứ tự priority
    PRIORITY_ORDER = ["urgent", "high", "medium", "low", "none"]
    start_idx = PRIORITY_ORDER.index("urgent")

    result = {"priority_matched": None, "issues": [], "metadata": []}

    all_docs = []  # Đổi tên để tránh ghi đè

    for level in PRIORITY_ORDER[start_idx:]:
        query = f"filter tasks with priority {level}"
        print(query)
        examples = [
            (
                "Find all tasks with priority high",
                {
                    "query": "get all tasks",
                    "filter": 'eq("priority", "high")',
                },
            ),
            (
                "Find all tasks with priority low",
                {
                    "query": "get all tasks",
                    "filter": 'eq("priority", "low")',
                },
            ),
        ]
        prompt = get_query_constructor_prompt(
            document_content_description, metadata_field_info, examples=examples
        )
        output_parser = StructuredQueryOutputParser.from_components()
        query_constructor = prompt | llm | output_parser
        # print(prompt.format(query=query))
        print(query_constructor.invoke(
            {
                "query": query
            }
        ))
        retriever = SelfQueryRetriever(
            query_constructor=query_constructor,
            vectorstore=vector_store,
            structured_query_translator=ChromaTranslator(),
        )

        level_docs = retriever.invoke(query)
        all_docs.extend(level_docs)  
        # Thu thập tất cả docs        
        # if docs:
        #     result = {
        #         "priority_matched": level,
        #         "issues": [d.page_content for d in docs],
        #         "metadata": [d.metadata for d in docs]
        #     }

    pprint.pprint(all_docs)
