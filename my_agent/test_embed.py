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
        content = f'This is a work item that has id {response["id"]}, name {response["name"]}, described as {response["description_stripped"]}, created by {response["created_by"]}'
        assignees = ' '.join(response["assignees"])
        metadata = {"state": response["state"], "priority": response["priority"], "assignees": assignees}
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
    # tasks = get_all_tasks()
    # add_to_vector_store()
    # pprint.pprint(tasks)
    # pprint.pprint(doc_to_text(tasks["result"]))
    embeddings = OllamaEmbeddings(model="all-minilm")
    tasks = get_all_tasks()
    # texts = doc_to_text(tasks)
    # for task in tasks:
    #     pprint.pprint(task)
    # index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    # vector_store = FAISS(
    #     embedding_function=embeddings,
    #     index=index,
    #     docstore=InMemoryDocstore(),
    #     index_to_docstore_id={},
    # )
    vector_store = Chroma.from_documents(tasks, embeddings)

    # vector_store = FAISS.from_documents(tasks, embeddings)

    # uuids = [str(uuid4()) for _ in range(len(tasks))]
    # for id in uuids:
    #     print(id)

    # vector_store.add_documents(documents=tasks)

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

    result = retriever.invoke("List all tasks that has 507817ec-d907-461e-a86a-be44fde519d3 as assignee")
    pprint.pprint(result)

    # query = "List all tasks with state fa8c251a-40e1-4f25-a438-213fc396f958"
    # filter_criteria = generate_filter_from_query(query)
    # print(f"Filter criteria: {filter_criteria}")

    # # Áp dụng filter vào similarity_search
    # results = vector_store.similarity_search(
    #     query,
    #     # search_kwargs={"k": 2},
    #     filter={'state': 'fa8c251a-40e1-4f25-a438-213fc396f958', 'priority': 'none'},
    # )
    
    # for res in results:
    #     print(f"* {res.page_content} [{res.metadata}]")

    # embedding = embeddings.embed_query("List all tasks that has 507817ec-d907-461e-a86a-be44fde519d3 as assignee")

    # results = vector_store.similarity_search_by_vector(embedding)
    # print(results)