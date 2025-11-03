from langchain.chains.query_constructor.schema import AttributeInfo
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from datetime import datetime
from langchain_community.query_constructors.chroma import ChromaTranslator

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

