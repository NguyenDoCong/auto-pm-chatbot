
from langchain_chroma import Chroma
from typing import List, Dict, Any, Optional, Literal
from .store import build_self_query_retriever
from langchain_openai import ChatOpenAI
import os
import re
from datetime import datetime
import requests
from dotenv import load_dotenv
from tools.store import build_vector_store
from data.table_union_plane import get_short_term_memory, get_long_term_memory, get_all_tasks

load_dotenv()
llm = ChatOpenAI(
  api_key=os.getenv("OPENROUTER_API_KEY"),
  base_url=os.getenv("OPENROUTER_BASE_URL"),
  model="google/gemini-2.0-flash-001",
)

tasks = get_all_tasks()

vector_store = build_vector_store(tasks)

def store_search(query: str) -> Dict[str, Any]:
    # , filter: Optional[Dict[str, str]] = None
    """
    Search for tasks related to query and return tasks.

    Parameters:
        query (str): User query string.

    Returns:
        Dict[str, Any]: Related tasks.
    """
    print(f"Query: {query}")
    

    retriever = build_self_query_retriever(llm, vector_store)
    

    result = retriever.invoke(query)
    # return compressed_docs
    return result

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

def add_report(id, report):
    """Add report by task id."""

    url = f"https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues/{id}/comments/"

    payload = {"comment_html": report}
    headers = {
        "x-api-key": "plane_api_29c429f9822146aba6782cba5d3c1a4a",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers)

    # print(response.json())
    return {"result": response.json()}