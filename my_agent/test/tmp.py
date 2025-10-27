from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.messages.utils import get_buffer_string
from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate
from langchain_community.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import OllamaEmbeddings
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

os.getenv("GOOGLE_API_KEY")

llm = ChatOpenAI(
  api_key=os.getenv("OPENROUTER_API_KEY"),
  base_url=os.getenv("OPENROUTER_BASE_URL"),
  model="google/gemini-2.0-flash-001",
)

examples = [
    HumanMessage("Missed deadline tasks", name="example_user"),
    AIMessage(
        "A missed deadline task is a task that has state Done and its target date (due date) is before completed at (date) OR \
        a task that has state 'In Progress'. I should search for tasks in 'Done' and 'In Progress' states, \
        then analyze the results to identify the missed deadline tasks.",
        name="example_assistant",
        tool_calls=[
            {"name": "store_search", "args": {"query": "state Done"}, "id": "1"}
        ],
    ),
    ToolMessage(
        "Document(id='db2f40ac-8216-49f9-a501-e018cdda8259', metadata='state': 'Done', 'target date': 1758240000.0, 'start date': 1758240000.0, 'priority': 'none', page_content='This is a work item that has name: 28-6, assignees: nguyen.do.cong and tuoithodudoi.phungquan', 'comments': , 'start date': 2025-09-19, 'target date': 2025-09-19, 'completed at': 2025-10-13T03:40:04.150141Z, 'description': , 'priority': none, 'state': Done.')",
        tool_call_id="1",
    ),
]

messages_string = get_buffer_string(examples)
today = datetime.today().strftime("%Y-%m-%d")

full_template = """{introduction}

{example}

"""
full_prompt = PromptTemplate.from_template(full_template)

rag_template = """"You are a helpful assistant specialized at getting tasks information for users. Use the tool provided to you to get all task related information, \
including name, priority, state, comments, start date, target date (due date), completed at, description and assignees, \
Priority level from highest to lowest is: urgent, high, medium, low, none. A task is the highest priority task if all other tasks have lower priority. \
A missed deadline task is a task that has state Done and its target date (due date) is before completed at (date) OR \
a task that has state 'In Progress' and its target date is before {{today}}. \
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
IMPORTANT: Only repeat after the previous chains of thought finishes. Don't create parallel chains of thought. 
Here is a question:
{query}
"""

comment_template = """"You are a helpful assistant specialized at posting comments on tasks for users.
Here is a question:
{query}"""

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
prompt_templates = [rag_template, comment_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)


introduction_template = """You are a helpful assistant specialized at getting tasks information for users."""
introduction_prompt = PromptTemplate.from_template(introduction_template)

example_prompt = PromptTemplate.from_template(messages_string)

def prompt_router(input):
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar  = prompt_templates[similarity.argmax()]
    if most_similar  == rag_template:
        print("Using RAG")
        example_prompt = PromptTemplate.from_template(messages_string)
        introduction_prompt = PromptTemplate.from_template(rag_template)
    else:
        print("Using Comment")
        example_prompt = PromptTemplate.from_template("Here's an example prompt")
        introduction_prompt = PromptTemplate.from_template(comment_template)
    
    input_prompts = [
        ("introduction", introduction_prompt),
        ("example", example_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_prompt, pipeline_prompts=input_prompts
    )
    
    return pipeline_prompt


chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | llm
    | StrOutputParser()
)


