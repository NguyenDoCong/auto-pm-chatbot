from typing import List, Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship, DeclarativeBase
from sqlalchemy import ForeignKey, String, insert, create_engine, text, MetaData, select
from my_agent.data.table_union_plane import unite_tables
import pprint
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

load_dotenv()
os.getenv("GOOGLE_API_KEY")
os.getenv("LANGSMITH_API_KEY")
X_API_KEY = os.getenv("X_API_KEY")

llm = ChatOpenAI(
  api_key=os.getenv("OPENROUTER_API_KEY"),
  base_url=os.getenv("OPENROUTER_BASE_URL"),
  model="google/gemini-2.0-flash-001",
)

metadata_obj = MetaData()

engine = create_engine(
    "sqlite:///myfile.db", connect_args={"autocommit": False}
)

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "tasks"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))
    priority: Mapped[Optional[str]]
    start_date: Mapped[Optional[str]]
    target_date: Mapped[Optional[str]]
    completed_at: Mapped[Optional[str]]
    description_stripped: Mapped[Optional[str]]
    # assignees: Mapped[Optional[list[str]]]
    state: Mapped[Optional[str]]
    comments: Mapped[Optional[str]]
    
    # addresses: Mapped[List["Address"]] = relationship(back_populates="user")
    # def __repr__(self) -> str:
    #     return f"User(id={self.id!r}, name={self.name!r}, fullname={self.fullname!r})"

Base.metadata.create_all(engine)

data = unite_tables()

with engine.connect() as conn:

    for item in data:
        # print(item)

        stmt = insert(User).values(name=item["name"], priority=item["priority"], 
        start_date=item["start_date"], target_date=item["target_date"], 
        completed_at=item["completed_at"], description_stripped=item["description"])

        result = conn.execute(stmt)
        conn.commit()


db = SQLDatabase.from_uri("sqlite:///myfile.db", sample_rows_in_table_info=3)

chain = create_sql_query_chain(llm, db)
# chain.get_prompts()[0].pretty_print()

context = db.get_context()
# print(list(context))
# print(context["table_info"])

prompt_with_context = chain.get_prompts()[0].partial(table_info=context["table_info"])
# print(prompt_with_context.pretty_repr()[:1500])

examples = [
    {
        "input": "List all tasks.", 
        "query": "SELECT * FROM tasks;"},
    {
        "input": "Find all tasks with priority high.",
        "query": "SELECT * FROM tasks WHERE priority = 'high';",
    },
    {
        "input": "Find all tasks with priority low and state is Todo.",
        "query": "SELECT * FROM tasks WHERE priority = 'low' AND state = 'Todo';",
    },
    {
        "input": "Tasks not in state Todo.",
        "query": "SELECT * FROM tasks WHERE NOT state = 'Todo';",
    },
    {
        "input": "How many tasks are there",
        "query": 'SELECT COUNT(*) FROM "tasks";',
    },
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    Chroma,
    k=5,
    input_keys=["input"],
)

example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")
prompt = FewShotPromptTemplate(
    examples=examples[:5],
    example_prompt=example_prompt,
    prefix="You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "top_k", "table_info"],
)

# print(prompt.format(input="How many tasks are there?", top_k=3, table_info="foo"))
# pprint.pprint(example_selector.select_examples({"input": "how many tasks are there?"}))
chain = create_sql_query_chain(llm, db, prompt)
query = chain.invoke({"question": "List all tasks?"})

print(db.run(query))