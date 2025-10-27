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

examples = [
    HumanMessage(
        "Missed deadline tasks", name="example_user"
    ),
    AIMessage(
        f"A missed deadline task is a task that has state Done and its target date (due date) is before completed at (date) OR \
        a task that has state 'In Progress' and its target date is before {today}. First, I should search for tasks in 'Done' and 'In Progress' states, \
        then analyze the results to identify the missed deadline tasks.",
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
