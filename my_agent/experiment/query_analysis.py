
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
