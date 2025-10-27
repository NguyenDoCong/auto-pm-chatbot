from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel
from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)

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