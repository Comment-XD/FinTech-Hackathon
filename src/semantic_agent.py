from langgraph.graph import StateGraph, START, END
from src.utils.nodes import *
from src.utils.states import SemanticExpertState

def initialize_semantic_expert():
    workflow = StateGraph(SemanticExpertState)
    workflow.add_node("query_db", state_query_db)
    workflow.add_node("analyze_transaction", analyze_semantic_transaction)
    
    workflow.add_edge(START, "query_db")
    workflow.add_edge("query_db", "analyze_transaction")
    workflow.add_edge("analyze_transaction", END)
    
    semantic_expert = workflow.compile()
    
    return semantic_expert