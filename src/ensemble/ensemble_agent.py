from langgraph.graph import StateGraph, START, END
from src.utils.nodes import *


def initialize_pattern_expert():
    """Initialize the ensemble agent with multiple models."""

    workflow = StateGraph(PatternExpertState)
    workflow.add_node("ensemble_prediction", ensemble_prediction)
    workflow.add_node("analyze_patterns", analyze_pattern_transaction)
    
    workflow.add_edge(START, "ensemble_prediction")
    workflow.add_edge("ensemble_prediction", "analyze_patterns")
    workflow.add_edge("analyze_patterns", END)
    
    pattern_expert = workflow.compile()
    return pattern_expert


