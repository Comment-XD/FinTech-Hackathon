from langgraph.graph import StateGraph, START, END
from src.utils.nodes import *
from src.utils.states import FinalAssessmentState

def initialize_decider_agent():
    workflow = StateGraph(FinalAssessmentState)
    workflow.add_node("determine_risk_assessment", determine_risk_assessment)
    workflow.add_node("fraud_decider", fraud_detection_decider)
    
    workflow.add_edge(START, "determine_risk_assessment")
    workflow.add_edge("determine_risk_assessment", "fraud_decider")
    workflow.add_edge("fraud_decider", END)
    
    decider = workflow.compile()
    
    return decider