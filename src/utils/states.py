from typing import Optional, Dict, Any, TypedDict, List

class SemanticExpertState(TypedDict):
    user_input: dict[str, Any]
    context: Optional[List[Dict[str, Any]]]
    risk_score: Optional[float]
    analysis: Optional[str]
    
class PatternExpertState(TypedDict):
    user_input: dict[str, Any]
    context: Optional[Dict[str, Any]]
    risk_score: Optional[float]
    analysis: Optional[str]
    
class FinalAssessmentState(TypedDict):
    semantic_risk_score: float
    pattern_risk_score: float
    semantic_analysis: str
    pattern_analysis: str
    final_risk_score: Optional[float]
    final_analysis: Optional[str]