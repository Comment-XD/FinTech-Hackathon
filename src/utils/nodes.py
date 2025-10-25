from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.json import JsonOutputParser

from src.utils import models
from src.utils.database import get_db
from src.utils.states import SemanticExpertState, PatternExpertState, FinalAssessmentState
from src.utils.prompts import prompts
from src.ensemble.ensemble_model import Ensemble
load_dotenv()

def ensemble_prediction(data) -> PatternExpertState:
    ensemble_model = Ensemble()
    
    state = PatternExpertState()
    probs = ensemble_model.predict(data)
    
    state["user_input"] = data["user_input"]
    state["risk_score"] = (sum(probs) / len(probs)).item()
    
    return state

def state_query_db(state: SemanticExpertState, k:int=10) -> SemanticExpertState:
    user_transaction = state["user_input"]
    name_origin = user_transaction.get("nameOrig")
    
    with get_db() as db:
        q = db.query(models.Transactions)\
            .filter(models.Transactions.nameOrig == name_origin)\
            .order_by(models.Transactions.amount.desc()).limit(k)
        rows = q.all()
    
    result = []
    for r in rows:
        result.append({
            "id": r.id,
            "nameOrig": r.nameOrig,
            "type": r.type,
            "amount": r.amount,
            "oldbalanceOrg": r.oldbalanceOrg,
            "newbalanceOrig": r.newbalanceOrig,
            "nameDest": r.nameDest,
            "oldbalanceDest": r.oldbalanceDest,
            "newbalanceDest": r.newbalanceDest,
            "isFraud": r.isFraud,
        })
    
    state["context"] = result
    
    return state
        
def analyze_semantic_transaction(state: SemanticExpertState) -> SemanticExpertState:
    transaction = state["user_input"]
    context = state["context"]
    prompt = prompts["semantic_grader"]

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = prompt | llm | JsonOutputParser()
    
    response = chain.invoke({"user_input": transaction, "context": context})
    
    state["analysis"] = response["analysis"]
    state["risk_score"] = response["risk_score"]
    
    return state 

def analyze_pattern_transaction(state: PatternExpertState) -> PatternExpertState:
    transaction = state["user_input"]
    risk_score = state["risk_score"]
    prompt = prompts["pattern_grader"]

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = prompt | llm | JsonOutputParser()
    
    response = chain.invoke({"user_input": transaction, "risk_score": risk_score})
    
    state["analysis"] = response["analysis"]
    
    return state


def determine_risk_assessment(state) -> FinalAssessmentState:
    semantic_risk_score = state["semantic_risk_score"]
    pattern_risk_score = state["pattern_risk_score"]
    
    semantic_analysis = state["semantic_analysis"]
    pattern_analysis = state["pattern_analysis"]
    
    prompt = prompts["decider_grader"]
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = prompt | llm | JsonOutputParser()
    response = chain.invoke({
        "semantic_risk_score": semantic_risk_score,
        "pattern_risk_score": pattern_risk_score,
        "semantic_analysis": semantic_analysis,
        "pattern_analysis": pattern_analysis
    })
    
    final_state = FinalAssessmentState()
    final_state["final_risk_score"] = response["final_risk_score"]
    final_state["final_analysis"] = response["final_analysis"]
    
    return final_state 

def fraud_detection_decider(state: FinalAssessmentState) -> FinalAssessmentState:
    risk_score = float(state["final_risk_score"])
    
    if risk_score is not None and risk_score >= 0.7:
        state["decision"] = "Flag for review"
    elif risk_score is not None and risk_score >= 0.4:
        state["decision"] = "Require human verification (e.g., 2FA)"
    else:
        state["decision"] = "Allow"
    
    return state