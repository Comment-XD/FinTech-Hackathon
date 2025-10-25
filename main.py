from src.semantic_agent import initialize_semantic_expert
from src.ensemble.ensemble_agent import initialize_pattern_expert
from src.decider_agent import initialize_decider_agent

def semantic_pattern_adverserial_analysis(new_tx):
    decider_agent = initialize_decider_agent()
    semantic_expert = initialize_semantic_expert()
    pattern_expert = initialize_pattern_expert()
    
    semantic_result = semantic_expert.invoke({"user_input": new_tx})
    pattern_result = pattern_expert.invoke({"user_input": new_tx})
    
    final_assessment = decider_agent.invoke({
        "semantic_risk_score": semantic_result["risk_score"],
        "pattern_risk_score": pattern_result["risk_score"],
        "semantic_analysis": semantic_result["analysis"],
        "pattern_analysis": pattern_result["analysis"]
    })
    
    return final_assessment
    
if __name__ == "__main__":
    
    new_tx = {
        "step": 1,
        "nameOrig": "sdv-pii-iwo48",
        "type": "TRANSFER",
        "amount": 8500.0,
        "oldbalanceOrg": 9000.0,
        "newbalanceOrig": 500.0,
        "nameDest": "C987654321",
        "oldbalanceDest": 10000.0,
        "newbalanceDest": 18500.0
    }

    # Build agent pipeline
    result = semantic_pattern_adverserial_analysis(new_tx)
    print(result)
    
    
    
    