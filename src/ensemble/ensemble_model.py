import pickle
import joblib
from typing import List

from src.ensemble.helper.metrics import sigmoid
from src.ensemble.helper.preprocess import preprocess_transform_pipeline


class Ensemble:
    def __init__(self):
        
        self.models = {
            "xgboost": joblib.load("src/ensemble/models/xgb_cifer.pkl"),
            "lightgbm": joblib.load("src/ensemble/models/lgb_cifer.pkl")
            # "tabnet": joblib.load("src/ensemble/models/tabnet_model.joblib"),
        }
        
        self.decision_weights = {
            "xgboost": 0.2,
            "lightgbm": 0.3,
            # "tabnet": 0.5,
        }
    
    def predict(self, data) -> List[float]:
        """Generate ensemble predictions from multiple models."""
        
        model_probs = {}
        data = preprocess_transform_pipeline(data)
        
        for model_name, model in self.models.items():
            if model_name in ["xgboost", "lightgbm"]:
                logits = model.predict(data)
                probs = [sigmoid(logit) for logit in logits]
            # else:
            #     probs = model.predict_proba(data)[:, 1].tolist()
            
            model_probs[model_name] = probs
        
        # Combine probabilities using weighted average
        final_probs = []
        num_samples = len(data)
        
        for i in range(num_samples):
            combined_prob = sum(
                model_probs[model_name][i] * self.decision_weights[model_name]
                for model_name in self.models.keys()
            )
            final_probs.append(combined_prob)
        
        return final_probs