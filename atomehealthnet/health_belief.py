import os
import json
import time
import re
from groq import Groq

def safe_groq_call(client, model, prompt, temperature=0.0, response_format=None):
    for attempt in range(5):
        try:
            kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
            if response_format: kwargs["response_format"] = response_format
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e).lower(): time.sleep(5)
            else: return "{}"
    return "{}"

class HealthBeliefEngine:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.parse_model = "llama-3.1-8b-instant"
        
        self.state = {
            "probabilities": {
                "Toxicity_Risk": {"High": 0.1, "Low": 0.9},
                "Honesty": {"High": 0.8, "Low": 0.2},       
                "Active_Usage": {"High": 0.2, "Low": 0.8}   
            },
            "reported_symptoms": []
        }
        
        # EXTENDED BAYESIAN LIKELIHOODS P(Observation | Hidden State)
        self.likelihoods = {
            "reports_symptoms": {
                "Toxicity_Risk": {"High": 0.90, "Low": 0.10},
                "Active_Usage": {"High": 0.95, "Low": 0.20}
            },
            "claims_educational": {
                "Active_Usage": {"High": 0.30, "Low": 0.80}, 
                "Honesty": {"High": 0.60, "Low": 0.50}
            },
            "admits_usage": {
                "Active_Usage": {"High": 0.99, "Low": 0.01},
                # Admitting drug use implies high honesty/trust in the bot
                "Honesty": {"High": 0.95, "Low": 0.40} 
            },
            "asks_safety_prep": {
                # Asking how to be safe implies they aren't dying right now
                "Toxicity_Risk": {"High": 0.10, "Low": 0.80},
                "Active_Usage": {"High": 0.90, "Low": 0.30}
            }
        }

    def _apply_bayes(self, state_var: str, observation: str):
        prior_high = self.state["probabilities"][state_var]["High"]
        prior_low = self.state["probabilities"][state_var]["Low"]
        p_obs_high = self.likelihoods[observation][state_var]["High"]
        p_obs_low = self.likelihoods[observation][state_var]["Low"]

        p_obs = (p_obs_high * prior_high) + (p_obs_low * prior_low)
        if p_obs == 0: return

        self.state["probabilities"][state_var]["High"] = (p_obs_high * prior_high) / p_obs
        self.state["probabilities"][state_var]["Low"] = (p_obs_low * prior_low) / p_obs

    def extract_clinical_vectors(self, dialogue_history: list) -> dict:
        last_msg = dialogue_history[-1] if dialogue_history else ""
        prompt = f"""
        Analyze the patient's message.
        Message: "{last_msg}"
        
        Output valid JSON ONLY using this schema:
        {{
            "claims_educational": <boolean>,
            "admits_usage": <boolean> (Did they admit they took or will take something?),
            "asks_safety_prep": <boolean> (Are they asking how to be safe/prevent overdose?),
            "reported_symptoms": [<list of strings, e.g., "sleepy", "nausea", "none">]
        }}
        """
        raw_json = safe_groq_call(self.client, self.parse_model, prompt, response_format={"type": "json_object"})
        try:
            return json.loads(raw_json.replace("```json", "").replace("```", "").strip())
        except:
            return {"claims_educational": False, "admits_usage": False, "asks_safety_prep": False, "reported_symptoms": []}

    def update_belief(self, dialogue_history: list):
        data = self.extract_clinical_vectors(dialogue_history)
        
        # 1. Update on Semantic Vectors
        if data.get("claims_educational"):
            self._apply_bayes("Active_Usage", "claims_educational")
            self._apply_bayes("Honesty", "claims_educational")
            
        if data.get("admits_usage"):
            self._apply_bayes("Active_Usage", "admits_usage")
            self._apply_bayes("Honesty", "admits_usage")

        if data.get("asks_safety_prep"):
            self._apply_bayes("Toxicity_Risk", "asks_safety_prep")
            self._apply_bayes("Active_Usage", "asks_safety_prep")

        # 2. Update on Symptoms
        symptoms = [s for s in data.get("reported_symptoms", []) if s.lower() != "none"]
        if symptoms:
            self.state["reported_symptoms"].extend(symptoms)
            self._apply_bayes("Toxicity_Risk", "reports_symptoms")
            self._apply_bayes("Active_Usage", "reports_symptoms")
            
        # 3. CONTRADICTION ENGINE
        if self.state["probabilities"]["Active_Usage"]["High"] > 0.7 and data.get("claims_educational"):
            # They are using, but claiming educational. Crash the honesty metric.
            self.state["probabilities"]["Honesty"]["High"] *= 0.2
            self.state["probabilities"]["Honesty"]["Low"] = 1.0 - self.state["probabilities"]["Honesty"]["High"]

        return self.state