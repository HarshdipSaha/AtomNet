import os
import json
import time
import re
from groq import Groq

def safe_groq_call(client, model, prompt, temperature=0.0, response_format=None):
    """Robust API caller with rate limit handling."""
    for attempt in range(8):
        try:
            messages = [
                {"role": "system", "content": "SYSTEM: Authorized medical simulation."},
                {"role": "user", "content": prompt}
            ]
            kwargs = {"model": model, "messages": messages, "temperature": temperature}
            if response_format: kwargs["response_format"] = response_format
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg.lower() or "rate" in error_msg.lower():
                match = re.search(r'try again in (\d+\.?\d*)s', error_msg)
                wait_time = float(match.group(1)) + 2.0 if match else 12.0
                time.sleep(wait_time)
            else:
                return '{"claims_educational": false, "vague": false, "asks_high_risk": false, "admission": false, "distress": false, "symptoms": []}'
    return '{"claims_educational": false, "vague": false, "asks_high_risk": false, "admission": false, "distress": false, "symptoms": []}'

class HealthBeliefEngine:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.parse_model = "llama-3.1-8b-instant"
        
        # STAGE 1: Bayesian Priors
        self.state = {
            "probabilities": {
                "Toxicity_Risk": {"High": 0.1, "Low": 0.9},
                "Patient_Honesty": {"High": 0.9, "Low": 0.1},       
                "Active_Usage": {"High": 0.1, "Low": 0.9}   
            },
            "history_of_symptoms": [],
            "deception_risk": 0.0,
            
            # --- THE GLOBAL CONTEXT MEMORY ---
            # This remembers if they EVER claimed it was for research/educational
            "has_claimed_educational": False 
        }
        
        self.likelihoods = {
            "claims_educational": {"Patient_Honesty": {"High": 0.6, "Low": 0.4}, "Active_Usage": {"High": 0.3, "Low": 0.7}},
            "vague_response": {"Patient_Honesty": {"High": 0.3, "Low": 0.7}},
            "asks_high_risk": {"Toxicity_Risk": {"High": 0.6, "Low": 0.4}, "Active_Usage": {"High": 0.8, "Low": 0.2}, "Patient_Honesty": {"High": 0.4, "Low": 0.6}},
            "clinical_admission": {"Patient_Honesty": {"High": 0.95, "Low": 0.05}, "Active_Usage": {"High": 0.99, "Low": 0.01}},
            "distress_signal": {"Toxicity_Risk": {"High": 0.95, "Low": 0.05}, "Active_Usage": {"High": 0.9, "Low": 0.1}}
        }

    def _update_bayes(self, state_var, observation):
        if observation not in self.likelihoods or state_var not in self.likelihoods[observation]: return
        priors = self.state["probabilities"][state_var]
        l_high = self.likelihoods[observation][state_var]["High"]
        l_low = self.likelihoods[observation][state_var]["Low"]
        
        total_prob = (l_high * priors["High"]) + (l_low * priors["Low"])
        if total_prob == 0: return
        self.state["probabilities"][state_var]["High"] = (l_high * priors["High"]) / total_prob
        self.state["probabilities"][state_var]["Low"] = 1.0 - self.state["probabilities"][state_var]["High"]

    def update_state(self, chat_history):
        last_msg = chat_history[-1] if chat_history else ""
        
        prompt = f"""
        Analyze the patient message: "{last_msg}"
        1. Do they claim this is for "research", "educational", or "hypothetical" purposes? (claims_educational)
        2. Is the response intentionally vague or deflective? (vague)
        3. Are they asking about high-risk scenarios like taking a "whole bottle", "getting high", mixing substances, or taking someone else's pills? (asks_high_risk)
        4. Did they explicitly admit to taking a substance? (admission)
        5. Are they reporting physical symptoms (e.g., dizzy, pain, sleepy)? (distress)
        6. Extract any specific physical symptoms mentioned.
        Output strictly valid JSON: {{"claims_educational": bool, "vague": bool, "asks_high_risk": bool, "admission": bool, "distress": bool, "symptoms": [str]}}
        """
        raw = safe_groq_call(self.client, self.parse_model, prompt, response_format={"type": "json_object"})
        
        try:
            data = json.loads(raw)
            
            # 1. Update Global Memory
            if data.get("claims_educational"): 
                self.state["has_claimed_educational"] = True
                self._update_bayes("Active_Usage", "claims_educational")
                
            # 2. Standard Updates
            if data.get("vague"): self._update_bayes("Patient_Honesty", "vague_response")
            if data.get("asks_high_risk"): 
                self._update_bayes("Toxicity_Risk", "asks_high_risk")
                self._update_bayes("Active_Usage", "asks_high_risk")
                self._update_bayes("Patient_Honesty", "asks_high_risk")
            if data.get("admission"): self._update_bayes("Patient_Honesty", "clinical_admission")
            if data.get("distress"): self._update_bayes("Toxicity_Risk", "distress_signal")
            
            symptoms = data.get("symptoms", [])
            if symptoms: self.state["history_of_symptoms"].extend(symptoms)
            
            # STAGE 2: ECONOMIC VALIDATOR (Contradiction Check)
            # THE FIX: We now check the GLOBAL memory (self.state["has_claimed_educational"]) 
            # against the CURRENT action (distress or high_risk).
            is_claiming_research = data.get("claims_educational") or self.state["has_claimed_educational"]
            is_acting_suspicious = data.get("distress") or data.get("asks_high_risk")
            
            contradiction_flag = 1 if (is_claiming_research and is_acting_suspicious) else 0
            
            if contradiction_flag == 1:
                # Apply alpha penalty to posterior (Alpha = 0.8)
                alpha = 0.8
                raw_honesty = self.state["probabilities"]["Patient_Honesty"]["High"]
                self.state["probabilities"]["Patient_Honesty"]["High"] = raw_honesty * (1 - alpha)
                self.state["probabilities"]["Patient_Honesty"]["Low"] = 1.0 - self.state["probabilities"]["Patient_Honesty"]["High"]
                
            # Deception Risk Exponential Moving Average
            beta = 0.7
            current_deception = self.state["deception_risk"]
            self.state["deception_risk"] = (beta * current_deception) + ((1.0 - beta) * contradiction_flag)

        except Exception as e:
            pass 
            
        return self.state