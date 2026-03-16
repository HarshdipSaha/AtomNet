import os
import json
import time
from groq import Groq

def safe_groq_call(client, model, prompt, temperature=0.0):
    for attempt in range(5):
        try:
            messages = [{"role": "user", "content": prompt}]
            kwargs = {"model": model, "messages": messages, "temperature": temperature, "response_format": {"type": "json_object"}}
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception:
            time.sleep(2)
    return '{"Generators": 0.5, "ICU_Beds": 0.5, "Vaccines": 0.5}'

class TriageBeliefEngine:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        self.items = ["Generators", "ICU_Beds", "Vaccines"]
        
        # Bayesian Priors: We don't know what the hospital actually needs most
        self.current_state = {
            "probabilities": {item: {"Critical": 0.5, "Low": 0.5} for item in self.items},
            "hoarding_risk": 0.0,
            "belief_honesty": 0.90
        }

    def _extract_cheap_talk(self, last_msg: str) -> dict:
        """Extracts claimed triage needs from natural language."""
        prompt = f"""
        Analyze the hospital's message: "{last_msg}"
        What do they CLAIM is their critical need? Output valid JSON: {{"Generators": 1.0, "ICU_Beds": 0.0, "Vaccines": 0.5}}
        1.0 = Claims critical life-or-death need. 0.0 = Willing to give it up. 0.5 = Unmentioned.
        """
        raw = safe_groq_call(self.client, self.model, prompt)
        try: return json.loads(raw)
        except: return {item: 0.5 for item in self.items}

    def validate_economic_consistency(self, stated_needs: dict, actual_offer: dict):
        """
        Stage 2: Economic Validator
        If they claim a critical need for Vaccines (1.0) but offer to give us 2 Vaccines,
        or if they claim they don't need ICU beds (0.0) but demand all of them, that's hoarding.
        """
        total_contradiction = 0.0
        
        for item in self.items:
            claimed_need = stated_needs.get(item, 0.5)
            # actual_offer is what the hospital KEEPS
            kept_amount = actual_offer.get(item, 1.5) 
            revealed_need = kept_amount / 3.0 # Normalize 0-3 to 0.0-1.0
            
            # Mathematical L1 Distance (Severity)
            contradiction = abs(claimed_need - revealed_need)
            total_contradiction += contradiction
            
        severity = min(1.0, total_contradiction / len(self.items))
        
        # Dynamic Bayesian Update
        if severity > 0.2: # Noise threshold
            self.current_state["belief_honesty"] *= (1.0 - severity)
            beta = 0.7
            self.current_state["hoarding_risk"] = (beta * self.current_state["hoarding_risk"]) + ((1 - beta) * severity)

        return severity