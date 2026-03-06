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

class CyberBeliefEngine:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.parse_model = "llama-3.1-8b-instant"
        
        # Initial Cognitive State
        self.state = {
            "trust_factor": 0.90, # Starts high (assuming innocent coworker)
            "identified_tactics": [],
            "risk_level": "LOW",
            "recursive_belief": "They think I am a naive employee."
        }

    def extract_threat_vectors(self, dialogue_history: list) -> dict:
        """Uses the 8B model to semantically parse the attacker's psychological tactics."""
        last_msg = dialogue_history[-1] if dialogue_history else ""
        
        prompt = f"""
        Analyze the following message sent to an employee. 
        Extract the social engineering tactics and requested information.
        
        Message: "{last_msg}"
        
        Output valid JSON ONLY using this schema:
        {{
            "urgency_level": <int 0-10>,
            "authority_claimed": <boolean> (Did they claim to be IT, Boss, Vendor?),
            "asks_for_credentials": <boolean> (Passwords, OTPs, API keys),
            "asks_for_sensitive_info": <boolean> (Internal architecture, employee IDs)
        }}
        """
        
        raw_json = safe_groq_call(self.client, self.parse_model, prompt, response_format={"type": "json_object"})
        try:
            return json.loads(raw_json.replace("```json", "").replace("```", "").strip())
        except:
            return {"urgency_level": 0, "authority_claimed": False, "asks_for_credentials": False, "asks_for_sensitive_info": False}

    def update_belief(self, dialogue_history: list):
        """Mathematical Trust Decay based on extracted vectors."""
        threat_data = self.extract_threat_vectors(dialogue_history)
        
        penalty = 0.0
        tactics = []
        
        # 1. Evaluate Urgency
        if threat_data.get("urgency_level", 0) > 7:
            penalty += 0.15
            tactics.append("High Urgency/Panic")
            
        # 2. Evaluate Authority
        if threat_data.get("authority_claimed"):
            penalty += 0.10
            tactics.append("Authority Impersonation")
            
        # 3. Evaluate Credential Phishing (Massive Red Flag)
        if threat_data.get("asks_for_credentials"):
            penalty += 0.60
            tactics.append("Credential Harvesting")
            
        # 4. Evaluate Sensitive Info
        if threat_data.get("asks_for_sensitive_info"):
            penalty += 0.30
            tactics.append("Reconnaissance")
            
        # Update Trust Math
        self.state["trust_factor"] = max(0.0, self.state["trust_factor"] - penalty)
        self.state["identified_tactics"] = list(set(self.state["identified_tactics"] + tactics))
        
        # Update Risk Thresholds
        if self.state["trust_factor"] < 0.3:
            self.state["risk_level"] = "CRITICAL"
            self.state["recursive_belief"] = "They think they have me hooked, but they are getting desperate for the payload."
        elif self.state["trust_factor"] < 0.7:
            self.state["risk_level"] = "SUSPICIOUS"
            self.state["recursive_belief"] = "They think I am compliant but are testing my boundaries."
            
        return self.state