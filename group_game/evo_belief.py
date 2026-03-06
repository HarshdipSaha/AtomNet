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
            if "429" in str(e).lower() or "rate" in str(e).lower():
                match = re.search(r'try again in (\d+\.?\d*)s', str(e))
                wait_time = float(match.group(1)) + 2.0 if match else 10.0
                time.sleep(wait_time)
            else: return "{}"
    return "{}"

class BeliefEngine:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        self.items = ["Food", "Water", "Firewood"]
        self.current_state = {
            "probabilities": {item: {"High": 0.5, "Low": 0.5} for item in self.items},
            "deception_risk": 0.0
        }
        self.likelihoods = {
            "offer_generous": {"High": 0.2, "Low": 0.8},
            "offer_stingy":   {"High": 0.8, "Low": 0.2},
            "offer_neutral":  {"High": 0.5, "Low": 0.5}
        }

    def _extract_cheap_talk(self, last_msg: str) -> dict:
        prompt = f"""
        Analyze the opponent's message: "{last_msg}"
        What do they CLAIM to value? Output valid JSON: {{"Food": 1.0, "Water": 0.0, "Firewood": 0.5}}
        1.0 = Claims they want it badly. 0.0 = Willing to give it away. 0.5 = Neutral/Unmentioned.
        """
        raw = safe_groq_call(self.client, self.model, prompt, response_format={"type": "json_object"})
        try: return json.loads(raw)
        except: return {item: 0.5 for item in self.items}

    def _apply_bayes(self, item: str, action: str):
        if item not in self.current_state["probabilities"]: return
        prior_h = self.current_state["probabilities"][item]["High"]
        prior_l = self.current_state["probabilities"][item]["Low"]
        p_act_h = self.likelihoods[action]["High"]
        p_act_l = self.likelihoods[action]["Low"]
        
        p_act = (p_act_h * prior_h) + (p_act_l * prior_l)
        if p_act == 0: return
        self.current_state["probabilities"][item]["High"] = (p_act_h * prior_h) / p_act
        self.current_state["probabilities"][item]["Low"] = (p_act_l * prior_l) / p_act

    def update_belief(self, chat_history: list, opp_offer: dict) -> dict:
        if not chat_history or not opp_offer: return self.current_state
        last_msg = chat_history[-1]
        stated_prefs = self._extract_cheap_talk(last_msg)
        total_contradiction = 0.0
        
        for item, count in opp_offer.items():
            if item not in self.items: continue
            if count >= 2:
                action, revealed_val = "offer_generous", 0.0
            elif count == 0:
                action, revealed_val = "offer_stingy", 1.0 
            else:
                action, revealed_val = "offer_neutral", 0.5
                
            self._apply_bayes(item, action)
            
            # STAGE 2: ECONOMIC VALIDATOR (Catches Machiavellians)
            stated_val = float(stated_prefs.get(item, 0.5))
            item_contradiction = abs(stated_val - revealed_val)
            total_contradiction += item_contradiction
            
        avg_contradiction = total_contradiction / len(self.items)
        curr_d = self.current_state["deception_risk"]
        self.current_state["deception_risk"] = (0.8 * curr_d) + (0.2 * avg_contradiction)
        return self.current_state