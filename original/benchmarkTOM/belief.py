import os
import json
from typing import List, Dict
from groq import Groq

class BeliefEngine:
    def __init__(self):
        from groq_manager import get_client
        #api_key = os.environ.get("GROQ_API_KEY")
        #if not api_key: raise ValueError("GROQ_API_KEY not found")
        self.client = get_client()
        self.model = "llama-3.1-8b-instant" 
        
        self.items = ["Food", "Water", "Firewood"]
        
        # 1. BAYESIAN PRIORS
        self.current_state = {
            "probabilities": {item: {"High": 0.5, "Low": 0.5} for item in self.items},
            "deception_risk": 0.0
        }
        
        # 2. LIKELIHOODS: P(Action | Type)
        self.likelihoods = {
            "offer_generous": {"High": 0.2, "Low": 0.8}, # Giving >= 2 
            "offer_stingy":   {"High": 0.8, "Low": 0.2}, # Giving 0 
            "offer_neutral":  {"High": 0.5, "Low": 0.5}, # Giving 1
            "reject_generous":{"High": 0.1, "Low": 0.9}  # NEW: Rejection Likelihood
        }
        
        # Hyperparameters for Deception Math
        self.alpha = 0.8
        self.beta = 0.2

    def _extract_cheap_talk(self, last_msg: str) -> dict:
        """Uses LLM strictly to parse Stated Preferences (Words)."""
        prompt = f"""
        Analyze the opponent's LAST message. What do they CLAIM to value?
        Output valid JSON only: {{"Food": 1.0, "Water": 0.0, "Firewood": 0.5}}
        1.0 = Claims they want it badly.
        0.0 = Claims they don't want it / willing to give it.
        0.5 = Didn't mention it / Neutral.
        
        MESSAGE: "{last_msg}"
        """
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(res.choices[0].message.content)
        except:
            return {item: 0.5 for item in self.items}

    def _apply_bayes(self, item: str, action: str):
        """Mathematical Bayesian Update"""
        if item not in self.current_state["probabilities"]:
            return

        prior_high = self.current_state["probabilities"][item]["High"]
        prior_low = self.current_state["probabilities"][item]["Low"]

        p_act_high = self.likelihoods[action]["High"]
        p_act_low = self.likelihoods[action]["Low"]

        p_act = (p_act_high * prior_high) + (p_act_low * prior_low)
        if p_act == 0: return

        self.current_state["probabilities"][item]["High"] = (p_act_high * prior_high) / p_act
        self.current_state["probabilities"][item]["Low"] = (p_act_low * prior_low) / p_act

    def update_on_rejection(self, my_rejected_offer: Dict[str, int]) -> dict:
        """
        Stage 2: Economic Validator (Costly Signal)
        Updates beliefs based on what the opponent REJECTED.
        """
        if not my_rejected_offer:
            return self.current_state

        for item, count in my_rejected_offer.items():
            # If we offered them a lot (>= 2) and they still rejected the deal,
            # it is a strong mathematical signal that they do not value this item highly.
            if count >= 2:
                self._apply_bayes(item, "reject_generous")
                
        return self.current_state

    def update_belief(self, chat_history: List[str], opp_offer: Dict[str, int] = None) -> dict:
        """Updates belief based on what the opponent OFFERED and SAID."""
        if not chat_history or not opp_offer: return self.current_state
        
        last_msg = chat_history[-1]
        stated_prefs = self._extract_cheap_talk(last_msg)
        total_contradiction = 0.0
        
        for item, count in opp_offer.items():
            if item not in self.items: continue

            if count >= 2:
                action = "offer_generous"
                revealed_val = 0.0
            elif count == 0:
                action = "offer_stingy"
                revealed_val = 1.0 
            else:
                action = "offer_neutral"
                revealed_val = 0.5
                
            self._apply_bayes(item, action)
            
            stated_val = float(stated_prefs.get(item, 0.5))
            item_contradiction = abs(stated_val - revealed_val)
            total_contradiction += item_contradiction
            
        avg_contradiction = total_contradiction / len(self.items)
        curr_d = self.current_state["deception_risk"]
        self.current_state["deception_risk"] = (self.alpha * curr_d) + (self.beta * avg_contradiction)
        
        return self.current_state
