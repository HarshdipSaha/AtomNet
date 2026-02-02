import os
import json
from typing import List
from groq import Groq

class BeliefEngine:
    def __init__(self):
        # Configure Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found")
            
        self.client = Groq(api_key=api_key)
        # LIGHTWEIGHT MODEL to avoid Rate Limits
        self.model = "llama-3.1-8b-instant" 
        
        self.current_state = {
            "books": {"probability_high": 0.5, "probability_low": 0.5, "reasoning": "Start"},
            "hats": {"probability_high": 0.5, "probability_low": 0.5, "reasoning": "Start"},
            "balls": {"probability_high": 0.5, "probability_low": 0.5, "reasoning": "Start"},
            "current_strategy": "Unknown",
            "deception_risk": 0.0,
            "predicted_next_move": "Wait"
        }

    def update_belief(self, chat_history: List[str], last_offer_details: str = "") -> dict:
        if not chat_history: return self.current_state

        prompt = f"""
        You are the Theory of Mind Engine.
        TASK: Update Opponent Profile based on LATEST TURN.
        
        PREVIOUS BELIEF:
        {json.dumps(self.current_state, indent=2)}
        
        LOGIC RULES:
        1. If they REJECT an offer for X, but previously said they need X -> deception_risk UP.
        2. High Value = they refuse to give it. Low Value = they offer it.
        
        LAST OFFER: {last_offer_details}
        CHAT LOG:
        {"\n".join(chat_history[-3:])}
        
        Output valid JSON only.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            self.current_state = json.loads(response.choices[0].message.content)
            return self.current_state
        except Exception as e:
            print(f"Belief Crash: {e}")
            return self.current_state
