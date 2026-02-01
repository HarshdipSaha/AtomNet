import os
import json
from groq import Groq
from belief import BeliefEngine
from decision import DecisionEngine

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

class BaseAgent:
    def __init__(self, name: str, point_card: dict):
        self.name = name
        self.point_card = point_card
        
        if not GROQ_API_KEY: raise ValueError("GROQ_API_KEY missing")
        self.client = Groq(api_key=GROQ_API_KEY)
        # Use 8B model to be safe on limits
        self.model = "llama-3.1-8b-instant"
        
        self.brain = BeliefEngine()
        self.math_engine = DecisionEngine(point_card)
        
        self.base_identity = f"""
        You are {name}, expert negotiator.
        OBJECTIVE: Maximize score.
        POINTS: {json.dumps(point_card)}
        ITEMS: 3 Books, 2 Hats, 1 Ball.
        """

# ... inside generate_response ...

    def generate_response(self, chat_history: list) -> str:
        # 1. BRAIN
        current_belief = self.brain.current_state
        if chat_history:
            current_belief = self.brain.update_belief(chat_history)
            
        # 2. MATH (Get BOTH dictionaries)
        # CHANGE THIS LINE:
        best_give, best_keep, math_reason = self.math_engine.calculate_best_offer(current_belief)
        
        # 3. RATIONALITY VETO (No changes needed here)
        last_msg = chat_history[-1] if chat_history else ""
        opp_offer_parsed = self.math_engine.parse_offer(last_msg)
        veto_msg = ""
        
        if opp_offer_parsed:
            my_score = self.math_engine.calculate_my_score(opp_offer_parsed)
            if my_score < 5:
                veto_msg = f"""
                !!! RATIONALITY CHECK !!!
                The opponent's offer gives you only {my_score} points.
                REJECT IT. Say: "I need a better deal."
                """

        # 4. TONE
        deception_risk = current_belief.get('deception_risk', 0.0)
        tone = "CAUTIOUS" if deception_risk > 0.4 else "COLLABORATIVE"

        # 5. GENERATE PROMPT (The Fix)
        prompt = f"""
        {self.base_identity}
        
        STATUS:
        - Opponent Profile: {json.dumps(current_belief)}
        - STRATEGY: Offer to GIVE {json.dumps(best_give)}.
        - REASON: {math_reason}
        
        {veto_msg}
        
        INSTRUCTIONS:
        - TONE: {tone}
        - Use Natural Language to propose the 'GIVE' offer.
        - CRITICAL: If you type 'DEAL:', you must list the items YOU KEEP.
        - DEAL FORMAT: "DEAL: books {best_keep['books']}, hats {best_keep['hats']}, balls {best_keep['balls']}"
        
        HISTORY:
        {"\n".join(chat_history)}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"
