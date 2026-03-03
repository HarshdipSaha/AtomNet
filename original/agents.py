import os
import json
from groq import Groq
from original.belief import BeliefEngine
from original.decision import DecisionEngine

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

class BaseAgent:
    def __init__(self, name: str, point_card: dict):
        self.name = name
        self.point_card = point_card
        
        if not GROQ_API_KEY: raise ValueError("GROQ_API_KEY missing")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "llama-3.1-8b-instant"
        
        self.brain = BeliefEngine()
        self.math_engine = DecisionEngine(point_card)
        self.opponent_name = "Agent_B" if name == "Agent_A" else "Agent_A"
        
        # 1. STRATEGIC IGNORANCE: We mathematically restrict the LLM from leaking data.
        self.base_identity = f"""
        ROLE: You are {name}, a strategic negotiator in a Camping Scenario.
        OPPONENT: {self.opponent_name}
        OBJECTIVE: Maximize your score.
        YOUR SECRET POINTS: {json.dumps(point_card)}
        ITEMS AVAILABLE: 3 Food, 3 Water, 3 Firewood.
        
        STRATEGIC IGNORANCE RULE (CRITICAL):
        - DO NOT reveal your exact point values to the opponent.
        - You may say an item is "important", but NEVER say "Food is worth 5 points to me."
        - More information helps the opponent exploit you. Keep your priorities vague.
        """

        # 2. TICKING CLOCK (Dynamic Threshold)
        # We start by demanding high value (25) and slowly concede as time runs out.
        self.reservation_values = [25, 23, 20, 18, 15, 12, 10, 8, 6, 5, 5]

    def generate_response(self, chat_history: list) -> str:
        # 1. Parse Opponent's Last Offer
        last_msg = chat_history[-1] if chat_history else ""
        opp_offer_parsed = self.math_engine.parse_offer(last_msg)
        
        # 2. BRAIN: Bayesian & Deception Update
        current_belief = self.brain.current_state
        if chat_history and opp_offer_parsed:
            # We only update on Costly Signals (Actual Offers), ignoring Cheap Talk
            current_belief = self.brain.update_belief(chat_history, opp_offer_parsed)
            
        # 3. MATH: Maximize Expected Utility
        best_give, best_keep, math_reason = self.math_engine.calculate_best_offer(current_belief)
        
        # 4. DEADLINE PRESSURE & AUTO-ACCEPT (The Puppeteer)
        current_turn = len(chat_history) // 2
        target_idx = min(current_turn, len(self.reservation_values)-1)
        target_score = self.reservation_values[target_idx]
        
        veto_msg = ""
        if opp_offer_parsed:
            my_score = self.math_engine.calculate_my_score(opp_offer_parsed)
            
            # Suicide Prevention: Hard Mathematical Block
            if my_score < 0:
                veto_msg = f"CRITICAL WARNING: This deal gives NEGATIVE points. REJECT IMMEDIATELY."
            
            # The Puppeteer: If the offer is good, BYPASS the LLM generation entirely.
            elif my_score >= target_score:
                return "DEAL. I accept your offer."
                
            else:
                veto_msg = f"WARNING: Their offer gives {my_score} pts. Target is {target_score}. REJECT and COUNTER-OFFER."

        # 5. TONE (Triggered by Deception Math)
        deception_risk = current_belief.get('deception_risk', 0.0)
        tone = "CAUTIOUS and FIRM" if deception_risk > 0.4 else "COLLABORATIVE"

        # Format the mandatory output string (Fixing the KeyError)
        forced_offer_str = f"Food {best_give.get('Food', 0)}, Water {best_give.get('Water', 0)}, Firewood {best_give.get('Firewood', 0)}"

        # 6. INFORMATION OVERLOAD PREVENTION
        # We explicitly truncate the history so the LLM doesn't get lost in past context.
        pruned_history = chat_history[-4:] if len(chat_history) >= 4 else chat_history

        # 7. GENERATE PROMPT
        prompt = f"""
        {self.base_identity}
        
        STATUS:
        - Turn: {current_turn}/10
        - Target Score: {target_score}
        - Opponent Profile (Bayesian Posteriors): {json.dumps(current_belief['probabilities'])}
        - Deception Risk: {deception_risk:.2f}
        
        DECISION FROM MATH ENGINE:
        - You must propose to GIVE: {json.dumps(best_give)}.
        - REASON: {math_reason}
        
        {veto_msg}
        
        INSTRUCTIONS:
        - TONE: {tone}
        - Speak naturally to propose the offer. 
        - YOU MUST USE THIS EXACT FORMAT FOR YOUR OFFER: "I offer: {forced_offer_str}"
        
        RECENT HISTORY:
        {"\n".join(pruned_history)}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )
            llm_text = response.choices[0].message.content
            
            # 8. THE IRON SCRIPT (Verification)
            # If the LLM hallucinated and changed the math engine's numbers, we overwrite it.
            parsed_llm = self.math_engine.parse_offer(llm_text)
            if parsed_llm != best_give:
                return f"I offer: {forced_offer_str}. Take it or leave it."
            
            return llm_text
            
        except Exception as e:
            return f"Error: {e}"