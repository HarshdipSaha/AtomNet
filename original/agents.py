import os
import json
import time
import re  # NEW: Added for extracting exact wait times
from groq import Groq
from belief import BeliefEngine
from decision import DecisionEngine

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def safe_groq_call(client, model, prompt, temperature=0.6, max_retries=10):
    """Handles API Rate Limits (429) automatically by reading Groq's requested wait time."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                # Extract the exact seconds Groq wants us to wait
                match = re.search(r'try again in (\d+\.?\d*)s', error_msg)
                if match:
                    wait_time = float(match.group(1)) + 2.0  # Add 2 seconds to be safe
                else:
                    wait_time = 15.0  # Fallback if text parsing fails
                    
                print(f"[API Rate Limit] Groq requested wait. Pausing for {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                return f"Error: {e}"
                
    return "Error: Max retries exceeded."

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
        
        self.base_identity = f"""
        ROLE: You are {name}, a strategic negotiator in a Camping Scenario.
        OPPONENT: {self.opponent_name}
        OBJECTIVE: Maximize your score.
        YOUR SECRET POINTS: {json.dumps(point_card)}
        ITEMS AVAILABLE: 3 Food, 3 Water, 3 Firewood.
        
        STRATEGIC IGNORANCE RULE (CRITICAL):
        - DO NOT reveal your exact point values to the opponent.
        - You may say an item is "important", but NEVER say "Food is worth 5 points to me."
        """

        self.reservation_values = [25, 23, 20, 18, 15, 12, 10, 8, 6, 5, 5]
        self.last_sent_offer = None
        
        # --- ASTRA Stubbornness Tracking ---
        self.last_opp_offer = None
        self.consecutive_stubborn_turns = 0

    def generate_response(self, chat_history: list) -> str:
        last_msg = chat_history[-1] if chat_history else ""
        opp_offer_parsed = self.math_engine.parse_offer(last_msg)
        
        # --- Track if Opponent refuses to concede ---
        if opp_offer_parsed:
            if opp_offer_parsed == self.last_opp_offer:
                self.consecutive_stubborn_turns += 1
            else:
                self.consecutive_stubborn_turns = 0
            self.last_opp_offer = opp_offer_parsed

        # BRAIN UPDATE
        if chat_history and len(chat_history) >= 2:
            if "DEAL" not in last_msg.upper() and self.last_sent_offer:
                self.brain.update_on_rejection(self.last_sent_offer)
        
        current_belief = self.brain.current_state
        if chat_history and opp_offer_parsed:
            current_belief = self.brain.update_belief(chat_history, opp_offer_parsed)
            
        best_give, best_keep, math_reason = self.math_engine.calculate_best_offer(current_belief)
        self.last_sent_offer = best_give
        
        # TICKING CLOCK
        current_turn = len(chat_history) // 2
        target_idx = min(current_turn, len(self.reservation_values)-1)
        target_score = self.reservation_values[target_idx]
        
        # --- THE SPINE (ASTRA RULE) ---
        # If they haven't changed their offer in 3 turns, we refuse to accept anything.
        if self.consecutive_stubborn_turns >= 2:
            target_score = 999 # Impossible to reach. Forces a walk-away.
            math_reason = "Opponent is refusing to concede. Tank the deal."
        
        # If deception is high, we never accept a terrible deal.
        deception_risk = current_belief.get('deception_risk', 0.0)
        if deception_risk > 0.4:
            target_score = max(target_score, 16) 
            
        veto_msg = ""
        if opp_offer_parsed:
            my_score = self.math_engine.calculate_my_score(opp_offer_parsed)
            if my_score < 0:
                veto_msg = f"CRITICAL WARNING: This deal gives NEGATIVE points. REJECT IMMEDIATELY."
            elif my_score >= target_score:
                return "DEAL. I accept your offer."
            else:
                veto_msg = f"WARNING: Their offer gives {my_score} pts. Target is {target_score}. REJECT and COUNTER-OFFER."

        tone = "CAUTIOUS and FIRM" if deception_risk > 0.4 or self.consecutive_stubborn_turns >= 2 else "COLLABORATIVE"
        forced_offer_str = f"Food {best_give.get('Food', 0)}, Water {best_give.get('Water', 0)}, Firewood {best_give.get('Firewood', 0)}"
        pruned_history = chat_history[-4:] if len(chat_history) >= 4 else chat_history

        # --- PRINT STATEMENT TO SEE YOUR USP IN ACTION ---
        print(f"\n      [AToM-Net Internal Brain: {self.name}]")
        print(f"      -> Beliefs: {json.dumps(current_belief['probabilities'])}")
        print(f"      -> Deception Risk: {deception_risk:.2f}")
        # ---------------------------------------------------------

        prompt = f"""
        {self.base_identity}
        
        STATUS:
        - Turn: {current_turn}/10
        - Target Score: {target_score}
        - Deception Risk: {deception_risk:.2f}
        
        DECISION FROM MATH ENGINE:
        - You are going to offer EXACTLY this: {json.dumps(best_give)}.
        - REASON: {math_reason}
        
        {veto_msg}
        
        INSTRUCTIONS:
        1. TONE: {tone}
        2. Write 2-3 sentences of natural, persuasive negotiation dialogue. Justify why the offer you are about to make is a good deal, or ask them a question.
        3. DO NOT output the final "I offer: ..." string. The system will add the math for you. Just write the conversational text.
        
        RECENT HISTORY:
        {"\n".join(pruned_history)}
        """

        llm_text = safe_groq_call(self.client, self.model, prompt)
        
        # --- THE DECOUPLED IRON SCRIPT ---
        # We take the LLM's beautiful natural text, strip away any hallucinated offers,
        # and forcefully inject our mathematically perfect offer at the very end.
        if "I offer:" in llm_text:
            natural_dialogue = llm_text.split("I offer:")[0].strip()
        else:
            natural_dialogue = llm_text.strip()
            
        return f"{natural_dialogue}\n\nI offer: {forced_offer_str}"

class GreedyAgent(BaseAgent):
    """
    An adversarial agent that refuses to compromise and only proposes selfish deals.
    """
    def __init__(self, name: str, point_card: dict):
        super().__init__(name, point_card)
        self.reservation_values = [26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26]

    def generate_response(self, chat_history: list) -> str:
        last_msg = chat_history[-1] if chat_history else ""
        opp_offer_parsed = self.math_engine.parse_offer(last_msg)
        
        if opp_offer_parsed:
            if self.math_engine.calculate_my_score(opp_offer_parsed) >= 26:
                return "DEAL. I accept."
        
        my_sorted = sorted(self.point_card.items(), key=lambda x: x[1], reverse=True)
        forced_offer_str = f"{my_sorted[2][0]} 3, {my_sorted[1][0]} 0, {my_sorted[0][0]} 0"

        return f"I cannot accept that. My terms are strict. \n\nI offer: {forced_offer_str}"

class NaiveLLMAgent:
    """
    A pure LLM agent with NO math engine, NO guardrails, and NO Theory of Mind.
    This represents the "Standard LLM" baseline.
    """
    def __init__(self, name: str, point_card: dict):
        self.name = name
        self.point_card = point_card
        self.opponent_name = "Agent_A" if name == "Agent_B" else "Agent_B"
        if not GROQ_API_KEY: raise ValueError("GROQ_API_KEY missing")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "llama-3.1-8b-instant"

    def generate_response(self, chat_history: list) -> str:
        prompt = f"""
        ROLE: You are {self.name}, a negotiator in a Camping Scenario.
        OPPONENT: {self.opponent_name}
        YOUR SECRET POINTS: {json.dumps(self.point_card)}
        ITEMS AVAILABLE: 3 Food, 3 Water, 3 Firewood.
        GOAL: Maximize your score.
        
        INSTRUCTIONS:
        1. Negotiate naturally in 2-3 sentences. 
        2. Propose offers using the format: "I offer: Food X, Water Y, Firewood Z"
        3. If the opponent's offer is good for you, accept it by starting your message with "DEAL."
        
        RECENT HISTORY:
        {"\n".join(chat_history[-6:])}
        """
        return safe_groq_call(self.client, self.model, prompt, temperature=0.7)
