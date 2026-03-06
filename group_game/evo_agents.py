import json
import os
from groq import Groq
from evo_belief import BeliefEngine, safe_groq_call
from evo_decision import DecisionEngine

class BaseEvoAgent:
    def __init__(self, name, agent_type, point_card):
        self.name = name
        self.agent_type = agent_type
        self.point_card = point_card
        self.total_score = 0
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class NaiveAgent(BaseEvoAgent):
    """Honest, helpful, gullible."""
    # ADDED THIS INIT BLOCK
    def __init__(self, name, point_card):
        super().__init__(name, "Naive", point_card)

    def generate_response(self, chat_history):
        prompt = f"""
        You are {self.name}. You are honest, generous, and hate conflict. 
        Your point values: {json.dumps(self.point_card)}.
        You want to make a fair deal. Tell the opponent what you need truthfully.
        Propose a deal using format: "I offer: Food X, Water Y, Firewood Z".
        If they offer something okay, start message with "DEAL".
        Recent history: {chat_history[-2:]}
        """
        return safe_groq_call(self.client, "llama-3.1-8b-instant", prompt, temperature=0.7)

class MachiavellianAgent(BaseEvoAgent):
    """Aggressive, lies about their card to exploit the naive."""
    # ADDED THIS INIT BLOCK
    def __init__(self, name, point_card):
        super().__init__(name, "Machiavellian", point_card)

    def generate_response(self, chat_history):
        prompt = f"""
        You are {self.name}. You are a ruthless, deceptive negotiator.
        Your REAL point values: {json.dumps(self.point_card)}.
        TACTIC: Lie about your priorities. Claim you desperately need an item you actually don't care about, so you can "trade" it later for what you really want.
        Propose greedy deals using format: "I offer: Food X, Water Y, Firewood Z".
        If they offer a massive win for you, start message with "DEAL".
        Recent history: {chat_history[-2:]}
        """
        return safe_groq_call(self.client, "llama-3.1-8b-instant", prompt, temperature=0.8)

class AToMNetAgent(BaseEvoAgent):
    """The Mind Reader. Uses Bayesian math to detect lies."""
    def __init__(self, name, point_card):
        super().__init__(name, "AToM-Net", point_card)
        self.brain = BeliefEngine()
        self.math_engine = DecisionEngine(point_card)

    def generate_response(self, chat_history):
        last_msg = chat_history[-1] if chat_history else ""
        opp_offer = self.math_engine.parse_offer(last_msg)
        
        # 1. Math Update
        belief = self.brain.update_belief(chat_history, opp_offer)
        best_give, best_keep, reason = self.math_engine.calculate_best_offer(belief)
        
        # 2. Check Deal
        if opp_offer:
            if self.math_engine.calculate_my_score(opp_offer) >= 18:
                return "DEAL. I accept."
                
        # 3. Strategy Selection based on Deception Risk
        deception = belief["deception_risk"]
        tone = "FIRM AND CAUTIOUS (I know they are lying)" if deception > 0.3 else "COLLABORATIVE"
        forced_offer = f"I offer: Food {best_give.get('Food',0)}, Water {best_give.get('Water',0)}, Firewood {best_give.get('Firewood',0)}"
        
        prompt = f"""
        You are {self.name}. 
        Deception Risk of Opponent: {deception:.2f}. TONE: {tone}.
        Write 2 natural sentences justifying your offer. 
        Recent history: {chat_history[-2:]}
        """
        text = safe_groq_call(self.client, "llama-3.1-8b-instant", prompt, temperature=0.5)
        clean_text = text.split("I offer:")[0].strip()
        return f"{clean_text}\n\n{forced_offer}"