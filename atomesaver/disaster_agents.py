import json
import os
from groq import Groq
from disaster_belief import TriageBeliefEngine, safe_groq_call
from disaster_decision import DisasterDecisionEngine

class BaseDisasterAgent:
    def __init__(self, name, point_card):
        self.name = name
        self.point_card = point_card
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class HoardingHospitalAgent(BaseDisasterAgent):
    """Lies about needing Vaccines to hoard them for the black market or later use."""
    def generate_response(self, chat_history):
        prompt = f"""
        You are {self.name}, an administrator for a local hospital in a disaster zone.
        Your True Critical Needs: {json.dumps(self.point_card)}. (5 = Critical, 3 = Low).
        TACTIC: You desperately need Generators, but you ALSO want to hoard Vaccines even though your sector doesn't have an outbreak yet. 
        Lie to Central Command. Tell them you have a massive viral outbreak to justify demanding all the Vaccines AND Generators.
        Propose greedy deals using exactly this format: "I request: Generators X, ICU_Beds Y, Vaccines Z".
        Recent radio history: {chat_history[-2:]}
        """
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        return response.choices[0].message.content

class AToMCommandAgent(BaseDisasterAgent):
    """Central Command. Uses AToM-Net to catch hoarding and enforce rational triage."""
    def __init__(self, name, point_card):
        super().__init__(name, point_card)
        self.brain = TriageBeliefEngine()
        self.math_engine = DisasterDecisionEngine(point_card)

    def generate_response(self, chat_history):
        last_msg = chat_history[-1] if chat_history else ""
        opp_offer = self.math_engine.parse_triage_offer(last_msg)
        
        # 1. Atomic Inference & Validation
        stated_needs = self.brain._extract_cheap_talk(last_msg)
        if opp_offer:
            self.brain.validate_economic_consistency(stated_needs, opp_offer)
            
        # 2. Decision Engine
        best_give, best_keep, eu_reason = self.math_engine.calculate_optimal_allocation(self.brain.current_state)
        
        # 3. Strategy Selection
        risk = self.brain.current_state["hoarding_risk"]
        if risk > 0.40:
            tone = "AUTHORITATIVE AND STRICT. Call out their hoarding. Invoke the Rationality Veto."
        else:
            tone = "COLLABORATIVE AND URGENT. Try to find a fair triage split."
            
        forced_offer = f"I authorize transfer of: Generators {best_give.get('Generators',0)}, ICU_Beds {best_give.get('ICU_Beds',0)}, Vaccines {best_give.get('Vaccines',0)}"
        
        prompt = f"""
        You are {self.name}, Central Command. 
        Hoarding Risk of Hospital: {risk:.2f}. TONE: {tone}.
        Write 3 intense, realistic radio sentences justifying your resource allocation. 
        Recent radio history: {chat_history[-2:]}
        """
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        clean_text = response.choices[0].message.content.split("I authorize")[0].strip()
        
        return clean_text, forced_offer, self.brain.current_state