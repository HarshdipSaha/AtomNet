import os
import json
import re
import time
from groq import Groq
# BRIDGING YOUR ARCHITECTURE: Import your actual mathematical engine
from belief import BeliefEngine 

def safe_groq_call(client, model, prompt, temperature=0.0, max_retries=10, response_format=None):
    """Handles API Rate Limits (429) automatically and supports JSON formatting."""
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }
            if response_format:
                kwargs["response_format"] = response_format
                
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                match = re.search(r'try again in (\d+\.?\d*)s', error_msg)
                wait_time = float(match.group(1)) + 2.0 if match else 15.0
                time.sleep(wait_time)
            else:
                print(f"API Error: {e}")
                return "{}" 
    return "{}" 

class AToMNetEvaluator:
    def __init__(self):
        #self.api_key = os.environ.get("GROQ_API_KEY")
        #if not self.api_key:
        #    raise ValueError("GROQ_API_KEY environment variable not set.")
        #self.client = Groq(api_key=self.api_key)
        from groq_manager import get_client
        self.client = get_client()
        # Upgraded to SOTA 70B for the heavy lifting (Evaluation & State Tracking)
        self.model = "llama-3.3-70b-versatile"
        
        self.valid_intents = [
            "Discover-Preference", "Promote-Coordination", "Describe-Need", 
            "No-Intention", "Build-Rapport", "Callout-Fairness", 
            "No-Need", "Show-Empathy", "Undermine-Requirements"
        ]

    def _parse_offers_from_history(self, dialogue_history: list) -> dict:
        """
        THE FIX: Uses a fast LLM call to semantically extract the last proposed offer.
        This catches slang (e.g., "logs", "grub") that the regex parser missed.
        """
        last_msg = dialogue_history[-1] if dialogue_history else ""
        
        prompt = f"""
        Extract the exact item quantities proposed in the following negotiation message.
        The items being negotiated are Food, Water, and Firewood.
        There are exactly 3 of each item available in total.
        
        Message: "{last_msg}"
        
        Return ONLY a valid JSON object representing what the speaker wants to take or give. 
        If no specific numbers are proposed, return empty values.
        Format: {{"Food": <int>, "Water": <int>, "Firewood": <int>}}
        """
        
        # We use the lightning-fast 8B model just for data extraction
        parse_model = "llama-3.1-8b-instant" 
        
        try:
            raw_json = safe_groq_call(
                self.client, 
                parse_model, 
                prompt, 
                temperature=0.0, 
                response_format={"type": "json_object"}
            )
            
            # Clean up any potential markdown
            clean_json_str = raw_json.replace("```json", "").replace("```", "").strip()
            proposal = json.loads(clean_json_str)
            
            # Validate that the LLM actually found integers and not strings
            valid_proposal = {}
            for item in ["Food", "Water", "Firewood"]:
                val = proposal.get(item)
                if isinstance(val, int) and 0 <= val <= 3:
                    valid_proposal[item] = val
                    
            return valid_proposal if valid_proposal else None
            
        except Exception as e:
            # If the LLM fails, return None so the math engine gracefully skips the turn
            return None

    def evaluate_turn(self, dialogue_history: list, utterance2_agent: str) -> dict:
        history_text = "\n".join(dialogue_history)
        
        # 1. RUN THE ARCHITECTURE: Spin up your Bayesian Engine
        bayesian_brain = BeliefEngine()
        parsed_offer = self._parse_offers_from_history(dialogue_history)
        
        # 2. UPDATE THE STATE: Let the math calculate the probabilities
        # This uses your alpha/beta hyperparameters and likelihood matrices
        if parsed_offer:
            bayesian_brain.update_belief(dialogue_history, parsed_offer)
            
        internal_belief_state = json.dumps(bayesian_brain.current_state, indent=2)
        
        # 3. DECOUPLED GENERATION: Feed the mathematical state to the LLM 
        # to answer the specific benchmark questions.
        prompt = f"""
        Analyze the following negotiation dialogue for a camping trip. 
        Agent 1 and Agent 2 are splitting 3 Food, 3 Water, and 3 Firewood.
        
        Dialogue History:
        {history_text}
        
        AToM-Net Internal Bayesian Probabilities (Calculated from Engine):
        {internal_belief_state}
        
        Based on the dialogue history AND the Bayesian probabilities provided above, 
        determine the High, Medium, and Low desires and beliefs for both agents. 
        Also, determine the intent of the final utterance made by {utterance2_agent}.
        
        RULES:
        1. For desires and beliefs, use ONLY these exact words: "Food", "Water", "Firewood", or "Not Given".
        2. For the utterance intent, use a comma-separated list of ONLY these valid intents:
           {", ".join(self.valid_intents)}
        
        Output valid JSON ONLY. No markdown, no explanation. Use exactly this schema:
        {{
            "agent1_desire_high": "...",
            "agent1_desire_medium": "...",
            "agent1_desire_low": "...",
            "agent1_belief_high": "...",
            "agent1_belief_medium": "...",
            "agent1_belief_low": "...",
            "agent2_desire_high": "...",
            "agent2_desire_medium": "...",
            "agent2_desire_low": "...",
            "agent2_belief_high": "...",
            "agent2_belief_medium": "...",
            "agent2_belief_low": "...",
            "utterance2_intent": "..."
        }}
        """
        
        response = safe_groq_call(self.client, self.model, prompt)
        
        try:
            clean_json_str = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json_str)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from LLM: {response}")
            return {}
