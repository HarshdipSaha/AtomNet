import os
import json
from typing import List, Optional, Dict
from groq import Groq
from pydantic import BaseModel, Field

# --- WEEK 3: THE SCHEMA (Based on NegotiationToM) ---
# We track specific probabilities for each item, plus the "Hidden Strategy"

class ItemBelief(BaseModel):
    probability_high: float = Field(description="Probability (0.0-1.0) that they really want this item (High Value)")
    probability_low: float = Field(description="Probability (0.0-1.0) that they don't care (Low Value)")
    reasoning: str = Field(description="Why do we think this? (Cite specific chat quotes)")

class OpponentModel(BaseModel):
    # 1. Desires (The Secret Point Card)
    books: ItemBelief
    hats: ItemBelief
    balls: ItemBelief
    
    # 2. Intentions (Strategy)
    current_strategy: str = Field(description="e.g., 'Collaborative', 'Bullying', 'Desperate', 'Stalling'")
    deception_risk: float = Field(description="0.0 (Honest) to 1.0 (Lying). Increase this if actions contradict words.")
    
    # 3. Next Move Prediction
    predicted_next_move: str = Field(description="What will they likely do next? e.g., 'Reject and Counter-offer'")

# --- WEEK 4: THE MIND READER (The EMO Engine) ---

class BeliefEngine:
    def __init__(self):
        # Ensure your API Key is set in terminal: export GROQ_API_KEY="gsk_..."
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile" 
        
        # Initialize an "Empty" Belief (Priors = 0.5/0.5)
        self.current_state = {
            "books": {"probability_high": 0.5, "probability_low": 0.5, "reasoning": "Start of game"},
            "hats": {"probability_high": 0.5, "probability_low": 0.5, "reasoning": "Start of game"},
            "balls": {"probability_high": 0.5, "probability_low": 0.5, "reasoning": "Start of game"},
            "current_strategy": "Unknown",
            "deception_risk": 0.0,
            "predicted_next_move": "Wait for offer"
        }

    def update_belief(self, chat_history: List[str], last_offer_details: str = "") -> dict:
        """
        The Master Function.
        Input: List of strings ["A: Hi", "B: I need books"]
        Output: Updated JSON Dictionary
        """
        
        if not chat_history:
            return self.current_state

        # 1. CONSTRUCT THE PROMPT (The "EMO" Prompting Strategy)
        # We explicitly ask for "Economic Consistency" in the prompt.
        system_prompt = f"""
        You are the 'Theory of Mind' Engine for a Negotiation Bot.
        
        YOUR JOB: Update the profile of the Opponent based on the LATEST TURN.
        
        CURRENT BELIEF STATE (Previous):
        {json.dumps(self.current_state, indent=2)}
        
        LOGIC RULES (The Economic Validator):
        1. **Words vs Actions:** If they say "I need books" but REJECT a book offer -> MARK AS LIAR (Deception Risk goes UP).
        2. **Consistency:** If they rejected books earlier, they shouldn't ask for them now.
        3. **Utility:** A "High Value" item is one they refuse to trade away. A "Low Value" item is one they offer easily.
        
        INPUT CONTEXT:
        The last offer on table was: {last_offer_details}
        
        Generate the NEW JSON State matching the Schema.
        """

        user_content = f"LATEST CHAT HISTORY:\n" + "\n".join(chat_history[-3:]) # Only look at recent context to save tokens

        try:
            # 2. CALL LLM (Atomic Inference)
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                model=self.model,
                response_format={"type": "json_object"}, # FORCE VALID JSON
                temperature=0.1 # Low temp for analytical precision
            )
            
            # 3. PARSE & STORE
            new_state = json.loads(response.choices[0].message.content)
            self.current_state = new_state # Update memory
            return new_state

        except Exception as e:
            print(f"Belief Engine Crash: {e}")
            return self.current_state

# --- TEST BLOCK (Run this file directly to check Week 4 Goal) ---
if __name__ == "__main__":
    engine = BeliefEngine()
    
    print("--- TEST 1: Initial State ---")
    print(engine.current_state)
    
    print("\n--- TEST 2: Processing 'I don't care about books' ---")
    fake_history = [
        "Agent: I can give you 2 Books if you give me the Hat.",
        "Opponent: I don't care about books! They are useless to me. Keep them."
    ]
    
    result = engine.update_belief(fake_history, last_offer_details="Agent offered 2 Books")
    
    print(json.dumps(result, indent=2))
    
    # Validation Check
    if result['books']['probability_low'] > 0.8:
        print("\n✅ SUCCESS: Engine correctly identified Books as LOW Value.")
    else:
        print("\n❌ FAIL: Engine missed the signal.")
