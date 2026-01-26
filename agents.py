import os
import json
from groq import Groq
from belief import BeliefEngine
from decision import DecisionEngine

# Ensure you have your API key set
# export GROQ_API_KEY="gsk_..."
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

class BaseAgent:
    def __init__(self, name: str, point_card: dict):
        self.name = name
        self.point_card = point_card
        
        # 1. THE BODY (API Connection)
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "llama-3.3-70b-versatile"
        
        # 2. THE BRAIN (Theory of Mind)
        self.brain = BeliefEngine()
        
        # 3. THE CALCULATOR (Game Theory Math)
        self.math_engine = DecisionEngine(point_card)
        
        # Base Identity (Static)
        self.base_identity = f"""
        You are {name}, an expert negotiator.
        OBJECTIVE: Maximize your score based on these Secret Points:
        {json.dumps(point_card, indent=2)}
        
        ITEMS AVAILABLE: 3 Books, 2 Hats, 1 Ball.
        
        PROTOCOL:
        1. You have an Internal Brain that analyzes the opponent.
        2. You have a Math Engine that calculates the Optimal Deal.
        3. Your job is to EXECUTE that strategy using Natural Language.
        """

    def generate_response(self, chat_history: list) -> str:
        # --- STEP 1: PERCEPTION & COGNITION (The Brain) ---
        # Only run analysis if there is history
        current_belief = self.brain.current_state
        if chat_history:
            # We assume the last message is from the opponent for the update
            # (In a real turn-based game, this is usually true)
            current_belief = self.brain.update_belief(chat_history)
            
        # --- STEP 2: DECISION MAKING (The Math) ---
        # Calculate the purely mathematical best move
        best_offer, math_reasoning = self.math_engine.calculate_best_offer(current_belief)
        
        # --- STEP 3: STRATEGY & TONE SELECTION (The Research) ---
        # "EmotionPrompt": Adjust tone based on Deception Risk
        deception_risk = current_belief.get('deception_risk', 0.0)
        
        if deception_risk > 0.6:
            tone = "AGGRESSIVE / SKEPTICAL"
            style_instruction = "The opponent is likely lying. Be short. Demand they make concessions. Use phrases like 'I'm not stupid', 'Stop wasting time'."
        elif deception_risk > 0.3:
            tone = "CAUTIOUS / FIRM"
            style_instruction = "They might be bluffing. Stick to your numbers. Don't be too friendly."
        else:
            tone = "COLLABORATIVE / PERSUASIVE"
            style_instruction = "They seem honest. Use 'We', 'Us'. Frame the deal as a win-win. Say things like 'I can help you with books if you help me with hats'."

        # --- STEP 4: ACTION GENERATION (The LLM) ---
        # Construct the "Neuro-Symbolic" System Prompt
        system_prompt = f"""
        {self.base_identity}
        
        [INTERNAL LIVE ANALYSIS]
        1. OPPONENT BELIEF: {json.dumps(current_belief, indent=2)}
        2. MATHEMATICAL OPTIMUM: Offer them {json.dumps(best_offer)}.
           (Reason: {math_reasoning})
        3. DETECTED DECEPTION RISK: {deception_risk}
        
        [EXECUTION INSTRUCTIONS]
        - CURRENT STRATEGY: Sell the "Mathematical Optimum" offer.
        - TONE: {tone}
        - STYLE: {style_instruction}
        
        CRITICAL RULES:
        - Do NOT reveal your internal math or probabilities.
        - If the Math Engine says to give 0 items, say "I can't give you that."
        - If you agree to a deal, YOU MUST OUTPUT: "DEAL: books X, hats Y, balls Z" (Exact format).
        - Use "Cheap Talk" (e.g., "I really need the hats for my collection") to justify the math.
        """
        
        # Build the message chain
        messages = [{"role": "system", "content": system_prompt}]
        for turn in chat_history:
            role = "assistant" if turn.startswith(f"{self.name}:") else "user"
            content = turn.split(": ", 1)[1] if ": " in turn else turn
            messages.append({"role": role, "content": content})

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.7 # Slight creativity for persuasion
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"
