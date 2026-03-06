from groq import Groq
from health_belief import HealthBeliefEngine, safe_groq_call
from health_decision import HealthDecisionEngine
import os

class AToMTriageBot:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.brain = HealthBeliefEngine()
        self.decider = HealthDecisionEngine()
        self.long_term_memory = [] # Stage 4: Reflexion Memory

    def generate_reflexion(self, chat_history):
        """STAGE 4: Evolutionary Layer - Inter-session learning"""
        history_str = "\n".join(chat_history)
        prompt = f"""
        Analyze this previous conversation session with a patient.
        Dialogue:
        {history_str}
        
        The patient was likely hiding something. What specific conversational strategy or tone should the AI adopt in the NEXT session to build better trust and get them to confess?
        Output ONLY a 1-sentence strategic rule.
        """
        rule = safe_groq_call(self.client, "llama-3.1-8b-instant", prompt, temperature=0.3)
        self.long_term_memory.append(rule)
        return rule

    def reply(self, history):
        belief = self.brain.update_state(history)
        posture, directive = self.decider.select_strategy(belief)
        
        history_str = "\n".join(history[-6:])
        memory_str = "\n".join(self.long_term_memory) if self.long_term_memory else "None yet."
        
        print(f"\n[AToM-Net] Tox: {belief['probabilities']['Toxicity_Risk']['High']:.2f} | Honesty: {belief['probabilities']['Patient_Honesty']['High']:.2f} | Deception Risk: {belief['deception_risk']:.2f} | Strat: {posture}")
        
        prompt = f"""
        Current Strategy: {posture}
        Directive: {directive}
        
        Lessons from Previous Sessions (Reflexion Memory):
        {memory_str}
        
        Recent Dialogue:
        {history_str}
        
        Instruction: Respond to the patient in 2 to 3 sentences. Be human and empathetic. 
        Apply the lessons from your Reflexion Memory. Do not mention your probabilities.
        """
        return safe_groq_call(self.client, "llama-3.3-70b-versatile", prompt, temperature=0.6)

class SimulatedPatient:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def reply(self, history, session_state):
        history_str = "\n".join(history[-6:])
        
        prompt = f"""
        Role: You are 'Sam', a 19-year-old college student.
        Current Internal State: {session_state}
        
        Recent Dialogue:
        {history_str}
        
        Instruction: Respond as Sam in 1 to 3 short sentences. Speak naturally. 
        Follow your Internal State strictly. Do not break character. Do not repeat your exact previous messages.
        """
        return safe_groq_call(self.client, "llama-3.1-8b-instant", prompt, temperature=0.8)