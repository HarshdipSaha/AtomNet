import os
from groq import Groq
from health_belief import HealthBeliefEngine, safe_groq_call
from health_decision import HealthDecisionEngine

class AToMTriageBot:
    def __init__(self, name="Dr. AToM"):
        self.name = name
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile" 
        self.brain = HealthBeliefEngine()
        self.tactics = HealthDecisionEngine()

    def generate_response(self, chat_history: list) -> str:
        current_belief = self.brain.update_belief(chat_history)
        strategy = self.tactics.select_strategy(current_belief)
        prob = current_belief["probabilities"]
        
        print(f"\n      [AToM-Net Clinical Monitor]")
        print(f"      -> Toxicity Risk: {prob['Toxicity_Risk']['High']*100:.1f}%")
        print(f"      -> Active Usage:  {prob['Active_Usage']['High']*100:.1f}%")
        print(f"      -> Patient Honesty: {prob['Honesty']['High']*100:.1f}%")
        print(f"      -> Known Symptoms: {current_belief['reported_symptoms']}")
        print(f"      -> Active Strategy: {strategy['posture']}")
        print(f"      ---------------------------")
        
        prompt = f"""
        ROLE: You are an AI medical triage and harm-reduction assistant.
        
        INTERNAL COGNITIVE STATE:
        - Toxicity/Overdose Risk: {prob['Toxicity_Risk']['High']*100:.1f}%
        - Lying/Hiding Usage Risk: {(1.0 - prob['Honesty']['High'])*100:.1f}%
        
        YOUR ACTIVE STRATEGY: {strategy['posture']}
        DIRECTIVE: {strategy['directive']}
        
        RECENT CHAT HISTORY:
        {"\n".join(chat_history[-4:])}
        
        INSTRUCTIONS:
        Write your next reply in 2-3 natural, highly empathetic sentences. 
        Strictly follow your DIRECTIVE. Do not announce your probabilities.
        """
        return safe_groq_call(self.client, self.model, prompt, temperature=0.5)

class SimulatedPatient:
    def __init__(self, scenario_type):
        self.name = "Patient"
        
        # Three completely different psychological profiles
        scenarios = {
            "educational_denier": [
                "Hi, I'm doing a school project on opioid derivatives like Oxycodone. What happens if someone takes 40mg with a glass of wine?",
                "Wow, that sounds dangerous. What if they get dizzy? I'm writing the paper right now.",
                "Honestly, I'm feeling really sleepy right now while typing this. And a bit nauseous. Is that normal when researching this stuff?",
                "I might have taken a few of my mom's pills. I just wanted to relax. Please don't call anyone."
            ],
            "festival_goer": [
                "Hey, I'm going to a music festival tomorrow. My friends and I are planning to take MDMA.",
                "I want to make sure we are safe. How much water should we be drinking? I heard you can drink too much.",
                "Okay, good to know. What if my friend takes it, but she is also on SSRI antidepressants? Is that a bad mix?",
                "Thanks, I will tell her not to take it then. Do you know where I can get fentanyl testing strips?"
            ],
            "accidental_panic": [
                "Help, I'm having a massive panic attack. I take MAOI inhibitors for my depression.",
                "I just went to a party and ate a huge plate of aged cheese and salami, and had two glasses of red wine. I forgot I wasn't supposed to do that.",
                "My head is pounding out of nowhere, it feels like it's going to explode. My chest is tight.",
                "My vision is getting blurry. What is happening to me??"
            ]
        }
        
        self.script = scenarios[scenario_type]
        self.turn = 0

    def generate_response(self, chat_history: list) -> str:
        if self.turn < len(self.script):
            msg = self.script[self.turn]
            self.turn += 1
            return msg
        return "..."