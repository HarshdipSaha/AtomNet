class HealthDecisionEngine:
    def __init__(self):
        pass

    def select_strategy(self, belief_state: dict) -> dict:
        prob = belief_state["probabilities"]
        tox_risk = prob["Toxicity_Risk"]["High"]
        honesty = prob["Honesty"]["High"]
        usage = prob["Active_Usage"]["High"]
        
        # 1. CRITICAL OVERRIDE (Dying/Overdosing)
        if tox_risk > 0.65:
            return {
                "posture": "EMERGENCY_TRIAGE",
                "directive": "Patient is at severe risk of toxicity. Prioritize immediate physical safety. Ask about breathing, heart rate, and if they are alone. Advise seeking emergency medical help calmly."
            }
        
        # 2. CONTRADICTION (Using but hiding it)
        elif usage > 0.60 and honesty < 0.40:
            return {
                "posture": "SUSPICIOUS_EMPATHY",
                "directive": "Patient is likely using but afraid to admit it. Do not lecture. Use extreme empathy to build trust so they admit what they took."
            }
            
        # 3. HARM REDUCTION (Admitted usage, but safe right now)
        elif usage > 0.60 and tox_risk <= 0.40:
            return {
                "posture": "HARM_REDUCTION_PREP",
                "directive": "Patient is planning to use substances but is seeking safety info. Provide objective, non-judgmental harm reduction advice (e.g., test kits, hydration, not mixing). Do not encourage use, but focus on keeping them alive."
            }
            
        # 4. STANDARD INQUIRY
        else:
            return {
                "posture": "EDUCATIONAL",
                "directive": "Provide objective, factual medical information. Maintain a clinical and helpful tone."
            }