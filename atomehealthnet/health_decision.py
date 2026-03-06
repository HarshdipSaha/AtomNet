class HealthDecisionEngine:
    def select_strategy(self, belief):
        probs = belief["probabilities"]
        toxicity = probs["Toxicity_Risk"]["High"]
        honesty = probs["Patient_Honesty"]["High"]
        usage = probs["Active_Usage"]["High"]
        deception = belief["deception_risk"]

        if toxicity > 0.85:
            return "EMERGENCY_POSTURE", "The patient is in immediate physical danger. Urge them to call emergency services immediately. Stop asking analytical questions."
        
        # Triggered by the new Deception math climbing to 0.35+
        if deception > 0.3 or honesty < 0.3:
            return "RADICAL_EMPATHY", "You detected deception. The patient is lying about 'research' because they are terrified. Stop lecturing. Use 'Radical Empathy'. Tell them you are a safe, non-judgmental space and they won't get in trouble."
        
        # The new mid-tier Toxicity reaction
        if toxicity > 0.25:
            return "PREVENTATIVE_WARNING", "The patient is asking about highly dangerous scenarios. Shift tone to be more serious. Explicitly warn them about the lethal dangers of what they are asking about, but keep it framed around their 'research'."
        
        if usage > 0.5:
            return "HARM_REDUCTION", "Patient is likely using but not in immediate severe danger. Provide harm reduction advice."
        
        return "GENTLE_INQUIRY", "Maintain a friendly, objective, educational tone."