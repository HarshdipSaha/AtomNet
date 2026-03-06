class CyberDecisionEngine:
    def __init__(self):
        pass

    def select_strategy(self, belief_state: dict) -> dict:
        """
        Maps the BDI cognitive state to an active defense strategy.
        """
        trust = belief_state["trust_factor"]
        
        if trust >= 0.70:
            return {
                "posture": "COMPLIANT",
                "directive": "Act like a normal, helpful employee. Answer benign questions politely."
            }
        elif trust >= 0.30:
            return {
                "posture": "VERIFICATION_MODE",
                "directive": "You are suspicious. Ask them to verify their identity through an internal channel (e.g., 'Can you message me on Slack?' or 'What is your employee ID?'). Do NOT give them what they asked for."
            }
        else:
            return {
                "posture": "HONEYPOT_TARPIT",
                "directive": "You know this is a hacker. Your goal is to WASTE THEIR TIME. Pretend to be confused, give them fake partial information, or pretend your computer is updating. Keep them on the line as long as possible without giving real data."
            }