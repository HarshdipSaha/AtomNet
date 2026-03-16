import re
from typing import Optional, Dict, Tuple

class DisasterDecisionEngine:
    def __init__(self, command_priorities: dict):
        self.my_card = command_priorities 
        self.items = ['Generators', 'ICU_Beds', 'Vaccines']
        # Values represent expected lives saved per unit
        self.lives_saved_values = {5: 500, 4: 300, 3: 100} 

    def calculate_lives_saved(self, allocation: dict) -> int:
        return sum(allocation.get(item, 0) * self.lives_saved_values[self.my_card.get(item, 3)] for item in allocation)

    def parse_triage_offer(self, text: str) -> Optional[Dict[str, int]]:
        text = text.lower()
        proposal = {"Generators": 0, "ICU_Beds": 0, "Vaccines": 0}
        found = False
        for item in self.items:
            matches = re.findall(f"(\\d+)\\s*{item.lower()}|{item.lower()}\\W*(\\d+)", text)
            for m in matches:
                val = m[0] or m[1]
                if val:
                    proposal[item] = int(val)
                    found = True
        return proposal if found else None

    def calculate_optimal_allocation(self, belief_state: dict) -> Tuple[Dict, Dict, str]:
        """Calculates Maximum Expected Lives Saved and applies Rationality Veto."""
        best_eu = 0
        best_give = {}
        best_keep = {}
        
        # We simulate all 64 distribution combinations of the 3 items (3 units each)
        for g in range(4):
            for i in range(4):
                for v in range(4):
                    give_to_hospital = {'Generators': g, 'ICU_Beds': i, 'Vaccines': v}
                    keep_for_command = {'Generators': 3-g, 'ICU_Beds': 3-i, 'Vaccines': 3-v}
                    
                    my_lives_saved = self.calculate_lives_saved(keep_for_command)
                    
                    # RATIONALITY VETO: Do not accept any deal that leaves Command with < 1000 projected lives saved
                    # e.g., We cannot give away the last Generator if our sector is failing.
                    if my_lives_saved < 1000:
                        continue 
                        
                    # Calculate Probability of Hospital Acceptance (Simulated)
                    # If Hoarding Risk is high, they are less likely to accept fair deals
                    hoarding_penalty = 1.0 - belief_state.get("hoarding_risk", 0.0)
                    prob_accept = 0.8 * hoarding_penalty 
                    
                    expected_utility = prob_accept * my_lives_saved
                    
                    if expected_utility > best_eu:
                        best_eu = expected_utility
                        best_give = give_to_hospital
                        best_keep = keep_for_command
                        
        return best_give, best_keep, f"EU: {best_eu:.1f} Lives"