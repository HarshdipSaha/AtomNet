import re
from typing import Optional, Dict, Tuple

class DecisionEngine:
    def __init__(self, my_card: dict):
        self.my_card = my_card 
        self.items = ['Food', 'Water', 'Firewood']
        self.total_items = {'Food': 3, 'Water': 3, 'Firewood': 3}

    def calculate_my_score(self, offer: dict) -> int:
        return sum(offer.get(item, 0) * self.my_card.get(item, 0) for item in offer)

    def parse_offer(self, text: str) -> Optional[Dict[str, int]]:
        text = text.lower()
        proposal = {"Food": 0, "Water": 0, "Firewood": 0}
        found = False
        for item in self.items:
            matches = re.findall(f"(\\d+)\\s*{item.lower()}|{item.lower()}\\W*(\\d+)", text)
            for m in matches:
                val = m[0] or m[1]
                if val:
                    proposal[item] = int(val)
                    found = True
        return proposal if found else None

    def calculate_best_offer(self, belief_state: dict) -> Tuple[Dict, Dict, str]:
        # Estimate Opponent Values
        opp_vals = {}
        for item in self.items:
            probs = belief_state['probabilities'].get(item, {})
            p_h, p_l = probs.get('High', 0.5), probs.get('Low', 0.5)
            p_m = max(0.0, 1.0 - p_h - p_l)
            opp_vals[item] = (p_h * 5) + (p_m * 4) + (p_l * 3)
            
        possible_deals = []
        for f in range(4):
            for w in range(4):
                for fw in range(4):
                    give = {'Food': f, 'Water': w, 'Firewood': fw}
                    keep = {'Food': 3-f, 'Water': 3-w, 'Firewood': 3-fw}
                    
                    my_score = self.calculate_my_score(keep)
                    opp_score = sum(give[k] * opp_vals.get(k, 0) for k in give)
                    
                    prob_accept = 1.0 if opp_score >= 18 else (0.5 if opp_score >= 14 else 0.0)
                    
                    # Tarpitting logic: If deception is high, heavily discount deals to force hardball
                    deception_penalty = 1.0 - belief_state.get('deception_risk', 0.0)
                    eu = prob_accept * my_score * deception_penalty
                    possible_deals.append({"give": give, "keep": keep, "eu": eu, "score": my_score})
                    
        best = max(possible_deals, key=lambda x: x['eu'])
        return best['give'], best['keep'], f"EU: {best['eu']:.1f}"