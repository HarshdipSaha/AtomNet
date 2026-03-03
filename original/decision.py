from typing import Dict, Tuple, Optional
import re

# CaSiNo Standard Point Values
VAL_HIGH = 5
VAL_MED = 4
VAL_LOW = 3

class DecisionEngine:
    def __init__(self, my_card: dict):
        self.my_card = my_card 
        self.items = ['Food', 'Water', 'Firewood']
        # CaSiNo always has exactly 3 of each item
        self.total_items = {'Food': 3, 'Water': 3, 'Firewood': 3}

    def _calculate_opponent_item_value(self, item_probs: dict) -> float:
        """Reads the Bayesian Posteriors to estimate opponent's hidden point card"""
        p_high = item_probs.get('High', 0.5)
        p_low = item_probs.get('Low', 0.5)
        p_med = max(0.0, 1.0 - p_high - p_low)
        return (p_high * VAL_HIGH) + (p_med * VAL_MED) + (p_low * VAL_LOW)

    def calculate_my_score(self, offer: dict) -> int:
        score = 0
        for item, count in offer.items():
            score += count * self.my_card.get(item, 0)
        return score

    def parse_offer(self, text: str) -> Optional[Dict[str, int]]:
        text = text.lower()
        proposal = {"Food": 0, "Water": 0, "Firewood": 0}
        found = False
        for item in self.items:
            # Regex to find "3 Food" or "Food: 3"
            pattern = f"(\\d+)\\s*{item.lower()}|{item.lower()}\\W*(\\d+)"
            matches = re.findall(pattern, text)
            for m in matches:
                val = m[0] or m[1]
                if val:
                    proposal[item] = int(val)
                    found = True
        return proposal if found else None

    def calculate_best_offer(self, belief_state: dict) -> Tuple[Dict, Dict, str]:
        # Estimate Opponent's Values using Bayesian Posteriors
        opp_vals = {
            item: self._calculate_opponent_item_value(belief_state['probabilities'].get(item, {}))
            for item in self.items
        }
        
        possible_deals = []
        # Brute Force Expected Utility (EU) Maximization
        for f in range(self.total_items['Food'] + 1):
            for w in range(self.total_items['Water'] + 1):
                for fw in range(self.total_items['Firewood'] + 1):
                    offer_to_give = {'Food': f, 'Water': w, 'Firewood': fw}
                    my_keep = {
                        'Food': self.total_items['Food'] - f,
                        'Water': self.total_items['Water'] - w,
                        'Firewood': self.total_items['Firewood'] - fw
                    }
                    
                    my_score = sum(my_keep[k] * self.my_card.get(k, 0) for k in my_keep)
                    opp_score = sum(offer_to_give[k] * opp_vals.get(k, 0) for k in offer_to_give)
                    
                    # Probability of Acceptance (Scaled for CaSiNo points)
                    # A fair deal usually gives the opponent 15-20 points.
                    if opp_score >= 18: prob_accept = 1.0
                    elif opp_score >= 14: prob_accept = 0.5
                    elif opp_score >= 10: prob_accept = 0.2
                    else: prob_accept = 0.0
                        
                    # Adjust EU by Deception Risk (Don't trust deals from liars)
                    deception_penalty = 1.0 - belief_state.get('deception_risk', 0.0)
                    eu = prob_accept * my_score * deception_penalty
                    
                    possible_deals.append({
                        "offer_give": offer_to_give,
                        "offer_keep": my_keep,
                        "eu": eu,
                        "my_score": my_score
                    })
        
        # Select the deal that maximizes Expected Utility
        best_deal = max(possible_deals, key=lambda x: x['eu'])
        return best_deal['offer_give'], best_deal['offer_keep'], f"EU: {best_deal['eu']:.1f} (My Score: {best_deal['my_score']})"