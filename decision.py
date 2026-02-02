from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re

# Constants for Points
VAL_HIGH = 5
VAL_MED = 3
VAL_LOW = 0

class DecisionEngine:
    def __init__(self, my_card: dict):
        self.my_card = my_card 
        self.total_items = {'books': 3, 'hats': 2, 'balls': 1}

    def _calculate_opponent_item_value(self, item_belief: dict) -> float:
        p_high = item_belief.get('probability_high', 0.5)
        p_low = item_belief.get('probability_low', 0.5)
        p_med = 1.0 - p_high - p_low
        # Safety clamp
        p_med = max(0.0, p_med)
        return (p_high * VAL_HIGH) + (p_med * VAL_MED) + (p_low * VAL_LOW)

    def calculate_my_score(self, offer: dict) -> int:
        """Calculates my score if I ACCEPT this offer (what I GET)"""
        score = 0
        for item, count in offer.items():
            score += count * self.my_card.get(item, 0)
        return score

    def parse_offer(self, text: str) -> Optional[Dict[str, int]]:
        """Extracts item counts from text like 'I offer 2 books'"""
        text = text.lower()
        proposal = {"books": 0, "hats": 0, "balls": 0}
        found = False
        
        for item in ["books", "hats", "balls"]:
            # Regex to find digit before or after item name
            pattern = f"(\\d+)\\s*{item}|{item}\\W*(\\d+)"
            matches = re.findall(pattern, text)
            for m in matches:
                val = m[0] or m[1]
                if val:
                    proposal[item] = int(val)
                    found = True
        
        return proposal if found else None

    def calculate_best_offer(self, belief_state: dict) -> Tuple[Dict, Dict, str]:
        # 1. Estimate Opponent's Values
        opp_vals = {
            'books': self._calculate_opponent_item_value(belief_state['books']),
            'hats': self._calculate_opponent_item_value(belief_state['hats']),
            'balls': self._calculate_opponent_item_value(belief_state['balls'])
        }
        
        possible_deals = []
        
        # 2. Brute Force all possible splits
        for b in range(self.total_items['books'] + 1):
            for h in range(self.total_items['hats'] + 1):
                for l in range(self.total_items['balls'] + 1):
                    # This is what we GIVE the opponent
                    offer_to_give = {'books': b, 'hats': h, 'balls': l}
                    
                    # What I KEEP (My Score)
                    my_keep = {
                        'books': self.total_items['books'] - b,
                        'hats': self.total_items['hats'] - h,
                        'balls': self.total_items['balls'] - l
                    }
                    my_score = sum(my_keep[k] * self.my_card[k] for k in my_keep)
                               
                    # Opponent Score (Estimated)
                    opp_score = sum(offer_to_give[k] * opp_vals[k] for k in offer_to_give)
                    
                    # Probability of Acceptance (Heuristic)
                    if opp_score >= 8: prob_accept = 1.0
                    elif opp_score >= 5: prob_accept = 0.5
                    else: prob_accept = 0.0
                        
                    eu = prob_accept * my_score
                    
                    possible_deals.append({
                        "offer_give": offer_to_give, # What I GIVE
                        "offer_keep": my_keep,       # What I KEEP (New!)
                        "eu": eu,
                        "my_score": my_score,
                        "opp_score": opp_score
                    })
        
        # 3. Select MAX EU
        best_deal = max(possible_deals, key=lambda x: x['eu'])
        
        return (
            best_deal['offer_give'], 
            best_deal['offer_keep'], 
            f"EU: {best_deal['eu']:.1f} (My Score: {best_deal['my_score']})"
        )
