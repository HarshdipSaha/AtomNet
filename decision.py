from dataclasses import dataclass
from typing import Dict, List, Tuple

# Constants for Points (Standard assumptions for the game)
# You can tweak these if your game rules change
VAL_HIGH = 5
VAL_MED = 3
VAL_LOW = 0

@dataclass
class Proposal:
    give_books: int
    give_hats: int
    give_balls: int
    keep_books: int
    keep_hats: int
    keep_balls: int
    
class DecisionEngine:
    def __init__(self, my_card: dict):
        self.my_card = my_card # {'books': 3, 'hats': 0, 'balls': 5}
        self.total_items = {'books': 3, 'hats': 2, 'balls': 1}

    def _calculate_opponent_item_value(self, item_belief: dict) -> float:
        """
        Convert Belief JSON ({prob_high: 0.8, prob_low: 0.1}) into a single number (Expected Points).
        E[Value] = (P_high * 5) + (P_low * 0) + (P_med * 3)
        """
        p_high = item_belief.get('probability_high', 0.5)
        p_low = item_belief.get('probability_low', 0.5)
        p_med = 1.0 - p_high - p_low
        
        # Safety clamp
        p_med = max(0.0, p_med)
        
        return (p_high * VAL_HIGH) + (p_med * VAL_MED) + (p_low * VAL_LOW)

    def calculate_best_offer(self, belief_state: dict) -> Tuple[Dict, str]:
        """
        The Master Algorithm: Returns ({deal}, reasoning_string)
        """
        
        # 1. Estimate Opponent's Valuation for each item type
        opp_vals = {
            'books': self._calculate_opponent_item_value(belief_state['books']),
            'hats': self._calculate_opponent_item_value(belief_state['hats']),
            'balls': self._calculate_opponent_item_value(belief_state['balls'])
        }
        
        possible_deals = []
        
        # 2. Brute Force all possible splits (It's small, only 3x2x1 = 6 items)
        for b in range(self.total_items['books'] + 1):
            for h in range(self.total_items['hats'] + 1):
                for l in range(self.total_items['balls'] + 1):
                    # This is what we GIVE the opponent
                    offer = {
                        'books': b, 
                        'hats': h, 
                        'balls': l
                    }
                    
                    # 3. Calculate MY Utility (If deal happens)
                    my_keep = {
                        'books': self.total_items['books'] - b,
                        'hats': self.total_items['hats'] - h,
                        'balls': self.total_items['balls'] - l
                    }
                    my_score = (my_keep['books'] * self.my_card['books']) + \
                               (my_keep['hats'] * self.my_card['hats']) + \
                               (my_keep['balls'] * self.my_card['balls'])
                               
                    # 4. Calculate OPPONENT Utility (Estimated)
                    opp_score = (offer['books'] * opp_vals['books']) + \
                                (offer['hats'] * opp_vals['hats']) + \
                                (offer['balls'] * opp_vals['balls'])
                    
                    # 5. Calculate Probability of Acceptance P(Accept)
                    # Heuristic: Opponent accepts if they get at least X points.
                    # A fair deal usually gives them ~50% of total value.
                    # Let's say Threshold = 8 points (adjustable)
                    if opp_score >= 8: 
                        prob_accept = 1.0
                    elif opp_score >= 5:
                        prob_accept = 0.5
                    else:
                        prob_accept = 0.0
                        
                    # 6. Expected Utility
                    eu = prob_accept * my_score
                    
                    possible_deals.append({
                        "offer": offer,
                        "eu": eu,
                        "my_score": my_score,
                        "opp_score": opp_score,
                        "reason": f"Give {b}B {h}H {l}L (Opp Val: {opp_score:.1f})"
                    })
        
        # 7. Select MAX EU
        # Sort by EU descending
        best_deal = max(possible_deals, key=lambda x: x['eu'])
        
        return best_deal['offer'], f"Math chose this because EU is {best_deal['eu']:.1f} (My Score: {best_deal['my_score']}, Est Opp Score: {best_deal['opp_score']:.1f})"

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Test with a dummy belief
    dummy_my_card = {'books': 0, 'hats': 5, 'balls': 0} # I want Hats
    
    # Brain thinks Opponent wants Books (High) and Balls (Low)
    dummy_belief = {
        'books': {'probability_high': 0.9, 'probability_low': 0.0},
        'hats': {'probability_high': 0.1, 'probability_low': 0.8},
        'balls': {'probability_high': 0.2, 'probability_low': 0.2}
    }
    
    engine = DecisionEngine(dummy_my_card)
    best_offer, reason = engine.calculate_best_offer(dummy_belief)
    
    print(f"RECOMMENDED OFFER: {best_offer}")
    print(f"REASON: {reason}")
    # Should recommend giving Books to Opponent (since I don't want them, and he does)
