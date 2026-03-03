import random
from dataclasses import dataclass
from typing import Dict, Tuple
import re
# Standard DealOrNoDeal Item Counts
TOTAL_ITEMS = {"books": 3, "hats": 2, "balls": 1}

@dataclass
class PointCard:
    """The Secret Preference Card for a Player"""
    values: Dict[str, int]

    def calculate_score(self, items_received: Dict[str, int]) -> int:
        score = 0
        for item, count in items_received.items():
            score += count * self.values.get(item, 0)
        return score

class NegotiationGame:
    def __init__(self):
        self.total_items = {"books": 3, "hats": 2, "balls": 1}
        self.agent_a_card = self._generate_card()
        self.agent_b_card = self._generate_card()
        self.history = []

    def _generate_card(self) -> PointCard:
        return PointCard({
            "books": random.randint(0, 5),
            "hats": random.randint(0, 5),
            "balls": random.randint(0, 5)
        })

    def get_ground_truth(self):
        return {
            "A": self.agent_a_card.values,
            "B": self.agent_b_card.values
        }
    
    def parse_proposal(self, text: str) -> Dict[str, int] | None:
        """
        ROBUST PARSER (The Fix)
        Extracts 'item count' patterns regardless of surrounding text.
        """
        text = text.lower()
        if "deal:" in text:
            # Normalize text to handle "books: 1" or "books 1"
            # Regex looks for: (word) followed by optional space/colon followed by (digits)
            pattern = r"(books|hats|balls)\W*(\d+)"
            matches = re.findall(pattern, text)
            
            if matches:
                proposal = {"books": 0, "hats": 0, "balls": 0}
                for item, count in matches:
                    proposal[item] = int(count)
                return proposal
        return None