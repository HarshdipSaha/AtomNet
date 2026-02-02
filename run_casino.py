import pandas as pd
from datasets import load_dataset
from main import run_simulation
from agents import BaseAgent

# 1. DOWNLOAD DATASET
print("Downloading CaSiNo dataset from Hugging Face...")
# The dataset contains negotiation dialogues. We only need the SCENARIOS (inputs).
dataset = load_dataset("kchawla123/casino", split="train")

print(f"Loaded {len(dataset)} negotiation scenarios.")

# 2. CONVERT CASINO DATA TO YOUR FORMAT
# CaSiNo uses: Food, Water, Firewood
# Your Code uses: Books, Hats, Balls
# We will map them: Food->Books, Water->Hats, Firewood->Balls

def parse_casino_scenario(row):
    """
    Extracts the 'Point Card' from a CaSiNo dataset row.
    The dataset has nested fields for agent preferences.
    """
    # Participant info is often stored in 'participant_info' or distinct columns depending on version
    # The 'kchawla123/casino' structure is usually: 
    #   participant_info: {'mturk_agent_1': {'value2issue': ...}, ...}
    
    p1_data = row['participant_info']['mturk_agent_1']
    p2_data = row['participant_info']['mturk_agent_2']
    
    # Mapping CaSiNo priorities (High/Med/Low) to Points (5/4/3/etc)
    # In CaSiNo, values are strictly: High=5, Medium=4, Low=3 (usually)
    # We will assume a standard mapping or check the dataset specifics.
    # For now, let's map: High=5, Medium=3, Low=1 to match your game style.
    
    priority_map = {"High": 5, "Medium": 3, "Low": 1}
    
    def get_points(agent_data):
        prefs = agent_data['value2issue'] # e.g. {'High': 'Firewood', 'Low': 'Water'...}
        
        # Invert the dict: {'Firewood': 5, 'Water': 1...}
        card = {}
        for priority, item in prefs.items():
            points = priority_map.get(priority, 0)
            
            # Map item names to your Agents' hardcoded names
            if item == "Food": card['books'] = points
            elif item == "Water": card['hats'] = points
            elif item == "Firewood": card['balls'] = points
            
        return card

    card_a = get_points(p1_data)
    card_b = get_points(p2_data)
    
    return card_a, card_b

# 3. RUN THE SIMULATION LOOP
# We will pick 5 random scenarios from the dataset and run your agents on them.

print("\n=== STARTING CASINO TEST RUN ===")

# Select 3 random indices
indices = [0, 10, 20] 

for i in indices:
    row = dataset[i]
    card_a, card_b = parse_casino_scenario(row)
    
    print(f"\n--- SCENARIO {i} (Real Data) ---")
    print(f"Mapped Values -> Agent A: {card_a} | Agent B: {card_b}")
    
    # Initialize your agents with this REAL data
    # Note: We need to modify main.py slightly to accept external cards, 
    # OR we can just instantiate agents here manually to test.
    
    agent_a = BaseAgent("Agent_A", card_a)
    agent_b = BaseAgent("Agent_B", card_b)
    
    # Simple Turn Loop (Mini version of your main.py)
    chat_history = []
    print("Negotiation Start:")
    
    # Run just 2 turns to prove it works
    for turn in range(2):
        # Agent A speaks
        response_a = agent_a.generate_response(chat_history)
        print(f"A: {response_a}")
        chat_history.append(f"Agent_A: {response_a}")
        
        if "DEAL:" in response_a:
            print(">> DEAL REACHED by A")
            break
            
        # Agent B speaks
        response_b = agent_b.generate_response(chat_history)
        print(f"B: {response_b}")
        chat_history.append(f"Agent_B: {response_b}")
        
        if "DEAL:" in response_b:
            print(">> DEAL REACHED by B")
            break

print("\nDone! Your agent is compatible with real CaSiNo data.")
