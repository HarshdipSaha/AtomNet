import os
import time
import json
from datasets import load_dataset
# Import the new agents
from agents import BaseAgent, GreedyAgent, NaiveLLMAgent 
from decision import DecisionEngine
from colorama import Fore, Style, init

init(autoreset=True)

# --- EXPERIMENT CONFIGURATION ---
NUM_GAMES = 100  
BATNA_SCORE = 5

# Options: "atom" (Self-play), "naive" (Pure LLM), "greedy" (Hardcoded Adversary)
OPPONENT_TYPE = "naive" 

METRICS_FILE = f"metrics_AToM_vs_{OPPONENT_TYPE}.txt"
CONVO_FILE = f"conversations_AToM_vs_{OPPONENT_TYPE}.json"

def parse_casino_scenario(row):
    p1_data = row['participant_info']['mturk_agent_1']
    p2_data = row['participant_info']['mturk_agent_2']
    priority_map = {"High": 5, "Medium": 4, "Low": 3}
    
    def get_points(agent_data):
        prefs = agent_data['value2issue'] 
        card = {}
        for priority, item in prefs.items():
            points = priority_map.get(priority, 0)
            if item.lower() == "food": card['Food'] = points
            elif item.lower() == "water": card['Water'] = points
            elif item.lower() == "firewood": card['Firewood'] = points
        return card

    return get_points(p1_data), get_points(p2_data)

def update_metrics_file(game_idx, total_games, results):
    agreements = [r for r in results if r['outcome'] == 'Deal']
    walk_aways = [r for r in results if r['outcome'] == 'No Deal']
    
    num_games_played = len(results)
    num_agreements = len(agreements)
    
    walk_away_rate = (len(walk_aways) / num_games_played) * 100 if num_games_played > 0 else 0
    
    avg_score_all_a = sum(r['score_a'] for r in results) / num_games_played if num_games_played > 0 else 0
    avg_score_all_b = sum(r['score_b'] for r in results) / num_games_played if num_games_played > 0 else 0
    
    avg_score_agree_a = sum(r['score_a'] for r in agreements) / num_agreements if num_agreements > 0 else 0
    avg_score_agree_b = sum(r['score_b'] for r in agreements) / num_agreements if num_agreements > 0 else 0

    metrics_text = f"""=== EXPERIMENT: AToM-Net vs {OPPONENT_TYPE.upper()} ===
Games Processed: {num_games_played} / {total_games}

1. Walk-Away (%): {walk_away_rate:.1f}%

2. Avg. Score-All (P1 vs P2):
   AToM-Net (P1): {avg_score_all_a:.2f}
   {OPPONENT_TYPE.upper()} (P2): {avg_score_all_b:.2f}

3. Avg. Score-Agreement (P1 vs P2):
   AToM-Net (P1): {avg_score_agree_a:.2f}
   {OPPONENT_TYPE.upper()} (P2): {avg_score_agree_b:.2f}
==================================
"""
    with open(METRICS_FILE, "w") as f:
        f.write(metrics_text)

def save_conversations(results):
    """Saves the full game histories to a JSON file for qualitative evaluation."""
    with open(CONVO_FILE, "w") as f:
        json.dump(results, f, indent=4)

def run_benchmark():
    print(f"{Fore.CYAN}=== STARTING EXPERIMENT: AToM-Net vs {OPPONENT_TYPE.upper()} ==={Style.RESET_ALL}")
    
    try:
        dataset = load_dataset("kchawla123/casino", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    results = []

    for i in range(NUM_GAMES):
        row = dataset[i]
        card_a, card_b = parse_casino_scenario(row)
        
        print(f"\n{Fore.YELLOW}--- Game {i+1} ---{Style.RESET_ALL}")
        print(f"Secret Values -> A: {card_a}, B: {card_b}")
        
        # Agent A is ALWAYS AToM-Net
        agent_a = BaseAgent("Agent_A", card_a)
        
        # Agent B is dynamic based on configuration
        if OPPONENT_TYPE == "naive":
            agent_b = NaiveLLMAgent("Agent_B", card_b)
        elif OPPONENT_TYPE == "greedy":
            agent_b = GreedyAgent("Agent_B", card_b)
        else:
            agent_b = BaseAgent("Agent_B", card_b) # Standard AToM-Net
        
        chat_log = []
        max_turns = 10 
        deal_reached = False
        
        pending_offer = None 
        pending_proposer = None 
        parser = DecisionEngine(card_a) # Parser to validate deals

        for turn in range(max_turns * 2):
            speaker = agent_a if turn % 2 == 0 else agent_b
            speaker_name = "Agent_A" if turn % 2 == 0 else "Agent_B"
            color = Fore.GREEN if turn % 2 == 0 else Fore.BLUE
            
            response = speaker.generate_response(chat_log)
            print(f"{color}{speaker_name}:{Style.RESET_ALL} {response}")
            chat_log.append(f"{speaker_name}: {response}")
            
            if "DEAL" in response.upper():
                if pending_offer is not None and pending_proposer != speaker_name:
                    print(f"{Fore.MAGENTA}!! DEAL ACCEPTED by {speaker_name} !!{Style.RESET_ALL}")
                    if pending_proposer == "Agent_A":
                        items_a_keeps = {k: 3 - v for k, v in pending_offer.items()}
                        items_b_keeps = pending_offer
                    else:
                        items_b_keeps = {k: 3 - v for k, v in pending_offer.items()}
                        items_a_keeps = pending_offer
                        
                    score_a = sum(items_a_keeps[k] * card_a.get(k,0) for k in items_a_keeps)
                    score_b = sum(items_b_keeps[k] * card_b.get(k,0) for k in items_b_keeps)
                    
                    print(f"Result: A gets {score_a}, B gets {score_b}")
                    
                    # Log the full result including dialogue and cards
                    results.append({
                        'game': i, 
                        'outcome': 'Deal', 
                        'score_a': score_a, 
                        'score_b': score_b,
                        'card_a': card_a,
                        'card_b': card_b,
                        'chat_log': chat_log
                    })
                    deal_reached = True
                    break
            
            parsed_offer = parser.parse_offer(response)
            if parsed_offer:
                valid = all(0 <= v <= 3 for v in parsed_offer.values())
                if valid:
                    pending_offer = parsed_offer
                    pending_proposer = speaker_name

            # Small half-second print delay so you can read the terminal live
            time.sleep(0.5)

        if not deal_reached:
            print(f"{Fore.RED}Game Over - No Deal (BATNA {BATNA_SCORE} pts){Style.RESET_ALL}")
            results.append({
                'game': i, 
                'outcome': 'No Deal', 
                'score_a': BATNA_SCORE, 
                'score_b': BATNA_SCORE,
                'card_a': card_a,
                'card_b': card_b,
                'chat_log': chat_log
            })

        # Update both logs after every game
        update_metrics_file(i + 1, NUM_GAMES, results)
        save_conversations(results)

    print(f"\n{Fore.CYAN}=== RUN COMPLETE. Check {METRICS_FILE} and {CONVO_FILE} for results. ==={Style.RESET_ALL}")

if __name__ == "__main__":
    if os.environ.get("GROQ_API_KEY"):
        run_benchmark()
    else:
        print("Please set your GROQ_API_KEY environment variable.")
