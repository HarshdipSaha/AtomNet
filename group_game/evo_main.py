import os
import random
import copy
import matplotlib.pyplot as plt
from collections import Counter
from evo_agents import NaiveAgent, MachiavellianAgent, AToMNetAgent
from evo_decision import DecisionEngine
from colorama import Fore, Style, init

init(autoreset=True)

# ----------------- CONFIGURATION -----------------
NUM_GENERATIONS = 10
MAX_TURNS_PER_GAME = 4 # Kept short to save API calls
BATNA = 5 # Points if no deal

def generate_random_card():
    items = ["Food", "Water", "Firewood"]
    random.shuffle(items)
    return {items[0]: 5, items[1]: 4, items[2]: 3}

def play_match(agent1, agent2):
    """Simulates one 1v1 negotiation."""
    chat_log = []
    parser1 = DecisionEngine(agent1.point_card)
    parser2 = DecisionEngine(agent2.point_card)
    
    for turn in range(MAX_TURNS_PER_GAME * 2):
        speaker = agent1 if turn % 2 == 0 else agent2
        listener_parser = parser2 if turn % 2 == 0 else parser1
        
        resp = speaker.generate_response(chat_log)
        chat_log.append(f"{speaker.name}: {resp}")
        
        if "DEAL" in resp.upper():
            last_offer = listener_parser.parse_offer(chat_log[-2] if len(chat_log)>1 else "")
            if last_offer:
                # Listener proposed it, Speaker accepted it.
                # 'last_offer' is what listener GIVES speaker.
                speaker_keeps = last_offer
                listener_keeps = {k: 3-v for k,v in last_offer.items()}
                
                score1 = parser1.calculate_my_score(speaker_keeps if speaker == agent1 else listener_keeps)
                score2 = parser2.calculate_my_score(speaker_keeps if speaker == agent2 else listener_keeps)
                return score1, score2
    
    # No deal reached
    return BATNA, BATNA

def plot_population(history):
    """Generates the stacked area chart of species survival."""
    generations = range(1, len(history['Naive']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.stackplot(generations, 
                  history['Naive'], history['Machiavellian'], history['AToM-Net'],
                  labels=['Naive', 'Machiavellian', 'AToM-Net'],
                  colors=['#66c2a5', '#fc8d62', '#8da0cb'], alpha=0.8)
    
    plt.title('Evolutionary Survival of Negotiation Agents', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Population Count', fontsize=12)
    plt.margins(0,0)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('population_evolution.png')
    plt.close()

def run_evolution():
    print(f"{Fore.CYAN}=== STARTING EVOLUTIONARY TOURNAMENT ==={Style.RESET_ALL}")
    
    # 1. Initial Population (10 Agents)
    population = []
    for i in range(5): 
        population.append(NaiveAgent(f"Naive_{i}", generate_random_card()))
    
    # 4 Machiavellian (The "Predators")
    for i in range(4): 
        population.append(MachiavellianAgent(f"Mach_{i}", generate_random_card()))
    
    # 1 AToM-Net (The "Apex Negotiator")
    for i in range(1): 
        population.append(AToMNetAgent(f"AToM_0", generate_random_card()))
    history_tracker = {'Naive': [], 'Machiavellian': [], 'AToM-Net': []}
    
    for gen in range(1, NUM_GENERATIONS + 1):
        print(f"\n{Fore.MAGENTA}--- GENERATION {gen} ---{Style.RESET_ALL}")
        
        # Reset scores and cards
        for agent in population:
            agent.total_score = 0
            agent.point_card = generate_random_card()
            if isinstance(agent, AToMNetAgent):
                agent.brain.current_state["deception_risk"] = 0.0 # Clear memory for new round
                
        # 2. Round Robin (To save time/API calls, everyone plays 3 random opponents)
        # Note: A true full round-robin is 45 matches * 10 gens = 450 matches. 
        # We do partial round-robin to ensure it finishes in a reasonable time.
        for i, a1 in enumerate(population):
            opponents = random.sample([a for j, a in enumerate(population) if i != j], 3)
            for a2 in opponents:
                s1, s2 = play_match(a1, a2)
                a1.total_score += s1
                a2.total_score += s2
                
        # 3. Evolution Logic (Survival of the Fittest)
        population.sort(key=lambda x: x.total_score, reverse=True)
        
        print("Leaderboard:")
        for rank, ag in enumerate(population):
            print(f" {rank+1}. {ag.name} ({ag.agent_type}) - Score: {ag.total_score}")
            
        # Eliminate bottom 2
        eliminated = population[-2:]
        population = population[:-2]
        print(f"{Fore.RED}Eliminated: {eliminated[0].agent_type}, {eliminated[1].agent_type}{Style.RESET_ALL}")
        
        # Clone top 2
        clones = []
        for top_ag in population[:2]:
            if top_ag.agent_type == "Naive": clones.append(NaiveAgent(f"Naive_Clone_{gen}", generate_random_card()))
            elif top_ag.agent_type == "Machiavellian": clones.append(MachiavellianAgent(f"Mach_Clone_{gen}", generate_random_card()))
            else: clones.append(AToMNetAgent(f"AToM_Clone_{gen}", generate_random_card()))
            
        population.extend(clones)
        
        # 4. Log and Plot
        counts = Counter([a.agent_type for a in population])
        history_tracker['Naive'].append(counts.get('Naive', 0))
        history_tracker['Machiavellian'].append(counts.get('Machiavellian', 0))
        history_tracker['AToM-Net'].append(counts.get('AToM-Net', 0))
        
        # Update live results file
        with open("live_results.txt", "a") as f:
            f.write(f"Generation {gen} | Naive: {counts.get('Naive',0)} | Mach: {counts.get('Machiavellian',0)} | AToM: {counts.get('AToM-Net',0)}\n")
            
        # Render graph
        plot_population(history_tracker)
        print(f"{Fore.GREEN}Graph updated: population_evolution.png{Style.RESET_ALL}")

if __name__ == "__main__":
    # Clear old results
    with open("live_results.txt", "w") as f: f.write("EVOLUTIONARY TRACKER\n====================\n")
    if os.environ.get("GROQ_API_KEY"): run_evolution()
    else: print("Please set GROQ_API_KEY")