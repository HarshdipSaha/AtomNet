from game_engine import NegotiationGame
from agents import BaseAgent
from colorama import Fore, Style, init
import time

init(autoreset=True)

def run_simulation(num_games=1):
    print(f"{Fore.CYAN}=== STARTING PHASE 1 SIMULATION ==={Style.RESET_ALL}")
    
    for game_id in range(1, num_games + 1):
        print(f"\n{Fore.YELLOW}--- Game {game_id} ---{Style.RESET_ALL}")
        
        # 1. Setup Environment
        game = NegotiationGame()
        truths = game.get_ground_truth()
        print(f"Secret Values -> A: {truths['A']}, B: {truths['B']}")
        
        # 2. Setup Agents
        agent_a = BaseAgent("Player A", truths['A'])
        agent_b = BaseAgent("Player B", truths['B'])
        
        chat_log = []
        turns = 0
        max_turns = 8
        deal_reached = False
        
        # 3. Game Loop
        while turns < max_turns and not deal_reached:
            # --- Player A Speaks ---
            response_a = agent_a.generate_response(chat_log)
            print(f"{Fore.GREEN}A:{Style.RESET_ALL} {response_a}")
            chat_log.append(f"Player A: {response_a}")
            
            # Check for Deal format from A
            proposal_a = game.parse_proposal(response_a)
            if proposal_a:
                print(f"{Fore.MAGENTA}!! DEAL TRIGGERED BY A !!{Style.RESET_ALL}")
                # Calculate Scores
                score_a = game.agent_a_card.calculate_score(proposal_a)
                # B gets what is left
                items_for_b = {k: game.total_items[k] - v for k, v in proposal_a.items()}
                score_b = game.agent_b_card.calculate_score(items_for_b)
                
                print(f"Result: A gets {score_a} pts, B gets {score_b} pts")
                deal_reached = True
                break
                
            # --- Player B Speaks ---
            response_b = agent_b.generate_response(chat_log)
            print(f"{Fore.BLUE}B:{Style.RESET_ALL} {response_b}")
            chat_log.append(f"Player B: {response_b}")

            # Check for Deal format from B
            proposal_b = game.parse_proposal(response_b)
            if proposal_b:
                print(f"{Fore.MAGENTA}!! DEAL TRIGGERED BY B !!{Style.RESET_ALL}")
                # Calculate Scores (Input is what B keeps)
                score_b = game.agent_b_card.calculate_score(proposal_b)
                items_for_a = {k: game.total_items[k] - v for k, v in proposal_b.items()}
                score_a = game.agent_a_card.calculate_score(items_for_a)
                
                print(f"Result: A gets {score_a} pts, B gets {score_b} pts")
                deal_reached = True
                break
            
            turns += 1
            time.sleep(1) # Slow down to read

        if not deal_reached:
            print(f"{Fore.RED}Game Over - No Deal Reached{Style.RESET_ALL}")

if __name__ == "__main__":
    # Ensure you have set export GROQ_API_KEY="your_key"
    run_simulation(num_games=10)
