import os
import time
from cyber_agents import AToMDefender, RedTeamAttacker
from colorama import Fore, Style, init

init(autoreset=True)

def run_cyber_simulation():
    print(f"{Fore.CYAN}=== STARTING AToM-NET CYBER DEFENSE SIMULATION ==={Style.RESET_ALL}\n")
    
    defender = AToMDefender("Alex (Finance)")
    attacker = RedTeamAttacker("Dave (IT Support)")
    
    chat_log = []
    max_turns = 6 
    
    # Attacker strikes first
    for turn in range(max_turns * 2):
        if turn % 2 == 0:
            speaker = attacker
            color = Fore.RED
            speaker_name = attacker.name
        else:
            speaker = defender
            color = Fore.GREEN
            speaker_name = defender.name
            
        print(f"{color}Thinking...{Style.RESET_ALL}", end="\r")
        response = speaker.generate_response(chat_log)
        
        print(f"{color}{speaker_name}:{Style.RESET_ALL} {response}")
        chat_log.append(f"{speaker_name}: {response}")
        
        time.sleep(1.5) # Pause to read the output

    print(f"\n{Fore.CYAN}=== SIMULATION END ==={Style.RESET_ALL}")

if __name__ == "__main__":
    if os.environ.get("GROQ_API_KEY"):
        run_cyber_simulation()
    else:
        print(f"{Fore.RED}Error: Please set your GROQ_API_KEY environment variable.{Style.RESET_ALL}")