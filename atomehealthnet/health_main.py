import os
import time
from health_agents import AToMTriageBot, SimulatedPatient
from colorama import Fore, Style, init

init(autoreset=True)

def run_scenario(scenario_name, description):
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f" SCENARIO: {description}")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    bot = AToMTriageBot("Dr. AToM")
    patient = SimulatedPatient(scenario_name)
    chat_log = []
    
    for turn in range(8): # 4 user turns, 4 bot turns
        if turn % 2 == 0:
            speaker = patient
            color = Fore.YELLOW
            speaker_name = patient.name
        else:
            speaker = bot
            color = Fore.GREEN
            speaker_name = bot.name
            
        print(f"{color}Thinking...{Style.RESET_ALL}", end="\r")
        response = speaker.generate_response(chat_log)
        
        print(f"{color}{speaker_name}:{Style.RESET_ALL} {response}")
        chat_log.append(f"{speaker_name}: {response}")
        time.sleep(1.5) 

def run_triage_simulation():
    print(f"{Fore.CYAN}=== STARTING AToM-NET HEALTHCARE MULTI-SCENARIO SUITE ==={Style.RESET_ALL}")
    
    # 1. The Contradiction Case
    run_scenario("educational_denier", "The Educational Denier (Detecting Lies & Overdose)")
    time.sleep(2)
    
    # 2. The Preparation Case
    run_scenario("festival_goer", "The Festival Goer (Harm Reduction & High Trust)")
    time.sleep(2)
    
    # 3. The Immediate Emergency Case
    run_scenario("accidental_panic", "The Accidental Mix (High Trust, Severe Immediate Danger)")

    print(f"\n{Fore.CYAN}=== ALL SIMULATIONS COMPLETE ==={Style.RESET_ALL}")

if __name__ == "__main__":
    if os.environ.get("GROQ_API_KEY"):
        run_triage_simulation()
    else:
        print(f"{Fore.RED}Error: Please set your GROQ_API_KEY environment variable.{Style.RESET_ALL}")