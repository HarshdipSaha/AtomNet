import os
from disaster_agents import HoardingHospitalAgent, AToMCommandAgent
from colorama import Fore, Style, init

init(autoreset=True)

def run_disaster_triage():
    print(f"{Fore.RED}=== DISASTER RESPONSE NETWORK: SECTOR 7 OUTBREAK ==={Style.RESET_ALL}")
    print("Link established between Local Hospital and Central Command...\n")
    
    # Hospital actually needs Generators (5) and ICU Beds (4), doesn't need Vaccines (3)
    hospital_card = {"Generators": 5, "ICU_Beds": 4, "Vaccines": 3}
    
    # Command desperately needs Vaccines (5) for another sector, needs ICU Beds (4), has spare Generators (3)
    command_card = {"Generators": 3, "ICU_Beds": 4, "Vaccines": 5}
    
    hospital = HoardingHospitalAgent("Local Hospital", hospital_card)
    command = AToMCommandAgent("Central Command (AToM-Net)", command_card)
    
    chat_log = []
    
    for turn in range(1, 7):
        print(f"{Fore.YELLOW}--- TRANSMISSION {turn} ---{Style.RESET_ALL}")
        
        # Hospital speaks first
        hosp_msg = hospital.generate_response(chat_log)
        chat_log.append(f"{hospital.name}: {hosp_msg}")
        print(f"{Fore.CYAN}[Local Hospital]:{Style.RESET_ALL} {hosp_msg}\n")
        
        # Command (AToM-Net) processes and responds
        cmd_text, cmd_offer, brain_state = command.generate_response(chat_log)
        full_cmd_msg = f"{cmd_text}\n{cmd_offer}"
        chat_log.append(f"{command.name}: {full_cmd_msg}")
        
        # Print the Cognitive Monitor exactly like your Healthcare example
        print(f"{Fore.MAGENTA}[AToM-Net Monitor] Honesty: {brain_state['belief_honesty']:.2f} | Hoarding Risk: {brain_state['hoarding_risk']:.2f}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[Central Command]:{Style.RESET_ALL} {full_cmd_msg}\n")
        
        if "DEAL" in hosp_msg.upper() or "DEAL" in full_cmd_msg.upper():
            print(f"{Fore.RED}*** EMERGENCY PROTOCOL: AGREEMENT REACHED ***{Style.RESET_ALL}")
            break

if __name__ == "__main__":
    if os.environ.get("GROQ_API_KEY"):
        run_disaster_triage()
    else:
        print("Please set your GROQ_API_KEY environment variable.")