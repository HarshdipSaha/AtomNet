import json
import os
from sklearn.metrics import f1_score
from atom_evaluator import AToMNetEvaluator
from colorama import Fore, Style, init

init(autoreset=True)

def binarize_intents(intent_string, valid_intents):
    """Converts a comma-separated intent string into a binary array for F1 calculation."""
    if intent_string == "None" or not intent_string:
        intent_string = "No-Intention"
        
    intents = [i.strip() for i in intent_string.split(",")]
    return [1 if valid in intents else 0 for valid in valid_intents]

def update_live_score_file(processed, total, exact_matches, intent_labels, intent_preds, dialogue_consistency, filepath="live_metrics.txt"):
    """Writes the running metrics to a text file in real-time."""
    if processed == 0: return

    # 1. Turn-Level Accuracy
    a1_d_acc = (exact_matches['a1_desire'] / processed) * 100
    a1_b_acc = (exact_matches['a1_belief'] / processed) * 100
    a2_d_acc = (exact_matches['a2_desire'] / processed) * 100
    a2_b_acc = (exact_matches['a2_belief'] / processed) * 100
    
    # 2. The "All" Score Accuracy
    all_acc = (exact_matches['all_score'] / processed) * 100

    # 3. Consistency Accuracy (Grouped by Dialogue)
    num_dialogues = len(dialogue_consistency)
    cons_desire = sum(1 for d in dialogue_consistency.values() if d['desire']) / num_dialogues * 100 if num_dialogues > 0 else 0
    cons_belief = sum(1 for d in dialogue_consistency.values() if d['belief']) / num_dialogues * 100 if num_dialogues > 0 else 0

    # 4. Intention F1
    try:
        micro_f1 = f1_score(intent_labels, intent_preds, average='micro', zero_division=0) * 100
        macro_f1 = f1_score(intent_labels, intent_preds, average='macro', zero_division=0) * 100
    except:
        micro_f1, macro_f1 = 0.0, 0.0

    live_text = f"""=== LIVE AToM-Net BENCHMARK SCORES ===
Progress: {processed} / {total} instances processed ({(processed/total)*100:.1f}%)

--- TURN-LEVEL EXACT MATCHES ---
Agent 1 Desire: {a1_d_acc:.2f}%
Agent 1 Belief: {a1_b_acc:.2f}%
Agent 2 Desire: {a2_d_acc:.2f}%
Agent 2 Belief: {a2_b_acc:.2f}%

--- THE "ALL" SCORE (The Ultimate Metric) ---
All Score (Desire+Belief+Intention Perfect Match): {all_acc:.2f}%

--- CONVERSATION CONSISTENCY ---
Unique Dialogues Tracked: {num_dialogues}
Desire Consistency: {cons_desire:.2f}%
Belief Consistency: {cons_belief:.2f}%

--- UTTERANCE INTENTION ---
Micro F1: {micro_f1:.2f}%
Macro F1: {macro_f1:.2f}%
======================================
* This file updates automatically after every dialogue turn *
"""
    with open(filepath, "w") as f:
        f.write(live_text)

def run_benchmark():
    print(f"{Fore.CYAN}=== STARTING LOCAL AToM-Net BENCHMARK ==={Style.RESET_ALL}")
    
    try:
        with open("data.json", "r") as f:
            dataset = json.load(f)
        total_instances = len(dataset)
        print(f"{Fore.GREEN}Success! Loaded {total_instances} evaluation instances.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Failed to load data.json: {e}{Style.RESET_ALL}")
        return

    evaluator = AToMNetEvaluator()
    valid_intents = evaluator.valid_intents
    saved_results = []
    
    # Trackers for ALL metrics
    exact_matches = {
        "a1_desire": 0, "a1_belief": 0,
        "a2_desire": 0, "a2_belief": 0,
        "all_score": 0
    }
    
    # Dictionary to track consistency across whole dialogues
    dialogue_consistency = {}
    intent_preds = []
    intent_labels = []

    for idx, row in enumerate(dataset):
        print(f"Processing Dialogue ID: {row['dialogue_id']} ({idx+1}/{total_instances})...")
        
        dialogue = row["dialogue"]
        u2_agent = row.get("utterance2_agent", "None")
        base_dialogue_id = str(row["dialogue_id"]).split("-")[0] # Extracts "0" from "0-1"
        
        # Initialize consistency tracker for new dialogues
        if base_dialogue_id not in dialogue_consistency:
            dialogue_consistency[base_dialogue_id] = {"desire": True, "belief": True}
        
        prediction = evaluator.evaluate_turn(dialogue, u2_agent)
        
        if not prediction:
            print(f"{Fore.RED}Skipping turn - API returned empty JSON.{Style.RESET_ALL}")
            # If the API fails, it ruins consistency for this dialogue
            dialogue_consistency[base_dialogue_id]["desire"] = False
            dialogue_consistency[base_dialogue_id]["belief"] = False
            continue 
            
        def norm(val):
            return "Not Given" if val == "None" or not val else val
            
        # 1. Calculate Exact Matches
        a1_d_correct = (prediction.get("agent1_desire_high") == norm(row.get("agent1_desire_high")) and
                        prediction.get("agent1_desire_medium") == norm(row.get("agent1_desire_medium")) and
                        prediction.get("agent1_desire_low") == norm(row.get("agent1_desire_low")))
        
        a1_b_correct = (prediction.get("agent1_belief_high") == norm(row.get("agent1_belief_high")) and
                        prediction.get("agent1_belief_medium") == norm(row.get("agent1_belief_medium")) and
                        prediction.get("agent1_belief_low") == norm(row.get("agent1_belief_low")))
        
        a2_d_correct = (prediction.get("agent2_desire_high") == norm(row.get("agent2_desire_high")) and
                        prediction.get("agent2_desire_medium") == norm(row.get("agent2_desire_medium")) and
                        prediction.get("agent2_desire_low") == norm(row.get("agent2_desire_low")))
        
        a2_b_correct = (prediction.get("agent2_belief_high") == norm(row.get("agent2_belief_high")) and
                        prediction.get("agent2_belief_medium") == norm(row.get("agent2_belief_medium")) and
                        prediction.get("agent2_belief_low") == norm(row.get("agent2_belief_low")))

        if a1_d_correct: exact_matches["a1_desire"] += 1
        if a1_b_correct: exact_matches["a1_belief"] += 1
        if a2_d_correct: exact_matches["a2_desire"] += 1
        if a2_b_correct: exact_matches["a2_belief"] += 1

        # 2. Update Consistency Tracker
        if not (a1_d_correct and a2_d_correct):
            dialogue_consistency[base_dialogue_id]["desire"] = False
        if not (a1_b_correct and a2_b_correct):
            dialogue_consistency[base_dialogue_id]["belief"] = False

        # 3. Calculate Intent Vectors
        pred_intent_str = prediction.get("utterance2_intent", "No-Intention")
        true_intent_str = row.get("utterance2_intent", "No-Intention")
        
        pred_vec = binarize_intents(pred_intent_str, valid_intents)
        true_vec = binarize_intents(true_intent_str, valid_intents)
        
        intent_preds.append(pred_vec)
        intent_labels.append(true_vec)
        
        intent_is_correct = (pred_vec == true_vec)

        # 4. Calculate "All" Score
        if a1_d_correct and a1_b_correct and a2_d_correct and a2_b_correct and intent_is_correct:
            exact_matches["all_score"] += 1
        
        # Save detailed payload
        saved_results.append({
            "dialogue_id": row["dialogue_id"],
            "dialogue": dialogue,
            "ground_truth": {
                "a1_desire_high": norm(row.get("agent1_desire_high")),
                "a1_belief_high": norm(row.get("agent1_belief_high")),
                "utterance2_intent": true_intent_str
            },
            "prediction": prediction
        })

        # Update the live metrics text file after every turn
        update_live_score_file(idx + 1, total_instances, exact_matches, intent_labels, intent_preds, dialogue_consistency)

        # Save JSON backups periodically
        if (idx + 1) % 10 == 0:
            with open("atom_local_results.json", "w") as f:
                json.dump(saved_results, f, indent=4)

    # Final Save
    with open("atom_local_results.json", "w") as f:
        json.dump(saved_results, f, indent=4)
        
    print(f"\n{Fore.GREEN}=== BENCHMARK COMPLETE ==={Style.RESET_ALL}")
    print(f"Final logs saved to {Fore.CYAN}atom_local_results.json{Style.RESET_ALL}")
    print(f"Final scores saved to {Fore.CYAN}live_metrics.txt{Style.RESET_ALL}")

if __name__ == "__main__":
    if os.environ.get("GROQ_API_KEY"):
        run_benchmark()
    else:
        print(f"{Fore.RED}Error: Please set your GROQ_API_KEY environment variable.{Style.RESET_ALL}")
