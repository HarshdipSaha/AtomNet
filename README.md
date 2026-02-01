My project adapts the EMO cognitive architecture—originally designed for social deduction—to solve the mental modeling failures identified in the NegotiationToM benchmark, while integrating Reflexion to enable inter-game strategic learning.
Current Theory-of-Mind benchmarks (NegotiationToM) lack solvable architectures. This study evaluates a novel adaptation of the EMO framework, originally designed for social deduction, to the domain of imperfect information bargaining. Furthermore, we introduce an adversarial memory module based on Reflexion to test if linguistic reinforcement learning is effective in non-cooperative settings.


## Stage 1: Atomic Inference (The Guess)

The LLM reads the latest message $m_t$ and produces a raw probability update.

$$\hat{q}_{raw} \sim LLM(m_t, H_t)$$

**Example:** "Opponent asked for books, so $P(\text{Books}=High)$ increases to 0.8."

---

## Stage 2: The Economic Validator ( Novel Contribution)

apply an **Economic Consistency Check** ($V_{econ}$).

We define a contradiction function $C(action, belief)$:

$$C(a_t, B_t) = \begin{cases} 1 & \text{if } a_t = \text{Reject}(X) \text{ AND } B_t(X) = \text{High} \\ 0 & \text{otherwise} \end{cases}$$

If a contradiction is found ($C=1$), we trigger a **Correction Factor** ($\alpha$):

$$B_{refined}(\theta) = B_{raw}(\theta) \times (1 - \alpha)$$

**Meaning:** If the math says they should accept, but they rejected, our belief that "They Value X" must be penalized.

---

## Stage 3: The Decision Engine (Game Theory)

Now that we have the Belief ($B_t$), how do we choose the move? We use **Expected Utility Maximization**.

### The Math

Let $A$ be the set of possible actions (Offer Splits). The **Expected Utility** ($EU$) of an action $a$ is:

$$EU(a) = \sum_{\theta \in \Theta} B_t(\theta) \times P(\text{Accept}|a, \theta) \times U_{self}(a)$$

| Symbol | Description |
|--------|-------------|
| $B_t(\theta)$ | The probability from your JSON (e.g., "50% chance he likes books") |
| $P(\text{Accept})$ | The probability he says "Yes" to offer $a$, given he is type $\theta$ |
| $U_{self}(a)$ | How many points you get from the deal |

> **Rule:** If (Points for him in $a$) $\geq$ (his Reservation Value), then $P=1$. Else $P=0$.

### The Algorithm

1. Loop through all valid deals (e.g., 20 possible splits).
2. Calculate $EU$ for each.
3. Select $a^* = \arg\max EU(a)$.

---

## Stage 4: The Evolutionary Layer (Reflexion)

This runs **between games**.

### The Reflexion Loop

Let $\tau$ be the trajectory of the game (the full chat log). Let $R$ be the final score (Reward).

If $R < \text{Threshold}$, generate a **Self-Reflection** ($S_{ref}$):

$$S_{ref} \sim LLM(\text{"Analyze } \tau \text{. Why did we fail?"})$$

This $S_{ref}$ is stored in **Long Term Memory** ($M$):

$$M_{new} = M_{old} \cup \{S_{ref}\}$$


---

## Complete Algorithm (Pseudocode)

```python
# AToM-Net Master Algorithm

# Initialize Memory
M = []

for G in range(1, 11):  # Games 1 to 10
    # Uniform Prior
    B = {"Book": 0.33, "Hat": 0.33, "Ball": 0.33}
    
    while game_is_active:
        # =========================================
        # 1. PERCEPTION (NegotiationToM)
        # =========================================
        opponent_move = get_chat()
        
        # =========================================
        # 2. COGNITION (EMO + Economic Validator)
        # =========================================
        raw_guess = LLM.predict_intent(opponent_move)
        consistency_error = logic_check(raw_guess, opponent_move)
        
        if consistency_error > 0:
            B = bayes_update(B, raw_guess, penalty="high")
            trigger_deception_flag()
        else:
            B = bayes_update(B, raw_guess, penalty=None)
        
        # =========================================
        # 3. DECISION (Game Theory)
        # =========================================
        best_offer = None
        max_EU = -1
        
        for potential_deal in all_possible_deals:
            prob_accept = calculate_accept_prob(potential_deal, B)
            my_payoff = calculate_my_score(potential_deal)
            EU = prob_accept * my_payoff
            
            if EU > max_EU:
                max_EU = EU
                best_offer = potential_deal
        
        # =========================================
        # 4. ACTION GENERATION
        # =========================================
        if best_offer > threshold:
            output = f"I offer {best_offer}"
        else:
            output = "No deal, give me more."
    
    # =========================================
    # 5. EVOLUTION (Reflexion) - Runs after game
    # =========================================
    score = calculate_final_score()
    if score < score_threshold:
        lesson = LLM.reflect(game_history)
        M.append(lesson)



    # 5. Evolution (Reflexion)

```

"Multi-Issue Bargaining Task" (specifically the CaSiNo environment ).

Here is the "Ez" explanation:

1. The Setup (The Pile of Junk)
Imagine there is a pile of random items on a table between two people. In your specific game, the pile always contains exactly:

3 Books

2 Hats

1 Ball

Total Items: 6.

2. The Secret (The "Point Card")
This is the most important part. You and your opponent have Secret Point Cards.

You might secretly love Hats. (1 Hat = 5 Points).

Your Opponent might secretly hate Hats. (1 Hat = 0 Points).


Crucially: You do NOT know each other's point cards. You have to guess what they want based on what they say.

3. The Goal
Your goal is NOT to split things 50/50. Your goal is to get the items that are High Value on your secret card, and dump the items that are Low Value on your card to the opponent.

4. Example Turn
The Pile: 3 Books, 2 Hats, 1 Ball.

You (Secretly): Love Books (5 pts), Hate Hats (0 pts).

Opponent (Secretly): Loves Hats (5 pts), Hates Books (0 pts).

The Stupid Way to Play:

"Let's split everything half and half." Result: You get 1.5 Books (7.5 pts) + 1 Hat (0 pts) = 7.5 Points.

The Smart Way (Theory of Mind):

You realize the opponent keeps asking for Hats. You think: "Aha! He loves Hats. I will give him all the Hats if he gives me all the Books." "DEAL: I keep 3 Books, you keep 2 Hats." Result: You get 3 Books (15 pts) + 0 Hats (0 pts) = 15 Points.
    Score = Calculate_Final_Score()

    if Score < Score_Threshold:
        Lesson = LLM.reflect(Game_History)
        M.append(Lesson)
