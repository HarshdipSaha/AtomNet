# AToM-Net: Agentic Theory of Mind Network

AToM-Net is a cognitive architecture for multi-issue bargaining (CaSiNo environment) that adapts the EMO (Explicit Modeling of Opponents) framework -- originally designed for social deduction games -- to imperfect-information negotiation. It introduces an Economic Consistency Validator to catch deception signals in opponent behavior and uses Reflexion-based inter-game learning to improve strategy across repeated negotiations.
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/41d2c71a-97f8-4ba9-9ae5-e9c914808b29" />

## Table of Contents

- [Motivation](#motivation)
- [Architecture Overview](#architecture-overview)
  - [Stage 1: Atomic Inference](#stage-1-atomic-inference-the-guess)
  - [Stage 2: Economic Validator](#stage-2-economic-validator-novel-contribution)
  - [Stage 3: Decision Engine](#stage-3-decision-engine-game-theory)
  - [Stage 4: Evolutionary Layer](#stage-4-evolutionary-layer-reflexion)
- [Flowchart](#flowchart)
- [Mathematical Formulation](#mathematical-formulation)
- [The CaSiNo Task Explained](#the-casino-task-explained)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [How to Run](#how-to-run)
- [Comparison with ASTRA](#comparison-with-astra)
- [References](#references)

---

## Motivation

Current Theory-of-Mind benchmarks such as NegotiationToM expose critical failure modes: LLM agents reveal private information, fail to model opponent beliefs, and accept dominated deals. AToM-Net addresses this by combining three components that no prior single architecture integrates:

1. Bayesian belief tracking over the opponent's hidden point card.
2. An economic contradiction detector that catches mismatches between observed actions and inferred types.
3. A Reflexion memory module that stores inter-game lessons and feeds them into future negotiations.

---

## Architecture Overview
<img width="1332" height="5066" alt="image" src="https://github.com/user-attachments/assets/f8acb7b8-0c3d-4e87-b246-96ebff64f10e" />

### Stage 1: Atomic Inference (The Guess)

At each dialogue turn t, the LLM reads the latest message m_t together with the full dialogue history H_t and produces a raw probability estimate over the opponent's type:

$$\hat{q}_{raw} \sim \text{LLM}(m_t, H_t)$$

Concretely, this outputs a JSON with P(Item = High) for each item (Food, Water, Firewood). The LLM is constrained to reason about "cheap talk" (what the opponent claims) and "costly signals" (what the opponent actually offers). Only costly signals trigger the formal Bayesian update in the BeliefEngine.

### Stage 2: Economic Validator (Novel Contribution)

After the raw inference, we apply an Economic Consistency Check V_econ that detects contradictions between observed actions and stated/inferred beliefs.

Define a contradiction function C mapping an action-belief pair to {0, 1}:

$$C(a_t, B_t) = \begin{cases} 1 & \text{if } a_t = \text{Reject}(X) \text{ AND } B_t(X) = \text{High} \\ 0 & \text{otherwise} \end{cases}$$

The intuition: if the agent's Bayesian model says the opponent values item X highly, but the opponent rejects a deal that is generous on X, then either (a) our belief is wrong, or (b) the opponent is bluffing. In both cases, the current belief needs correction.

When C = 1, a correction factor alpha in (0, 1) is applied:

$$B_{refined}(\theta) = B_{raw}(\theta) \times (1 - \alpha)$$

where alpha = 0.8 by default (configurable in belief.py). The posterior is then re-normalized to sum to 1. Additionally, a deception risk score is incremented:

$$d_{t+1} = \beta \cdot d_t + (1 - \beta) \cdot C(a_t, B_t)$$

where beta = 0.2 is an exponential smoothing factor. This deception score is injected into the LLM's system prompt to make it more cautious in subsequent turns.

### Stage 3: Decision Engine (Game Theory)

Given the corrected belief vector B_t (a probability distribution over opponent types theta in Theta), the agent selects an offer a from the set A of all feasible item allocations by maximizing Expected Utility:

$$EU(a) = \sum_{\theta \in \Theta} B_t(\theta) \times P(\text{Accept} \mid a, \theta) \times U_{self}(a)$$

where:

| Symbol | Definition |
|--------|-----------|
| B_t(theta) | Posterior probability that the opponent is type theta (e.g., "values Food = High"). This comes from the Bayesian tracker. |
| P(Accept given a, theta) | Probability the opponent accepts offer a if their true type is theta. Computed as a step function: if the deal gives the opponent at least their reservation value under type theta, P = 1; otherwise P = 0. |
| U_self(a) | The agent's own payoff under offer a, computed as sum over items of (items_kept * own_point_value). |

The optimization is exhaustive over the discrete deal space. In CaSiNo with 3 items and 3 units each, there are at most (3+1)^3 = 64 possible allocations, so brute-force enumeration is tractable:

```
a* = argmax_{a in A} EU(a)
```

The acceptance thresholds are calibrated for CaSiNo point ranges:

| Opponent estimated score | P(Accept) |
|-------------------------|-----------|
| >= 18                   | 1.0       |
| >= 14                   | 0.5       |
| >= 10                   | 0.1       |
| < 10                    | 0.0       |

### Stage 4: Evolutionary Layer (Reflexion)

This module runs between games, not within a single negotiation. After game G finishes with trajectory tau (the full dialogue log) and reward R (the agent's final score):

If R < threshold (default = 15 points):

$$S_{ref} \sim \text{LLM}(\text{"Analyze } \tau \text{. Why did we fail?"})$$

This self-reflection S_ref is appended to long-term memory M:

$$M_{new} = M_{old} \cup \{S_{ref}\}$$

In subsequent games, M is injected into the system prompt. The LLM can thus learn patterns like "When the opponent aggressively claims to want Food, they are usually bluffing -- focus on Firewood instead." Over 10 games, this accumulates a small set of strategic lessons that bias future play without changing model weights (linguistic reinforcement learning).

---

## Flowchart

```
                     +---------------------------+
                     |    Game Starts (Round t)   |
                     +---------------------------+
                                  |
                                  v
                     +---------------------------+
                     | 1. PERCEPTION              |
                     |    Read opponent message    |
                     |    m_t from dialogue        |
                     +---------------------------+
                                  |
                                  v
                     +---------------------------+
                     | 2. ATOMIC INFERENCE         |
                     |    LLM extracts cheap talk  |
                     |    Parse actual offer       |
                     |    (costly signal)          |
                     +---------------------------+
                                  |
                                  v
                     +---------------------------+
                     | 3. BAYESIAN UPDATE          |
                     |    Apply Bayes' rule:       |
                     |    P(H|action) =            |
                     |    P(action|H)*P(H)/P(act)  |
                     +---------------------------+
                                  |
                                  v
                     +---------------------------+
                     | 4. ECONOMIC VALIDATOR       |
                     |    C(action, belief) = ?    |
                     |    If C=1:                  |
                     |      B *= (1 - alpha)       |
                     |      deception_risk += 1    |
                     +---------------------------+
                                  |
                                  v
                     +---------------------------+
                     | 5. DECISION ENGINE          |
                     |    For each possible deal:  |
                     |      Compute EU(a)          |
                     |    Select a* = argmax EU    |
                     +---------------------------+
                                  |
                                  v
                     +---------------------------+
                     | 6. ACTION GENERATION        |
                     |    If EU(a*) > threshold:   |
                     |      Propose offer a*       |
                     |    Else:                    |
                     |      Request better terms   |
                     +---------------------------+
                                  |
                                  v
                     +---------------------------+
                     | 7. DEADLINE PRESSURE        |
                     |    Reservation value drops  |
                     |    each turn:               |
                     |    [25,23,20,18,15,12,10,.] |
                     |    Auto-accept if score >=  |
                     |    current reservation      |
                     +---------------------------+
                                  |
                                  v
                   +-----------+     +-------------+
                   | Deal?     |---->| Next Turn   |
                   | No        |     +-------------+
                   +-----------+
                        |
                        v (Game Over)
                     +---------------------------+
                     | 8. REFLEXION (Inter-Game)   |
                     |    If score < threshold:    |
                     |      Generate S_ref         |
                     |      M = M + {S_ref}        |
                     |    Feed M into next game    |
                     +---------------------------+
```

---

## Mathematical Formulation

**Bayesian Update Rule (BeliefEngine):**

For each item i in {Food, Water, Firewood} and each action type a in {offer_generous, offer_stingy, offer_neutral, reject_generous}:

$$P(H_i = \text{High} \mid a) = \frac{P(a \mid H_i = \text{High}) \cdot P(H_i = \text{High})}{P(a \mid H_i = \text{High}) \cdot P(H_i = \text{High}) + P(a \mid H_i = \text{Low}) \cdot P(H_i = \text{Low})}$$

The likelihoods are fixed priors:

| Action type       | P(action given High) | P(action given Low) |
|-------------------|---------------------|---------------------|
| offer_generous    | 0.2                 | 0.8                 |
| offer_stingy      | 0.8                 | 0.2                 |
| offer_neutral     | 0.5                 | 0.5                 |
| reject_generous   | 0.1                 | 0.9                 |

**Expected Utility**

$$
EU(a) = \sum_{\theta \in \Theta} B_t(\theta) \cdot \mathbb{1}[opp(a,\theta) \ge RV_{\theta}] \cdot U_{self}(a)
$$

where RV_theta is the opponent's reservation value under type theta.

**Estimated Opponent Value per Item:**

$$\hat{V}_{\text{opp}}(i) = P(H_i = \text{High}) \times 5 + P(H_i = \text{Med}) \times 4 + P(H_i = \text{Low}) \times 3$$

where P(Med) = max(0, 1 - P(High) - P(Low)).

**Deception Risk (Exponential Moving Average):**

$$d_{t+1} = 0.2 \cdot d_t + 0.8 \cdot C(a_t, B_t)$$

---

## The CaSiNo Task Explained

CaSiNo (Camp Site Negotiation) is a multi-issue bargaining benchmark. Two players negotiate over a fixed pool of items:

- 3 packages of Food
- 3 packages of Water
- 3 packages of Firewood

Each player has a secret "point card" that assigns different values to each item type. The standard CaSiNo mapping is:

| Priority level | Points |
|---------------|--------|
| High          | 5      |
| Medium        | 4      |
| Low           | 3      |

The total pool value for any player is always 3*5 + 3*4 + 3*3 = 36 points. The goal is to claim items that are high on your own card while giving away items that are low on your card, by inferring what the opponent values through dialogue.

A "naive" split (50/50 of everything) typically yields around 18 points per player. A Theory-of-Mind agent that correctly infers the opponent's priorities can achieve 25+ points by trading low-value items for high-value items (logrolling).

---

## Repository Structure

```
AtomNet/
+-- README.md                  # This file. Project documentation.
+-- run_casino.py              # Entry point for testing the agent on real CaSiNo data
|                                from HuggingFace. Downloads the dataset, maps
|                                CaSiNo items (Food/Water/Firewood) to the agent's
|                                internal representation, and runs a mini negotiation
|                                loop to verify compatibility.
|
+-- original/                  # Core AToM-Net implementation
|   +-- main.py                # Main simulation driver. Loads CaSiNo dataset,
|   |                            runs NUM_GAMES (default=100) negotiations, computes
|   |                            ASTRA-format metrics (Walk-Away %, Avg Score All,
|   |                            Avg Score Agreement), and saves results to JSON.
|   +-- agents.py              # BaseAgent class. Wraps the BeliefEngine and
|   |                            DecisionEngine with an LLM (Groq/LLaMA). Implements
|   |                            Strategic Ignorance (prevents info leaking), Ticking
|   |                            Clock (reservation values decay per turn), and
|   |                            auto-accept logic (bypasses LLM if offer exceeds
|   |                            reservation value).
|   +-- belief.py              # BeliefEngine. Bayesian probability tracker with
|   |                            LLM-based cheap talk extraction, fixed likelihood
|   |                            tables, and the Economic Validator (deception
|   |                            detection via contradiction function C).
|   +-- decision.py            # DecisionEngine. Brute-force Expected Utility
|   |                            maximizer over all 64 possible deals. Estimates
|   |                            opponent values from Bayesian posteriors and
|   |                            computes P(Accept) via threshold-based step function.
|   +-- game_engine.py         # NegotiationGame and PointCard classes. Handles
|                                random card generation and proposal parsing for
|                                the Books/Hats/Balls variant of the game.
|
+-- results/                   # Experiment output files
|   +-- metrics_AToM_vs_atom.txt         # Summary metrics: AToM-Net vs ATOM baseline
|   |                                      (100 games). Walk-Away, Avg Score All,
|   |                                      Avg Score Agreement.
|   +-- conversations_AToM_vs_atom.json  # Full dialogue logs and per-game scores
|   |                                      for AToM-Net vs ATOM.
|   +-- metrics_AToM_vs_naive.txt        # Summary metrics: AToM-Net vs Naive baseline
|   |                                      (100 games).
|   +-- conversations_AToM_vs_naive.json # Full dialogue logs and per-game scores
|                                          for AToM-Net vs Naive.
|
+-- __pycache__/               # Python bytecode cache (auto-generated, can be
                                 gitignored).
```

---

## Results

### AToM-Net vs Naive Baseline (100 games)

| Metric                     | AToM-Net (P1) | Naive (P2) |
|---------------------------|---------------|------------|
| Walk-Away (%)             | 1.0%          | --         |
| Avg Score - All           | 23.30         | 12.53      |
| Avg Score - Agreement     | 23.48         | 12.61      |

### AToM-Net vs ATOM Baseline (100 games)

| Metric                     | AToM-Net (P1) | ATOM (P2)  |
|---------------------------|---------------|------------|
| Walk-Away (%)             | 0.0%          | --         |
| Avg Score - All           | 17.36         | 21.40      |
| Avg Score - Agreement     | 17.36         | 21.40      |

---

## How to Run

Prerequisites: Python 3.8+, a Groq API key (for LLaMA inference).

```bash
# 1. Clone the repo
git clone https://github.com/HARSHDIPSAHA/AtomNet.git
cd AtomNet

# 2. Install dependencies
pip install datasets groq colorama pandas

# 3. Set your API key
export GROQ_API_KEY="your-groq-api-key"

# 4. Quick compatibility test (3 scenarios)
python run_casino.py

# 5. Full 100-game benchmark
python -m original.main
```

---

## Comparison with ASTRA

See the analysis section below the README for a detailed comparison between AToM-Net and ASTRA (EMNLP 2025).

---

## References

- CaSiNo: A Corpus of Campsite Negotiation Dialogues (Chawla et al., 2021)
- Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)
- EMO: Earth-Mover-based Opponent Modeling (Xu et al., 2023)
- ASTRA: A Negotiation Agent with Adaptive and Strategic Reasoning (Kwon et al., EMNLP 2025)
