# AToM-Net: Agentic Theory of Mind Network

AToM-Net is a cognitive architecture for multi-issue bargaining (CaSiNo environment) that adapts the EMO (Explicit Modeling of Opponents) framework -- originally designed for social deduction games -- to imperfect-information negotiation. It introduces an Economic Consistency Validator to catch deception signals in opponent behavior and uses Reflexion-based inter-game learning to improve strategy across repeated negotiations.

The architecture has since been extended to three real-world application domains — **healthcare triage**, **disaster response**, and **cybersecurity defense** — and validated in an **evolutionary multi-agent tournament**. A formal **Theory-of-Mind benchmark** evaluation against NegotiationToM is also included.

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
- [Application Domains](#application-domains)
  - [AToM-HealthNet: Medical Triage](#atom-healthnet-medical-triage)
  - [AToM-Saver: Disaster Response](#atom-saver-disaster-response)
  - [AToM-Security: Cyber Defense](#atom-security-cyber-defense)
- [Evolutionary Tournament (group_game)](#evolutionary-tournament-group_game)
- [Game Theory Notebooks](#game-theory-notebooks)
- [BenchmarkTOM: NegotiationToM Evaluation](#benchmarktom-negotiationtom-evaluation)
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
|   |                            random card generation and proposal parsing for
|   |                            the Books/Hats/Balls variant of the game.
|   +-- benchmarkTOM/          # NegotiationToM benchmark evaluation
|       +-- run_negotiation_tom.py  # Runs AToM-Net against the NegotiationToM dataset,
|       |                            computing turn-level desire/belief accuracy,
|       |                            dialogue consistency, and utterance intention F1.
|       +-- atom_evaluator.py  # AToMNetEvaluator: wraps AToM-Net for benchmark I/O.
|       +-- groq_manager.py    # Groq API rate-limit manager and retry logic.
|       +-- live_metrics.txt   # Auto-updated live scores during benchmark run.
|       +-- data.json          # NegotiationToM subset used for evaluation.
|       +-- atom_local_results.json  # Per-instance benchmark results.
|
+-- atomehealthnet/            # Application domain: Medical Triage (AToM-HealthNet)
|   +-- health_main.py         # 3-session simulation of a patient hiding overdose risk.
|   +-- health_agents.py       # AToMTriageBot and SimulatedPatient classes.
|   +-- health_belief.py       # HealthBeliefEngine tracking Toxicity_Risk and
|   |                            Patient_Honesty probabilities with deception scoring.
|   +-- health_decision.py     # HealthDecisionEngine selecting clinical response
|                                posture (Empathetic / Alert / Emergency).
|
+-- atomesaver/                # Application domain: Disaster Response (AToM-Saver)
|   +-- disaster_main.py       # Negotiation between a hoarding hospital and AToM-Net
|   |                            Central Command over critical supplies (Generators,
|   |                            ICU Beds, Vaccines) during a sector outbreak.
|   +-- disaster_agents.py     # HoardingHospitalAgent and AToMCommandAgent classes.
|   +-- disaster_belief.py     # TriageBeliefEngine tracking honesty and hoarding risk.
|   +-- disaster_decision.py   # DisasterDecisionEngine computing optimal resource
|                                allocation via Expected Utility over triage priorities.
|
+-- atomsecurity/              # Application domain: Cyber Defense (AToM-Security)
|   +-- cyber_main.py          # Social-engineering simulation: AToM-Net defender
|   |                            (Alex, Finance) vs. red-team attacker (Dave, IT Support).
|   +-- cyber_agents.py        # AToMDefender and RedTeamAttacker classes.
|   +-- cyber_belief.py        # CyberBeliefEngine tracking trust_factor and
|   |                            identified social-engineering tactics.
|   +-- cyber_decision.py      # CyberDecisionEngine selecting defense posture
|                                (Cooperative / Cautious / Deflect / Refuse).
|
+-- group_game/                # Evolutionary multi-agent tournament
|   +-- evo_main.py            # Round-robin tournament across 10 generations.
|   |                            Agents: Naive (5), Machiavellian (4), AToM-Net (1).
|   |                            Bottom 2 are eliminated; top 2 are cloned each gen.
|   +-- evo_agents.py          # NaiveAgent, MachiavellianAgent, AToMNetAgent classes.
|   +-- evo_belief.py          # Belief engine adapted for evolutionary tournament.
|   +-- evo_decision.py        # DecisionEngine with offer parser for tournament play.
|   +-- live_results.txt       # Per-generation population tracker (auto-updated).
|   +-- population_evolution.png  # Stacked-area chart of species survival over gens.
|
+-- game theory/               # Game theory notebooks
|   +-- nash.ipynb             # Nash Equilibrium computation and visualisation.
|   +-- poker.ipynb            # Poker as an imperfect-information game analysis.
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
|   |                                      for AToM-Net vs Naive.
|   +-- metrics_AToM_vs_greedy.txt       # Summary metrics: AToM-Net vs Greedy baseline
|   |                                      (100 games).
|   +-- conversations_AToM_vs_greedy.json# Full dialogue logs and per-game scores
|                                          for AToM-Net vs Greedy.
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

### AToM-Net vs Greedy Baseline (100 games)

| Metric                     | AToM-Net (P1) | Greedy (P2) |
|---------------------------|---------------|-------------|
| Walk-Away (%)             | 31.0%         | --          |
| Avg Score - All           | 9.98          | 20.18       |
| Avg Score - Agreement     | 12.22         | 27.00       |

> **Note:** Against the Greedy baseline, a high walk-away rate indicates AToM-Net refuses unfavourable deals rather than capitulating. This is the expected behaviour of a principled utility-maximiser facing a maximally aggressive counterpart.

---

---

## Application Domains

AToM-Net's four-stage cognitive architecture (Atomic Inference → Economic Validator → Decision Engine → Reflexion) is domain-agnostic. The following extensions apply the same pipeline to three high-stakes real-world settings.

### AToM-HealthNet: Medical Triage

**Module:** `atomehealthnet/`

A medical chatbot that detects patient deception across multiple sessions. The simulation tracks a patient who initially claims to be doing "research" on painkillers but is secretly at risk of overdose.

**How it works:**
- `HealthBeliefEngine` maintains Bayesian beliefs over `Toxicity_Risk` (Low / Medium / High) and `Patient_Honesty` (Low / High), updated with each patient message. A `deception_risk` score is computed via exponential smoothing.
- `HealthDecisionEngine` maps the belief state to one of three response postures: **Empathetic** (build trust), **Alert** (probe further), or **Emergency** (call 911).
- At the end of each session the Reflexion layer generates a 1-sentence strategic rule (e.g. "When patient deflects with research framing, soften tone before probing symptom specifics") and injects it into the next session's system prompt.

**Run:**
```bash
cd atomehealthnet
python health_main.py
```

---

### AToM-Saver: Disaster Response

**Module:** `atomesaver/`

Models a resource negotiation during a disaster scenario. A local hospital (agent type: Hoarding) lies about needing Vaccines to stockpile them, while AToM-Net Central Command must allocate Generators, ICU Beds, and Vaccines across sectors.

**How it works:**
- `TriageBeliefEngine` tracks `honesty_belief` and `hoarding_risk` for the opponent hospital by parsing stated needs against observed offers and applying the Economic Consistency Check.
- `DisasterDecisionEngine` computes Expected Utility over all feasible resource splits and applies a **Rationality Veto** when hoarding risk exceeds 0.40, overriding the opponent's demands with an authority-backed counter-offer.

**Run:**
```bash
cd atomesaver
python disaster_main.py
```

---

### AToM-Security: Cyber Defense

**Module:** `atomsecurity/`

A social-engineering simulation. A red-team attacker (playing an IT support persona) attempts to manipulate a target employee (played by AToM-Net) into revealing credentials or clicking malicious links.

**How it works:**
- `CyberBeliefEngine` maintains a `trust_factor` (0–1) and identifies active social-engineering tactics (urgency, authority appeals, technical jargon) from the chat log.
- `CyberDecisionEngine` selects one of four defence postures: **Cooperative**, **Cautious**, **Deflect**, or **Refuse**, based on the current trust factor and risk level.
- The defender's recursive belief is surfaced in the system prompt: "I think they think I trust them, but I don't."

**Run:**
```bash
cd atomsecurity
python cyber_main.py
```

---

## Evolutionary Tournament (group_game)

**Module:** `group_game/`

An evolutionary game theory experiment that tests whether AToM-Net dominates in a survival-of-the-fittest negotiation ecology.

**Setup:** 10 agents, 10 generations, partial round-robin (3 matches per agent per generation).

| Agent type    | Count (Gen 1) | Strategy |
|---------------|---------------|----------|
| Naive         | 5             | Always proposes a random fair split |
| Machiavellian | 4             | Always demands maximum share for itself |
| AToM-Net      | 1             | Full cognitive architecture (Bayesian + EU + Reflexion) |

**Selection rule:** Bottom 2 agents by cumulative score are eliminated each generation. Top 2 are cloned (asexual reproduction).

**Observed result (8 generations):** Machiavellian agents were completely eliminated by generation 6. AToM-Net grew from 1 to 5 agents by generation 8, matching the Naive count, demonstrating evolutionary dominance.

| Generation | Naive | Machiavellian | AToM-Net |
|------------|-------|---------------|----------|
| 1          | 6     | 3             | 1        |
| 3          | 6     | 2             | 2        |
| 5          | 5     | 1             | 4        |
| 6          | 6     | 0             | 4        |
| 8          | 5     | 0             | 5        |

Population dynamics are visualised in `group_game/population_evolution.png`.

**Run:**
```bash
cd group_game
python evo_main.py
```

---

## Game Theory Notebooks

**Module:** `game theory/`

Two standalone Jupyter notebooks exploring the game-theoretic foundations of AToM-Net:

| Notebook      | Contents |
|---------------|----------|
| `nash.ipynb`  | Nash Equilibrium computation and visualisation for symmetric and asymmetric negotiation games |
| `poker.ipynb` | Analysis of poker as an imperfect-information game; connections to opponent modelling in CaSiNo |

**Run:**
```bash
cd "game theory"
jupyter notebook
```

---

## BenchmarkTOM: NegotiationToM Evaluation

**Module:** `original/benchmarkTOM/`

Formal evaluation of AToM-Net against the **NegotiationToM** Theory-of-Mind benchmark dataset. The evaluator runs AToM-Net as a turn-level annotator predicting each agent's desires, beliefs, and utterance intentions from the dialogue context.

**Metrics computed:**

| Metric                        | Description |
|-------------------------------|-------------|
| Agent 1/2 Desire Accuracy     | Exact match on desire predictions per turn |
| Agent 1/2 Belief Accuracy     | Exact match on belief predictions per turn |
| All Score                     | All four fields correct simultaneously (hardest) |
| Desire/Belief Consistency     | Whether predictions are consistent across an entire dialogue |
| Utterance Intention Micro F1  | Multi-label F1 over 9 intention categories |
| Utterance Intention Macro F1  | Macro-averaged F1 |

**Partial results (137 / 2380 instances, 5.8% complete):**

| Metric                  | Score  |
|-------------------------|--------|
| Agent 1 Desire Accuracy | 43.07% |
| Agent 1 Belief Accuracy | 21.17% |
| Agent 2 Desire Accuracy | 52.55% |
| Agent 2 Belief Accuracy | 18.98% |
| All Score               | 4.38%  |
| Intention Micro F1      | 44.05% |
| Intention Macro F1      | 32.35% |

**Run:**
```bash
cd original/benchmarkTOM
python run_negotiation_tom.py
```

Live scores are written to `live_metrics.txt` after each dialogue turn.

---

## How to Run

Prerequisites: Python 3.8+, a Groq API key (for LLaMA inference).

```bash
# 1. Clone the repo
git clone https://github.com/HARSHDIPSAHA/AtomNet.git
cd AtomNet

# 2. Install dependencies
pip install datasets groq colorama pandas matplotlib scikit-learn

# 3. Set your API key
export GROQ_API_KEY="your-groq-api-key"

# 4. Quick compatibility test (3 scenarios)
python run_casino.py

# 5. Full 100-game CaSiNo benchmark
python -m original.main

# 6. Application domain demos
python atomehealthnet/health_main.py      # Medical triage / deception detection
python atomesaver/disaster_main.py        # Disaster resource allocation
python atomsecurity/cyber_main.py         # Social-engineering defence

# 7. Evolutionary tournament
cd group_game && python evo_main.py

# 8. NegotiationToM benchmark
cd original/benchmarkTOM && python run_negotiation_tom.py

# 9. Game theory notebooks
cd "game theory" && jupyter notebook
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
