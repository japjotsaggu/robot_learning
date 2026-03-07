# Robot Learning — Imitation Learning & Residual Reinforcement Learning

A robot learning system that trains an agent to navigate a 2D terrain environment using three algorithms: **Behavioural Cloning**, **DAgger**, and **Residual Reinforcement Learning**. All three are implemented with a budget-aware training loop.


## The Task

The robot starts on a blue line on the left of a 2D environment and must reach a red goal line on the right. The terrain is textured, dark regions are faster to traverse, light regions are slower. The robot observes a 3-dimensional observation vector rather than its raw state, and outputs a 2D continuous action each step.

**Training** is budget-limited (see below). **Testing** measures how quickly the robot crosses the goal line, or how close it got if it didn't reach it.

---

## Project Structure

```
robot.py   ← all implementation
robot-learning.py  ← main loop 
environment.py     ← terrain dynamics
constants.py       ← costs, dimensions, CEM parameters
graphics.py        ← VisualisationLine
```

Switch algorithm by changing one line at the top of `robot.py`:

```python
ALGORITHM = "bc"          # or "dagger" or "residual_rl"
```
---

## Budget System

Training is constrained by a budget of **$100**. Every operation has a cost:

| Operation | Cost |
|---|---|
| Demo request (flat fee) | $10.00 |
| Each step inside a demo | $0.30 |
| Each env step (action type 1) | $0.002 |
| Compute time | $0.03 / second |
| Environment reset | $5.00 |

A demo of 30 steps costs `10 + 30 × 0.3 = $19.00`. Going below zero incurs a penalty. The code maintains a `SAFETY_BUFFER = $6.00` and checks `money` before every expensive operation.

---

## Algorithms

### 1. Behavioural Cloning (`"bc"`)

The simplest approach. The robot requests one expert demonstration and trains a policy to imitate it using supervised learning (MSE loss).

**Training procedure:**
1. Request 1 demonstration of length `DEMO_LENGTH`
2. Train `PolicyNetwork` by minimising `MSE(π(obs), expert_action)`
3. End training

**Architecture:** 3-layer MLP → Tanh → scaled by `MAX_ACTION_MAGNITUDE`

**Budget spend:** ~$21 / $100

**Limitation:** BC suffers from *covariate shift*. The policy is only trained on states the expert visits, not the states the robot itself ends up in. Errors compound at test time.

---

### 2. DAgger — Dataset Aggregation (`"dagger"`)

DAgger addresses the covariate shift problem of BC by iteratively expanding the training dataset with states the *learner* visits, not just states from the initial demo.

**Training procedure (each round `i`):**
1. Request a correction demo from the robot's current position
2. Aggregate `(obs, expert_action)` pairs into a growing dataset
3. Retrain the policy on the full aggregated dataset
4. Repeat for `DAGGER_MAX_ROUNDS` rounds (stopping early if budget runs low)

The mixing coefficient `β_i = β^i` (where `β = 0.5`) controls how often the expert would be queried in a full DAgger implementation. Here it decays each round and is logged for reference.

**Budget spend:** ~$81 / $100 (4 demos × $19 + compute)

**Why it works:** As rounds progress, the dataset covers more of the states the learner actually reaches, making the policy progressively more robust.

---

### 3. Residual Reinforcement Learning (`"residual_rl"`)

Combines the sample-efficiency of imitation learning with the flexibility of RL. A frozen BC policy provides a reasonable base, and a small residual actor is trained with **TD3** to correct it.

```
π_total(obs)  =  π_bc(obs)  +  π_residual(obs)
                 [frozen]       [trained with TD3]
```

**Training procedure:**
1. **Phase 1:** Request 1 demo → train and *freeze* `π_bc`
2. **Phase 2 (warmup):** Take random residual actions for `RL_WARMUP_STEPS` steps to fill the replay buffer
3. **Phase 2 (RL):** At each env step:
   - Select `action = clip(π_bc(obs) + π_residual(obs) + noise)`
   - Receive reward `r = −distance_to_goal`
   - Store `(obs, residual_action, r, next_obs, done)` in replay buffer
   - Run a TD3 update

**TD3 details:**
- Twin critics Q1, Q2 to reduce overestimation bias
- Delayed actor updates every `POLICY_UPDATE_FREQ` critic steps
- Target policy smoothing with clipped Gaussian noise
- Soft target network updates with coefficient `τ = 0.005`
- Residual action clipped to `±0.5 × MAX_ACTION_MAGNITUDE`

**Budget spend:** ~$30 / $100 (1 demo + 3k steps + compute)

---

## Budget-Aware Guards

Every trainer checks the budget before committing to any operation:

```python
# Before requesting a demo
length = affordable_demo_length(money)   # returns None if unaffordable
if length is None:
    return 4, None                       # end training cleanly

# Before every RL env step
if not can_afford_step(money):
    return 4, None
```

`affordable_demo_length(money)` calculates the longest demo that fits within the remaining budget minus the safety buffer, so demo length shrinks gracefully as money runs low rather than causing a hard failure.

---

## Module Layout

```
Networks
├── PolicyNetwork     shared MLP for BC and DAgger
├── ResidualActor     small correction network for Residual RL
└── TwinCritic        twin Q-networks for TD3

ReplayBuffer          uniform experience replay (deque-based)

BCMixin               shared BC logic inherited by all three trainers
├── _train_bc()
├── _bc_action()
└── _accumulate_demo()

Trainers
├── BCTrainer
├── DAggerTrainer
└── ResidualRLTrainer

Budget helpers
├── demo_cost(length)
├── can_afford_demo(money)
├── can_afford_step(money)
└── affordable_demo_length(money)

Robot                 thin framework shell — delegates to self.trainer
```

---

## Visualisation

During testing, each algorithm draws a fading position trail in a distinct colour:

| Algorithm | Trail colour |
|---|---|
| BC | Blue |
| DAgger | Red |
| Residual RL | Green |

An arrow shows the current action direction, transitioning from red → green as the robot approaches the goal line.

---

## Running

```bash
# Set MODE = 'development' in config.py
python robot-learning.py

```

---

## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `DEMO_LENGTH` | 30 | steps per demo |
| `BC_EPOCHS` | 100 | kept low to limit compute cost |
| `DAGGER_MAX_ROUNDS` | 4 | 4 × $19 = $76 on demos |
| `SAFETY_BUFFER` | $6.00 | never spend below this |
| `RL_MAX_STEPS` | 3000 | hard cap on RL env steps |
| `RL_WARMUP_STEPS` | 200 | random residual before TD3 kicks in |
| `RESIDUAL_MAX` | `0.5 × MAX_ACT` | maximum residual correction |
| `TAU` | 0.005 | soft target update rate |
| `GAMMA` | 0.99 | discount factor |