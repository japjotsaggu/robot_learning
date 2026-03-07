"""
robot_modular.py  –  Unified Robot Agent (Budget-Aware)
===============================================================

Switch algorithm by changing ALGORITHM in config.
    "bc"          – Behavioural Cloning only.
    "dagger"      – Dataset Aggregation (DAgger).
    "residual_rl" – Frozen BC policy + TD3-trained residual actor.

Cost of one demo of length L:  10 + L × 0.3
Cost of N RL steps:            N × 0.002
Compute time is charged automatically by the framework.

Key budget decisions
---------------------
- Every trainer checks money before any expensive operation.
- A safety_buffer is kept so we never risk going negative.
- Demo length is chosen dynamically based on remaining budget.
- DAgger rounds are cut early if we can no longer afford another demo.
- RL training steps are capped so they fit in the remaining budget.
- BC epochs are kept modest to limit compute cost (0.03/s).
"""

from __future__ import annotations

import copy
import random
from collections import deque
import config

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from graphics import VisualisationLine
import constants

algorithm = config.ALGORITHM 

safety_buffer= config.SAFETY_BUFFER 

# BC hyper-parameters 
BC_EPOCHS     = 100     # kept low: compute time costs 0.03/s
BC_BATCH_SIZE = 32

# Demo settings 
DEMO_LENGTH = 30        # steps per demo  →  cost = 10 + 30×0.3 = 19.0

# DAgger hyper-parameters 
# With 4 demos: 4×19 = 76, ~5 compute, leaving ~19 headroom
DAGGER_MAX_ROUNDS = 4
DAGGER_BETA_DECAY = 0.5

# Residual RL (TD3) hyper-parameters 
BUFFER_SIZE        = 50_000
BATCH_SIZE         = 64
GAMMA              = 0.99
TAU                = 0.005
ACTOR_LR           = 3e-4
CRITIC_LR          = 3e-4
POLICY_UPDATE_FREQ = 2
EXPLORATION_NOISE  = 0.02
TARGET_NOISE       = 0.01
TARGET_NOISE_CLIP  = 0.02
RESIDUAL_MAX       = 0.5 * constants.MAX_ACTION_MAGNITUDE

# 1 demo (19) + steps×0.002 + compute ~5 → fits comfortably in 100
RL_MAX_STEPS    = 3000
RL_WARMUP_STEPS = 200

OBS_DIM = constants.OBSERVATION_DIMENSION
ACT_DIM = constants.ACTION_DIMENSION
MAX_ACT = constants.MAX_ACTION_MAGNITUDE


#  budget helpers

def demo_cost(length: int = DEMO_LENGTH) -> float:
    """Total cost of one demo request of `length` steps."""
    return constants.COST_PER_DEMO + length * constants.COST_PER_DEMO_STEP


def can_afford_demo(money: float, length: int = DEMO_LENGTH) -> bool:
    return money - demo_cost(length) >= safety_buffer


def can_afford_step(money: float) -> bool:
    return money - constants.COST_PER_STEP >= safety_buffer


def affordable_demo_length(money: float) -> int | None:
    """
    Longest demo we can afford given remaining budget.
    Returns None if we cannot afford even a 1-step demo.
    """
    available = money - safety_buffer - constants.COST_PER_DEMO
    if available <= 0:
        return None
    return max(1, min(DEMO_LENGTH, int(available / constants.COST_PER_DEMO_STEP)))


#  networks

class PolicyNetwork(nn.Module):
    """Shared MLP: observation → scaled action in [-MAX_ACT, MAX_ACT]."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, ACT_DIM), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * MAX_ACT


class ResidualActor(nn.Module):
    """Correction network; output in [-RESIDUAL_MAX, RESIDUAL_MAX]."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, ACT_DIM), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * RESIDUAL_MAX


class TwinCritic(nn.Module):
    """Twin Q-networks for TD3."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        inp = OBS_DIM + ACT_DIM
        self.q1 = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs, act) -> torch.Tensor:
        return self.q1(torch.cat([obs, act], dim=-1))


#  replay buffer

class ReplayBuffer:
    def __init__(self, maxlen: int = BUFFER_SIZE):
        self._buf: deque = deque(maxlen=maxlen)

    def add(self, obs, act, reward: float, next_obs, done: float):
        self._buf.append((obs, act, reward, next_obs, done))

    def sample(self, n: int = BATCH_SIZE):
        batch = random.sample(self._buf, n)
        obs, act, rew, nobs, done = zip(*batch)
        t = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
        return t(obs), t(act), t(rew).unsqueeze(1), t(nobs), t(done).unsqueeze(1)

    def __len__(self) -> int:
        return len(self._buf)


#  shared bc 

class BCMixin:
    """
    Reusable behavioural-cloning logic.

    Subclasses must initialise:
        self.bc_policy    – PolicyNetwork
        self.bc_opt       – its Adam optimiser
        self.agg_obs      – list
        self.agg_actions  – list
    """

    def _train_bc(self, epochs: int = BC_EPOCHS, freeze: bool = False):
        if not self.agg_obs:
            return
        obs_t = torch.tensor(np.array(self.agg_obs),     dtype=torch.float32)
        act_t = torch.tensor(np.array(self.agg_actions), dtype=torch.float32)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(obs_t, act_t),
            batch_size=BC_BATCH_SIZE, shuffle=True,
        )
        for _ in range(epochs):
            for ob, ac in loader:
                loss = nn.MSELoss()(self.bc_policy(ob), ac)
                self.bc_opt.zero_grad()
                loss.backward()
                self.bc_opt.step()
        if freeze:
            for p in self.bc_policy.parameters():
                p.requires_grad = False
        print(f"[BC] trained  samples={len(self.agg_obs)}  epochs={epochs}"
              + ("  → frozen" if freeze else ""))

    def _bc_action(self, obs_np: np.ndarray) -> np.ndarray:
        t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.bc_policy(t).squeeze(0).numpy()

    def _accumulate_demo(self, demo):
        for obs, action in demo:
            self.agg_obs.append(obs)
            self.agg_actions.append(action)


#  algorithm trainers

class BCTrainer(BCMixin):
    """
    Pure Behavioural Cloning.

    Budget:  1 demo (≈19) + compute (≈2)  ≈  21 / 100
    """

    def __init__(self):
        self.bc_policy    = PolicyNetwork()
        self.bc_opt       = optim.Adam(self.bc_policy.parameters(), lr=1e-3)
        self.agg_obs:     list = []
        self.agg_actions: list = []
        self._requested   = False
        self._trained     = False

    def training_action(self, observation, money):
        if not self._requested:
            length = affordable_demo_length(money)
            if length is None:
                print(f"[BC] Cannot afford demo (money={money:.2f}) — done.")
                return 4, None
            self._requested = True
            print(f"[BC] Requesting demo  length={length}  "
                  f"cost≈{demo_cost(length):.1f}  money={money:.1f}")
            return 3, length

        if self._trained:
            return 4, None

        return 1, np.zeros(ACT_DIM)     # safe fallback

    def receive_demo(self, demo):
        self._accumulate_demo(demo)
        self._train_bc(BC_EPOCHS)
        self._trained = True

    def receive_transition(self, obs, act, next_obs, distance):
        pass

    def testing_action(self, observation) -> np.ndarray:
        return np.clip(self._bc_action(observation), -MAX_ACT, MAX_ACT)

    @property
    def trail_colour(self):
        return (30, 30, 220)    # blue



class DAggerTrainer(BCMixin):
    """
    DAgger (Dataset Aggregation).

    Budget:  DAGGER_MAX_ROUNDS demos × 19 + compute
             4 rounds → 76 + ~5  ≈  81 / 100
    Each round is skipped if the remaining budget cannot cover a demo.
    """

    def __init__(self):
        self.bc_policy    = PolicyNetwork()
        self.bc_opt       = optim.Adam(self.bc_policy.parameters(), lr=1e-3)
        self.agg_obs:     list = []
        self.agg_actions: list = []
        self._round         = 0
        self._training_done = False

    def training_action(self, observation, money):
        if self._training_done:
            return 4, None

        if self._round < DAGGER_MAX_ROUNDS:
            length = affordable_demo_length(money)
            if length is None:
                print(f"[DAgger] Round {self._round}: budget too low "
                      f"(money={money:.2f}) — ending training.")
                self._training_done = True
                return 4, None

            beta = DAGGER_BETA_DECAY ** self._round
            print(f"[DAgger] Round {self._round}  β={beta:.3f}  "
                  f"length={length}  cost≈{demo_cost(length):.1f}  "
                  f"money={money:.1f}")
            self._round += 1
            return 3, length

        self._training_done = True
        print("[DAgger] All rounds complete.")
        return 4, None

    def receive_demo(self, demo):
        self._accumulate_demo(demo)
        print(f"[DAgger] Dataset size: {len(self.agg_obs)}. Retraining…")
        self._train_bc(BC_EPOCHS)

    def receive_transition(self, obs, act, next_obs, distance):
        pass

    def testing_action(self, observation) -> np.ndarray:
        return np.clip(self._bc_action(observation), -MAX_ACT, MAX_ACT)

    @property
    def trail_colour(self):
        return (220, 30, 30)    # red


class ResidualRLTrainer(BCMixin):
    """
    Residual RL:  π_total = π_bc (frozen)  +  π_residual (TD3).

    Budget:  1 demo (≈19) + RL_MAX_STEPS × 0.002 (≈6) + compute (≈5)  ≈  30 / 100
    money is checked before every RL step — training stops cleanly
    before the budget hits zero.
    """

    def __init__(self):
        self.bc_policy    = PolicyNetwork()
        self.bc_opt       = optim.Adam(self.bc_policy.parameters(), lr=1e-3)
        self.agg_obs:     list = []
        self.agg_actions: list = []
        self._bc_trained  = False

        self.actor         = ResidualActor()
        self.actor_target  = copy.deepcopy(self.actor)
        self.actor_opt     = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.critic        = TwinCritic()
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt    = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.buffer        = ReplayBuffer()

        self._rl_steps      = 0
        self._update_count  = 0
        self._prev_obs      = None
        self._prev_residual = None
        self._demo_done     = False
        self._training_done = False

    # helpers 
    def _residual_action(self, obs_np: np.ndarray) -> np.ndarray:
        t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.actor(t).squeeze(0).numpy()

    def _total_action(self, bc_a: np.ndarray, res_a: np.ndarray) -> np.ndarray:
        return np.clip(bc_a + res_a, -MAX_ACT, MAX_ACT)

    # interface

    def training_action(self, observation, money):
        # Phase 1: one demo
        if not self._demo_done:
            length = affordable_demo_length(money)
            if length is None:
                print(f"[ResidualRL] Cannot afford demo (money={money:.2f}) — done.")
                return 4, None
            self._demo_done = True
            print(f"[ResidualRL] Requesting demo  length={length}  "
                  f"cost≈{demo_cost(length):.1f}  money={money:.1f}")
            return 3, length

        if self._training_done:
            return 4, None

        # Step cap
        if self._rl_steps >= RL_MAX_STEPS:
            self._training_done = True
            print(f"[ResidualRL] Step cap ({RL_MAX_STEPS}) reached — done.")
            return 4, None

        # Budget gate — checked every step
        if not can_afford_step(money):
            self._training_done = True
            print(f"[ResidualRL] Budget low (money={money:.2f}) — ending training.")
            return 4, None

        # Compute action
        bc_a = self._bc_action(observation)
        if self._rl_steps < RL_WARMUP_STEPS:
            res_a = np.random.uniform(-RESIDUAL_MAX, RESIDUAL_MAX, ACT_DIM)
        else:
            res_a = self._residual_action(observation)
            res_a = res_a + np.random.normal(0, EXPLORATION_NOISE, ACT_DIM)
            res_a = np.clip(res_a, -RESIDUAL_MAX, RESIDUAL_MAX)

        self._prev_obs      = observation
        self._prev_residual = res_a
        self._rl_steps     += 1

        return 1, self._total_action(bc_a, res_a)

    def receive_demo(self, demo):
        self._accumulate_demo(demo)
        self._train_bc(BC_EPOCHS, freeze=True)
        self._bc_trained = True

    def receive_transition(self, obs, act, next_obs, distance):
        if not self._bc_trained or self._prev_obs is None:
            return
        reward = -distance
        done   = float(distance < 0.05)
        self.buffer.add(self._prev_obs, self._prev_residual,
                        reward, next_obs, done)
        if len(self.buffer) >= BATCH_SIZE:
            self._td3_update()

    def testing_action(self, observation) -> np.ndarray:
        bc_a  = self._bc_action(observation)
        res_a = self._residual_action(observation)
        return self._total_action(bc_a, res_a)

    @property
    def trail_colour(self):
        return (30, 220, 30)    # green

    # TD3 update 

    def _td3_update(self):
        obs, act, rew, nobs, done = self.buffer.sample()

        with torch.no_grad():
            noise    = (torch.randn_like(act) * TARGET_NOISE).clamp(
                -TARGET_NOISE_CLIP, TARGET_NOISE_CLIP)
            next_res = (self.actor_target(nobs) + noise).clamp(-RESIDUAL_MAX, RESIDUAL_MAX)
            next_bc  = torch.tensor(
                np.array([self._bc_action(o) for o in nobs.numpy()]),
                dtype=torch.float32)
            next_act = torch.clamp(next_bc + next_res, -MAX_ACT, MAX_ACT)
            q1_t, q2_t = self.critic_target(nobs, next_act)
            y = rew + GAMMA * (1 - done) * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(obs, act)
        c_loss = nn.MSELoss()(q1, y) + nn.MSELoss()(q2, y)
        self.critic_opt.zero_grad(); c_loss.backward(); self.critic_opt.step()

        self._update_count += 1

        if self._update_count % POLICY_UPDATE_FREQ == 0:
            bc_a  = torch.tensor(
                np.array([self._bc_action(o) for o in obs.numpy()]),
                dtype=torch.float32)
            res   = self.actor(obs)
            total = torch.clamp(bc_a + res, -MAX_ACT, MAX_ACT)
            a_loss = -self.critic.q1_only(obs, total).mean()
            self.actor_opt.zero_grad(); a_loss.backward(); self.actor_opt.step()

            for src, tgt in [(self.critic, self.critic_target),
                             (self.actor,  self.actor_target)]:
                for p, pt in zip(src.parameters(), tgt.parameters()):
                    pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)


#  ROBOT: delegating to the chosen trainer

_TRAINERS = {
    "bc":          BCTrainer,
    "dagger":      DAggerTrainer,
    "residual_rl": ResidualRLTrainer,
}


class Robot:
    """
    Framework-facing Robot class.

    Change algorithm at the top of this file to switch between
    "bc", "dagger", and "residual_rl".

    money is forwarded to every training_action() call so trainers can
    make fully budget-aware decisions at every step.
    """

    def __init__(self):
        self.visualisation_lines: list = []
        self.environment = None
        self._trail: list = []

        trainer_cls = _TRAINERS.get(algorithm)
        if trainer_cls is None:
            raise ValueError(f"Unknown algorithm '{algorithm}'. "
                             f"Choose from: {list(_TRAINERS)}")
        self.trainer = trainer_cls()
        print(f"[Robot] Algorithm : {algorithm}")
        print(f"[Robot] Budget    : {constants.INIT_MONEY}")
        print(f"[Robot] Demo cost : {demo_cost(DEMO_LENGTH):.1f}  "
              f"(flat={constants.COST_PER_DEMO} + "
              f"{DEMO_LENGTH}×{constants.COST_PER_DEMO_STEP})")
        print(f"[Robot] Step cost : {constants.COST_PER_STEP}  "
              f"| Compute: {constants.COST_PER_SECOND}/s")

    # framework interface 

    def training_action(self, observation, money):
        return self.trainer.training_action(observation, money)

    def receive_demo(self, demo):
        self.trainer.receive_demo(demo)

    def receive_transition(self, observation, action, next_observation, distance):
        self.trainer.receive_transition(observation, action, next_observation, distance)

    def testing_action(self, observation):
        action = self.trainer.testing_action(observation)
        self._update_visualisation(action)
        return action

    # visualisation

    def _update_visualisation(self, action: np.ndarray):
        if self.environment is None:
            return

        self.visualisation_lines = []
        x, y = self.environment.state

        self._trail.append((x, y))
        if len(self._trail) > 60:
            self._trail.pop(0)

        base = self.trainer.trail_colour
        for i in range(1, len(self._trail)):
            x1, y1 = self._trail[i - 1]
            x2, y2 = self._trail[i]
            alpha = i / len(self._trail)
            c = tuple(max(20, int(v * alpha)) for v in base)
            self.visualisation_lines.append(
                VisualisationLine(x1, y1, x2, y2, colour=c, width=0.004)
            )

        progress = np.clip(x / constants.GOAL_LINE_X, 0, 1)
        r = int(255 * (1 - progress))
        g = int(255 * progress)
        self.visualisation_lines.append(
            VisualisationLine(
                x, y,
                x + action[0] * 8.0,
                y + action[1] * 8.0,
                colour=(r, g, 50),
                width=0.008,
            )
        )