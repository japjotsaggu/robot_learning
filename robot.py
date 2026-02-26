import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from graphics import VisualisationLine
import constants

EPOCHS = 150
SEED = 1


class PolicyNetwork(nn.Module):
    """
    Simple feedforward neural network for the robot policy.

    Takes an observation as input and outputs a continuous action.
    The final action is scaled by MAX_ACTION_MAGNITUDE.
    """

    def __init__(self):
        super().__init__()

        # 3-layer MLP with ReLU activations
        self.net = nn.Sequential(
            nn.Linear(constants.OBSERVATION_DIMENSION, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, constants.ACTION_DIMENSION),
            nn.Tanh()  # keeps output in [-1, 1]
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Observation tensor

        Returns:
            torch.Tensor: Scaled action
        """
        return self.net(x) * constants.MAX_ACTION_MAGNITUDE


class Robot:
    """
    Robot agent that learns using Behavioural Cloning (BC).

    Training process:
    1. Request demonstrations.
    2. Train policy using BC (supervised learning).
    3. Use trained policy at test time.
    """

    def __init__(self):
        # Lines used for drawing robot movement
        self.visualisation_lines = []

        # Policy network and optimizer
        self.policy = PolicyNetwork()
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=1e-3)

        # Storage for demonstration data
        self.demo_obs = []
        self.demo_actions = []

        # Demo settings
        self.num_demos = 1
        self.demo_length = 30
        self.demos_requested = 0
        self.demos_received = 0
        self.bc_trained = False

        # Environment reference (set externally)
        self.environment = None

        # Stores recent positions for trail drawing
        self._trail = []

    # TRAINING

    def training_action(self, observation, money):
        """
        Decides what to do during training.

        Returns:
            (action_type, value)
            action_type:
                1 -> normal action
                3 -> request demo
                4 -> end training
        """

        # Phase 1: request demonstrations
        if self.demos_requested < self.num_demos:
            self.demos_requested += 1
            print(f"Requesting demo {self.demos_requested}/{self.num_demos}")
            return 3, self.demo_length

        # Phase 2: training finished
        if self.bc_trained:
            return 4, None

        # Default fallback
        return 1, np.zeros(constants.ACTION_DIMENSION)

    # RECEIVE DEMO

    def receive_demo(self, demo):
        """
        Receives a demonstration and stores the data.

        Args:
            demo: list of (observation, action) pairs
        """

        self.demos_received += 1

        # Store all demo transitions
        for obs, action in demo:
            self.demo_obs.append(obs)
            self.demo_actions.append(action)

        print(f"Demo {self.demos_received} received. "
              f"Training BC on {len(self.demo_obs)} samples...")

        # Train using behavioural cloning
        self.train_bc(EPOCHS)
        self.bc_trained = True

    # RECEIVE TRANSITION

    def receive_transition(self, observation, action,
                           next_observation, distance_to_goal):
        """
        Not used in this implementation.
        Included for compatibility with framework.
        """
        pass

    # BEHAVIOURAL CLONING

    def train_bc(self, epochs=EPOCHS):
        """
        Train the policy using Behavioural Cloning.

        This is supervised learning:
        Minimise MSE between predicted action and demo action.
        """

        if not self.demo_obs:
            return

        # Convert stored data to tensors
        obs = torch.tensor(np.array(self.demo_obs), dtype=torch.float32)
        actions = torch.tensor(np.array(self.demo_actions), dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(obs, actions)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=32,
                                             shuffle=True)

        # Standard supervised training loop
        for _ in range(epochs):
            for obs_batch, act_batch in loader:

                # Forward pass
                pred = self.policy(obs_batch)

                # Mean Squared Error loss
                loss = nn.MSELoss()(pred, act_batch)

                # Backpropagation
                self.policy_opt.zero_grad()
                loss.backward()
                self.policy_opt.step()

        print(f"BC training done "
              f"({len(self.demo_obs)} samples, {epochs} epochs).")
        
    # POLICY ACTION

    def policy_action(self, observation):
        """
        Returns action from trained policy.

        Args:
            observation (np.array)

        Returns:
            np.array: clipped action
        """

        obs_t = torch.tensor(observation,
                             dtype=torch.float32).unsqueeze(0)

        # Disable gradients during inference
        with torch.no_grad():
            action = self.policy(obs_t).squeeze(0).numpy()

        # Clip action to valid range
        return np.clip(action,
                       -constants.MAX_ACTION_MAGNITUDE,
                       constants.MAX_ACTION_MAGNITUDE)

    # TESTING

    def testing_action(self, observation):
        """
        Called during testing phase.
        Uses trained policy and updates visualisation.
        """
        action = self.policy_action(observation)
        self._update_visualisation(action)
        return action

    # VISUALISATION

    def _update_visualisation(self, action):
        """
        Updates visualisation lines.

        Draws:
        - A fading trail of past positions
        - Current action direction (colour shows progress)
        """

        if self.environment is None:
            return

        self.visualisation_lines = []
        x, y = self.environment.state

        # Store current position
        self._trail.append((x, y))

        # Keep only last 60 positions
        if len(self._trail) > 60:
            self._trail.pop(0)

        # Draw trail
        for i in range(1, len(self._trail)):
            x1, y1 = self._trail[i - 1]
            x2t, y2t = self._trail[i]

            # Fade colour over time
            alpha = i / len(self._trail)
            c = int(80 + 175 * alpha)

            self.visualisation_lines.append(
                VisualisationLine(
                    x1, y1, x2t, y2t,
                    colour=(30, 30, c),
                    width=0.004
                )
            )

        # Colour changes from red to green as it progresses
        progress = np.clip(x / constants.GOAL_LINE_X, 0, 1)
        r = int(255 * (1 - progress))
        g = int(255 * progress)

        # Draw action arrow
        x2 = x + action[0] * 8.0
        y2 = y + action[1] * 8.0

        self.visualisation_lines.append(
            VisualisationLine(
                x, y, x2, y2,
                colour=(r, g, 50),
                width=0.008
            )
        )