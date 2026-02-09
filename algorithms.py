"""
Algorithmes de RL étudiés dans notre rapport.


Nous inspirons notre implémentation de Q-Learning et SARSA du pattern de la documentation Gymnasium:
https://gymnasium.farama.org/introduction/train_agent/
"""

from dataclasses import dataclass
import numpy as np
from utils import state_to_id


@dataclass
class RLConfig:
    """Configuration des algorithmes de RL (Q-Learning, SARSA)."""

    learning_rate: float = 0.1
    initial_epsilon: float = 1.0
    epsilon_decay: float = 0.001
    final_epsilon: float = 0.05
    discount_factor: float = 0.99


class QLearningAgent:
    """Agent Q-Learning tabulaire (off-policy)."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        bin_dicts: dict,
        configs: RLConfig,
    ):
        """Initialise l'agent Q-Learning.

        Args:
            n_states: Nombre total d'états discrétisés.
            n_actions: Nombre total d'actions discrétisées.
            bin_dicts: Dictionnaire des bins pour S, E, I.
        """
        self.q_values = np.zeros((n_states, n_actions), dtype=np.float64)
        self.bin_dicts = bin_dicts
        self.actions_grid = bin_dicts["actions"]
        self.n_actions = n_actions

        self.lr = configs.learning_rate
        self.discount_factor = configs.discount_factor

        self.epsilon = configs.initial_epsilon
        self.epsilon_decay = configs.epsilon_decay
        self.final_epsilon = configs.final_epsilon

        self.training_error = []

    def get_action(self, obs: np.ndarray) -> tuple[int, np.ndarray]:
        """Sélection ε-greedy.

        Returns:
            (action_id, action_continue) : indice et vecteur action [u_conf, u_vacc].
        """
        state_id = state_to_id(obs, self.bin_dicts)

        if np.random.random() < self.epsilon:
            action_id = np.random.randint(self.n_actions)
        else:
            action_id = int(np.argmax(self.q_values[state_id]))

        return action_id, self.actions_grid[action_id]

    def update(
        self,
        obs: np.ndarray,
        action_id: int,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray,
    ):
        """Mise à jour Q-Learning (off-policy)"""
        state_id = state_to_id(obs, self.bin_dicts)
        next_state_id = state_to_id(next_obs, self.bin_dicts)

        future_q_value = (not terminated) * np.max(self.q_values[next_state_id])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[state_id, action_id]

        self.q_values[state_id, action_id] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Réduit epsilon après chaque épisode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def greedy_policy(self, obs: np.ndarray) -> np.ndarray:
        """Politique greedy déterministe."""
        state_id = state_to_id(obs, self.bin_dicts)
        action_id = int(np.argmax(self.q_values[state_id]))
        return self.actions_grid[action_id]


class SarsaAgent:
    """Agent SARSA tabulaire (on-policy)."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        bin_dicts: dict,
        configs: RLConfig,
    ):
        self.q_values = np.zeros((n_states, n_actions), dtype=np.float64)
        self.bin_dicts = bin_dicts
        self.actions_grid = bin_dicts["actions"]
        self.n_actions = n_actions

        self.lr = configs.learning_rate
        self.discount_factor = configs.discount_factor

        self.epsilon = configs.initial_epsilon
        self.epsilon_decay = configs.epsilon_decay
        self.final_epsilon = configs.final_epsilon

        self.training_error = []

    def get_action(self, obs: np.ndarray) -> tuple[int, np.ndarray]:
        """Sélection ε-greedy."""
        state_id = state_to_id(obs, self.bin_dicts)

        if np.random.random() < self.epsilon:
            action_id = np.random.randint(self.n_actions)
        else:
            action_id = int(np.argmax(self.q_values[state_id]))

        return action_id, self.actions_grid[action_id]

    def update(
        self,
        obs: np.ndarray,
        action_id: int,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray,
        next_action_id: int,
    ):
        """Mise à jour SARSA"""
        state_id = state_to_id(obs, self.bin_dicts)
        next_state_id = state_to_id(next_obs, self.bin_dicts)

        future_q_value = (not terminated) * self.q_values[next_state_id, next_action_id]
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[state_id, action_id]

        self.q_values[state_id, action_id] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Réduit epsilon après chaque épisode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def greedy_policy(self, obs: np.ndarray) -> np.ndarray:
        """Politique greedy déterministe."""
        state_id = state_to_id(obs, self.bin_dicts)
        action_id = int(np.argmax(self.q_values[state_id]))
        return self.actions_grid[action_id]
