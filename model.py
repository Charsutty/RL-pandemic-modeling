from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class ProblemConfig:
    beta: float = 0.5  # Infectiosité
    sigma: float = 1 / 5  # Incubation (passer de E à I)
    gamma: float = 1 / 10  # Guérison (passer de I à R)
    dt: float = 1.0  # Pas de temps (1 jour)


class SEIREnv(gym.Env):
    def __init__(self, config: ProblemConfig = ProblemConfig()):
        super(SEIREnv, self).__init__()
        self.config = config
        # Actions: [Confinement (0-1), Vaccination (0-1)]
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        # État: [S, E, I, R] (Normalisé entre 0 et 1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def step(self, action):
        u_conf, u_vacc = action
        S, E, I, R = self.state

        # 1. Dynamique SEIR (Euler discret)
        # Bruit stochastique (demandé par le sujet)
        noise = np.random.normal(0, 0.001, 4)

        new_S = (
            S
            + self.config.dt * (-self.config.beta * (1 - u_conf) * S * I - u_vacc * S)
            + noise[0]
        )
        new_E = (
            E
            + self.config.dt
            * (self.config.beta * (1 - u_conf) * S * I - self.config.sigma * E)
            + noise[1]
        )
        new_I = (
            I
            + self.config.dt * (self.config.sigma * E - self.config.gamma * I)
            + noise[2]
        )
        new_R = R + self.config.dt * (self.config.gamma * I + u_vacc * S) + noise[3]

        # Contraintes : On reste entre 0 et 1, et la somme fait 1
        self.state = np.clip([new_S, new_E, new_I, new_R], 0, 1)
        self.state = (self.state / np.sum(self.state)).astype(
            np.float32
        )  # Renormalisation

        # 2. Calcul de la Récompense
        # On veut minimiser I (santé) et minimiser l'effort u (économie)
        # Reward = - Cost
        sanitary_cost = 10.0 * new_I  # Poids fort sur la santé
        economic_cost = 1.0 * (u_conf**2) + 0.5 * u_vacc
        reward = -(sanitary_cost + economic_cost)

        # 3. Terminaison (Si l'épidémie est finie ou temps écoulé)
        terminated = False
        if new_I < 0.001 and new_E < 0.001:  # Épidémie éradiquée
            terminated = True
            reward += 10  # Bonus de victoire

        return self.state, reward, terminated, False, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Initialisation : 99% Sains, 1% Exposés
        self.state = np.array([0.99, 0.01, 0.0, 0.0], dtype=np.float32)
        return self.state, {}
