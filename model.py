from dataclasses import dataclass, field
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class SocioEconomicConfig:
    # On réduit le coût du confinement (était 5.0) pour inciter l'agent à l'utiliser
    confinement_eco_cost: float = 1.0
    # On réduit drastiquement le coût du vaccin (était 2.0) pour le rendre viable
    vaccination_eco_cost: float = 0.1
    # On augmente massivement le coût de la vie (était 100.0)
    # 1000.0 signifie que sauver 1% de la population vaut 1000 jours de PIB.
    life_cost: float = 1000.0
    # On augmente le coût de l'infection (était 5.0) pour éviter la saturation
    infection_cost: float = 50.0
    max_vacc_rate: float = 0.005  # 0.5% par jour max (plus réaliste que 2%)


@dataclass
class ProblemConfig:
    beta: float = 0.27
    sigma: float = 0.14
    gamma: float = 0.1
    mu: float = 0.01
    dt: float = 1.0
    max_steps: int = 365
    socio_eco_config: SocioEconomicConfig = field(default_factory=SocioEconomicConfig)
    scale: float = 10.0


class SEIREnv(gym.Env):
    def __init__(self, config: ProblemConfig = ProblemConfig()):
        super(SEIREnv, self).__init__()
        self.config = config
        self.current_step = 0

        # Actions: [Confinement (0-1), Vaccination (0-1)]
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        # État: [S, E, I, R, D]
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

        self.state = None

    def step(self, action):
        self.current_step += 1
        u_conf, u_vacc_raw = action

        # Application du taux max de vaccination
        u_vacc = u_vacc_raw * self.config.socio_eco_config.max_vacc_rate

        S, E, I, R, D = self.state

        # --- CALCULS DÉTERMINISTES (POUR LA RÉCOMPENSE) ---
        # On calcule les flux théoriques pour avoir un signal d'apprentissage propre
        # sans le bruit aléatoire qui perturbe l'agent.

        infection_rate = self.config.beta * (1 - u_conf)
        flow_inf = infection_rate * S * I
        flow_vacc = u_vacc * S
        flow_death = self.config.gamma * self.config.mu * I  # Flux de décès théorique

        # --- MISE À JOUR DE L'ÉTAT (AVEC BRUIT) ---
        # Le bruit s'applique à la dynamique de l'environnement, pas à la récompense
        noise = np.random.normal(0, 1e-5, 5)  # Bruit réduit pour stabilité

        new_S = S + self.config.dt * (-flow_inf - flow_vacc) + noise[0]
        new_E = E + self.config.dt * (flow_inf - self.config.sigma * E) + noise[1]
        new_I = (
            I
            + self.config.dt * (self.config.sigma * E - self.config.gamma * I)
            + noise[2]
        )
        new_R = (
            R
            + self.config.dt
            * (self.config.gamma * (1 - self.config.mu) * I + flow_vacc)
            + noise[3]
        )

        # Pour D, on s'assure qu'il ne diminue pas
        new_D = D + self.config.dt * flow_death + noise[4]
        if new_D < D:
            new_D = D

        self.state = np.clip([new_S, new_E, new_I, new_R, new_D], 0, 1)
        self.state = (self.state / np.sum(self.state)).astype(np.float32)

        # --- CALCUL DE LA RÉCOMPENSE (PROPRE) ---

        # Coût économique quadratique
        L_eco = (
            self.config.scale
            * self.config.socio_eco_config.confinement_eco_cost
            * (u_conf**2)
        )

        # Coût vaccination
        L_vacc = (
            self.config.scale
            * self.config.socio_eco_config.vaccination_eco_cost
            * u_vacc_raw
        )

        # Coût décès : On utilise le FLUX théorique (flow_death)
        # C'est la dérivée instantanée des morts. C'est un signal très propre pour le gradient.
        L_deaths = (
            self.config.scale
            * self.config.socio_eco_config.life_cost
            * (flow_death * self.config.dt)
        )

        # Coût infection (Saturation hôpital)
        L_infection = (
            self.config.scale * self.config.socio_eco_config.infection_cost * I
        )

        reward = -(L_eco + L_vacc + L_deaths + L_infection)

        # Terminaison
        terminated = False
        truncated = self.current_step >= self.config.max_steps

        # Si épidémie éteinte
        if new_I < 1e-4 and new_E < 1e-4:
            terminated = True
            reward += 10.0

        return self.state, float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Initialisation réaliste
        self.state = np.array([0.98, 0.01, 0.01, 0.0, 0.0], dtype=np.float32)
        return self.state, {}
