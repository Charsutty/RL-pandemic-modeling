from dataclasses import dataclass, field
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class SocioEconomicConfig:
    """
    Paramètres socio-économiques pour la fonction de récompense.
    Afin de rendre les coûts comparables et d'obtenir des récompenses de l'ordre de 1,
    on normalise pour que les coûts soient sur une échelle similaire

    La justification des normalisations est détaillée dans le rapport.
    """

    confinement_eco_cost: float = 1.0

    vaccination_eco_cost: float = 0.02

    life_cost: float = 1000.0

    infection_cost: float = 20

    max_vacc_rate: float = 0.001


@dataclass
class ProblemConfig:
    """
    Paramètres épidémiologiques SEIRD
    Leurs valeurs sont indiquées dans le rapport et inspirées de la littérature.
    """

    beta: float = 0.27  # Taux de transmission (R0 = β/γ ≈ 2.7)
    sigma: float = 0.14  # 1 / temps d'incubation (7 jours)
    gamma: float = 0.1  # 1 / temps de guérison  (10 jours)
    mu: float = 0.01  # Taux de mortalité (1 %)
    dt: float = 1.0  # Pas de temps (1 jour)
    max_steps: int = 365  # Horizon (1 an)
    socio_eco_config: SocioEconomicConfig = field(default_factory=SocioEconomicConfig)
    # field pour avoir les valeurs par défaut d'un dataclass imbriqué


class SEIREnv(gym.Env):
    """Environnement Gymnasium SEIRD avec contrôle (confinement, vaccination).

    Inspiré de https://gymnasium.farama.org/introduction/create_custom_env/

    Les états et les coûts proviennent du rapport.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: ProblemConfig = ProblemConfig()):
        super().__init__()
        self.config = config
        self.current_step = 0

        # Actions : [confinement (0-1), vaccination (0-1)]
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        # Observation : [S, E, I, R, D]
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

        self.state = None

    def _get_obs(self):
        """Convertit l'état interne en observation.

        Ref: https://gymnasium.farama.org/introduction/create_custom_env/
        """
        return self.state.copy()

    def _get_info(self, L_eco=0.0, L_vacc=0.0, L_deaths=0.0, L_infection=0.0):
        """Retourne les informations auxiliaires (décomposition des coûts)."""
        return {
            "L_eco": L_eco,
            "L_vacc": L_vacc,
            "L_deaths": L_deaths,
            "L_infection": L_infection,
        }

    def step(self, action):
        self.current_step += 1
        u_conf = float(np.clip(action[0], 0, 1))
        u_vacc_raw = float(np.clip(action[1], 0, 1))

        S, E, I, R, D = self.state

        # ── Flux déterministes ──
        infection_rate = self.config.beta * (1 - u_conf)
        flow_inf = infection_rate * S * I

        # Contrainte logistique : plafond absolu sur les vaccinations/jour,
        # saturé par les susceptibles disponibles.
        # v(t) = min(u_vacc * v_max, S)
        max_vacc = self.config.socio_eco_config.max_vacc_rate
        flow_vacc = min(u_vacc_raw * max_vacc, S)

        flow_death = self.config.gamma * self.config.mu * I

        # Les demandes du projet sont d'introduire du bruit dans la dynamique pour simuler l'incertitude.
        noise = self.np_random.normal(0, 1e-5, 5)

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
        new_D = max(D, D + self.config.dt * flow_death + noise[4])

        self.state = np.clip([new_S, new_E, new_I, new_R, new_D], 0, 1)
        total = np.sum(self.state)
        if total > 0:
            self.state = (self.state / total).astype(np.float32)
        cfg = self.config.socio_eco_config

        L_eco = cfg.confinement_eco_cost * (u_conf**2)
        # Coût proportionnel aux vaccinations réellement administrées
        L_vacc = cfg.vaccination_eco_cost * (flow_vacc / max_vacc)
        L_deaths = cfg.life_cost * (flow_death * self.config.dt)
        L_infection = cfg.infection_cost * I

        reward = -(L_eco + L_vacc + L_deaths + L_infection)

        terminated = False
        truncated = self.current_step >= self.config.max_steps

        if new_I < 1e-4 and new_E < 1e-4:
            terminated = True
            reward += 10.0

        observation = self._get_obs()
        info = self._get_info(L_eco, L_vacc, L_deaths, L_infection)

        return observation, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.array([0.98, 0.01, 0.01, 0.0, 0.0], dtype=np.float32)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
