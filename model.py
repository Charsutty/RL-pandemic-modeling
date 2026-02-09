from dataclasses import dataclass, field
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class SocioEconomicConfig:
    """
    Paramètres économiques normalisés en "jours de PIB à confinement total".
    Unité de référence : C_eco = 1 ≡ coût d'un confinement total / jour (~2 Md€).
    Voir rapport projet.tex, sections 2.3–2.4.
    """

    # C_eco : coût confinement total / jour (unité de référence = 1)
    confinement_eco_cost: float = 1.0

    # C_vacc : ~10 M€/jour à plein régime / 2 Md€/jour ≈ 0.005
    #   (21.5€/dose × 2 doses + logistique ≈ 53€/pers, ×335k pers/jour)
    vaccination_eco_cost: float = 0.005

    # C_vie : valeur stat. vie humaine (3 M€) × pop / C_eco_jour ≈ 100
    #   On amplifie (×10) pour prioriser les vies humaines
    life_cost: float = 1000.0

    # C_hosp : coût hospitalier + arrêt maladie par infecté / jour
    #   ~580€/patient/jour (source éval. sécu. sociale) → normalisé ≈ 3–6
    infection_cost: float = 3.0

    # Taux de vaccination max : 0.5 % de la pop / jour
    max_vacc_rate: float = 0.005


@dataclass
class ProblemConfig:
    """
    Paramètres épidémiologiques SEIRD (Covid-19, France).
    Sources : Wu et al. (2020), The Lancet ; rapport projet.tex §2.4.
    """

    beta: float = 0.27  # Taux de transmission (R0 = β/γ ≈ 2.7)
    sigma: float = 0.14  # 1 / temps d'incubation (~7 jours)
    gamma: float = 0.1  # 1 / temps de guérison  (~10 jours)
    mu: float = 0.01  # Taux de mortalité (1 %)
    dt: float = 1.0  # Pas de temps (1 jour)
    max_steps: int = 365  # Horizon (1 an)
    socio_eco_config: SocioEconomicConfig = field(default_factory=SocioEconomicConfig)


class SEIREnv(gym.Env):
    """
    Environnement Gym SEIRD avec contrôle (confinement, vaccination).

    Simulation complète : [S, E, I, R, D]
    État utile pour l'agent : [S, E, I]  (obs[:3])
      → Les équations de S, E, I ne dépendent ni de R ni de D.
      → Les coûts dépendent de I et des actions uniquement.

    Actions : [u_conf, u_vacc_raw] ∈ [0, 1]²
      u_vacc effectif = u_vacc_raw × max_vacc_rate

    Reward = −(L_eco + L_vacc + L_deaths + L_infection)
    Composantes retournées dans info pour analyse.
    """

    def __init__(self, config: ProblemConfig = ProblemConfig()):
        super(SEIREnv, self).__init__()
        self.config = config
        self.current_step = 0

        # Actions : [confinement (0-1), vaccination (0-1)]
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        # Observation : [S, E, I, R, D]
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

        self.state = None

    def step(self, action):
        self.current_step += 1
        u_conf = float(np.clip(action[0], 0, 1))
        u_vacc_raw = float(np.clip(action[1], 0, 1))

        u_vacc = u_vacc_raw * self.config.socio_eco_config.max_vacc_rate

        S, E, I, R, D = self.state

        # ── Flux déterministes ──
        infection_rate = self.config.beta * (1 - u_conf)
        flow_inf = infection_rate * S * I
        flow_vacc = u_vacc * S
        flow_death = self.config.gamma * self.config.mu * I

        # ── Mise à jour SEIRD (bruit léger pour stochasticité) ──
        noise = np.random.normal(0, 1e-5, 5)

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

        # ── Récompense (coûts normalisés, voir rapport §2.3) ──
        cfg = self.config.socio_eco_config

        L_eco = cfg.confinement_eco_cost * (u_conf**2)
        L_vacc = cfg.vaccination_eco_cost * u_vacc_raw
        L_deaths = cfg.life_cost * (flow_death * self.config.dt)
        L_infection = cfg.infection_cost * I

        reward = -(L_eco + L_vacc + L_deaths + L_infection)

        terminated = False
        truncated = self.current_step >= self.config.max_steps

        if new_I < 1e-4 and new_E < 1e-4:
            terminated = True
            reward += 10.0

        return (
            self.state.copy(),
            float(reward),
            terminated,
            truncated,
            {
                "L_eco": L_eco,
                "L_vacc": L_vacc,
                "L_deaths": L_deaths,
                "L_infection": L_infection,
            },
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.array([0.98, 0.01, 0.01, 0.0, 0.0], dtype=np.float32)
        return self.state.copy(), {}
