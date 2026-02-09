"""
Fonctions utilitaires pour la discrétisation et l'enregistrement d'épisodes.

La discrétisation de l'espace d'états continu (S, E, I) en bins non-uniformes
permet d'utiliser des méthodes tabulaires (Q-Learning, SARSA).

Ref: https://gymnasium.farama.org/introduction/train_agent/
Ref: https://gymnasium.farama.org/introduction/record_agent/
"""

import numpy as np
import gymnasium as gym


# ─── Discrétisation ──────────────────────────────────────────────


def default_bin_dicts() -> dict:
    """Retourne la configuration de discrétisation par défaut.

    - S : 11 bins, plus fins autour des valeurs hautes (début d'épidémie).
    - E, I : 11 bins log-espacés, plus fins près de 0.
    - actions : grille 5×5 (confinement × vaccination).
    """
    S_bins = np.array([0, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.985, 1.0])
    EI_bins = np.array(
        [0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
    )

    grid_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    actions = np.array(
        [(uc, uv) for uc in grid_vals for uv in grid_vals], dtype=np.float32
    )

    return {
        "S_bins": S_bins,
        "E_bins": EI_bins,
        "I_bins": EI_bins,
        "actions": actions,
    }


def bin_index(x: float, bins: np.ndarray) -> int:
    """Retourne l'indice du bin contenant x (0-indexé)."""
    return int(np.clip(np.digitize([x], bins)[0] - 1, 0, len(bins) - 2))


def state_to_id(obs: np.ndarray, bin_dicts: dict) -> int:
    """Convertit une observation [S, E, I, ...] en un indice entier."""
    iS = bin_index(float(obs[0]), bin_dicts["S_bins"])
    iE = bin_index(float(obs[1]), bin_dicts["E_bins"])
    iI = bin_index(float(obs[2]), bin_dicts["I_bins"])

    nE = len(bin_dicts["E_bins"]) - 1
    nI = len(bin_dicts["I_bins"]) - 1

    return (iS * nE + iE) * nI + iI


def bin_center(bins: np.ndarray, idx: int) -> float:
    """Retourne le centre du bin d'indice idx."""
    return 0.5 * (bins[idx] + bins[idx + 1])


def problem_sizes(bin_dicts: dict) -> tuple[int, int]:
    """Retourne (n_states, n_actions) à partir de la configuration de bins."""
    nS = len(bin_dicts["S_bins"]) - 1
    nE = len(bin_dicts["E_bins"]) - 1
    nI = len(bin_dicts["I_bins"]) - 1
    n_states = nS * nE * nI
    n_actions = len(bin_dicts["actions"])
    return n_states, n_actions


# ─── Enregistrement d'épisodes ──────────────────────────────────


def record_episode(
    env: gym.Env,
    policy_fn,
    record: bool = True,
) -> tuple[float, dict]:
    """Simule un épisode complet et enregistre l'historique.

    A partir de la documentation Gymnasium:
    https://gymnasium.farama.org/introduction/record_agent/

    Args:
        env: Environnement Gymnasium.
        policy_fn: Fonction obs → action (np.ndarray de shape (2,)).
        record: Si True, enregistre l'historique détaillé.
    """
    obs, info = env.reset()
    episode_over = False
    total_reward = 0.0

    history = {
        "S": [],
        "E": [],
        "I": [],
        "R": [],
        "D": [],
        "u_conf": [],
        "u_vacc": [],
        "reward": [],
        "L_eco": [],
        "L_vacc": [],
        "L_deaths": [],
        "L_infection": [],
    }

    while not episode_over:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        total_reward += reward

        if record:
            history["S"].append(obs[0])
            history["E"].append(obs[1])
            history["I"].append(obs[2])
            history["R"].append(obs[3])
            history["D"].append(obs[4])
            history["u_conf"].append(float(action[0]))
            history["u_vacc"].append(float(action[1]))
            history["reward"].append(reward)
            history["L_eco"].append(info.get("L_eco", 0))
            history["L_vacc"].append(info.get("L_vacc", 0))
            history["L_deaths"].append(info.get("L_deaths", 0))
            history["L_infection"].append(info.get("L_infection", 0))

    if record:
        for k in history:
            history[k] = np.array(history[k])

    return total_reward, history
