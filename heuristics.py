"""
Politiques heuristiques de référence (baselines).

Ces politiques simples permettent d'évaluer les agents RL par comparaison :
    1. Aucun contrôle (laissez-faire)
    2. Vaccination maximale (sans confinement)
    3. Confinement à seuil (réactif sur I)
    4. Double seuil adaptatif (confinement modulé + vaccination décroissante)
"""

import numpy as np


def policy_no_control(obs: np.ndarray) -> np.ndarray:
    """Aucun contrôle : u_conf=0, u_vacc=0."""
    return np.array([0.0, 0.0], dtype=np.float32)


def policy_vacc_max(obs: np.ndarray) -> np.ndarray:
    """Vaccination maximale, pas de confinement."""
    return np.array([0.0, 1.0], dtype=np.float32)


def policy_seuil_conf(obs: np.ndarray) -> np.ndarray:
    """Confinement fort si I > 2%, vaccination à 50%."""
    I = float(obs[2])
    u_conf = 1.0 if I > 0.02 else 0.0
    return np.array([u_conf, 0.5], dtype=np.float32)


def policy_double_seuil(obs: np.ndarray) -> np.ndarray:
    """Double seuil adaptatif.

    - Confinement modulé selon le niveau d'infection.
    - Vaccination forte quand S est grand (beaucoup de gens à protéger).
    """
    S, E, I = float(obs[0]), float(obs[1]), float(obs[2])

    if I > 0.02:
        u_conf = 0.75
    elif I > 0.005:
        u_conf = 0.25
    else:
        u_conf = 0.0

    # Vaccination : on utilise 100% de la capacité si S est grand, sinon on ralentit
    u_vacc = 1.0 if S > 0.3 else 0.25

    return np.array([u_conf, u_vacc], dtype=np.float32)
