"""
Fonctions de visualisation pour l'analyse des résultats.

Chaque fonction produit un graphique matplotlib autonome.

NOTE: Nous avons utilisé ici assez fortement l'IA afin d'obtenir des graphiques
pertinents et surtout esthétiques.
"""

import numpy as np
import matplotlib.pyplot as plt


# ─── Courbes d'apprentissage ─────────────────────────────────────


def smooth(x, window: int = 100) -> np.ndarray:
    """Moyenne glissante (convolution valide)."""
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_learning_curves(
    rewards_dict: dict[str, list],
    window: int = 100,
    title: str = "Courbes d'apprentissage",
):
    """Trace les courbes de reward moyen glissant pour plusieurs agents.

    Args:
        rewards_dict: {"nom_agent": [reward_ep0, reward_ep1, ...]}.
        window: Taille de la fenêtre glissante.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    for name, rewards in rewards_dict.items():
        ax.plot(smooth(rewards, window), label=name, alpha=0.9)

    ax.set_xlabel("Épisode")
    ax.set_ylabel(f"Reward moyen (fenêtre glissante {window})")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ─── Évolution des compartiments SEIRD ───────────────────────────


def plot_seird_evolution(results: dict[str, dict], policy_names: list[str]):
    """Trace l'évolution S, E, I, R, D pour chaque politique (un subplot par politique)."""
    compartments = ["S", "E", "I", "R", "D"]
    colors = {
        "S": "tab:blue",
        "E": "tab:orange",
        "I": "tab:red",
        "R": "tab:green",
        "D": "tab:gray",
    }

    n_pol = len(policy_names)
    fig, axes = plt.subplots(n_pol, 1, figsize=(14, 3.5 * n_pol), sharex=True)
    if n_pol == 1:
        axes = [axes]

    for i, name in enumerate(policy_names):
        ax = axes[i]
        h = results[name]
        days = np.arange(len(h["S"]))
        for c in compartments:
            ax.plot(days, h[c], label=c, color=colors[c], linewidth=1.5)
        ax.set_ylabel("Proportion")
        ax.set_title(name, fontweight="bold")
        ax.legend(loc="center right", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, 365)

    axes[-1].set_xlabel("Jours")
    fig.suptitle(
        "Évolution des compartiments SEIRD selon la politique", fontsize=14, y=1.01
    )
    plt.tight_layout()
    plt.show()


# ─── Commandes appliquées ────────────────────────────────────────


def plot_commands(results: dict[str, dict], policy_names: list[str]):
    """Trace u_conf et u_vacc au cours du temps pour chaque politique."""
    n_pol = len(policy_names)
    fig, axes = plt.subplots(n_pol, 2, figsize=(14, 3 * n_pol), sharex=True)
    if n_pol == 1:
        axes = axes[np.newaxis, :]

    for i, name in enumerate(policy_names):
        h = results[name]
        days = np.arange(len(h["u_conf"]))

        axes[i, 0].step(days, h["u_conf"], where="post", color="tab:red", linewidth=1.2)
        axes[i, 0].set_ylabel("$u_{conf}$")
        axes[i, 0].set_ylim(-0.05, 1.05)
        axes[i, 0].set_title(f"{name} — Confinement", fontsize=10)

        axes[i, 1].step(
            days, h["u_vacc"], where="post", color="tab:green", linewidth=1.2
        )
        axes[i, 1].set_ylabel("$u_{vacc}$")
        axes[i, 1].set_ylim(-0.05, 1.05)
        axes[i, 1].set_title(f"{name} — Vaccination", fontsize=10)

        for ax in axes[i]:
            ax.set_xlim(0, 365)

    axes[-1, 0].set_xlabel("Jours")
    axes[-1, 1].set_xlabel("Jours")
    fig.suptitle("Commandes appliquées au cours du temps", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# ─── Comparaison I(t) et D(t) ───────────────────────────────────


def plot_epidemic_comparison(results: dict[str, dict], policy_names: list[str]):
    """Compare la courbe épidémique I(t) et les décès D(t) entre politiques."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    n_pol = len(policy_names)
    cmap = plt.cm.Set1

    for i, name in enumerate(policy_names):
        h = results[name]
        days = np.arange(len(h["I"]))
        ls = "--" if name in ["Aucun contrôle", "Vaccination max"] else "-"
        color = cmap(i / max(n_pol, 1))
        ax1.plot(days, h["I"], label=name, color=color, linestyle=ls, linewidth=1.5)
        ax2.plot(days, h["D"], label=name, color=color, linestyle=ls, linewidth=1.5)

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 365)

    ax1.set_title("Courbe épidémique I(t)")
    ax1.set_xlabel("Jours")
    ax1.set_ylabel("Proportion d'infectés")
    ax1.legend(fontsize=8)

    ax2.set_title("Décès cumulés D(t)")
    ax2.set_xlabel("Jours")
    ax2.set_ylabel("Proportion de décès")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ─── Décomposition des coûts ─────────────────────────────────────


def plot_cost_decomposition(results: dict[str, dict], policy_names: list[str]):
    """Barres empilées des coûts cumulés par composante."""
    cost_keys = ["L_eco", "L_vacc", "L_deaths", "L_infection"]
    cost_labels = [
        r"$\mathcal{L}_{eco}$",
        r"$\mathcal{L}_{vacc}$",
        r"$\mathcal{L}_{deaths}$",
        r"$\mathcal{L}_{hosp}$",
    ]
    cost_colors = ["tab:blue", "tab:green", "tab:gray", "tab:red"]

    n_pol = len(policy_names)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_pol)
    width = 0.6
    bottoms = np.zeros(n_pol)

    for ck, cl, cc in zip(cost_keys, cost_labels, cost_colors):
        vals = np.array([float(np.sum(results[name][ck])) for name in policy_names])
        ax.bar(x, vals, width, bottom=bottoms, label=cl, color=cc, alpha=0.85)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(policy_names, rotation=25, ha="right")
    ax.set_ylabel("Coût cumulé (unités $C_{eco}$·jours)")
    ax.set_title("Décomposition des coûts cumulés par politique")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Tableau récapitulatif
    print(
        f"\n{'Politique':25s} | {'L_eco':>10s} | {'L_vacc':>10s} "
        f"| {'L_deaths':>10s} | {'L_hosp':>10s} | {'TOTAL':>10s}"
    )
    print("-" * 85)
    for name in policy_names:
        h = results[name]
        le = np.sum(h["L_eco"])
        lv = np.sum(h["L_vacc"])
        ld = np.sum(h["L_deaths"])
        li = np.sum(h["L_infection"])
        print(
            f"{name:25s} | {le:10.1f} | {lv:10.1f} "
            f"| {ld:10.1f} | {li:10.1f} | {le + lv + ld + li:10.1f}"
        )


# ─── Sensibilité ────────────────────────────────────────────────


def plot_sensitivity(
    sensitivity_results: dict[str, dict],
    multipliers: list[float],
):
    """Grille 3×3 montrant I(t) et commandes pour chaque combinaison (C_vacc, C_hosp)."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)

    for i, m_vacc in enumerate(multipliers):
        for j, m_hosp in enumerate(multipliers):
            ax = axes[i, j]
            label = f"C_vacc×{m_vacc}, C_hosp×{m_hosp}"
            h = sensitivity_results[label]
            days = np.arange(len(h["I"]))

            ax.plot(days, h["I"], color="tab:red", label="I(t)", linewidth=1.5)
            ax.plot(
                days,
                h["u_conf"],
                color="tab:blue",
                label="$u_{conf}$",
                alpha=0.6,
                linewidth=1,
            )
            ax.plot(
                days,
                h["u_vacc"],
                color="tab:green",
                label="$u_{vacc}$",
                alpha=0.6,
                linewidth=1,
            )
            ax.set_title(f"$C_{{vacc}}$×{m_vacc}, $C_{{hosp}}$×{m_hosp}", fontsize=10)
            ax.set_xlim(0, 365)

            if i == 0 and j == 0:
                ax.legend(fontsize=7)

    axes[2, 1].set_xlabel("Jours")
    axes[1, 0].set_ylabel("Proportion / Intensité")
    fig.suptitle(
        "Sensibilité de la politique Q-Learning à $C_{vacc}$ et $C_{hosp}$",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()
    plt.show()
