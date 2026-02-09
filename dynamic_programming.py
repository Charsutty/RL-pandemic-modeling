import numpy as np
from tqdm.auto import tqdm
from model import SEIREnv
from utils import bin_center, state_to_id, problem_sizes


def run_approximate_dp(config, bin_dicts, K=3, gamma=0.99):
    """
    Performs backward induction (Approximate Dynamic Programming) on a finite horizon.

    Args:
        config: ProblemConfig object.
        bin_dicts: Dictionary containing S_bins, E_bins, I_bins and actions.
        K: Number of Monte-Carlo samples per (state, action) pair.
        gamma: Discount factor.

    Returns:
        V_dp: (T+1, n_states) array of values.
        pi_dp: (T, n_states) array of optimal action indices.
    """
    T = config.max_steps
    S_bins = bin_dicts["S_bins"]
    E_bins = bin_dicts["E_bins"]
    I_bins = bin_dicts["I_bins"]
    actions = bin_dicts["actions"]

    n_states, n_actions = problem_sizes(bin_dicts)
    nS, nE, nI = len(S_bins) - 1, len(E_bins) - 1, len(I_bins) - 1

    # Tables de valeurs et politique
    V_dp = np.zeros((T + 1, n_states), dtype=np.float64)
    pi_dp = np.zeros((T, n_states), dtype=np.int32)

    # États plausibles (S+E+I ≤ 1.05)
    plausible = []
    for iS in range(nS):
        for iE in range(nE):
            for iI in range(nI):
                s_val = bin_center(S_bins, iS)
                e_val = bin_center(E_bins, iE)
                i_val = bin_center(I_bins, iI)
                if s_val + e_val + i_val <= 1.05:
                    sid = (iS * nE + iE) * nI + iI
                    plausible.append((sid, s_val, e_val, i_val))

    print(f"États plausibles : {len(plausible)} / {n_states}")

    env_dp = SEIREnv(config)

    for t in tqdm(range(T - 1, -1, -1), desc="DP backward"):
        for sid, s_val, e_val, i_val in plausible:
            best_val = -np.inf
            best_a = 0

            for a_idx in range(n_actions):
                total_r = 0.0
                total_v_next = 0.0

                for _ in range(K):
                    # For setting the internal state, we need R as well
                    r_val = max(0, 1.0 - s_val - e_val - i_val)
                    env_dp.state = np.array(
                        [s_val, e_val, i_val, r_val, 0.0], dtype=np.float32
                    )
                    env_dp.current_step = t
                    obs2, r, _, _, _ = env_dp.step(actions[a_idx])
                    s2 = state_to_id(obs2, bin_dicts)
                    total_r += r
                    total_v_next += V_dp[t + 1, s2]

                q_val = total_r / K + gamma * total_v_next / K

                if q_val > best_val:
                    best_val = q_val
                    best_a = a_idx

            V_dp[t, sid] = best_val
            pi_dp[t, sid] = best_a

    return V_dp, pi_dp
