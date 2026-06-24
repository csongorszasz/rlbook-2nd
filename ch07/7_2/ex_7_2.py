"""
Exercise 7.2
---

Devised example: Random Walk (Example 6.2 from the book)

    A Markov reward process. Capital letters are states; starting state is C; numbers above edges represent rewards of state-transitions;
    in any given non-terminal state, there is an equal probability of moving 'left' or 'right' (10 possible transitions in total); 
    an episode ends when a terminal state ([end]) is reached.
    

                  0     0     0     0     0     1
            [end] <- A <-> B <-> C <-> D <-> E -> [end]
                              (start)
"""

import numpy as np
from enum import Enum, auto
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#---
# metrics
#---
class EvalMetrics:
    def __init__(self, alpha, n, error_formula):
        self.alpha = alpha
        self.n = n
        self.error_formula = error_formula
        self.rms = []

    def step_episode(self, estimate, target):
        rms = 0
        for i in range(len(estimate)):
            rms += ((target[i] - estimate[i]) ** 2) / len(estimate)
        rms = np.sqrt(rms)
        self.rms.append(rms)

    def get_display_name(self):
        return fr"$\alpha={self.alpha}, n={self.n},$ error={self.error_formula.name}"

class AveragedEvalMetrics(EvalMetrics):
    def step_run(self, run_eval_metrics: EvalMetrics):
        if self.rms == []:
            self.rms = run_eval_metrics.rms
        else:
            for i in range(len(run_eval_metrics.rms)):
                self.rms[i] += run_eval_metrics.rms[i]
    
    def finalize(self, n_runs):
        for i in range(len(self.rms)):
            self.rms[i] /= n_runs
#.

#---
# environment
#---
RNG = np.random.default_rng(0)
STATE_SPACE = np.arange(7)
START_STATE = 3
TERM_STATES = set([0, 6])
VALUES_REAL = np.array([0.0, 1/6, 2/6, 3/6, 4/6, 5/6, 0.0])

def get_reward(old_state, new_state):
    if old_state == 5 and new_state == 6:
        return 1.0
    return 0.0

def transition(state):
    if RNG.uniform(0.0, 1.0) <= 0.5:
        new_state = state - 1
    else:
        new_state = state + 1
    reward = get_reward(state, new_state)
    return new_state, reward

def is_terminal(state):
    return state in TERM_STATES
#.

#---
# error computation
#---
class ErrorFormula(Enum):
    REGULAR_DIFF = auto()
    TD_SUM = auto()

def _error_regular_diff(past_states, past_rewards, n, T, tao, values, *args):
    G = 0
    for i in range(tao+1, min(tao+n, T)+1):
        G += past_rewards[i % (n+1)] # undiscounted
    if tao + n < T:
        G += values[past_states[(tao + n) % (n+1)]] # undiscounted
    return G - values[past_states[tao % (n+1)]]

def _error_td_sum(past_states, past_rewards, n, T, tao, values, values_episode_start):
    error = 0
    for i in range(tao, min(tao+n, T)):
        error += past_rewards[(i+1) % (n+1)] + values_episode_start[past_states[(i+1) % (n+1)]] - values_episode_start[past_states[i % (n+1)]] # undiscounted
    return error

ERROR_FN = {
    ErrorFormula.REGULAR_DIFF: _error_regular_diff,
    ErrorFormula.TD_SUM: _error_td_sum
}

def compute_error(past_states, past_rewards, n, T, tao, values, values_episode_start, error_formula: ErrorFormula):
    return ERROR_FN[error_formula](past_states, past_rewards, n, T, tao, values, values_episode_start)
#.

#---
# algorithm
#---
def n_step_td_for_values(alpha, n, n_episodes, error_formula: ErrorFormula, reset_rng=False):
    if reset_rng:
        global RNG
        RNG = np.random.default_rng(0)
        
    metrics = EvalMetrics(alpha, n, error_formula)
    values = np.full(STATE_SPACE.shape, 0.5)
    for s in iter(TERM_STATES):
        values[s] = 0.0
    metrics.step_episode(values, VALUES_REAL)
    past_states = np.full(n+1, -1, dtype=int)
    past_rewards = np.full(n+1, 0.0)
    
    for i in range(n_episodes):
        print(f"Episode [{i+1}]")
        values_episode_start = values.copy()
        state = START_STATE
        past_states[0] = state
        T = np.inf
        t = 0
        while True:
            if t < T:
                new_state, reward = transition(state)
                past_states[(t+1) % (n+1)] = new_state
                past_rewards[(t+1) % (n+1)] = reward
                if is_terminal(new_state):
                    T = t+1
            tao = t-n+1
            if tao >= 0:
                error = compute_error(past_states, past_rewards, n, T, tao, values, values_episode_start, error_formula)
                state_to_update = past_states[tao % (n+1)]
                values[state_to_update] = values[state_to_update] + alpha * error
            if tao == T-1:
                break
            state = new_state
            t += 1
        print(f"Ended after {t+1} steps")
        metrics.step_episode(values, VALUES_REAL)
    return values, metrics
#.

#---
# visualization
#---
def plot_rms_over_episodes(*metrics: EvalMetrics):
    plt.figure(figsize=(16,10)) # 1600x1000
    unique_combos = sorted(list(set([(m.alpha, m.n) for m in metrics])))
    palette = sns.color_palette("tab10", len(unique_combos))
    color_map = {combo: palette[i] for i, combo in enumerate(unique_combos)}

    for m in metrics:
        if m.error_formula == ErrorFormula.REGULAR_DIFF:
            linestyle = "-"
        else:
            linestyle = "--"
    
        sns.lineplot(
            x=np.arange(len(m.rms)), 
            y=m.rms, 
            label=m.get_display_name(), 
            linestyle=linestyle,
            linewidth=1,
            color=color_map[(m.alpha, m.n)],
            alpha=0.8
        )
    plt.xlabel("Episodes")
    plt.ylabel("Average RMS error")
    plt.title(
        "RMS error over episodes",
        fontsize=16, 
        pad=15, 
    )
    plt.legend(framealpha=0.3)
    path = "rms_over_episodes.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved RMS error over episodes plot to: {path}")

if __name__ == "__main__":
    metrics_all_runs = []
    alpha = [0.05] * 8 + [0.1] * 8 + [0.15] * 8
    n = [1, 2, 4, 8] * 6
    error_formula = ([ErrorFormula.REGULAR_DIFF] * 4 + [ErrorFormula.TD_SUM] * 4) * 3
    n_episodes = 100
    n_runs = 100

    for i in range(len(alpha)):
        avg_metrics = AveragedEvalMetrics(alpha[i], n[i], error_formula[i])
        for _ in range(n_runs):
            values, metrics = n_step_td_for_values(alpha[i], n[i], n_episodes, error_formula[i], reset_rng=False)
            avg_metrics.step_run(metrics)
        avg_metrics.finalize(n_runs)
        metrics_all_runs.append(avg_metrics)
    
    plot_rms_over_episodes(*metrics_all_runs)

    #---
    # testing single run 
    #---
    # values = n_step_td_for_values(0.05, 1, 100, ErrorFormula.TD_SUM)
    # with np.printoptions(precision=3):
    #     print(f"Real      : {VALUES_REAL}")
    #     print(f"Estimates : {values}")
    #.