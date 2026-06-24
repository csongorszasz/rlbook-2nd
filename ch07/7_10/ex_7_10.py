"""
Exercise 7.10
---

Devised example: Random Walk (Example 6.2 from the book)

    A Markov decision process. Capital letters are states; starting state is C; numbers above edges represent rewards of state-transitions;
    in any given non-terminal state; possible actions in any non-terminal state are moving 'left' or 'right' (10 possible transitions in total); 
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
    def __init__(self, alpha, n, error_formula, with_control_variates):
        self.alpha = alpha
        self.n = n
        self.error_formula = error_formula
        self.with_control_variates = with_control_variates
        self.timesteps = [] # length of episodes
        self.rewards = [] # total rewards per episode

    def step_episode(self, reward, timesteps):
        self.rewards.append(reward)
        self.timesteps.append(timesteps)

    def get_display_name(self):
        return fr"$\alpha={self.alpha}, n={self.n},$ control_variates={self.with_control_variates}"

class AverageEvalMetrics(EvalMetrics):
    def __init__(self, alpha, n, error_formula, with_control_variates):
        super().__init__(alpha, n, error_formula, with_control_variates)
        self.reward_std = None

class BufferEvalMetrics:
    def __init__(self):
        self._buffer = []

    def step_run(self, run_eval_metrics: EvalMetrics):
        self._buffer.append(run_eval_metrics)

    def average_mean_reward_last_10(self):
        avg_metrics = AverageEvalMetrics(self._buffer[0].alpha, self._buffer[0].n, self._buffer[0].error_formula, self._buffer[0].with_control_variates)
        rewards_all = np.array([m.rewards for m in self._buffer])
        avg_metrics.rewards = np.mean(rewards_all, axis=0)
        avg_metrics.reward_std = np.std(rewards_all, axis=0)
        return avg_metrics
#.

#---
# environment
#---
RNG = np.random.default_rng(0)
STATE_SPACE = np.arange(7)
START_STATE = 3
TERM_STATES = set([0, 6])
ACTION_SPACE = np.array([-1, 1])

def get_reward(old_state, new_state):
    if old_state == 5 and new_state == 6:
        return 1.0
    return 0.0

def transition(state, action_ind):
    action = ACTION_SPACE[action_ind]
    new_state = state + action
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

def _error_regular_diff(past_states, past_rewards, n, T, tao, values, *args, **kwargs):
    G = 0
    with_control_variates = kwargs.get("with_control_variates", False)
    if not with_control_variates:
        for i in range(tao+1, min(tao+n, T)+1):
            G += past_rewards[i % (n+1)] # undiscounted
        if tao + n < T:
            G += values[past_states[(tao + n) % (n+1)]] # undiscounted
    else:
        ...
    return G - values[past_states[tao % (n+1)]]

def _error_td_sum(past_states, past_rewards, n, T, tao, values, values_episode_start, *args, **kwargs):
    error = 0
    for i in range(tao, min(tao+n, T)):
        error += past_rewards[(i+1) % (n+1)] + values_episode_start[past_states[(i+1) % (n+1)]] - values_episode_start[past_states[i % (n+1)]] # undiscounted
    return error

ERROR_FN = {
    ErrorFormula.REGULAR_DIFF: _error_regular_diff,
    ErrorFormula.TD_SUM: _error_td_sum
}

def compute_error(past_states, past_rewards, n, T, tao, values, values_episode_start, error_formula: ErrorFormula, with_control_variates=False):
    return ERROR_FN[error_formula](past_states, past_rewards, n, T, tao, values, values_episode_start, with_control_variates=with_control_variates)
#.

#---
# algorithm
#---
def sample_action(state, policy):
    return RNG.choice(len(policy[state, :]), p=policy[state, :])

def update_greedily(policy, values):
    for s in range(STATE_SPACE.size):
        if s in TERM_STATES:
            continue
        max_ind = np.argmax([values[s + a] for a in ACTION_SPACE])
        policy[s, :] = 0.0
        policy[s, max_ind] = 1.0

def compute_importance_ratio(past_states, past_actions, behavior_policy, target_policy, n, T, tao):
    importance = 1
    # ratios = [None] * (min(tao+n, T-1) - tao)
    for i in range(tao, min(tao+n, T-1)): # TODO verify
        state = past_states[i % (n+1)]
        action = past_actions[i % (n+1)]
        ratio = target_policy[state, action] / behavior_policy[state, action]
        importance *= ratio
        # print(f"Importance ratio at step {i}: state={state}, action={action}, target_policy={target_policy[state, action]}, behavior_policy={behavior_policy[state, action]}, ratio={ratio}, cumulative_importance={importance}")
    return importance

def n_step_td_for_values(alpha, n, n_episodes, error_formula: ErrorFormula, with_control_variates: bool, reset_rng=False):
    if reset_rng:
        global RNG
        RNG = np.random.default_rng(0)
        
    metrics = EvalMetrics(alpha, n, error_formula, with_control_variates)
    
    # values = np.full(STATE_SPACE.shape, 0.0)
    # values = np.full(STATE_SPACE.shape, 0.5)
    values = np.full(STATE_SPACE.shape, 2.0)
    behavior_policy = np.full((*STATE_SPACE.shape, *ACTION_SPACE.shape), 1 / ACTION_SPACE.size)
    for s in iter(TERM_STATES):
        # values[s] = 0.0
        behavior_policy[s] = 0.0
    target_policy = np.full_like(behavior_policy, 0.0)
    update_greedily(target_policy, values)
    
    past_states = np.full(n+1, -1, dtype=int)
    past_actions = np.full(n+1, -1, dtype=int)
    past_rewards = np.full(n+1, 0.0)
    
    for i in range(n_episodes):
        print(f"Episode [{i+1}]")
        values_episode_start = values.copy()
        state = START_STATE
        past_states[0] = state
        action = sample_action(state, behavior_policy)
        past_actions[0] = action
        total_reward = 0
        T = np.inf
        t = 0
        while True:
            if t < T:
                new_state, reward = transition(state, action)
                total_reward += reward
                past_states[(t+1) % (n+1)] = new_state
                past_rewards[(t+1) % (n+1)] = reward
                if is_terminal(new_state):
                    T = t+1
                else:
                    next_action = sample_action(new_state, behavior_policy)
                    past_actions[(t+1) % (n+1)] = next_action
            tao = t-n+1
            if tao >= 0:
                # TODO: introduce control variates
                
                importance_ratio = compute_importance_ratio(past_states, past_actions, behavior_policy, target_policy, n, T, tao)
                error = compute_error(past_states, past_rewards, n, T, tao, values, values_episode_start, error_formula, with_control_variates)
                state_to_update = past_states[tao % (n+1)]
                values[state_to_update] = values[state_to_update] + alpha * importance_ratio * error
                # print(f"target_policy={target_policy}, importance_ratio={importance_ratio}, state_to_update={state_to_update}, error={error}, values={values}")
                update_greedily(target_policy, values)
            if tao == T-1:
                break
            state = new_state
            action = next_action
            t += 1
        # print(f"Ended after {T} steps")
        eval_rewards, eval_timesteps = evaluate_estimates(values)
        print(f"Evaluation: total_reward={eval_rewards}, episode_length={eval_timesteps}")
        metrics.step_episode(eval_rewards, eval_timesteps)
    return values, metrics

def evaluate_estimates(values):
    """Runs an episode following a greedy policy w.r.t. given value estimates. Returns the total reward and episode length."""
    # print(f"Evaluating the target policy... [state values: {values}]")
    total_reward = 0
    state = START_STATE
    max_steps = 1000
    t = 0
    while t < max_steps:
        action_ind = np.argmax([values[state + a] for a in ACTION_SPACE])
        new_state, reward = transition(state, action_ind)
        total_reward += reward
        # print(f"Step [{t}]: state={state}, action={ACTION_SPACE[action_ind]}, reward={reward}, new_state={new_state}")
        if is_terminal(new_state):
            break
        state = new_state
        t += 1
    return total_reward, t+1
#.

#---
# visualization
#---
def plot_mean_reward(*metrics: AverageEvalMetrics):
    plt.figure(figsize=(16,10)) # 1600x1000
    unique_combos = sorted(list(set([(m.alpha, m.n) for m in metrics])))
    palette = sns.color_palette("tab10", len(unique_combos))
    color_map = {combo: palette[i] for i, combo in enumerate(unique_combos)}

    for m in metrics:
        if not m.with_control_variates:
            linestyle = "-"
        else:
            linestyle = "--"
        
        x_values = np.arange(len(m.rewards))
        y_values = m.rewards
        y_std = m.reward_std

        sns.lineplot(
            x=x_values, 
            y=y_values,
            label=m.get_display_name(), 
            linestyle=linestyle,
            linewidth=1,
            color=color_map[(m.alpha, m.n)],
            alpha=0.8
        )

        plt.fill_between(
            x_values,
            np.clip(y_values - y_std, 0.0, 1.0),
            np.clip(y_values + y_std, 0.0, 1.0),
            color=color_map[(m.alpha, m.n)],
            alpha=0.2,
            edgecolor=None
        )
    
    plt.xlabel("Episodes")
    plt.ylabel("Mean reward")
    plt.title(
        "Reward distribution",
        fontsize=16,
        pad=15, 
    )
    plt.legend(framealpha=0.3)
    path = "reward_distribution.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved reward plot to: {path}")

if __name__ == "__main__":
    metrics_all_runs = []
    alpha = [0.05] * 4# * 2
    n = [1, 2, 4, 8]# * 2
    error_formula = [ErrorFormula.REGULAR_DIFF] * 4# * 2
    with_control_variates = [False] * 4# + [True] * 4
    n_episodes = 250
    n_runs = 10

    for i in range(len(alpha)):
        buffer_metrics = BufferEvalMetrics()
        for _ in range(n_runs):
            values, metrics = n_step_td_for_values(alpha[i], n[i], n_episodes, error_formula[i], with_control_variates[i], reset_rng=False)
            buffer_metrics.step_run(metrics)
        avg_metrics = buffer_metrics.average_mean_reward_last_10()
        metrics_all_runs.append(avg_metrics)
    
    plot_mean_reward(*metrics_all_runs)

    #---
    # testing single configuration 
    #---
    # metrics_all_runs = []
    # buffer_metrics = BufferEvalMetrics()
    # for _ in range(100):
    #     values, metrics = n_step_td_for_values(0.05, 1, 150, ErrorFormula.REGULAR_DIFF, False)
    #     buffer_metrics.step_run(metrics)
    # metrics_all_runs.append(buffer_metrics.average_mean_reward_last_10())
    # plot_mean_reward(*metrics_all_runs)
    #.