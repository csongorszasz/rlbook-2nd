"""
Exercise 7.10
---

Devised example: Random Walk (Example 6.2 from the book)

    A Markov decision process. Capital letters are states; starting state is C; numbers above edges represent rewards of state-transitions;
    in any given non-terminal state; possible actions in any non-terminal state are moving 'left' or 'right' (10 possible transitions in total); 
    an episode ends when a terminal state ([end]) is reached. Returns are discounted with $gamma=0.9$ to encourage shorter paths to termination.

                  0     0     0     0     0     1
            [end] <- A <-> B <-> C <-> D <-> E -> [end]
                              (start)
"""

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#---
# metrics
#---
class EvalMetrics:
    def __init__(self, config):
        self.alpha = config.alpha
        self.n = config.n
        self.with_control_variates = config.with_control_variates
        self.timesteps = [] # length of episodes
        self.rewards = [] # total rewards per episode

    def step_episode(self, reward, timesteps):
        self.rewards.append(reward)
        self.timesteps.append(timesteps)

    def get_display_name(self):
        return fr"$\alpha={self.alpha}, n={self.n},$ control_variates={self.with_control_variates}"

class AverageEvalMetrics(EvalMetrics):
    def __init__(self, config):
        super().__init__(config)
        self.reward_std = None

class BufferEvalMetrics:
    def __init__(self, config):
        self._config = config
        self._buffer = []

    def step_run(self, run_eval_metrics: EvalMetrics):
        self._buffer.append(run_eval_metrics)

    def average_mean_reward_per_episodes(self):
        avg_metrics = AverageEvalMetrics(self._config)

        rewards_all = np.array([m.rewards for m in self._buffer])
        avg_metrics.rewards = np.mean(rewards_all, axis=0)
        avg_metrics.reward_std = np.std(rewards_all, axis=0)
        
        return avg_metrics
    
    def average_mean_reward_per_timesteps(self):
        avg_metrics = AverageEvalMetrics(self._config)

        rewards_runs = [
            np.repeat(np.asarray(m.rewards, dtype=float), np.asarray(m.timesteps, dtype=int))
            for m in self._buffer
        ]

        max_len = max(map(len, rewards_runs))
        rewards_padded = np.full((len(rewards_runs), max_len), np.nan)
        for i, r in enumerate(rewards_runs):
            rewards_padded[i, :len(r)] = r

        last_full_column = np.where(~np.isnan(rewards_padded).all(axis=0))[0][-1]

        avg_metrics.rewards = np.nanmean(rewards_padded[:, :last_full_column+1], axis=0)
        avg_metrics.reward_std = np.nanstd(rewards_padded[:, :last_full_column+1], axis=0)

        return avg_metrics

#.

#---
# environment
#---
STATE_SPACE = np.arange(7)
START_STATE = 3
TERM_STATES = set([0, 6])
ACTION_SPACE = np.array([-1, 1])
DISCOUNT = 0.9

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

@dataclass(frozen=True)
class ExperimentConfig:
    alpha: float
    n: int
    with_control_variates: bool
    n_episodes: int = 150
    n_runs: int = 100
    seed: int = 0
    label: Optional[str] = None

    def display_name(self):
        if self.label is not None:
            return self.label
        return fr"$\alpha={self.alpha}, n={self.n},$ control_variates={self.with_control_variates}"

def compute_error(past_states, past_rewards, n, T, tao, values, with_control_variates=False, importance_components=None):
    G = 0.0
    if not with_control_variates:
        discount = 1.0
        for i in range(tao+1, min(tao+n, T)+1):
            G += discount * past_rewards[i % (n+1)]
            discount *= DISCOUNT
        if tao + n < T:
            G += discount * values[past_states[(tao + n) % (n+1)]]
    else:
        if tao + n >= T:
            G = values[past_states[T % (n+1)]]
            back = T-1
        else:
            back = tao+n
        for i in range(back, tao, -1):
            importance = importance_components[i % (n+1)]
            G = importance * (past_rewards[(i+1) % (n+1)] + DISCOUNT * G) + (1 - importance) * values[past_states[i % (n+1)]]
    return G - values[past_states[tao % (n+1)]]

#.

#---
# algorithm
#---
def sample_action(state, policy, rng_agent):
    return rng_agent.choice(len(policy[state, :]), p=policy[state, :])

def update_policy_greedily(policy, values):
    for s in range(STATE_SPACE.size):
        if s in TERM_STATES:
            continue
        max_ind = np.argmax([values[s + a] for a in ACTION_SPACE])
        policy[s, :] = 0.0
        policy[s, max_ind] = 1.0

def compute_importance_ratio(past_states, past_actions, behavior_policy, target_policy, n, T, tao):
    importance = 1.0
    upper_bound = min(tao+n-1, T-1)
    components = np.ones_like(past_states, dtype=float)
    for i in range(tao, upper_bound + 1):
        state = past_states[i % (n+1)]
        action = past_actions[i % (n+1)]
        ratio = target_policy[state, action] / behavior_policy[state, action]
        importance *= ratio
        components[i % (n+1)] = ratio
        # print(f"Importance ratio at step {i}: state={state}, action={action}, target_policy={target_policy[state, action]}, behavior_policy={behavior_policy[state, action]}, ratio={ratio}, cumulative_importance={importance}")
    return importance, components

def update_value(values, state, error, alpha, importance_ratio, with_control_variates):
    if not with_control_variates:
        values[state] += alpha * importance_ratio * error
    else:
        values[state] += alpha * error

def n_step_td_for_values(config: ExperimentConfig, rng_agent):
    metrics = EvalMetrics(config)
    
    # values = np.full(STATE_SPACE.shape, 0.0)
    # values = np.full(STATE_SPACE.shape, 0.5)
    values = np.full(STATE_SPACE.shape, 2.0)
    behavior_policy = np.full((*STATE_SPACE.shape, *ACTION_SPACE.shape), 1 / ACTION_SPACE.size)
    for s in iter(TERM_STATES):
        # values[s] = 0.0
        behavior_policy[s, :] = 0.0
    target_policy = np.full_like(behavior_policy, 0.0)
    update_policy_greedily(target_policy, values)
    
    past_states = np.full(config.n+1, -1, dtype=int)
    past_actions = np.full(config.n+1, -1, dtype=int)
    past_rewards = np.full(config.n+1, 0.0)
    
    for i in range(config.n_episodes):
        print(f"Episode [{i+1}]")
        state = START_STATE
        past_states[0] = state
        action = sample_action(state, behavior_policy, rng_agent)
        past_actions[0] = action
        total_reward = 0
        T = np.inf
        t = 0
        while True:
            if t < T:
                new_state, reward = transition(state, action)
                total_reward += reward
                past_states[(t+1) % (config.n+1)] = new_state
                past_rewards[(t+1) % (config.n+1)] = reward
                if is_terminal(new_state):
                    T = t+1
                else:
                    next_action = sample_action(new_state, behavior_policy, rng_agent)
                    past_actions[(t+1) % (config.n+1)] = next_action
            tao = t-config.n+1
            if tao >= 0:
                importance_ratio, importance_components = compute_importance_ratio(past_states, past_actions, behavior_policy, target_policy, config.n, T, tao)
                error = compute_error(past_states, past_rewards, config.n, T, tao, values, config.with_control_variates, importance_components)
                state_to_update = past_states[tao % (config.n+1)]
                update_value(values, state_to_update, error, config.alpha, importance_ratio, config.with_control_variates)
                update_policy_greedily(target_policy, values)
                # print(f"target_policy={target_policy}, importance_ratio={importance_ratio}, state_to_update={state_to_update}, error={error}, values={values}")
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
    max_steps = 100 # cut off non-terminating episode
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
def plot_mean_reward(*metrics: AverageEvalMetrics, configs: list[ExperimentConfig], xlabel="Episodes"):
    plt.figure(figsize=(16,10)) # 1600x1000
    unique_combos = sorted(list(set([(m.alpha, m.n, m.with_control_variates) for m in metrics])))
    palette = sns.color_palette("tab10", len(unique_combos))
    color_map = {combo: palette[i] for i, combo in enumerate(unique_combos)}

    last_common_idx = min(map(len, [m.rewards for m in metrics]))

    for m in metrics:
        if not m.with_control_variates:
            linestyle = "-"
        else:
            linestyle = "--"
        
        x_values = np.arange(len(m.rewards))
        y_values = m.rewards
        y_std = m.reward_std

        x_values = x_values[:last_common_idx]
        y_values = y_values[:last_common_idx]
        y_std = y_std[:last_common_idx]

        sns.lineplot(
            x=x_values, 
            y=y_values,
            label=m.get_display_name(), 
            linestyle=linestyle,
            linewidth=1,
            color=color_map[(m.alpha, m.n, m.with_control_variates)],
            alpha=0.8
        )

        plt.fill_between(
            x_values,
            np.clip(y_values - y_std, 0.0, 1.0),
            np.clip(y_values + y_std, 0.0, 1.0),
            color=color_map[(m.alpha, m.n, m.with_control_variates)],
            alpha=0.2,
            edgecolor=None
        )
    
    # unique n values and n_runs from configs for filename
    n_values = sorted(set(config.n for config in configs))
    n_runs = configs[0].n_runs if configs else 1
    n_str = "_".join(map(str, n_values))
    alpha = configs[0].alpha if configs else 0.0
    path = f"reward_dist_alpha={alpha}_n={n_str}_runs={n_runs}.png"
    
    plt.xlabel(xlabel=xlabel)
    plt.ylabel("Mean reward")
    plt.title(
        "Reward distribution",
        fontsize=16,
        pad=15, 
    )
    plt.legend(framealpha=0.3)
    plt.savefig(path)
    plt.close()
    print(f"Saved reward plot to: {path}")

def build_test_experiments():
    return [
        ExperimentConfig(alpha=0.05, n=1, with_control_variates=False, n_episodes=20, n_runs=2),
        ExperimentConfig(alpha=0.05, n=1, with_control_variates=True, n_episodes=20, n_runs=2),
    ]

def build_comparison_experiments(n, alpha):
    return [
        ExperimentConfig(alpha=alpha, n=n, with_control_variates=False),
        ExperimentConfig(alpha=alpha, n=n, with_control_variates=True),
    ]


def run_experiment(config: ExperimentConfig):
    rng_agent = np.random.default_rng(config.seed)
    buffer_metrics = BufferEvalMetrics(config)

    for _ in range(config.n_runs):
        _, metrics = n_step_td_for_values(config, rng_agent)
        buffer_metrics.step_run(metrics)

    return buffer_metrics.average_mean_reward_per_episodes()

if __name__ == "__main__":
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
    # for n in (1,2):
        for alpha in (0.05, 0.1, 0.15, 0.20, 0.25, 0.5, 0.9):
        # for alpha in (0.05,):
            configs = build_comparison_experiments(n, alpha)
            metrics_all_runs = [run_experiment(config) for config in configs]
            plot_mean_reward(*metrics_all_runs, configs=configs)

    #---
    # testing 
    #---
    # configs = build_test_experiments()
    # metrics_all_runs = [run_experiment(config) for config in configs]
    # plot_mean_reward(*metrics_all_runs, configs=configs)
    #.