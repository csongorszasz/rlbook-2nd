# Exercise 4.9

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


THETA = 1e-30
P_HEAD = 0.51
GOAL = 100
REWARD_GOAL = 1
REWARD_TRANSITION = 0
DISCOUNT = 1.0

STATES = np.arange(0, GOAL+1, dtype=np.int32) # dummy states: 0, GOAL
ACTIONS = np.arange(0, GOAL, dtype=np.int32) # all actions

def legal_actions(state):
    return ACTIONS[1:min(state, GOAL-state)+1] # exclude action=0

def expected_return(state, action, value):
    ret = 0.0
    
    # (next_state, reward, probability)
    transitions = [(state + action, (state + action == GOAL) * REWARD_GOAL, P_HEAD), 
                   (state - action, REWARD_TRANSITION, 1 - P_HEAD)]

    for next_state, reward, probability in transitions:
        ret += probability * (reward + DISCOUNT * value[next_state])

    return ret

def value_iteration():
    value_history = []
    value = np.zeros_like(STATES, dtype=np.float128)
    delta = np.inf
    while delta >= THETA:
        delta = 0.0
        for s in STATES[1:-1]:
            v = value[s]
            max_ = 0.0
            for a in legal_actions(s):
                ret = expected_return(s, a, value)
                if ret > max_:
                    max_ = ret
            value[s] = max_
            delta = max(delta, np.abs(v - value[s]))
        value_history.append(value.copy())
    return np.array(value_history)

def extract_optimal_policy(value, alternative=0):
    policy = np.zeros_like(STATES)
    for s in STATES[1:GOAL]:
        leg_actions = legal_actions(s)
        action_values = []
        max_val = -np.inf
        for a in leg_actions:
            ret = expected_return(s, a, value)
            if ret > max_val:
                max_val = ret
            action_values.append((a, ret))
    
        optimal_actions = []
        for i, (a, a_val) in enumerate(action_values):
            if np.isclose(a_val, max_val):
                optimal_actions.append(a)
        print(f" found [{len(optimal_actions)}] optimal actions")

        # if multiple optimal policies, select the i'th alternative
        alt_cnt = 0
        for i, a in enumerate(optimal_actions):
            if alt_cnt < alternative and i+1 < len(optimal_actions):
                alt_cnt += 1
                continue
            print(f"state={s}: Selected optimal action alternative nr. [{alt_cnt}]")
            policy[s] = a
            break
        if policy[s] == 0:
            raise ValueError(f"Failed to select an optimal action for state [{s}] with alternative [{alternative}] | max_val={max_val} | action_values={action_values}")

    return policy

def plot_value(value, id=0):
    """
    :param value: batch of values in the shape of (sweep_nr, values)
    """
    x = STATES[:GOAL]
    selected = [0, 1, 2, value.shape[0]-1]
    plt.clf()
    for i in selected:
        y = value[i, x]
        sns.lineplot(x=x, y=y, label=f"sweep {i+1}")

    plt.title(rf"State value estimates ($p_h={P_HEAD}$)")
    plt.xlabel("Capital")
    plt.xlim((1, 99))
    plt.xticks([1, 25, 50, 75, 99])
    plt.ylabel("Probability of win")
    plt.ylim((-0.01, 1.0))
    plt.savefig(f"value_{str(id).zfill(2)}.png")

def plot_policy(policy, id=0, alternative_optimal_policy=None):
    x = STATES[:GOAL]
    y = policy[x]

    plt.clf()
    sns.barplot(x=x, y=y)
    title = "Optimal policy"
    if alternative_optimal_policy is not None:
        title = rf"{title} (ignoring first {alternative_optimal_policy} optimal actions)"
    plt.title(title)
    plt.xlabel("Capital")
    plt.xlim((1, 99))
    plt.xticks([1, 25, 50, 75, 99])
    plt.ylabel("Stake")
    plt.ylim((-0.01, GOAL-1))
    plt.yticks([0, 25, 50, 75, 99])
    plt.savefig(f"policy_{str(id).zfill(2)}.png")

if __name__ == "__main__":
    value = value_iteration()
    print(f"Finished value iteration after [{value.shape[0]}] sweeps")
    for opt_alternative in range(GOAL//2):
        policy = extract_optimal_policy(value[-1, :], opt_alternative)
        plot_policy(policy, opt_alternative, alternative_optimal_policy=opt_alternative)
    plot_value(value)
