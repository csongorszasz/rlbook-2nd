# 1. Reproduction of Jack's Car Rental problem solution (Example 4.2)
# [NOT DONE] 2. Adaptation for Exercise 4.7
########################################################################

import numpy as np
from scipy.stats import poisson
import copy
import matplotlib.pyplot as plt

REWARD_RENT = 10
REWARD_CANNOT_RENT = 0
REWARD_MOVE = -2
EXPECTED_REQUESTS = (3, 4)
EXPECTED_REQUESTS_NEG = tuple(-item for item in EXPECTED_REQUESTS)
EXPECTED_RETURNS = (3, 2)
MAX_CARS_PER_LOC = 20
MAX_MOVES_PER_NIGHT_PER_LOC = 5
GAMMA = 0.9
THETA = 0.1
POISSON_UPPER_BOUND = 11
POISSON_PROBA_REQUESTS1 = [poisson.pmf(i, EXPECTED_REQUESTS[0]) for i in range(POISSON_UPPER_BOUND)]
POISSON_PROBA_REQUESTS2 = [poisson.pmf(i, EXPECTED_REQUESTS[1]) for i in range(POISSON_UPPER_BOUND)]
POISSON_PROBA_RETURNS1 = [poisson.pmf(i, EXPECTED_RETURNS[0]) for i in range(POISSON_UPPER_BOUND)]
POISSON_PROBA_RETURNS2 = [poisson.pmf(i, EXPECTED_RETURNS[1]) for i in range(POISSON_UPPER_BOUND)]

rng = np.random.default_rng(0)

class State:
    """Number of cars at each location at the end of the day."""
    SHAPE = (2,)
    def __init__(self, x: tuple):
        self.x = tuple(min(MAX_CARS_PER_LOC, val) for val in x)

    @classmethod
    def space(cls):
        # TODO: generalize to n dimensions
        for i in range(MAX_CARS_PER_LOC+1):
            for j in range(MAX_CARS_PER_LOC+1):
                yield State((i, j))

    def add(self, u: tuple):
        new_x = np.zeros(self.SHAPE, dtype=int)
        for i in range(len(self.x)):
            new_x[i] = self.x[i] + u[i]
        new_x = np.clip(new_x, 0, MAX_CARS_PER_LOC)
        self.x = tuple(new_x)

    def subtract(self, u: tuple):
        u_neg = tuple(-item for item in u)
        self.add(u_neg)

class Action:
    """Net number of cars moved between the locations overnight."""
    SHAPE = (2,2)
    def __init__(self, x: np.ndarray = None):
        self.x = np.zeros(self.SHAPE) if x is None else x

    def __eq__(self, value):
        return np.array_equal(self.x, value.x)

    @classmethod
    def space(cls):
        # TODO: generalize to n dimensions
        for i in range(MAX_MOVES_PER_NIGHT_PER_LOC+1):
            for j in range(MAX_MOVES_PER_NIGHT_PER_LOC+1):
                a = Action()
                a.x[0, 1] = i
                a.x[1, 0] = j
                yield a

    @classmethod
    def legal_actions(cls, s: State):
        # TODO: generalize to n dimensions
        for i in range(MAX_MOVES_PER_NIGHT_PER_LOC+1):
            for j in range(MAX_MOVES_PER_NIGHT_PER_LOC+1):
                a = Action()
                a.x[0, 1] = i
                a.x[1, 0] = j
                if (a.x[0, 1] <= s.x[0]) and (a.x[1, 0] <= s.x[1]):
                    yield a

def transition(s: State, a: np.ndarray, requests: tuple, returns: tuple):
    # [time]: night
    new_s = State((s.x[0] - a.x[0, 1] + a.x[1, 0], 
                   s.x[1] + a.x[0, 1] - a.x[1, 0]))
    move_reward = REWARD_MOVE * (a.x[0, 1] + a.x[1, 0])
    
    # [time]: middle of day

    new_s_before_requests = copy.deepcopy(new_s)
    new_s.subtract(requests)
    
    rented = (new_s_before_requests.x[0] - new_s.x[0], new_s_before_requests.x[1] - new_s.x[1])
    rent_reward = REWARD_RENT * sum(rented)
    
    ### [omitted because reward for lost business is 0]
    # lost = (np.abs(EXPECTED_REQUESTS[0]) - rented[0], np.abs(EXPECTED_REQUESTS[1]) - rented[1])
    # lose_reward = REWARD_CANNOT_RENT * sum(lost)
    
    new_s.add(returns) # new cars are available for rent only from the next day

    # [time]: end of day (new state represents end of next day)
    
    reward = move_reward + rent_reward #+ lose_reward
    return new_s, reward

class Policy:
    def __init__(self):
        self.x = np.zeros((*tuple(MAX_CARS_PER_LOC+1 for _ in range(State.SHAPE[0])), *Action.SHAPE))

    def __call__(self, s: State):
        return Action(self.x[s.x])
    
    def update(self, s: State, value: "Value"):
        optimal_actions = []
        legal_actions = list(Action.legal_actions(s))
        action_values = np.zeros(len(legal_actions))
        
        for i, a in enumerate(legal_actions):
            expected_value = 0
            for requests1 in range(POISSON_UPPER_BOUND):
                for requests2 in range(POISSON_UPPER_BOUND):
                    for returns1 in range(POISSON_UPPER_BOUND):
                        for returns2 in range(POISSON_UPPER_BOUND):
                            probability = (POISSON_PROBA_REQUESTS1[requests1] * POISSON_PROBA_REQUESTS2[requests2]
                                           *POISSON_PROBA_RETURNS1[returns1] * POISSON_PROBA_RETURNS2[returns2])
                            new_s, reward = transition(s, a, (requests1, requests2), (returns1, returns2))
                            expected_value += probability * (reward + GAMMA * value(new_s))
            action_values[i] = expected_value
        
        opt_action_val = np.max(action_values)
        for i, val in enumerate(action_values):
            if np.isclose(val, opt_action_val):
                optimal_actions.append(legal_actions[i])

        self.x[s.x] = optimal_actions[0].x

        return optimal_actions

class Value:
    def __init__(self):
        self.x = np.zeros(tuple(MAX_CARS_PER_LOC+1 for _ in range(State.SHAPE[0])))

    def __call__(self, s: State):
        return self.x[s.x]
    
    def update(self, s: State, policy: Policy):
        a = policy(s)
        
        expected_value = 0
        for requests1 in range(POISSON_UPPER_BOUND):
            for requests2 in range(POISSON_UPPER_BOUND):
                for returns1 in range(POISSON_UPPER_BOUND):
                    for returns2 in range(POISSON_UPPER_BOUND):
                        probability = (POISSON_PROBA_REQUESTS1[requests1] * POISSON_PROBA_REQUESTS2[requests2]
                                           *POISSON_PROBA_RETURNS1[returns1] * POISSON_PROBA_RETURNS2[returns2])
                        new_s, reward = transition(s, a, (requests1, requests2), (returns1, returns2))
                        expected_value += probability * (reward + GAMMA * self(new_s))

        self.x[s.x] = expected_value

def policy_evaluation(policy: Policy, value: Value):
    i = 0
    while True:
        print(f"Policy evaluation step [{i}]")
        delta = 0
        for s in State.space():
            v = value(s)
            value.update(s, policy)
            delta = max(delta, np.abs(v - value(s)))
        i += 1
        print(f"delta={delta}")
        if delta < THETA:
            print(f"Policy evaluation stopped: delta[{delta}] < theta[{THETA}]")
            break
    return value

def policy_improvement(policy: Policy, value: Value):
    stable = True
    for s in State.space():
        old_action = policy(s)
        optimal_actions = policy.update(s, value)
        if old_action not in optimal_actions:
            stable = False
    return policy, stable

def policy_iteration(policy0):
    policies = [policy0]
    values = []
    
    po = policies[-1]
    v = Value()
    stable = False
    i = 0
    while not stable:
        print(f"Policy iteration round [{i}]")
        stable = True
        v = policy_evaluation(po, v)
        po, stable = policy_improvement(po, v)
        values.append(v)
        policies.append(po)
        i += 1
    values.append(policy_evaluation(policies[-1], values[-1]))
    print(f"Policy iteration finished after [{i}] iterations")

    return policies, values

def log_summary(policies, values):
    for i in range(len(policies)):
        print(f"Iteration: [{i}]")
        print(f"  value_min={np.min(values[i].x)} value_max={np.max(values[i].x)} value_mean={np.mean(values[i].x)}")
        print(f"  policy_min={np.min(policies[i].x)} policy_max={np.max(policies[i].x)} policies_mean={np.mean(policies[i].x)}")

def plot_policy(policy: Policy, title: str, id: int):
    fig = plt.figure()
    data = policy.x
    data[:, :, 1, :] = -data[:, :, 1, :]
    data = data.sum(axis=(2, 3))
    ax = plt.gca()
    ax.set_title(title)
    im = ax.imshow(data)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("net no. cars moved between locations", rotation=-90)
    plt.xlabel("no. cars at location 1")
    plt.ylabel("no. cars at location 2")
    plt.savefig(f"policy_{id}.png")

def plot_value(value: Value):
    ...

if __name__ == "__main__":
    policies, values = policy_iteration(Policy())
    log_summary(policies, values)
    for i, po in enumerate(policies):
        plot_policy(po, rf"$\pi_{i}$", i)
    # dummy_po = Policy()
    # dummy_po.x[10, 10, 0, 1] = 5
    # dummy_po.x[10, 5, 1, 0] = 3
    # plot_policy(dummy_po, rf"dummy policy $\pi_d$", 0)
    # dummy_val = Value()
    # plot_value(dummy_val)

"""
Output
-------

Policy iteration round [0]
Policy evaluation step [0]
delta=191.14044425450055
...
Policy evaluation step [36]
delta=0.08413336513081049
Policy evaluation stopped: delta[0.08413336513081049] < theta[0.1]
Policy evaluation step [0]
delta=64.13722433726383
...
Policy evaluation step [17]
delta=0.09384874370039142
Policy evaluation stopped: delta[0.09384874370039142] < theta[0.1]
Policy iteration finished after [1] iterations
Iteration: [0]
  value_min=402.10287936952636 value_max=607.2791629089214 value_mean=538.9271065616259
  policy_min=0.0 policy_max=5.0 policies_mean=0.6377551020408163
Iteration: [1]
  value_min=402.10287936952636 value_max=607.2791629089214 value_mean=538.9271065616259
  policy_min=0.0 policy_max=5.0 policies_mean=0.6377551020408163

------
Figures: policy_0.png; policy_1.png

"""





