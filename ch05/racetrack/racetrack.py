# Exercise 5.12: Racetrack
"""
STATE SPACE:

- 2D grid, where:
    
    0: off-track
    1: on-track
    2: start
    3: finish

- velocity: (vertical, horizontal), values in the discrete range [0, 4]

    - value (0, 0) is allowed only in start positions

state = (car_location, velocity)


ACTION SPACE:

   [[-1 -1]
    [ 0 -1]
    [ 1 -1]
    [-1  0]
    [ 0  0]
    [ 1  0]
    [-1  1]
    [ 0  1]
    [ 1  1]]

, where 
    
    row = (vertical_velocity_increment, horizontal_velocity_increment)
"""

import numpy as np
from typing import Generator
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path

GRID_PATH = Path(__file__).parent / "track_example_2.txt"
REWARD_TRANSITION = -1
DISCOUNT = 1.0
MAX_EPISODES = 10000
VELOCITY_MIN = 0
VELOCITY_MAX = 4
NOISE_ON = False
NOISE_PROBA = 0.1
LEARN = False

rng = np.random.default_rng(0)

class State:
    def __init__(self, car_loc=None, velocity=None):
        self.car_loc = car_loc if car_loc is not None else np.zeros(2, dtype=np.int32)
        self.velocity = velocity if velocity is not None else np.zeros(2, dtype=np.int8)

    def idx(self, action=None):
        if action is None:
            return (self.car_loc[0], self.car_loc[1], self.velocity[0], self.velocity[1])
        return (self.car_loc[0], self.car_loc[1], self.velocity[0], self.velocity[1], action)

class StateSpace:
    def __init__(self, track: np.ndarray):
        self.track = track
        self.velocity_shape = (5, 5)

        self._shape = (*self.track.shape, *self.velocity_shape)
        self._start_positions = np.argwhere(self.track == 2)

    @property
    def shape(self):
        return self._shape

    @property
    def start_positions(self):
        return self._start_positions
        
    @property
    def start_velocity(self):
        return np.zeros(2, dtype=np.int8)

    def sample_start(self):
        return State(
            rng.choice(self.start_positions),
            self.start_velocity
        )
    
    def choose_start(self, position):
        return State(
            np.array(position, dtype=np.int32),
            self.start_velocity
        )

class Controller:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    def project_path(self, start_loc, end_loc) -> Generator:
        last_loc = start_loc
        distance = np.sqrt(np.sum((end_loc - start_loc) ** 2))
        steps = distance.astype(np.int32) * 100 # arbitrary boundary to reduce chance of missed cells
        for step in range(1, steps+1):
            coord = start_loc + (step / steps) * (end_loc - start_loc)
            loc = np.round(coord, 5).astype(np.int32)
            # print(f"  Projecting path...coord={coord} loc={loc}")
            if np.any(loc != last_loc):
                yield loc
            if np.all(loc == end_loc):
                break
            last_loc = loc

    def is_off_track(self, loc):
        if not all(0 <= ind < dim for dim, ind in zip(self.state_space.track.shape, loc)):
            return True
        return self.state_space.track[loc[0], loc[1]] == 0
    
    def is_terminal(self, loc):
        return self.state_space.track[loc[0], loc[1]] == 3

class Episode:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = [0] # dummy first value to align indices
    
    @property
    def length(self):
        return len(self.actions)
    
    def add(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def add_state(self, s):
        self.states.append(s)

class CumulativeWeight:
    def __init__(self, state_space, action_space):
        self.value = np.zeros((*state_space.shape, action_space.size), dtype=np.float128)
    
    def update(self, state, action: int, weight):
        # print(f"C: idx={state.idx(action)}")
        self.value[state.idx(action)] += weight

    def at(self, state, action: int):
        return self.value[state.idx(action)]

class QEstimate:
    def __init__(self, state_space, action_space):
        # self.value = np.zeros((*state_space.shape, action_space.size), dtype=np.float128)
        self.value = np.full((*state_space.shape, action_space.size), -10e6, dtype=np.float128) # initialize to very low value because '-1', for example, has to be considered optimal over others
        self.state_space = state_space
        self.action_space = action_space
    
    def update(self, state, action: int, step_size, target):
        idx = state.idx(action)
        old = self.value[idx]
        self.value[idx] = old + step_size * (target - old)
        if not np.isclose(old, self.value[idx]):
            print(f"  Updated Q...idx={idx} old={old} new={self.value[idx]} step_size={step_size} target={target}")

    def at(self, state, action: int):
        return self.value[state.idx(action)]
    
    def log_start(self):
        print("Q-values at start positions:")
        for start_pos in self.state_space.start_positions:
            for velocity in [(0, 0)]:
                state = State(car_loc=start_pos, velocity=velocity)
                idx_legal = legal_actions_idx(state, self.action_space)
                q_values = [self.at(state, a_ind) for a_ind in idx_legal]
                print(f"  start_pos={start_pos} velocity={velocity} legal_action_indices={idx_legal} q_values={q_values}")


class DetPolicy:
    def __init__(self, state_space, action_space, Q: QEstimate = None):
        self.value = np.zeros(state_space.shape, dtype=np.int8)
        for ci in range(self.value.shape[0]):
            for cj in range(self.value.shape[1]):
                for vi in range(self.value.shape[2]):
                    for vj in range(self.value.shape[3]):
                        state = State(car_loc=np.array([ci, cj]), velocity=np.array([vi, vj]))
                        ind_legal = legal_actions_idx(state, action_space)
                        ind_legal_optimal = np.argmax([Q.at(state, a_ind) for a_ind in ind_legal])
                        ind_optimal = ind_legal[ind_legal_optimal]
                        self.value[ci, cj, vi, vj] = ind_optimal
                
    def update(self, state, value):
        old = self.value[state.idx()]
        if old == value:
            return
        print(f"  Updating target policy...old={old} new={value}")
        self.value[state.idx()] = value

    def at(self, state):
        return self.value[state.idx()]
    
def read_input(path):
    return np.loadtxt(path, dtype=int)

def legal_actions_idx(state: State, action_space):
    indices_legal = []
    for i, a in enumerate(action_space):
        # new_velocity = np.clip(state.velocity + a, VELOCITY_MIN, VELOCITY_MAX)
        new_velocity = state.velocity + a
        if np.any(new_velocity < VELOCITY_MIN) or np.any(new_velocity > VELOCITY_MAX) or np.all(new_velocity == 0):
            continue
        indices_legal.append(i)
    return indices_legal

def generate_episode(state_space, action_space, policy: None|DetPolicy, start_position=None):
    """
    :param policy: if None, uses random behavior policy
    """
    # print("Generating episode")
    episode = Episode()
    controller = Controller(state_space, action_space)

    def step(state, action):
        next_state = State()

        if NOISE_ON:
            if rng.uniform(0.0, 1.0) <= NOISE_PROBA:
                action = np.zeros(2, dtype=np.int8)
        
        next_state.velocity = state.velocity + action
        next_state.car_loc[0] = state.car_loc[0] - next_state.velocity[0] # moving upward
        next_state.car_loc[1] = state.car_loc[1] + next_state.velocity[1] # moving to the right
        
        reward = REWARD_TRANSITION
        terminated = False
        for loc in controller.project_path(state.car_loc, next_state.car_loc):
            if controller.is_off_track(loc):
                # print(f"  Went off-track at loc={loc}, velocity={next_state.velocity}. Resetting...")
                next_state = state_space.sample_start()
                break
            if controller.is_terminal(loc):
                # print(f"  Reached terminal state at loc={loc}, velocity={next_state.velocity}")
                terminated = True
                next_state.car_loc = loc
                break

        return next_state, reward, terminated

    def behavior_policy(state):
        """Random behavior policy (satisfies the soft criterion)."""
        ind_legal = legal_actions_idx(state, action_space)
        return rng.choice(ind_legal)

    chosen_policy = behavior_policy if policy is None else policy.at
    if start_position is None:
        state = state_space.sample_start()
    else:
        state = state_space.choose_start(start_position)
    terminated = False
    i = 0
    while not terminated:
        action_idx = chosen_policy(state)
        action = action_space[action_idx]
        # print(f" Step [{i}] | car_loc={state.car_loc} velocity={state.velocity} action={action}")
        next_state, reward, terminated = step(state, action)
        episode.add(state, action_idx, reward)
        state = next_state
        i += 1
    episode.add_state(state) # add terminal state
    # print(f"Episode terminated after [{i}] steps")

    return episode

def off_policy_mc_control(state_space: StateSpace, action_space: np.ndarray):
    print(f"Action space: {action_space}")
    Q = QEstimate(state_space, action_space)
    C = CumulativeWeight(state_space, action_space)
    t_policy = DetPolicy(state_space, action_space, Q)

    ep_cnt = 0
    avg_episode_length = 0
    avg_episode_lengths = []

    
    while ep_cnt < MAX_EPISODES:
        print(f"Episode {ep_cnt+1}/{MAX_EPISODES}")
        Q.log_start()
            
        episode = generate_episode(state_space, action_space, None)
        G = 0
        W = 1
        print("Updating estimates")
        for t in range(episode.length-1, -1, -1):
            # print(f"  t={t}")
            G = DISCOUNT * G + episode.rewards[t+1]
            s_t = episode.states[t]
            a_t_idx = episode.actions[t]
            C.update(s_t, a_t_idx, W)
            # print(f" Updating...t={t} s_t={s_t.idx()} a_t={a_t} a_t_idx={a_t_idx} G={G} W={W} C={C.at(s_t, a_t)}")
            Q.update(s_t, a_t_idx, W/C.at(s_t, a_t_idx), G)
            ind_legal = legal_actions_idx(s_t, action_space)
            ind_legal_optimal = np.argmax([Q.at(s_t, a_ind) for a_ind in ind_legal])
            ind_optimal = ind_legal[ind_legal_optimal]
            # print(f"   state={s_t.idx()} optimal action indices={ind_legal} optimal action index={ind_optimal} optimal action={action_space[ind_optimal]}")
            t_policy.update(s_t, ind_optimal)
            # print(f" Step [{t}] | G={G} W={W} C={C.at(s_t, a_t)} Q={Q.at(s_t, a_t)} t_policy={t_policy.at(s_t)} a_t={a_t}")
            if a_t_idx != t_policy.at(s_t):
                break
            W *= 1 / (1 / len(legal_actions_idx(s_t, action_space))) # random behavior policy
        avg_episode_length = (avg_episode_length * ep_cnt + episode.length) / (ep_cnt + 1)
        avg_episode_lengths.append(avg_episode_length)
        # print(f"avg_episode_length={avg_episode_length}")
        ep_cnt += 1

    return t_policy, avg_episode_lengths

def save_policy(policy, dir_path):
    path = dir_path / "optimal_policy.npy"
    np.save(path, policy.value)
    print(f"Saved optimal policy to {path}")

def load_policy(dir_path, state_space, action_space):
    policy = DetPolicy(state_space, action_space, QEstimate(state_space, action_space))
    path = dir_path / "optimal_policy.npy"
    policy.value = np.load(path)
    print(f"Loaded optimal policy from {path}")
    return policy

def plot_avg_episode_length(avg_episode_lengths, dir_path):
    plt.clf()
    sns.lineplot(x=range(1, len(avg_episode_lengths)+1), y=avg_episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Average Episode Length (steps)")
    plt.title("Average Episode Length Over Time")
    path = dir_path / "avg_episode_length.png"
    plt.savefig(path)
    print(f"Saved average episode length plot to {path}")

def visualize_optimal_path(state_space, action_space, policy, dir_path, start_position=None):
    episode = generate_episode(state_space, action_space, policy, start_position)

    path = np.array([state.car_loc for state in episode.states])
    
    plt.clf()
    cmap = ListedColormap(['grey', 'white', 'coral', 'lightgreen'])
    plt.imshow(state_space.track, cmap=cmap, origin='upper')

    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, state_space.track.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, state_space.track.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)

    print(f"path={path}")
    
    if len(path) > 1:
        # color gradient
        colors = plt.cm.cool(np.linspace(0, 1, len(path)))
        for i in range(len(path) - 1):
            plt.plot(path[i:i+2, 1], path[i:i+2, 0], color=colors[i], linewidth=1)
        # markers on top
        plt.scatter(path[:, 1], path[:, 0], color=colors, s=5, zorder=5)
    elif len(path) == 1:
        plt.plot(path[:, 1], path[:, 0], color='black', marker='o', markersize=2)

    plt.text(1.05, 0.95, f"Path length: {len(path)} steps", transform=ax.transAxes, 
             fontsize=10, verticalalignment='top')
    
    plt.text(1.05, 0.90, f"Noise: {'ON' if NOISE_ON else 'OFF'}", transform=ax.transAxes, 
             fontsize=10, verticalalignment='top')
    if NOISE_ON:
        plt.text(1.05, 0.85, f"Noise probability: {NOISE_PROBA}", transform=ax.transAxes, 
             fontsize=10, verticalalignment='top')

    plt.title("Sample path using optimal policy")
    name = "optimal_path"
    if start_position is not None:
        name += "_start-{}".format("_".join(map(str, start_position)))
    if NOISE_ON:
        name += "_noise-{}".format(NOISE_PROBA)
    else:
        name += "_noise-off"
    name += ".png"

    path = dir_path / name
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved optimal path visualization to {path}")

def visualize_optimal_path_foreach_start_position(state_space, action_space, policy, dir_path):
    for start_pos in state_space.start_positions:
        visualize_optimal_path(state_space, action_space, policy, dir_path, start_position=start_pos)

if __name__ == "__main__":
    dir_path = Path(GRID_PATH.stem)
    dir_path.mkdir(exist_ok=True)
    state_space = StateSpace(track=read_input(GRID_PATH))
    action_space = np.array([(i,j) for j in range(-1, 2) for i in range(-1, 2)])

    if LEARN:
        optimal_policy, avg_episode_length = off_policy_mc_control(state_space, action_space)
        save_policy(optimal_policy, dir_path)
        plot_avg_episode_length(avg_episode_length, dir_path)
    
    optimal_policy = load_policy(dir_path, state_space, action_space)
    visualize_optimal_path_foreach_start_position(state_space, action_space, optimal_policy, dir_path)
