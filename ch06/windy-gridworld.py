# Exercies 6.9 and 6.10

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

rng = np.random.default_rng(0)

class World:
    def __init__(self, deterministic_wind):
        self.deterministic_wind = deterministic_wind
        self.grid_shape = (7, 10)
        self.wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
        # self.actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]) # UP/DOWN/LEFT/RIGHT
        self.actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1],    # + UP-LEFT/UP-RIGHT/DOWN-LEFT/DOWN-RIGHT
                                 [-1, -1], [-1, 1], [1, -1], [1, 1]])
        # self.actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1],    # + NO-OP
        #                          [-1, -1], [-1, 1], [1, -1], [1, 1],
        #                          [0, 0]])
        self.reward_transition = -1.0
        self.goal_states = {(3,7)}

        self._wind_fn = self._get_deterministic_wind if deterministic_wind else self._get_stochastic_wind

    def _get_deterministic_wind(self, column):
        return self.wind[column]

    def _get_stochastic_wind(self, column):
        sample = rng.uniform(0.0, 1.0)
        if sample <= 1/3:
            return self.wind[column]
        if sample <= 2/3:
            return self.wind[column] + 1
        return self.wind[column] - 1

    def legal_actions(self, state):
        legal_inds = []
        for ind, a in enumerate(self.actions):
            new_state = state + a
            if not all(pos >= 0 and pos < bound for pos, bound in zip(new_state, self.grid_shape)):
                continue
            legal_inds.append(ind)
        return legal_inds
    
    def is_final(self, state):
        return tuple(state) in self.goal_states

    def get_wind(self, column):
        return self._wind_fn(column)        

    def step(self, state, action_ind):
        new_state = state + self.actions[action_ind]
        new_state[0] = np.clip(new_state[0] - self.get_wind(state[1]), 0, self.grid_shape[0]-1)
        return new_state, self.reward_transition, self.is_final(new_state)

class Metrics:
    def __init__(self):
        self.episodes = []
        self.timesteps = []

    def record_episode(self, episode_cnt, timestep_cnt_global):
        self.episodes.append(episode_cnt)
        self.timesteps.append(timestep_cnt_global)

    def plot_episodes_for_timesteps(self):
        plt.clf()
        plt.plot(self.timesteps, self.episodes)
        plt.xlabel("Time steps")
        plt.ylabel("Episodes")
        plt.title(
            "Episode length",
            fontsize=16, 
            pad=15, 
        )
        path = "episodes_timesteps.png"
        plt.savefig(path)
        print(f"Saved episodes vs timesteps plot to: {path}")

class Episode:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = [0.0] # dummy
    
    def add_step(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

class Sarsa:
    def __init__(self, world, alpha, eps, discount, n_episodes=None, n_timesteps=None):
        self.world = world
        self.alpha = alpha
        self.eps = eps
        self.discount = discount
        self.n_episodes = n_episodes if n_episodes else np.inf
        self.n_timesteps = n_timesteps if n_timesteps else np.inf

        self.ep_cnt = 0
        self.timestep_cnt_global = 0
        
    def __call__(self, start_state):
        metrics = Metrics()
        Q = np.zeros((*self.world.grid_shape, len(self.world.actions)), dtype=np.float128) # actions are represented by their indices
        self.ep_cnt = 0
        self.timestep_cnt_global = 0
        while self.ep_cnt < self.n_episodes and self.timestep_cnt_global < self.n_timesteps:
            print(f"Episode: {self.ep_cnt}, Timesteps: {self.timestep_cnt_global}")
            _ = self.generate_episode(start_state, Q, early_stop=True)
            metrics.record_episode(self.ep_cnt, self.timestep_cnt_global)
            self.ep_cnt += 1
        print(f"Stopped training after {self.ep_cnt} episodes and {self.timestep_cnt_global} timesteps.")
        return Q, metrics
    
    def select_action_eps_greedy(self, state, Q, eps):
        all_action_values = Q[state[0], state[1], :]
        legal_action_inds = self.world.legal_actions(state)
        if rng.uniform(0, 1) <= eps:
            return rng.choice(legal_action_inds)
        selected_ind = -1
        max_val = -np.inf
        for ind in legal_action_inds:
            val = all_action_values[ind]
            if val > max_val:
                max_val = val
                selected_ind = ind
        return selected_ind
    
    def generate_episode(self, start_state, Q, early_stop=False):
        episode = Episode()
        s = start_state
        a = self.select_action_eps_greedy(s, Q, self.eps)
        terminated = False
        t = 0
        while not terminated:
            new_s, reward, terminated = self.world.step(s, a)
            print(f"t: {t} | s: {s}, a: {self.world.actions[a]}, a_ind: {a}, new_s: {new_s}, reward: {reward}")
            episode.add_step(s, a, reward)
            Q_s_a = Q[s[0], s[1], a]
            if not terminated:
                new_a = self.select_action_eps_greedy(new_s, Q, self.eps)
                Q_s_a_next = Q[new_s[0], new_s[1], new_a]
            else:
                new_a = -1
                Q_s_a_next = 0.0
            Q[s[0], s[1], a] = Q_s_a + self.alpha * (reward + self.discount * Q_s_a_next - Q_s_a)
            print(f" old Q(s,a): {Q_s_a}, new Q(s,a): {Q[s[0], s[1], a]}")
            s = new_s
            a = new_a
            t += 1
            self.timestep_cnt_global += 1
            if early_stop and self.timestep_cnt_global >= self.n_timesteps:
                break
        episode.add_step(s, a, reward)
        return episode

def plot_optimal_policy_trajectory(start_state, sarsa, Q):
    print(f"Creating trajectory following optimal policy from start state: {start_state}")

    # generate episode following greedy policy
    sarsa.eps = 0.0
    episode = sarsa.generate_episode(start_state, Q)
    trajectory = np.array(episode.states)
    trajectory_length = len(trajectory)-1
    print(f"trajectory length: {trajectory_length}")

    # create grid
    grid = np.zeros(sarsa.world.grid_shape)

    # highlight start state
    grid[start_state[0], start_state[1]] = 1.0
    
    # highlight final state
    goal = next(iter(sarsa.world.goal_states))
    grid[goal[0], goal[1]] = -1.0
    
    plt.clf()
    plt.figure()
    
    cmap = ListedColormap(['palegreen', 'white', 'darksalmon'])
    ax = sns.heatmap(grid, cmap=cmap, cbar=False, linewidths=0.5, linecolor="gray")
    
    # thick outline around the grid
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('black')

    # start/end label
    plt.text(start_state[1] + 0.5, start_state[0] + 0.5, "START", ha="center", va="center", color="black", fontsize=8, zorder=10)
    plt.text(goal[1] + 0.5, goal[0] + 0.5, "GOAL", ha="center", va="center", color="black", fontsize=8, zorder=10)
    
    # draw trajectory
    x_coords = [state[1] + 0.5 for state in trajectory]
    y_coords = [state[0] + 0.5 for state in trajectory]
    # plt.plot(x_coords, y_coords, color='black', linewidth=1) # black line
    # gradient line
    colors = plt.cm.cool(np.linspace(0, 1, len(trajectory)))
    for i in range(len(trajectory) - 1):
        plt.plot(trajectory[i:i+2, 1] + 0.5, trajectory[i:i+2, 0] + 0.5, color=colors[i], linewidth=1)
    plt.scatter(trajectory[1:-1, 1] + 0.5, trajectory[1:-1, 0] + 0.5, color='black', s=5, zorder=5)

    # arrow at the end of the line
    if len(x_coords) > 1:
        plt.annotate('', xy=(x_coords[-1], y_coords[-1]), xytext=(x_coords[-2], y_coords[-2]),
                     arrowprops=dict(arrowstyle="->", color='black', lw=1, mutation_scale=20))

    # wind labels    
    plt.xticks([])
    plt.yticks([])
    plt.xticks(np.arange(sarsa.world.grid_shape[1]) + 0.5, labels=[f"wind: {w}" for w in sarsa.world.wind], rotation=45)
    
    plt.title(
        "Optimal trajectory",
        fontsize=16, 
        pad=10, 
        loc="left"
    )

    plt.text(0.65, 1.03, f"Trajectory length: {trajectory_length} steps", transform=ax.transAxes,
             fontsize=10)

    path = "optimal_policy_trajectory.png"
    plt.savefig(path)
    print(f"Saved optimal policy trajectory plot to: {path}")

if __name__ == "__main__":
    world = World(deterministic_wind=False)
    sarsa = Sarsa(world, alpha=0.5, eps=0.1, discount=1.0, n_timesteps=8000)
    start_state = np.array([3, 0])
    Q, metrics = sarsa(start_state)
    metrics.plot_episodes_for_timesteps()
    plot_optimal_policy_trajectory(start_state, sarsa, Q)
    