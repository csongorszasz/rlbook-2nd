## On-policy TD control (Sarsa)

Solutions to _Exercise 6.9_ (Windy Gridworld problem)

### Four possible actions (vertical+horizontal) _[reproducing book results]_

![](4actions/episodes_timesteps.png)
![](4actions/optimal_policy_trajectory.png)

### Eight possible actions (vertical+horizontal+diagonal)

![](8actions/episodes_timesteps.png)
![](8actions/optimal_policy_trajectory.png)

### Nine possible actions (vertical+horizontal+diagonal+noop)

![](9actions/episodes_timesteps.png)
![](9actions/optimal_policy_trajectory.png)

- found better path than with 8 actions - the no-op action helps obtain more accurate estimates of action values

---

Solutions to _Exercise 6.10_ (Stochastic Wind)

![](8actions-stochastic/episodes_timesteps.png)
![](8actions-stochastic/optimal_policy_trajectory.png)