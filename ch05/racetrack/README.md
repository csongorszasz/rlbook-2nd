## Learning near-optimal policies via Off-policy Monte Carlo Control

Example solutions to _Exercise 5.12_ (Racetrack problem)

### Example track nr. 1

Optimal trajectories with starting position $(31,6)$

> Noise turned OFF

![](track_example_1_episodes-1000/optimal_path_start-31_6_noise-off.png)

> Noise turned ON: probability 0.1 of intented action being replaced by neutral action 

![](track_example_1_episodes-1000/optimal_path_start-31_6_noise-0.1.png)

- car goes off-track twice - once, close to the finish, then again, immediately after being reset to the start at $(31,8)$

### Example track nr. 2

Optimal trajectories with starting position $(29,2)$

> Noise turned OFF

![](track_example_2-episodes-10000/optimal_path_start-29_2_noise-off.png)

> Noise turned ON: probability 0.1 of intented action being replaced by neutral action 

![](track_example_2-episodes-10000/optimal_path_start-29_2_noise-0.1.png)

- car goes off-track once, near the finish

[trained with NOISE=0.1]