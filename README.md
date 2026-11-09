# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement

To find the optimal policy for an agent navigating through a CliffWalking environment using the Value Iteration algorithm. The agent must reach the goal while avoiding the cliff by maximizing the expected cumulative reward.


## Software Requirements
- **Programming Language:** Python
- **Platform:** Google Colab / Jupyter Notebook
- **Libraries:** Gymnasium, NumPy
- **Environment:** CliffWalking-v0
## Environment Description


## MDP Representation

The Cliff Walking environment is represented using the Markov Decision Process (MDP).

MDP = (S, A, P, R, γ)

- **S (States):** 48 grid positions (4 × 12).
- **A (Actions):** Up, Right, Down and Left.
- **P (Transition Probability):** Probability of moving from one state to another.
- **R (Reward):** -1 for each movement and -100 for falling into the cliff.
- **γ (Discount Factor):** 0.99.
## Theory

Value Iteration is a model-based Reinforcement Learning algorithm used to find the optimal policy of an environment.

It repeatedly updates the value of each state using the Bellman Optimality Equation until the values become stable.

In CliffWalking, the algorithm calculates the expected reward for every possible action. It selects the action with the highest value to help the agent reach the goal while avoiding the cliff.

### Bellman Optimality Equation

$$
V(s)=\max_a\sum_{s'}P(s'|s,a)[R+\gamma V(s')]
$$

Where:

- **V(s)** – Value of the current state.
- **P(s'|s,a)** – Transition probability.
- **R** – Immediate reward.
- **γ** – Discount factor.
- **V(s')** – Value of the next state.



## Algorithm

1. Start the program.
2. Import Gymnasium and NumPy libraries.
3. Create the CliffWalking environment.
4. Initialize the state-value function with zeros.
5. Set the discount factor and convergence threshold.
6. For every state, calculate the expected value of all possible actions.
7. Update the state value using the maximum action value.
8. Repeat the process until the maximum change is less than the threshold.
9. Calculate the optimal policy using the Q-values.
10. Display the number of iterations, optimal state-value function and optimal policy.
11. Stop the program.





## Python Program

```



import gymnasium as gym
import numpy as np

# Create CliffWalking Environment
env = gym.make("CliffWalking-v1")

# Value Iteration Algorithm
def value_iteration(env, gamma=0.99, theta=1e-8):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    iteration = 0

    while True:
        delta = 0
        V_new = np.copy(V)

        for s in range(n_states):
            for a in range(n_actions):

                Q[s, a] = sum(
                    prob * (reward + gamma * V[next_state] * (not done))
                    for prob, next_state, reward, done
                    in env.unwrapped.P[s][a]
                )

            V_new[s] = np.max(Q[s])
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        iteration += 1

        if delta < theta:
            break

    # Calculate final Q-values
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = sum(
                prob * (reward + gamma * V[next_state] * (not done))
                for prob, next_state, reward, done
                in env.unwrapped.P[s][a]
            )

    # Derive optimal policy
    policy = np.argmax(Q, axis=1)

    return V, policy, iteration


# Execute Value Iteration
V, policy, iterations = value_iteration(env)

print("Name: S R NIVEDHITHA")
print("Register Number: 212225240102")

print("Value Iteration Completed")
print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 12), 2))

# Action Symbols
action_symbols = {
    0: "U",
    1: "R",
    2: "D",
    3: "L"
}

policy_grid = np.array(
    [action_symbols[a] for a in policy]
).reshape(4, 12)

print("\nOptimal Policy:")
print(policy_grid)

env.close()




```

---

## Output
<img width="332" height="201" alt="Screenshot 2026-11-09 202922" src="https://github.com/user-attachments/assets/86190264-7ad7-4de9-8c33-716cd1b80493" />


## Result
Thus, the Value Iteration algorithm was successfully implemented using the Gymnasium CliffWalking environment. The optimal state-value function and optimal policy were computed using the Bellman Optimality Equation.

## Inference
Value Iteration repeatedly updates the state values until convergence. It helps the agent identify the best action in each state to reach the goal while avoiding the cliff. Thus, the agent learns an optimal policy that maximizes the expected cumulative reward.



