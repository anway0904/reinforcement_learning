import copy
import numpy as np
import random
from environment import FourRooms, FunctionApproximation
import gymnasium as gym


def sarsa_state_agg(env:FourRooms, w:dict, policy:dict, gamma:float, num_episodes:int, eps:float, alpha:float):
    sarsa_policy = copy.deepcopy(policy)
    rewards = np.zeros(num_episodes)

    for n in range(num_episodes):
        state, _ = env.reset()
        action = env.get_action(state, sarsa_policy)
        terminated = False
        steps = 0
        while not terminated:
            next_state, reward, terminated, _, _ = env.step(action)
            steps += 1
            curr_gradient = env.state_agg(state)
            curr_q = env.get_approx_q(state, action, w[action])

            if terminated:
                w[action] += alpha*(reward - curr_q)*curr_gradient
                rewards[n:] += gamma**steps
                

            else:
                next_action = env.get_action(next_state, sarsa_policy)
                next_q = env.get_approx_q(next_state, next_action, w[next_action])

                w[action] += alpha*(reward + (gamma*next_q) - curr_q)*curr_gradient

                opt_Q = -np.inf
                for a in env.action_space:
                    sarsa_policy[state][a] = eps/len(env.action_space)

                    Q = env.get_approx_q(state, a, w[a])
                    if Q > opt_Q:
                        opt_action = [a]
                        opt_Q = Q

                    elif Q == opt_Q:
                        opt_action.append(a)
                sarsa_policy[state][random.choice(opt_action)] = 1 - eps + (eps/len(env.action_space))
                
                state = next_state
                action = next_action

    return w, rewards

def sarsa_func_approx(env:FourRooms, fa:FunctionApproximation, w:dict, policy:dict, gamma:float, num_episodes:int, eps:float, alpha:float):
    sarsa_policy = copy.deepcopy(policy)
    rewards = np.zeros(num_episodes)

    for n in range(num_episodes):
        state, _ = env.reset()
        action = env.get_action(state, sarsa_policy)
        terminated = False
        steps = 0
        while not terminated:
            next_state, reward, terminated, _, _ = env.step(action)
            steps += 1
            curr_gradient = fa.get_gradient(state, action)
            curr_q = fa.get_approx_q(state, action, w[action])

            if terminated:
                w[action] += alpha*(reward - curr_q)*curr_gradient
                rewards[n:] += gamma ** steps

            else:
                next_action = env.get_action(next_state, sarsa_policy)
                next_q = fa.get_approx_q(next_state, next_action, w[next_action])
                
                w[action] += alpha*(reward + (gamma*next_q) - curr_q)*curr_gradient

                opt_Q = -np.inf
                for a in env.action_space:
                    sarsa_policy[state][a] = eps/len(env.action_space)

                    Q = fa.get_approx_q(state, a, w[a])
                    if Q > opt_Q:
                        opt_action = [a]
                        opt_Q = Q

                    elif Q == opt_Q:
                        opt_action.append(a)

                sarsa_policy[state][random.choice(opt_action)] = 1 - eps + (eps/len(env.action_space))
                
                state = next_state
                action = next_action

    return w, rewards
