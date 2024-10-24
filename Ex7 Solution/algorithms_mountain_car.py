import grid_tiling
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
from tqdm import trange

iht_size = 4096
tilings = 8

iht = grid_tiling.IHT(iht_size)
feature_array = np.zeros([tilings, iht_size])
action_space = None

def get_feature_vec(state:tuple, action:int):
    global feature_array
    pos = state[0]
    vel = state[1]
    feature_array = np.multiply(feature_array, 0)
    
    ones_idices = grid_tiling.tiles(iht, tilings, [tilings*pos/(0.5 + 1.2),tilings*vel/(0.07 + 0.07)],[action])
    # print(ones_idices)
    for tiling_idx, ones_idx in enumerate(ones_idices):
        feature_array[tiling_idx, ones_idx] = 1

    
    return np.reshape(feature_array, np.size(feature_array))

def get_approx_q(state:tuple, action:int, w:np.ndarray):
    f = get_feature_vec(state, action)
    return np.sum(np.multiply(f, w))

def get_action(env:gym.Env, state:tuple, w:np.ndarray, eps:float):
    global action_space
    if action_space is None:
        action_space = list(range(env.action_space.start, env.action_space.n))

    q_vals = [get_approx_q(state, a, w) for a in action_space]
    action = np.argmax(q_vals)
    return action

def sarsa_mountain_car(env:gym.Env, gamma:float, num_episodes:int, eps:float, alpha:float):
    rewards = np.full(num_episodes, -1)
    w = np.zeros(iht_size*tilings)
    steps_per_episode = np.zeros(num_episodes)

    for n in range(num_episodes):
        state, _ = env.reset()
        action = get_action(env, state, w, eps)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            next_state, reward, terminated, truncated, _ = env.step(action)
            steps_per_episode[n] += 1
            curr_gradient = get_feature_vec(state, action)
            curr_q = get_approx_q(state, action, w)

            if terminated:
                w += alpha*(reward - curr_q)*curr_gradient
                rewards[n] = 0

            else:
                next_action = get_action(env, next_state, w, eps)
                next_q = get_approx_q(next_state, next_action, w)

                w += alpha*(reward + (gamma*next_q) - curr_q)*curr_gradient
                state = next_state
                action = next_action
        
    return w, steps_per_episode

def n_step_sarsa(env:gym.Env, gamma:float, num_episodes:int, eps:float, alpha:float, n:int):

    steps_per_episode = np.zeros(num_episodes, dtype=int)
    w = np.zeros(iht_size*tilings)

    for n_e in range(num_episodes):
        states  = deque()
        actions = deque()
        rewards = deque()

        state, _ = env.reset()
        action = get_action(env, state, w, eps)
        states.append(state)
        actions.append(action)

        T = np.inf
        tau = -np.inf
        t = 0
        
        while tau != T-1:
            if t < T:
                next_state, reward, terminated, truncated, _ = env.step(action)
                steps_per_episode[n_e] += 1

                states.append(next_state)
                rewards.append(reward)

                if terminated:
                    T = t + 1
                
                else:
                    next_action = get_action(env, next_state, w, eps)
                    actions.append(next_action)
            
            tau = t - n + 1
            if tau >= 0:
                G = np.sum([(gamma**i)*rewards[i] for i in range(len(rewards))])

                if tau + n < T:
                    q = get_approx_q(states[-1], actions[-1], w)
                    G += (gamma**n)*q

                curr_gradient = get_feature_vec(state, action)
                q = get_approx_q(states[0], actions[0], w)
                w += alpha*(G - q)*curr_gradient
                    
                states.popleft()
                actions.popleft()
                rewards.popleft()

            state = next_state
            action = next_action
            t += 1

    return w, steps_per_episode