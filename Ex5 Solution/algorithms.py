import numpy as np
from tqdm import trange
from collections import defaultdict
import copy
import random

def td_0(env, V:dict, num_episodes:int, alpha:float, gamma:float, calc_rms:bool = False, policy:defaultdict = None, ideal_V:np.ndarray = None) -> tuple[defaultdict, np.ndarray]:
    val_func = copy.deepcopy(V)
    rms = np.zeros(num_episodes)

    for n in range(num_episodes):
        state, _ = env.reset()
        terminated = False

        while not terminated:
            action = env.get_action(state)
            next_state, reward, terminated, _, _ = env.step(action)
            val_func[state] += alpha*((reward + gamma*val_func[next_state]) - val_func[state])
            state = next_state

        if calc_rms:
            predictions = list(val_func.values())[1:-1]
            rms[n] =  np.sqrt((np.square(predictions - ideal_V)).mean())

    return val_func, rms

def generate_episode(env, policy:dict) -> list[tuple[tuple, tuple, int]]:
    episode = []
    state, _ = env.reset()
    terminated = False

    while not terminated:
        action = env.action_space.get(random.choices(range(4), policy[state])[0])
        next_state, reward, terminated, _, _ = env.step(action)
        episode.insert(0, (state, action, reward))
        state = next_state

    return episode

def mc_off_policy(env, episodes:list, b_policy:dict, t_policy:dict, num_episodes:int, gamma:float) -> dict:
    Q = {(s, a.value):0 for s in env.state_space for a in env.action_space}
    C = defaultdict(int)

    for i in range(num_episodes):  
        episode = episodes[i]
        G = 0
        W = 1
        for idx_e, e in enumerate(episode):
            if W == 0:
                break
            state = e[0]
            action = e[1]
            next_reward = e[2]

            G = gamma*G + next_reward
            C[(state, action)] += W
            Q[(state,action)] += (W/C[(state, action)])*(G - Q[(state,action)])
            
            W *= ((t_policy[state][env.action_space.get_idx(action)])/(b_policy[state][env.action_space.get_idx(action)]))

    return Q

def mc_on_policy_pred(env, policy:dict, num_episodes:int, gamma:float) -> dict:
    Q = {(s, a.value):0 for s in env.state_space for a in env.action_space}
    N = defaultdict(int)

    for i in range(num_episodes):  
        episode = generate_episode(env, policy)
        visited = []
        G = 0
        
        for idx_e, e in enumerate(episode):
            state = e[0]
            action = e[1]
            next_reward = e[2]

            G = gamma*G + next_reward

            if (state, action) not in visited:
                N[(state, action)] += 1
                Q[(state,action)] += (1/N[(state, action)])*(G - Q[(state,action)])
                visited.append((state,action))
    return Q

def mc_e_soft(env, policy:dict, gamma:float, num_episodes:int, num_trials:int, eps:float) -> tuple[dict, np.ndarray, dict, list]:
    discounted_return = np.zeros([num_trials, num_episodes])
    
    for n_t in range(num_trials):
        new_policy = copy.deepcopy(policy)
        action_value_function = {(s, a.value):0 for s in env.state_space for a in env.action_space}
        N = copy.deepcopy(action_value_function)
        episodes = []

        for n_e in range(num_episodes):
            episode = generate_episode(env, new_policy)
            episodes.append(episode)
            visited = []
            g = 0

            ret = (gamma**(len(episode)))*episode[0][2]
            discounted_return[n_t, n_e] = ret

            for i, e in enumerate(episode):
                state = episode[i][0]
                action = episode[i][1]
                next_reward = episode[i][2]

                g = (gamma*g) + next_reward

                if (state, action) not in visited:
                    N[(state, action)] += 1
                    
                    action_value_function[(state, action)] += (1/N[(state, action)])*(g - action_value_function[(state, action)])
                    
                    q_val = [action_value_function[(state, a.value)] for a in env.action_space]
                    optimal_action_idx = np.argmax([action_value_function[(state, a.value)] for a in env.action_space])

                    if not all(q == 0 for q in q_val):
                        new_policy[state] = [eps for i in range(len(env.action_space))]
                        new_policy[state][optimal_action_idx] = 1 - eps
                    
                    visited.append((state, action))
    
    return new_policy, discounted_return, action_value_function, episodes