import numpy as np
from collections import defaultdict
import random 
import copy

def generate_episode(env, policy, exploring_starts = False, e = False):
    episode = []
    state, _ = env.reset()
    terminated = False

    while not terminated:
        if exploring_starts:
            action =  env.action_space.sample()
            exploring_starts = False
        else:
            action = policy[state] if not e else env.action_space.get(random.choices(range(4), policy[state])[0])


        next_state, reward, terminated, _, _ = env.step(action)
        episode.insert(0, (state, action, reward))
        state = next_state

    return episode

def mc_first_visit(env, state_space, state_space_shape, action_space, policy, gamma, num_episodes):
    returns = {s:[] for s in state_space}
    value_function = np.zeros(state_space_shape)

    for _ in range(num_episodes):
        episode = generate_episode(env, policy)
        visited = []
        g = 0

        for i in range(len(episode)):
            state = episode[i][0]
            next_reward = episode[i][2]

            g = (gamma*g) + next_reward

            if state not in visited:
                returns[state].append(g)
                value_function[state] = np.mean(returns[state])
                visited.append(state)

    return value_function

def mc_exploring_start(env, state_space, state_space_shape, action_space, policy, gamma, num_episodes):
    returns = {(s, a):[] for s in state_space for a in action_space}
    action_value_function = np.zeros(state_space_shape+(len(action_space),))
    value_function = np.zeros(state_space_shape)
    new_policy = np.ndarray.copy(policy)

    for _ in range(num_episodes):
        episode = generate_episode(env, new_policy, exploring_starts=True)
        visited = []
        g = 0

        for i in range(len(episode)):
            state = episode[i][0]
            action = episode[i][1]
            next_reward = episode[i][2]

            g = (gamma*g) + next_reward

            if (state, action) not in visited:
                returns[(state, action)].append(g)
                action_value_function[state + (action,)] = np.mean(returns[(state, action)])

                new_policy[state] = np.argmax([action_value_function[state + (a,)] for a in action_space])
                value_function[state] = action_value_function[state + (new_policy[state],)]
                
                visited.append((state, action))
    

    return value_function, new_policy

def mc_e_soft(env, policy, gamma, num_episodes, num_trials, eps):
    
    discounted_return = np.zeros([num_trials, num_episodes])
    
    for n_t in range(num_trials):
        new_policy = copy.deepcopy(policy)
        action_value_function = {(s, a.value):0 for s in env.state_space for a in env.action_space}
        N = copy.deepcopy(action_value_function)

        for n_e in range(num_episodes):
            episode = generate_episode(env, new_policy, e=True)
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
    
    return new_policy, discounted_return