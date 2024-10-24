import numpy as np
import copy
from collections import defaultdict, deque
import random

def generate_episode(env, policy:dict = None):
    episode = deque()
    state, _ = env.reset()
    terminated = False

    while not terminated:
        action = env.get_action(state, policy)
        next_state, reward, terminated, _, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state

    # Appending the terminal state with no action and the corresponding reward
    episode.append((state, None, reward))

    return episode

def mc_e_soft(env, policy:dict, gamma:float, num_steps:int, eps:float):
    e_soft_policy = copy.deepcopy(policy)
    Q = defaultdict(int)
    N = defaultdict(int)

    steps_vs_episode = np.zeros(num_steps, dtype=int)
    episode = deque()
    state, _ = env.reset()
    terminated = False

    for n in range(num_steps):
        action = env.get_action(state, e_soft_policy)
        next_state, reward, terminated, _, _ = env.step(action)
        episode.appendleft((state, action, reward))
        state = next_state

        if terminated:
            G = 0
            visited = set()
            
            for _, e in enumerate(episode):
                state = e[0]
                action = e[1]
                next_reward = e[2]

                G = (gamma*G) + next_reward

                if (state, action) not in visited:
                    N[(state, action)] += 1
                    Q[(state, action)] += (1/N[(state, action)])*(G - Q[(state, action)])
                    
                    opt_Q = -np.inf
                    for a in env.action_space:
                        e_soft_policy[state][a] = eps/len(env.action_space)

                        if Q[(state, a)] > opt_Q:
                            opt_action = [a]
                            opt_Q = Q[(state, a)]

                        elif Q[(state, a)] == opt_Q:
                            opt_action.append(a)

                    e_soft_policy[state][random.choice(opt_action)] = 1 - eps + (eps/len(env.action_space))
                    visited.add((state, action))
            
            episode = deque()
            state, _ = env.reset()
            terminated = False
            steps_vs_episode[n:] += 1

    return e_soft_policy, steps_vs_episode

def sarsa(env, policy:dict, gamma:float, num_steps:int, eps:float, alpha:float):
    sarsa_policy = copy.deepcopy(policy)
    Q = defaultdict(int)
    steps_vs_episode = np.zeros(num_steps, dtype=int)

    state, _ = env.reset()
    terminated = False

    for n in range(num_steps):
        action = env.get_action(state, sarsa_policy)
        next_state, reward, terminated, _, _ = env.step(action)

        if terminated:
            state, _ = env.reset()
            terminated = False
            steps_vs_episode[n:] += 1

        if not terminated:
            next_action = env.get_action(next_state, sarsa_policy)

            Q[(state, action)] += alpha*(reward + gamma*Q[(next_state, next_action)] - Q[(state, action)])

            opt_Q = -np.inf
            for a in env.action_space:
                sarsa_policy[state][a] = eps/len(env.action_space)

                if Q[(state, a)] > opt_Q:
                    opt_action = [a]
                    opt_Q = Q[(state, a)]

                elif Q[(state, a)] == opt_Q:
                    opt_action.append(a)

            sarsa_policy[state][random.choice(opt_action)] = 1 - eps + (eps/len(env.action_space))
            state = next_state

    return sarsa_policy, steps_vs_episode, Q

def q_learning(env, policy:dict, gamma:float, num_steps:int, eps:float, alpha:float):
    q_policy = copy.deepcopy(policy)
    Q = defaultdict(int)
    steps_vs_episode = np.zeros(num_steps, dtype=int)

    state, _ = env.reset()
    terminated = False

    for n in range(num_steps):
        if terminated:
            state, _ = env.reset()
            terminated = False
            steps_vs_episode[n:] += 1

        if not terminated:
            action = env.get_action(state, q_policy)
            next_state, reward, terminated, _, _ = env.step(action)

            opt_next_Q = -np.inf
            for a in env.action_space:
                if Q[(next_state, a)] > opt_next_Q:
                    opt_action = [a]
                    opt_next_Q = Q[(next_state, a)]

                elif Q[(next_state, a)] == opt_next_Q:
                    opt_action.append(a)

            Q[(state, action)] += alpha*(reward + gamma*Q[(next_state, random.choice(opt_action))] - Q[(state, action)])

            opt_Q = -np.inf
            for a in env.action_space:
                q_policy[state][a] = eps/len(env.action_space)

                if Q[(state, a)] > opt_Q:
                    opt_action = [a]
                    opt_Q = Q[(state, a)]

                elif Q[(state, a)] == opt_Q:
                    opt_action.append(a)

            q_policy[state][random.choice(opt_action)] = 1 - eps + (eps/len(env.action_space))
            state = next_state

    return q_policy, steps_vs_episode, Q

def exp_sarsa(env, policy:dict, gamma:float, num_steps:int, eps:float, alpha:float):
    exp_sarsa = copy.deepcopy(policy)
    Q = defaultdict(int)
    steps_vs_episode = np.zeros(num_steps, dtype=int)

    state, _ = env.reset()
    terminated = False

    for n in range(num_steps):
        if terminated:
            state, _ = env.reset()
            terminated = False
            steps_vs_episode[n:] += 1

        if not terminated:
            action = env.get_action(state, exp_sarsa)
            next_state, reward, terminated, _, _ = env.step(action)
            steps_vs_episode[n] += 1

            target = np.sum(np.multiply(list(exp_sarsa[next_state].values()), [Q[(next_state, a)] for a in env.action_space]))
            Q[(state, action)] += alpha*(reward + (gamma*target) - Q[(state, action)])

            opt_Q = -np.inf
            for a in env.action_space:
                exp_sarsa[state][a] = eps/len(env.action_space)

                if Q[(state, a)] > opt_Q:
                    opt_action = [a]
                    opt_Q = Q[(state, a)]

                elif Q[(state, a)] == opt_Q:
                    opt_action.append(a)

            exp_sarsa[state][random.choice(opt_action)] = 1 - eps + (eps/len(env.action_space))
            state = next_state

    return exp_sarsa, steps_vs_episode, Q

def n_step_sarsa(env, policy:dict, gamma:float, num_steps:int, eps:float, alpha:float, n:int):
    Q = defaultdict(int)
    nsarsa_policy = copy.deepcopy(policy)
    steps_vs_episode = np.zeros(num_steps, dtype=int)
    step_count = 0

    while step_count < num_steps:
        states  = deque()
        actions = deque()
        rewards = deque()

        state, _ = env.reset()
        action = env.get_action(state, nsarsa_policy)
        states.append(state)
        actions.append(action)

        T = np.inf
        tau = -np.inf
        t = 0
        
        while tau != T-1 and step_count < num_steps:
            if t < T:
                next_state, reward, _, _, _ = env.step(action)
                step_count += 1

                states.append(next_state)
                rewards.append(reward)

                if env.is_terminal(next_state):
                    steps_vs_episode[step_count:] += 1
                    T = t + 1
                
                else:
                    next_action = env.get_action(next_state, nsarsa_policy)
                    actions.append(next_action)
            
            tau = t - n + 1
            if tau >= 0:
                G = np.sum([(gamma**i)*rewards[i] for i in range(len(rewards))])

                if tau + n < T:
                    G += (gamma**n)*Q[(states[-1], actions[-1])]

                Q[(states[0], actions[0])] += alpha*(G - Q[(states[0], actions[0])])

                opt_Q = -np.inf
                for a in env.action_space:
                    nsarsa_policy[states[0]][a] = eps/len(env.action_space)

                    if Q[(states[0], a)] > opt_Q:
                        opt_action = [a]
                        opt_Q = Q[(states[0], a)]

                    elif Q[(states[0], a)] == opt_Q:
                        opt_action.append(a)

                nsarsa_policy[states[0]][random.choice(opt_action)] = 1 - eps + (eps/len(env.action_space))
                    
                states.popleft()
                actions.popleft()
                rewards.popleft()

            state = next_state
            action = next_action
            t += 1

    return nsarsa_policy, steps_vs_episode, Q

def n_step_td(env, gamma:float, alpha:float, n:int, episodes:deque = None, num_episodes:int = None, policy:dict = None, V:dict=None):
    if V is None:
        V = {s:0 for s in env.state_space}
    
    if episodes is None:
        episodes = [None]*num_episodes
    
    for i_e, episode in enumerate(episodes):
        states  = deque()
        rewards = deque()

        if episode is None:
            state, _ = env.reset()  
        else:
            state = episode[0][0]
        
        states.append(state)

        T = np.inf
        tau = -np.inf
        t = 0
        
        while tau != T-1:
            if t < T:
                if episode is None:
                    action = env.get_action(state, policy)
                    next_state, reward, _, _, _ = env.step(action)

                else:
                    next_state = episode[t+1][0]
                    reward = episode[t][2]

                states.append(next_state)
                rewards.append(reward)

                if env.is_terminal(next_state):
                    T = t + 1
            
            tau = t - n + 1
            if tau >= 0:
                G = np.sum([(gamma**i) * rewards[i] for i in range(len(rewards))])

                if tau + n < T:
                    G += (gamma**n) * V[states[-1]]

                V[states[0]] += alpha*(G - V[states[0]])
            
                states.popleft()
                rewards.popleft()

            state = next_state
            t += 1

    return V

def mc_prediction(env, gamma:float, episodes:deque):
    V = {s:0 for s in env.state_space}
    N = defaultdict(int)

    for episode in episodes:
        episode.reverse()
        visited = set()
        g = 0

        for i in range(1, len(episode)):
            state = episode[i][0]
            reward = episode[i][2]

            g = (gamma*g) + reward

            if state not in visited:
                N[state] += 1
                V[state] += (1/N[state])*(g - V[state])
                visited.add(state)

        episode.reverse()

    return V