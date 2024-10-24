import numpy as np
import environment as env
import copy

world = env.World()
# jack = env.JacksCarRental()
# jack.modified = True

def policy_eval(grid, policy:dict, threshold:float, gamma:float):
    grid_iter = np.nditer(grid, flags=['multi_index'], op_flags=['readwrite'])
    grid_copy = np.ndarray.copy(grid)
    delta = np.inf

    while delta > threshold:
        delta = 0
        grid_iter.reset()

        while not grid_iter.finished:
            state = grid_iter.multi_index
            v = grid_copy[state]
            new_v = 0

            for a in env.Action:
                new_state, reward = world.take_action(state, a)
                prob_a = 1/len(policy[state]) if a.name in policy[state] else 0
                new_v += prob_a*(reward + gamma*grid_copy[new_state])

            grid_copy[state] = new_v

            delta = max(delta, abs(new_v - v))
            grid_iter.iternext()
        
    return grid_copy

def get_optimal_policy(grid_copy):
    optimal_policy = {}
    grid_iter = np.nditer(grid_copy, flags=['multi_index'], op_flags=['readwrite'])
    while not grid_iter.finished:
        state = grid_iter.multi_index
        
        best_value = -np.inf
        for a in env.Action:
            next_state, _ = world.take_action(state, a)
            value = round(grid_copy[next_state], 1)

            if value > best_value:
                optimal_policy[state] = [a.name]
                best_value = value
            
            elif value == best_value:
                optimal_policy[state].append(a.name)

        grid_iter.iternext()

    return grid_copy, optimal_policy

def value_iter(grid, threshold:float, gamma:float):
    grid_iter = np.nditer(grid, flags=['multi_index'], op_flags=['readwrite'])
    grid_copy = np.ndarray.copy(grid)
    delta = np.inf

    while delta > threshold:
        delta = 0
        grid_iter.reset()

        while not grid_iter.finished:
            state = grid_iter.multi_index
            v = grid_copy[state]
            new_v = -np.inf

            for a in env.Action:
                new_state, reward = world.take_action(state, a)
                new_v = max(new_v,(reward + gamma*grid_copy[new_state]))

            grid_copy[state] = new_v

            delta = max(delta, abs(new_v - v))
            grid_iter.iternext()
    
    grid_iter.reset()
    optimal_value_function, optimal_policy = get_optimal_policy(grid_copy)

    return optimal_value_function, optimal_policy

def policy_iter(grid, policy:dict, threshold:float, gamma:float):
    grid_iter = np.nditer(grid, flags=['multi_index'], op_flags=['readwrite'])
    value_function = np.ndarray.copy(grid)
    policy_stable = False
    policy_copy = copy.deepcopy(policy)

    while not policy_stable:
        value_function = np.round(policy_eval(value_function, policy_copy, threshold, gamma), 1)
        grid_iter.reset()

        while not grid_iter.finished:
            state = grid_iter.multi_index
            old_action = policy_copy.get(state, [])
            
            best_value = -np.inf
            for a in env.Action:
                next_state, reward = world.take_action(state, a)
                value = reward + gamma*value_function[next_state]
                if value > best_value:
                    policy_copy[state] = [a.name]
                    best_value = value
            
                elif value == best_value:
                    policy_copy[state].append(a.name)
            
            if old_action == policy_copy[state]:
                policy_stable = True 

            else:
                policy_stable = False

            grid_iter.iternext()

    return value_function, policy_copy     

def policy_eval_jack(jack, state_space, action_space, policy, threshold:float, gamma:float):
    value_function = np.zeros([jack.max_cars_end+1, jack.max_cars_end+1])
    policy_copy = np.copy(policy)
    delta = np.inf

    while delta > threshold:
        delta = 0
        for state in state_space:
            v = value_function[state]
            
            new_v = 0

            for action in action_space:
                prob_action = 1 if action == policy_copy[state] else 0
                new_v += prob_action*np.sum(jack.transition(state, action)*(jack.get_rewards(state, action) + gamma*value_function))
            
            value_function[state] = new_v
            delta = max(delta, abs(new_v - v))
        
    return value_function

def policy_iter_jack(jack, state_space, action_space, policy, threshold:float, gamma:float):
    policy_stable = False
    policy_copy = np.copy(policy)

    policies = [np.copy(policy_copy)]
    while not policy_stable:
        value_function = policy_eval_jack(jack, state_space, action_space, policy_copy, threshold, gamma)
        policy_stable = True
        for state in state_space:
            old_action = policy_copy[state]

            best_value = -np.inf
            best_action = None
            for action in action_space:
                value = np.sum(jack.transition(state, action)*(jack.get_rewards(state, action) + gamma*value_function))
                
                if value > best_value:
                    policy_copy[state] = action
                    best_value = value
            
            if old_action != policy_copy[state]:
                policy_stable = False 

        policies.append(np.copy(policy_copy))

    return value_function, policies