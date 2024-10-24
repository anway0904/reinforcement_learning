import numpy as np
from enum import Enum
from scipy.stats import poisson

class Policy(Enum):
    NORTH = 0.25
    SOUTH = 0.25
    EAST  = 0.25
    WEST  = 0.25

class Action(Enum):
    NORTH = [-1, 0]
    SOUTH = [1, 0]
    EAST  = [0, 1]  
    WEST  = [0, -1]

class World:
    def __init__(self) -> None:
        self.dim = 5
        # Special states
        self.A  = (0, 1)
        self.B  = (0, 3)

        # Teleport states
        self.A_ = (4, 1)
        self.B_ = (2, 3)

    def take_action(self, state:tuple, action:Action):
        if state == self.A:
            new_state = self.A_
            reward = 10
            return new_state, reward
        
        if state == self.B:
            new_state = self.B_
            reward = 5
            return new_state, reward
        
        new_state = tuple(state + np.array(action.value))
        reward = 0

        if not all(n >= 0 and n < self.dim for n in new_state):
            new_state = state
            reward = -1

        return new_state, reward

class Location(Enum):
    A = 0
    B = 1

class Type(Enum):
    RENT    = 0
    RETURN  = 1

class JacksCarRental:
    def __init__(self, modified=False):
        self.max_cars_end = 20
        self.max_cars_start = 25

        self.state_space = [(x, y) for x in range(self.max_cars_end+1) for y in range(self.max_cars_end+1)]
        self.action_space = list(range(-5, 6))
        self.policy_space = np.zeros([self.max_cars_end+1, self.max_cars_end+1])

        self.mean_rent_A = 3
        self.mean_rent_B = 4
        self.mean_retn_A = 3
        self.mean_retn_B = 2

        # Number of max rent/return requests at each location
        self.poisson_limit = 25

        self.reward_rent = 10
        self.reward_move = -2
        self.overflow_cars = 10
        self.overflow_cost = 4

        self.transitions = np.zeros([self.max_cars_end+1, self.max_cars_end+1, len(self.action_space),
                                     self.max_cars_end+1, self.max_cars_end+1 ])
        
        self.rewards = np.zeros([self.max_cars_end+1, self.max_cars_end+1, len(self.action_space)])
        
        self.modified = modified
        self.precompute_transitions()
        
    def get_poisson_prob(self, n:int, l:int):
        '''
        n is the number of cars and l(lambda) is the expected value
        '''
        return poisson.pmf(n, l)
    
    def open_to_close(self, loc:Location):
        probs = np.zeros([self.max_cars_start+1, self.max_cars_end+1])
        rewards = np.zeros([self.max_cars_start+1])

        for num_cars_start in range(self.max_cars_start+1):
            for num_cars_requested in range(self.poisson_limit+1):
                for num_cars_returned in range(self.poisson_limit+1):
                    prob_event = self.get_poisson_prob(num_cars_requested, self.mean_rent_A if loc == Location.A else self.mean_rent_B)*\
                                 self.get_poisson_prob(num_cars_returned, self.mean_retn_A if loc == Location.A else self.mean_retn_B)
                    
                    actual_cars_rented = min(num_cars_start, num_cars_requested)
                    reward = actual_cars_rented*self.reward_rent
                    num_cars_end = max(min(num_cars_start - actual_cars_rented + num_cars_returned, self.max_cars_end), 0)
                    
                    probs[num_cars_start][num_cars_end] += prob_event
                    rewards[num_cars_start] += prob_event*reward
        
        return probs, rewards

    def valid_action(self, state_A:int, state_B:int, action:int):
        if state_A< action or state_B < -action:
            return False
        else:
            return True

    def precompute_transitions(self):
        prob_A, rewards_A = self.open_to_close(Location.A)
        prob_B, rewards_B = self.open_to_close(Location.B)
 
        for state_A, state_B in self.state_space:
            for action in self.action_space:
                if not self.valid_action(state_A, state_B, action):
                    self.transitions[state_A, state_B, action+5, :, :] = 0
                
                else:
                    new_state_A = state_A - action
                    new_state_B = state_B + action

                    prob_end_state_A = prob_A[new_state_A]
                    prob_end_state_B = prob_B[new_state_B]

                    self.transitions[state_A, state_B, action+5,:,:] = prob_end_state_A[:, np.newaxis]*prob_end_state_B[np.newaxis, :]

                    if not self.modified:
                        reward = rewards_A[new_state_A] + rewards_B[new_state_B] + (abs(action)*self.reward_move)

                    if self.modified:
                        cars_to_move = action-1 if action > 0 else abs(action)
                        reward = rewards_A[new_state_A] + rewards_B[new_state_B] + (cars_to_move*self.reward_move)
                        
                        if new_state_A>self.overflow_cars and new_state_B>self.overflow_cars:
                            reward -= self.overflow_cost*2
                        elif new_state_A>self.overflow_cars or new_state_B>self.overflow_cars:
                            reward -= self.overflow_cost
                            
                    self.rewards[state_A, state_B, action+5] = reward

    def transition(self, state:tuple, action:int):
        return self.transitions[state[0], state[1], action+5]
    
    def get_rewards(self, state:tuple, action:int):        
        return self.rewards[state[0], state[1], action+5]