from enum import Enum
import numpy as np
import random

class FourRooms:
    def __init__(self, goal = (10, 10)) -> None:
        
        self.dim = 11
        self.WALLS = [
            (0, 5), (2, 5), (3, 5), (4, 5), (5, 5),
            (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 9), (5, 10),
            (6, 4), (7, 4), (9, 4), (10, 4)
        ]

        self.GOAL = goal
        self.state = (0, 0)
        self.max_steps = 459
        self.steps = 0
        
        self.action_space = self.Action
        self.state_space = [(x, y) for x in range(self.dim) for y in range(self.dim) if (x, y) not in self.WALLS]

    def __get_state(self, states:list):
        weights = [0.9, 0.1] if len(states) == 2 else [0.8, 0.1, 0.1]
        chosen_state = random.choices(states, weights)
        return chosen_state[0]
    
    def __get_possible_states(self, state, action):
        specified_state = tuple(map(sum, zip(state, action.value)))
        
        if not all(coordinate >= 0 and coordinate <= 10 for coordinate in specified_state) or specified_state in self.WALLS:
            specified_state = state

        noisy_actions = [self.Action.UP, self.Action.DOWN] if action == self.Action.RIGHT or action == self.Action.LEFT else [self.Action.LEFT, self.Action.RIGHT]
        noisy_states_unfiltered  = [tuple(map(sum, zip(state, noisy_action.value))) for noisy_action in noisy_actions]
        noisy_states = list(filter(lambda noisy_state: all(coordinate >= 0 and coordinate <= 10 for coordinate in noisy_state) and noisy_state not in self.WALLS,
                                    noisy_states_unfiltered))
        if noisy_states == []:
            noisy_states.append(state)
        
        possible_states = [specified_state] + noisy_states
        
        return possible_states
    
    def step(self, action:list):
        possible_states = self.__get_possible_states(self.state, self.Action(action))
        self.state = self.__get_state(possible_states)
        reward = 1 if self.state == self.GOAL else 0
        self.steps += 1

        terminated = True if (self.steps == self.max_steps or self.state == self.GOAL) else False

        return self.state, reward, terminated, None, None

    def reset(self):
        self.state = (0, 0)
        self.steps = 0

        return self.state, None

    class Action(Enum):
        UP      = ( 0,  1)
        DOWN    = ( 0, -1)
        LEFT    = (-1,  0)
        RIGHT   = (1,  0)

        @classmethod
        def sample(cls):
            s = random.choice(list(cls))
            return tuple(s.value)
        
        @classmethod
        def get(cls, idx):
            return list(cls)[idx].value

        