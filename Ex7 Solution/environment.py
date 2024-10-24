from enum import Enum
import numpy as np
import random
from collections import deque

class FourRooms:
    def __init__(self, goal = (10, 10)) -> None:
        
        self.dim = 11

        self.WALLS = [
            (0, 5), (2, 5), (3, 5), (4, 5), (5, 5),
            (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 9), (5, 10),
            (6, 4), (7, 4), (9, 4), (10, 4)
        ]

        self.GOAL = goal
        self.start_state = (0, 0)
        self.state = self.start_state
        self.max_steps = 459
        self.steps = 0
        
        self.action_space = list(self.Actions)
        self.state_space = [(x, y) for x in range(self.dim) for y in range(self.dim) if (x, y) not in self.WALLS]

        self.state_agg_scale = None

    class Actions(Enum):
        UP      = ( 0,  1)
        DOWN    = ( 0, -1)
        LEFT    = (-1,  0)
        RIGHT   = (1,  0)

    def __get_state(self, states:list) -> tuple:
        weights = [0.9, 0.1] if len(states) == 2 else [0.8, 0.1, 0.1]
        chosen_state = random.choices(states, weights)
        return chosen_state[0]
    
    def __get_possible_states(self, state:tuple, action:Actions) -> list[tuple]:
        specified_state = tuple(map(sum, zip(state, action.value)))
        
        if not all(coordinate >= 0 and coordinate <= 10 for coordinate in specified_state) or specified_state in self.WALLS:
            specified_state = state

        noisy_actions = [self.Actions.UP, self.Actions.DOWN] if action == self.Actions.RIGHT or action == self.Actions.LEFT else [self.Actions.LEFT, self.Actions.RIGHT]
        noisy_states_unfiltered  = [tuple(map(sum, zip(state, noisy_action.value))) for noisy_action in noisy_actions]
        noisy_states = list(filter(lambda noisy_state: all(coordinate >= 0 and coordinate <= 10 for coordinate in noisy_state) and noisy_state not in self.WALLS,
                                    noisy_states_unfiltered))
        if noisy_states == []:
            noisy_states.append(state)
        
        possible_states = [specified_state] + noisy_states
        
        return possible_states
    
    def step(self, action:Actions) -> tuple[tuple, int, bool, None, None]:
        possible_states = self.__get_possible_states(self.state, action)
        self.state = self.__get_state(possible_states)
        reward = 1 if self.state == self.GOAL else 0
        self.steps += 1

        terminated = True if (self.steps == self.max_steps or self.state == self.GOAL) else False

        return self.state, reward, terminated, None, None

    def reset(self) -> tuple[tuple, None]:
        self.state = self.start_state
        self.steps = 0

        return self.state, None

    def get_action(self, state:tuple, policy:dict[dict]) -> Actions:
        return random.choices(self.action_space, weights=list(policy[state].values()))[0]

    def init_state_agg(self, scale:int):
        self.state_agg_scale = scale
        feature_vec_dim = len(self.state_agg((0,0)))
        weight_vec = {a:np.zeros([feature_vec_dim]) for a in self.action_space}
        return weight_vec

    def state_agg(self, state:tuple) -> np.ndarray:
        scale = self.state_agg_scale
        grid_x = np.arange(0, self.dim, scale)
        grid_y = grid_x

        feature_vec = np.zeros([len(grid_x), len(grid_y)])

        state_x = state[0]
        state_y = state[1]

        feature_x = max([x for x in grid_x if x <= state_x])
        feature_y = max([y for y in grid_y if y <= state_y])

        feature_vec[(int(feature_y/scale), int(feature_x/scale))] = 1

        return np.reshape(feature_vec, np.size(feature_vec))
    
    def get_approx_q(self, state:tuple, action:Actions, w:np.ndarray):
        f = self.state_agg(state)
        return np.sum(np.multiply(f, w))
    
class FunctionApproximation():
    def __init__(self, env:FourRooms) -> None:
        self.feature_vec = deque()
        self.env = env

    def init_func_approx(self):
        feature_vec_dim = len(self.calculate_features((0,0)))
        weight_vec = {a:np.zeros([feature_vec_dim]) for a in self.env.action_space}
        return weight_vec
    
    def wall_up(self, s:tuple) -> int:
        if (s[0], s[1] + 1) not in self.env.state_space:
            self.feature_vec.append(1)
        else:
            self.feature_vec.append(0)
        
    def wall_down(self, s:tuple) -> int:
        if (s[0], s[1] - 1) not in self.env.state_space:
            self.feature_vec.append(1)
        else:
            self.feature_vec.append(0)
            
    def wall_left(self, s:tuple) -> int:
        if (s[0] - 1 , s[1]) not in self.env.state_space:
            self.feature_vec.append(1)
    
        else:
            self.feature_vec.append(0)
        
    def wall_right(self, s:tuple) -> int:
        if (s[0] + 1, s[1]) not in self.env.state_space:
            self.feature_vec.append(1)
        else:
            self.feature_vec.append(0)
    
    def x_coord(self, state:tuple):
        self.feature_vec.append(state[0])

    def y_coord(self, state:tuple):
        self.feature_vec.append(state[1])

    def constant(self, a:int):
        self.feature_vec.append(a)

    def dist_to_goal(self, s:tuple):
        goal = self.env.GOAL
        dist = (goal[0] - s[0]) + (goal[1] - s[1])
        self.feature_vec.append(dist)

    def calculate_features(self, state:tuple, action:FourRooms.Actions = None):
        self.feature_vec = deque()
        self.x_coord(state)
        self.y_coord(state)
        self.constant(1)
        self.wall_up(state)
        self.wall_down(state)
        self.wall_left(state)
        self.wall_right(state)
        self.dist_to_goal(state)
        return self.feature_vec
    
    def get_gradient(self, state:tuple, action:FourRooms.Actions = None):
        return np.array(self.calculate_features(state, action))

    def get_approx_q(self, state:tuple, action:FourRooms.Actions, w:np.ndarray):
        f = self.calculate_features(state, action)
        return np.sum(np.multiply(f, w))