import numpy as np
from enum import Enum
import random

class WindyWorld():
    def __init__(self, stoc_wind, kings_moves, ninth_move = False) -> None:
        self.rows = 7
        self.cols = 10
        self.wind = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:1, 9:0}

        self.start = (3, 0)
        self.goal  = (3, 7)
        self.state = self.start

        self.state_space  = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        self.udlr_actions = [a for a in self.Actions if a.name not in ["UR", "UL", "DR", "DL"]]
        self.action_space = list(self.Actions) if kings_moves else self.udlr_actions
        if not ninth_move:
            self.action_space.remove(self.Actions.XX) 

        self.stoc_wind = stoc_wind
        self.kings_moves = kings_moves

        self.max_steps = 2000
        self.steps = 0

    class Actions(Enum):
        UP = (-1,  0)
        DN = ( 1,  0)
        LT = ( 0, -1)
        RT = ( 0,  1)
        UR = (-1,  1)
        UL = (-1, -1)
        DR = ( 1,  1)
        DL = ( 1, -1)
        XX = ( 0,  0)

    def is_terminal(self, state:tuple) -> bool:
        result = True if state == self.goal else False
        return result

    def step(self, action:Actions) -> tuple[tuple, int, bool, None, None]:
        reward = -1 
        terminated = False
        action_tup = action.value
        wind_idx = self.state[1]
        stoc_val = random.choice([-1,0,1]) if self.stoc_wind else 0
        wind_val = self.wind[wind_idx] + stoc_val

        new_state = (self.state[0] + action_tup[0] - wind_val,
                     self.state[1] + action_tup[1])

        if new_state[0] < 0:
            new_state = (0, new_state[1])
        
        elif new_state[0] > self.rows-1:
            new_state = (self.rows-1, new_state[1])

        if new_state[1] < 0:
            new_state = (new_state[0], 0)
        
        elif new_state[1] > self.cols-1:
            new_state = (new_state[0], self.cols-1)

        if new_state == self.goal:
            terminated = True
            reward = 0

        self.steps += 1
        if self.steps > self.max_steps:
            terminated = True

        self.state = new_state
        return new_state, reward, terminated, None, None
    
    def reset(self) -> tuple[tuple, None]:
        self.state = self.start
        self.steps = 0
        return self.state, None

    def get_action(self, state:tuple, policy:dict[dict]) -> Actions:
        return random.choices(self.action_space, weights=list(policy[state].values()))[0]

class RandomWalk:
    def __init__(self) -> None:
        self.state_space = ["T0", "A", "B", "C", "D", "E", "T1"]
        self.action_space = [self.ActionsRW.LEFT, self.ActionsRW.RIGHT]
        self.ter_state_1 = "T1"
        self.ter_state_0 = "T0"
        self.state = "C"

    class ActionsRW(Enum):
        LEFT  = 0
        RIGHT = 1

    def step(self, action:ActionsRW) -> tuple[str, int, bool, None, None]:
        new_idx = 1 if action==self.ActionsRW.RIGHT else -1
        new_state = self.state_space[self.state_space.index(self.state)+new_idx]

        reward = 1 if new_state == self.ter_state_1 else 0 
        terminated = True if new_state == self.ter_state_0 or new_state == self.ter_state_1 else False

        self.state = new_state
        return (new_state, reward, terminated, None, None)

    def reset(self) -> tuple[str, None]:
        self.state = "C"
        return self.state, None

    def get_action(self, state = None, policy = None) -> ActionsRW:
        action = random.choice([self.ActionsRW.LEFT, self.ActionsRW.RIGHT])
        return action
    
    def is_terminal(self, state:str) -> bool:
        result = True if state == self.ter_state_0 or state == self.ter_state_1 else False
        return result
    
class LargeRandomWalk:
    def __init__(self) -> None:
        self.state_space = ["T0", "A", "B", "C", "D", "E", "F", "G", "H", "I", 
                            "J" , "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T1"]
        self.action_space = [self.ActionsRW.LEFT, self.ActionsRW.RIGHT]
        self.ter_state_1 = "T1"
        self.ter_state_0 = "T0"
        self.state = "C"

    class ActionsRW(Enum):
        LEFT  = 0
        RIGHT = 1

    def step(self, action:ActionsRW) -> tuple[str, int, bool, None, None]:
        new_idx = 1 if action==self.ActionsRW.RIGHT else -1
        new_state = self.state_space[self.state_space.index(self.state)+new_idx]

        reward = 0
        terminated = False

        if new_state == self.ter_state_0:
            reward = -1
            terminated = True

        elif new_state == self.ter_state_1:
            reward = 1
            terminated = True
        
        self.state = new_state
        return (new_state, reward, terminated, None, None)

    def reset(self) -> tuple[str, None]:
        self.state = "J"
        return self.state, None

    def get_action(self, state = None, policy = None) -> ActionsRW:
        action = random.choice([self.ActionsRW.LEFT, self.ActionsRW.RIGHT])
        return action
    
    def is_terminal(self, state:str) -> bool:
        result = True if state == self.ter_state_0 or state == self.ter_state_1 else False
        return result