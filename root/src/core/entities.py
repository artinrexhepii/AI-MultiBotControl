from enum import Enum
from collections import defaultdict
import time
import numpy as np
import torch

class CellType(Enum):
    EMPTY = 0
    ROBOT = 1
    OBSTACLE = 2
    TARGET = 3
    TASK = 4

class Task:
    def __init__(self, x, y, priority=1):
        self.x = x
        self.y = y
        self.priority = priority  # 1 (low) to 3 (high)
        self.creation_time = time.time()
        
    def get_position(self):
        return (self.x, self.y)
        
    def get_waiting_time(self):
        return time.time() - self.creation_time

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.target = None  # Will now store Task object instead of just position
        self.path = []
        self.waiting = False
        self.waiting_time = 0
        self.last_waiting_start = None
        self.q_table = {}  # Changed to use string keys
        self.last_move_time = time.time()
        self.completed_tasks = 0
        self.total_distance = 0
        self.start_time = time.time()
        self.status_message = ""
        self.id = None
        self.last_action = None
        self.last_state = None
        
    def set_target(self, task):
        self.target = task
        self.status_message = f"Assigned to task at ({task.x}, {task.y}) with priority {task.priority}"
        
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
    def update_waiting(self, is_waiting, current_time):
        if is_waiting:
            if not self.waiting:  # Just started waiting
                self.last_waiting_start = current_time
            self.waiting = True
        else:
            if self.waiting:  # Just stopped waiting
                self.waiting_time += current_time - self.last_waiting_start
            self.waiting = False
            self.last_waiting_start = None
            
    def get_state_key(self, state):
        """Convert state array to hashable key"""
        if isinstance(state, np.ndarray):
            return tuple(state.tolist())
        elif isinstance(state, torch.Tensor):
            return tuple(state.cpu().numpy().tolist())
        return tuple(state)
        
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair"""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        return self.q_table[state_key].get(str(action), 0.0)
        
    def update_q_value(self, state, action, value):
        """Update Q-value for a state-action pair"""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][str(action)] = value