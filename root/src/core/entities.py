import random
from enum import Enum
import time
import numpy as np

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
    def __init__(self, x, y, id):
        # Use a tuple for position instead of separate x and y
        self.position = (x, y)
        self.x=x
        self.y=y
        self.target = None  # Will now store Task object instead of just position
        self.path = []
        self.waiting = False
        self.waiting_time = 0
        self.last_waiting_start = None
        self.last_move_time = time.time()
        self.completed_tasks = 0
        self.total_distance = 0
        self.start_time = time.time()
        self.status_message = ""
        self.id = id
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

    def act(self, tasks, epsilon=0.1):
        """
        The robot decides how much to bid for each task using the DQN model.
        """
        bids = {}
        for task in tasks:
            state = np.array([self.position[0], self.position[1], task.x, task.y, task.priority], dtype=np.float32)
            if random.random() < epsilon:  # Exploration: random bid
                bid = random.uniform(0, 1)
            else:  # Exploitation: use the DQN to decide on bid
                bid = self.dqn.predict(state)

            # Normalize bid based on distance and priority
            distance = self.manhattan_distance(self.position, (task.x, task.y))
            bid += (task.priority * 0.1)  # Priority increases the bid
            bid -= (distance * 0.05)  # Distance decreases the bid

            bids[task] = bid
        return bids

    def get_state(self):
        """
        Return the robot's state as a vector.
        """
        if self.target:
            target_x, target_y = self.target.get_position()
            target_priority = self.target.priority
        else:
            target_x, target_y = -1, -1  # No target assigned
            target_priority = 0

        state = np.array([
            self.position[0], self.position[1],  # Robot's position
            target_x, target_y,   # Target position
            target_priority       # Target priority
        ], dtype=np.float32)
        return state

    def learn(self, state, action, reward, next_state, done):
        self.dqn.train(state, action, reward, next_state, done)

class DQN:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def predict(self, state):
        state = np.reshape(state, [1, self.state_size])
        return self.model.predict(state, verbose=0)[0][0]

    def train(self, state, action, reward, next_state, done, gamma=0.95):
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])
        target = reward
        if not done:
            target = reward + gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
        target_f = self.model.predict(state, verbose=0)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

# Market-based auction method
def auction(tasks, robots, epsilon=0.1):
    task_allocations = {}
    bids = {}

    # Each robot bids for each task
    for robot in robots:
        task_bids = robot.act(tasks, epsilon)  # Robot bids for each task
        for task, bid in task_bids.items():
            bids[(robot.id, task)] = bid

    # Allocate tasks to the robot with the highest bid for each task
    for task in tasks:
        # Find the robot with the highest bid for this task
        highest_bid_robot = max(robots, key=lambda robot: bids[(robot.id, task)])
        task_allocations[task] = highest_bid_robot
        highest_bid_robot.set_target(task)  # Assign the task to the robot

        # Simulate completing the task and receiving a reward
        distance_to_task = highest_bid_robot.manhattan_distance((highest_bid_robot.x, highest_bid_robot.y), (task.x, task.y))
        reward = 1 if distance_to_task == 0 else -1  # Reward for successful task completion or penalty for failure
        state = np.array([highest_bid_robot.x, highest_bid_robot.y, task.x, task.y, task.priority], dtype=np.float32)
        next_state = state  # For simplicity, assume state does not change much
        done = True  # Assume task is completed immediately
        highest_bid_robot.learn(state, 0, reward, next_state, done)  # Use DQN to learn

    return task_allocations
