import random
import numpy as np
from src.core.constants import EPSILON, LEARNING_RATE, DISCOUNT_FACTOR, GRID_SIZE, MAX_TASKS
from src.core.entities import CellType, Task
from src.agents.dqn import CentralizedDQN
import time
import torch
from collections import deque
import tensorflow as tf

class MADQLAgent:
    def __init__(self, game, state_size=None, action_size=None):
        self.game = game
        self.total_tasks_completed = 0  # Initialize total tasks completed
        self.state_size = state_size or self._calculate_state_size()
        self.action_size = action_size or MAX_TASKS
        
        # Initialize learning parameters with dynamic tuning
        self.learning_params = {
            'epsilon': {
                'value': 1.0,
                'min': 0.01,
                'decay': 0.995,
                'dynamic_adjust': True
            },
            'learning_rate': {
                'value': 0.001,
                'min': 0.0001,
                'max': 0.01,
                'dynamic_adjust': True
            },
            'gamma': {
                'value': 0.99,
                'min': 0.9,
                'max': 0.999,
                'dynamic_adjust': True
            }
        }
        
        # Q-Network parameters
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.min_replay_size = 1000
        
        # Performance tracking
        self.performance_window = deque(maxlen=100)
        self.update_counter = 0
        self.target_update_frequency = 10
        self.start_time = time.time()
        
        # Initialize metrics
        self.metrics = {
            'episode_rewards': [],
            'avg_q_values': [],
            'loss_history': [],
            'parameter_history': []
        }
    
    def update(self, robot, state, action, reward, next_state, done):
        """Store experience and perform learning updates"""
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))
        
        # Only update if we have enough samples
        if len(self.memory) < self.min_replay_size:
            return
        
        # Sample batch and perform update
        self._replay_experience()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.update_target_network()
        
        # Track performance and adjust parameters
        self._update_performance_metrics(reward)
        if len(self.performance_window) >= 50:  # Wait for enough samples
            self._adjust_parameters()
    
    def _replay_experience(self):
        """Perform experience replay update"""
        # Sample random batch
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Current Q values
        current_q = self.q_network.predict(states)
        
        # Target Q values
        target_q = self.target_network.predict(next_states)
        max_target_q = np.max(target_q, axis=1)
        
        # Update Q values
        for i in range(len(batch)):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                current_q[i][actions[i]] = rewards[i] + self.learning_params['gamma']['value'] * max_target_q[i]
        
        # Train network
        history = self.q_network.fit(states, current_q, verbose=0)
        self.metrics['loss_history'].append(history.history['loss'][0])
        self.metrics['avg_q_values'].append(np.mean(current_q))
    
    def _update_performance_metrics(self, reward):
        """Update performance tracking metrics"""
        self.performance_window.append(reward)
        self.metrics['episode_rewards'].append(reward)
        
        # Record current parameter values
        self.metrics['parameter_history'].append({
            'epsilon': self.learning_params['epsilon']['value'],
            'learning_rate': self.learning_params['learning_rate']['value'],
            'gamma': self.learning_params['gamma']['value']
        })
    
    def _adjust_parameters(self):
        """Dynamically adjust learning parameters based on performance"""
        recent_avg_reward = np.mean(list(self.performance_window))
        reward_std = np.std(list(self.performance_window))
        
        # Adjust epsilon based on performance stability
        if self.learning_params['epsilon']['dynamic_adjust']:
            if reward_std < 5.0:  # Stable performance
                self.learning_params['epsilon']['value'] *= self.learning_params['epsilon']['decay']
            else:  # Unstable performance
                self.learning_params['epsilon']['value'] = min(
                    1.0,
                    self.learning_params['epsilon']['value'] * 1.1
                )
            self.learning_params['epsilon']['value'] = max(
                self.learning_params['epsilon']['min'],
                self.learning_params['epsilon']['value']
            )
        
        # Adjust learning rate based on loss trend
        if self.learning_params['learning_rate']['dynamic_adjust'] and len(self.metrics['loss_history']) > 10:
            recent_losses = self.metrics['loss_history'][-10:]
            loss_trend = np.mean(recent_losses[5:]) - np.mean(recent_losses[:5])
            
            if loss_trend > 0:  # Loss increasing
                self.learning_params['learning_rate']['value'] *= 0.9
            elif loss_trend < -0.1:  # Loss decreasing significantly
                self.learning_params['learning_rate']['value'] *= 1.1
            
            self.learning_params['learning_rate']['value'] = np.clip(
                self.learning_params['learning_rate']['value'],
                self.learning_params['learning_rate']['min'],
                self.learning_params['learning_rate']['max']
            )
        
        # Adjust gamma based on average reward trend
        if self.learning_params['gamma']['dynamic_adjust'] and len(self.metrics['episode_rewards']) > 20:
            reward_trend = np.mean(self.metrics['episode_rewards'][-10:]) - np.mean(self.metrics['episode_rewards'][-20:-10])
            
            if reward_trend > 0:  # Improving performance
                self.learning_params['gamma']['value'] = min(
                    self.learning_params['gamma']['max'],
                    self.learning_params['gamma']['value'] * 1.001
                )
            else:  # Degrading performance
                self.learning_params['gamma']['value'] = max(
                    self.learning_params['gamma']['min'],
                    self.learning_params['gamma']['value'] * 0.999
                )
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_network.set_weights(self.q_network.get_weights())

    def _calculate_state_size(self):
        """Calculate the size of the state space"""
        # 1. Robot's current state (x, y, completed_tasks_ratio, waiting_time)
        robot_state_size = 4

        # 2. Local environment (3x3 grid around robot)
        # Each cell has 4 features: robot, obstacle, task/target, priority
        local_env_size = 9 * 4

        # 3. Task-related features
        # nearest_task: x, y, priority, waiting_time, path_congestion
        task_features_size = 5

        # 4. Team coordination features (for each other robot)
        # Basic info: x, y, has_target, completed_tasks_ratio, waiting_time
        # Target info: x, y, priority
        robot_features = 5 + 3
        max_other_robots = MAX_TASKS - 1  # Maximum number of other robots
        team_features_size = robot_features * max_other_robots

        # 5. Global environment features
        # task_density, avg_priority, global_congestion, epsilon, time_elapsed
        global_features_size = 5

        total_size = (
            robot_state_size +
            local_env_size +
            task_features_size +
            team_features_size +
            global_features_size
        )

        return total_size

    def _build_network(self):
        """Build the Q-network with a simpler architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_params['learning_rate']['value'])
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def predict(self, state):
        """Make a prediction with proper input formatting"""
        state = np.array(state).reshape(1, -1).astype(np.float32)
        return self.q_network(state, training=False).numpy()

    def get_state(self, robot):
        """Get the current state representation for a robot"""
        state = []
        
        print(f"\nCalculating state for Robot {robot.id}")
        
        # 1. Robot's current state
        robot_state = [
            robot.x/GRID_SIZE,  # Normalized position
            robot.y/GRID_SIZE,
            getattr(robot, 'completed_tasks', 0) / max(1, self.total_tasks_completed),  # Efficiency metric
            getattr(robot, 'waiting_time', 0) / 10.0  # Normalized waiting time
        ]
        state.extend(robot_state)
        print(f"Robot state: {robot_state}")
        
        # 2. Local environment (3x3 grid around robot)
        local_env = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                x, y = robot.x + dx, robot.y + dy
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    cell = self.game.grid.get_cell(x, y)
                    cell_state = [
                        1.0 if cell == CellType.ROBOT else 0.0,
                        1.0 if cell == CellType.OBSTACLE else 0.0,
                        1.0 if cell in [CellType.TASK, CellType.TARGET] else 0.0,
                        self._get_cell_priority(x, y) / 3.0
                    ]
                else:
                    cell_state = [0.0, 0.0, 0.0, 0.0]
                local_env.extend(cell_state)
        state.extend(local_env)
        print(f"Local environment features: {len(local_env)}")
        
        # 3. Task-related features
        nearest_task = self._find_nearest_task(robot)
        if nearest_task:
            path = self.game.astar.find_path(
                (robot.x, robot.y),
                nearest_task.get_position(),
                ignore_robots=True,
                ignore_tasks=True
            )
            task_features = [
                nearest_task.x/GRID_SIZE,
                nearest_task.y/GRID_SIZE,
                nearest_task.priority/3.0,
                nearest_task.get_waiting_time()/10.0,
                self._calculate_path_congestion(path)/10.0 if path else 1.0
            ]
        else:
            task_features = [0.0, 0.0, 0.0, 0.0, 0.0]
        state.extend(task_features)
        print(f"Task features: {task_features}")
        
        # 4. Team coordination features
        team_features = []
        for other_robot in self.game.robots:
            if other_robot != robot and len(team_features) < 8 * (MAX_TASKS - 1):
                robot_features = [
                    other_robot.x/GRID_SIZE,
                    other_robot.y/GRID_SIZE,
                    1.0 if other_robot.target else 0.0,
                    getattr(other_robot, 'completed_tasks', 0) / max(1, self.total_tasks_completed),
                    getattr(other_robot, 'waiting_time', 0)/10.0
                ]
                
                # Add target information if exists
                if other_robot.target:
                    target_features = [
                        other_robot.target.x/GRID_SIZE,
                        other_robot.target.y/GRID_SIZE,
                        other_robot.target.priority/3.0
                    ]
                else:
                    target_features = [0.0, 0.0, 0.0]
                
                team_features.extend(robot_features + target_features)
        
        # Pad team features if needed
        padding_needed = 8 * (MAX_TASKS - 1) - len(team_features)
        if padding_needed > 0:
            team_features.extend([0.0] * padding_needed)
        
        state.extend(team_features)
        print(f"Team features: {len(team_features)}")
        
        # 5. Global environment features
        global_features = [
            len(self.game.grid.tasks)/MAX_TASKS,  # Task density
            sum(t.priority for t in self.game.grid.tasks)/(3 * MAX_TASKS),  # Average priority
            self._calculate_global_congestion()/10.0,  # Global congestion
            self.learning_params['epsilon']['value'],  # Current exploration rate
            (time.time() - self.start_time) / 3600.0  # Time elapsed in hours
        ]
        state.extend(global_features)
        print(f"Global features: {global_features}")
        
        state_array = np.array(state, dtype=np.float32)
        print(f"Total state size: {len(state)}")
        
        return state_array

    def _get_cell_priority(self, x, y):
        """Get priority of task at cell if it exists"""
        for task in self.game.grid.tasks:
            if task.x == x and task.y == y:
                return task.priority
        return 0

    def _find_nearest_task(self, robot):
        """Find nearest available task"""
        if not self.game.grid.tasks:
            return None
        return min(self.game.grid.tasks, 
                  key=lambda t: robot.manhattan_distance((robot.x, robot.y), t.get_position()))

    def _calculate_path_congestion(self, path):
        """Calculate congestion level along a path"""
        if not path:
            return 0
        
        congestion = 0
        for x, y in path:
            # Check adjacent cells for robots and obstacles
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                check_x, check_y = x + dx, y + dy
                if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                    cell = self.game.grid.get_cell(check_x, check_y)
                    if cell in [CellType.ROBOT, CellType.OBSTACLE]:
                        congestion += 1
        return congestion

    def _calculate_global_congestion(self):
        """Calculate global congestion level"""
        congestion = 0
        for robot in self.game.robots:
            if robot.target:
                path = self.game.astar.find_path(
                    (robot.x, robot.y),
                    robot.target.get_position(),
                    ignore_robots=True,
                    ignore_tasks=True
                )
                if path:
                    congestion += self._calculate_path_congestion(path)
        return congestion / max(1, len(self.game.robots))

    def choose_action(self, robot):
        """Choose an action for the given robot using epsilon-greedy policy"""
        # Get available tasks
        available_tasks = [task for task in self.game.grid.tasks 
                          if not any(r.target == task for r in self.game.robots)]
        
        if not available_tasks:
            return None
        
        state = self.get_state(robot)
        
        # Epsilon-greedy action selection
        if random.random() < self.learning_params['epsilon']['value']:
            # Exploration: choose a random available task
            return random.choice(available_tasks)
        
        # Exploitation: choose best action based on Q-values
        q_values = self.predict(state)[0]
        
        # Filter Q-values for only available tasks
        available_actions = []
        for task_idx, task in enumerate(available_tasks):
            if task_idx < len(q_values):
                available_actions.append((task, q_values[task_idx]))
        
        if not available_actions:
            return None
        
        # Choose task with highest Q-value
        chosen_task = max(available_actions, key=lambda x: x[1])[0]
        
        return chosen_task

    def calculate_bid(self, robot, available_tasks=None):
        """Calculate bid value for available tasks using Q-values"""
        if available_tasks is None:
            available_tasks = [task for task in self.game.grid.tasks 
                             if not any(r.target == task for r in self.game.robots)]
        
        if not available_tasks:
            print(f"No available tasks for Robot {robot.id}")
            return None, 0.0
        
        # Get current state and Q-values
        state = self.get_state(robot)
        try:
            q_values = self.predict(state)[0]
            print(f"Robot {robot.id} Q-values shape: {q_values.shape}")
        except Exception as e:
            print(f"Error predicting Q-values for Robot {robot.id}: {e}")
            return None, 0.0
        
        # Calculate bids for each available task
        task_bids = []
        for task_idx, task in enumerate(available_tasks):
            try:
                print(f"\nRobot {robot.id} evaluating task at ({task.x}, {task.y})")
                # First check if path exists
                path = self.game.astar.find_path(
                    (robot.x, robot.y),
                    task.get_position(),
                    ignore_robots=True,  # Ignore other robots when planning
                    ignore_tasks=True    # Ignore tasks as obstacles
                )
                
                if path:  # Only bid if path exists
                    print(f"Found path of length {len(path)} to task")
                    # Use task index directly instead of grid position
                    if task_idx < len(q_values):
                        # Get base Q-value
                        q_value = q_values[task_idx]
                        print(f"Q-value for task: {q_value}")
                        
                        # Calculate priority factor (higher priority = higher bid)
                        priority_factor = task.priority / 3.0
                        
                        # Calculate distance factor (shorter distance = higher bid)
                        distance = len(path)
                        distance_factor = 1.0 - (distance / (2 * GRID_SIZE))
                        
                        # Calculate congestion factor (less congestion = higher bid)
                        congestion = self._calculate_path_congestion(path)
                        congestion_factor = max(0.1, 1.0 - (congestion / (4 * len(path))))
                        
                        # Normalize Q-value to [0,1] range, with a minimum value to ensure some exploration
                        q_min, q_max = q_values.min(), q_values.max()
                        if abs(q_max - q_min) < 1e-6:  # If Q-values are very similar
                            normalized_q = 0.5  # Use a neutral value
                        else:
                            normalized_q = max(0.1, (q_value - q_min) / (q_max - q_min))
                        
                        # Calculate final bid value with weighted components
                        bid_value = (
                            0.3 * normalized_q +           # Q-value contribution
                            0.3 * priority_factor +        # Priority contribution
                            0.2 * distance_factor +        # Distance contribution
                            0.2 * congestion_factor        # Congestion contribution
                        )
                        
                        print(f"Bid components for Robot {robot.id} -> Task ({task.x}, {task.y}):")
                        print(f"  Q-value: {normalized_q:.2f}")
                        print(f"  Priority: {priority_factor:.2f}")
                        print(f"  Distance: {distance_factor:.2f}")
                        print(f"  Congestion: {congestion_factor:.2f}")
                        print(f"  Final bid: {bid_value:.2f}")
                        
                        # Only add bid if it's above a minimum threshold
                        if bid_value >= 0.1:
                            task_bids.append((task, bid_value))
                        else:
                            print(f"Bid value {bid_value:.2f} too low, skipping")
                    else:
                        print(f"Task index {task_idx} out of bounds for Q-values length {len(q_values)}")
                else:
                    print(f"No valid path found to task at ({task.x}, {task.y})")
            except Exception as e:
                print(f"Error calculating bid for task at ({task.x}, {task.y}): {e}")
                continue
        
        if not task_bids:
            print(f"No valid bids generated for Robot {robot.id}")
            return None, 0.0
        
        # Choose task with highest bid
        chosen_task, highest_bid = max(task_bids, key=lambda x: x[1])
        print(f"Robot {robot.id} chose task at ({chosen_task.x}, {chosen_task.y}) with bid {highest_bid:.2f}")
        
        return chosen_task, highest_bid