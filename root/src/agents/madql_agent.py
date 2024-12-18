import random
import numpy as np
from src.core.constants import EPSILON, LEARNING_RATE, DISCOUNT_FACTOR, GRID_SIZE, MAX_TASKS
from src.core.entities import CellType, Task
from src.agents.dqn import CentralizedDQN
import time
import torch

class MADQLAgent:
    def __init__(self, game):
        self.game = game
        self.episode_count = 0
        self.total_tasks_completed = 0  # Initialize total_tasks_completed
        self.learning_stats = {
            'episode_rewards': [],
            'q_value_history': [],
            'bid_history': [],
            'task_distribution': {},  # Track task distribution among robots
            'completion_times': [],   # Track task completion times
            'congestion_levels': []   # Track congestion levels
        }
        
        # Initialize learned weights for different factors
        self.factor_weights = {
            'q_value': 1.0,
            'priority': 1.0,
            'distance': 1.0,
            'congestion': 1.0,
            'waiting_time': 1.0
        }
        
        # Experience prioritization parameters
        self.priority_alpha = 0.6    # Priority exponent
        self.priority_beta = 0.4     # Importance sampling weight
        self.max_priority = 1.0      # Maximum priority
        
        # Calculate state size with local and global components
        self.local_state_size = (
            2 +  # Robot position (x, y)
            4 * 9 +  # 3x3 grid around robot (4 features per cell)
            5  # Local task features
        )
        self.global_state_size = (
            3 +  # Global metrics (task count, avg priority, congestion)
            5 * (MAX_TASKS - 1)  # Other robots' compressed info
        )
        self.state_size = self.local_state_size + self.global_state_size
        self.action_size = GRID_SIZE * GRID_SIZE
        
        # Initialize networks
        self.dqn = CentralizedDQN(
            state_size=self.state_size,
            action_size=self.action_size,
            num_agents=len(game.robots)
        )
        
        self.prev_states = [None] * len(game.robots)
        self.prev_actions = [None] * len(game.robots)
        
    def get_state(self, robot):
        """Enhanced state representation with local and global features"""
        robot_idx = self.game.robots.index(robot)
        state = []
        
        # Local state components
        # 1. Robot's position
        state.extend([robot.x/GRID_SIZE, robot.y/GRID_SIZE])
        
        # 2. 3x3 grid around robot
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                x, y = robot.x + dx, robot.y + dy
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    cell = self.game.grid[y][x]
                    # Features: is_robot, is_obstacle, is_task, task_priority
                    state.extend([
                        1.0 if cell == CellType.ROBOT else 0.0,
                        1.0 if cell == CellType.OBSTACLE else 0.0,
                        1.0 if cell in [CellType.TASK, CellType.TARGET] else 0.0,
                        self._get_cell_priority(x, y) / 3.0
                    ])
                else:
                    state.extend([0.0, 0.0, 0.0, 0.0])  # Out of bounds
        
        # 3. Local task features
        nearest_task = self._find_nearest_task(robot)
        if nearest_task:
            state.extend([
                nearest_task.x/GRID_SIZE,
                nearest_task.y/GRID_SIZE,
                nearest_task.priority/3.0,
                nearest_task.get_waiting_time()/10.0,
                self._calculate_path_congestion(
                    self.game.astar.find_path((robot.x, robot.y), nearest_task.get_position())
                )/10.0
            ])
        else:
            state.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Global state components
        # 1. Global metrics
        state.extend([
            len(self.game.tasks)/MAX_TASKS,
            sum(t.priority for t in self.game.tasks)/(3 * MAX_TASKS),
            self._calculate_global_congestion()/10.0
        ])
        
        # 2. Compressed info about other robots
        other_robots_info = []
        for i, other_robot in enumerate(self.game.robots):
            if i != robot_idx and len(other_robots_info) < 5 * (MAX_TASKS - 1):
                other_robots_info.extend([
                    other_robot.x/GRID_SIZE,
                    other_robot.y/GRID_SIZE,
                    1.0 if other_robot.target else 0.0,
                    other_robot.target.priority/3.0 if other_robot.target else 0.0,
                    other_robot.waiting_time/10.0
                ])
        
        # Pad if needed
        padding_needed = 5 * (MAX_TASKS - 1) - len(other_robots_info)
        if padding_needed > 0:
            other_robots_info.extend([0.0] * padding_needed)
        
        state.extend(other_robots_info)
        
        return np.array(state, dtype=np.float32)
    
    def _get_cell_priority(self, x, y):
        """Get priority of task at cell if it exists"""
        for task in self.game.tasks:
            if task.x == x and task.y == y:
                return task.priority
        return 0
    
    def _find_nearest_task(self, robot):
        """Find nearest available task"""
        if not self.game.tasks:
            return None
        return min(self.game.tasks, 
                  key=lambda t: robot.manhattan_distance((robot.x, robot.y), t.get_position()))
    
    def _calculate_global_congestion(self):
        """Calculate global congestion level"""
        congestion = 0
        for robot in self.game.robots:
            if robot.target:
                path = self.game.astar.find_path(
                    (robot.x, robot.y),
                    robot.target.get_position()
                )
                if path:
                    congestion += self._calculate_path_congestion(path)
        return congestion / len(self.game.robots) if self.game.robots else 0
    
    def calculate_bid(self, robot, available_tasks=None):
        """Centralized method to calculate bids for tasks"""
        if available_tasks is None:
            available_tasks = self.get_available_tasks(robot)
            
        if not available_tasks:
            return None, 0.0
            
        try:
            # Get current state and Q-values
            state = self.get_state(robot)
            q_values = self.dqn.get_q_values(state)
            
            # Calculate bids for each task
            bids = []
            for task in available_tasks:
                task_idx = task.y * GRID_SIZE + task.x
                q_value = float(q_values[task_idx])
                
                # Get path and congestion information
                distance = robot.manhattan_distance((robot.x, robot.y), task.get_position())
                path = self.game.astar.find_path((robot.x, robot.y), task.get_position())
                congestion = self._calculate_path_congestion(path) if path else float('inf')
                
                # Calculate normalized factors
                factors = {
                    'q_value': q_value / max(1e-6, q_values.max()),
                    'priority': task.priority / 3.0,
                    'distance': 1.0 - (distance / (2 * GRID_SIZE)),
                    'congestion': 1.0 - (congestion / (len(path) if path else 1)),
                    'waiting_time': min(task.get_waiting_time() / 10.0, 1.0)
                }
                
                # Calculate weighted bid
                bid_value = sum(self.factor_weights[k] * v for k, v in factors.items())
                
                # Apply fairness adjustment
                completion_count = self.learning_stats['task_distribution'].get(robot.id, 0)
                fairness_factor = 1.0 + (0.1 * (1.0 - completion_count / max(1, self.total_tasks_completed)))
                bid_value *= fairness_factor
                
                bids.append((task, bid_value, factors))
            
            if not bids:
                return None, 0.0
                
            # Track bid history
            self.learning_stats['bid_history'].append(max(b[1] for b in bids))
            
            # Choose task with highest bid
            chosen_task, bid_value, factors = max(bids, key=lambda x: x[1])
            
            # Update factor weights based on success/failure
            if robot.id in self.learning_stats['task_distribution']:
                success_rate = self.learning_stats['task_distribution'][robot.id] / max(1, self.episode_count)
                self._update_factor_weights(factors, success_rate)
            
            return chosen_task, bid_value
            
        except Exception as e:
            print(f"Warning: Error calculating bid: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0
            
    def choose_action(self, robot):
        """Enhanced action selection using centralized bidding"""
        # Check if number of robots has changed and reinitialize DQN if needed
        if len(self.game.robots) != self.dqn.num_agents:
            print(f"Reinitializing DQN for {len(self.game.robots)} agents")
            self.dqn = CentralizedDQN(
                state_size=self.state_size,
                action_size=self.action_size,
                num_agents=len(self.game.robots)
            )
            self.prev_states = [None] * len(self.game.robots)
            self.prev_actions = [None] * len(self.game.robots)
        
        # Get available tasks
        available_tasks = self.get_available_tasks(robot)
        if not available_tasks:
            return None
            
        try:
            # Use centralized bidding
            chosen_task, _ = self.calculate_bid(robot, available_tasks)
            
            if chosen_task:
                robot_idx = self.game.robots.index(robot)
                action_idx = chosen_task.y * GRID_SIZE + chosen_task.x
                
                # Store state and action
                self.prev_states[robot_idx] = self.get_state(robot)
                self.prev_actions[robot_idx] = action_idx
            
            return chosen_task
            
        except Exception as e:
            print(f"Warning: Error during action selection: {e}")
            return random.choice(available_tasks) if available_tasks else None
    
    def _update_factor_weights(self, factors, success_rate):
        """Update weights of different factors based on success rate"""
        learning_rate = 0.01
        for factor, value in factors.items():
            if success_rate > 0.5:  # If robot is successful
                # Increase weights of factors that contributed to success
                self.factor_weights[factor] += learning_rate * value * (success_rate - 0.5)
            else:
                # Decrease weights of factors that may have contributed to failure
                self.factor_weights[factor] -= learning_rate * value * (0.5 - success_rate)
            
            # Normalize weights
            total = sum(self.factor_weights.values())
            for k in self.factor_weights:
                self.factor_weights[k] /= total
    
    def update(self, robot, old_state, action, reward, new_state):
        """Enhanced update with prioritized experience replay"""
        robot_idx = self.game.robots.index(robot)
        
        # Check if number of robots has changed and reinitialize DQN if needed
        if len(self.game.robots) != self.dqn.num_agents:
            print(f"Reinitializing DQN for {len(self.game.robots)} agents")
            self.dqn = CentralizedDQN(
                state_size=self.state_size,
                action_size=self.action_size,
                num_agents=len(self.game.robots)
            )
            self.prev_states = [None] * len(self.game.robots)
            self.prev_actions = [None] * len(self.game.robots)
            return  # Skip this update as we've reinitialized
        
        try:
            # Convert action to index if it's a Task object
            if isinstance(action, Task):
                action_idx = int(action.y * GRID_SIZE + action.x)
            elif isinstance(action, tuple):
                x, y = action
                action_idx = int(y * GRID_SIZE + x)
            else:
                action_idx = int(action)
                
            # Calculate TD error for prioritization
            with torch.no_grad():
                # Get Q-values for current state and ensure it's a tensor
                current_q = self.dqn.get_q_values(old_state)
                if not isinstance(current_q, torch.Tensor):
                    current_q = torch.tensor(current_q)
                
                # Ensure action_idx is within bounds and is an integer
                max_idx = current_q.numel() - 1
                action_idx = max(0, min(action_idx, max_idx))
                
                # Get Q-value for the action taken
                try:
                    current_q_value = float(current_q[action_idx])
                except Exception as e:
                    print(f"Warning: Error accessing Q-value: {e}")
                    print(f"current_q shape: {current_q.shape}, action_idx: {action_idx}")
                    current_q_value = 0.0
                
                # Get max Q-value for next state
                next_q = self.dqn.get_q_values(new_state)
                if not isinstance(next_q, torch.Tensor):
                    next_q = torch.tensor(next_q)
                next_q_value = float(next_q.max())
                
                # Calculate TD error
                td_error = abs(reward + self.dqn.gamma * next_q_value - current_q_value)
            
            # Update task distribution statistics
            if isinstance(action, Task) and robot.target == action:
                self.learning_stats['task_distribution'][robot.id] = \
                    self.learning_stats['task_distribution'].get(robot.id, 0) + 1
                self.total_tasks_completed += 1  # Increment total_tasks_completed
                
                # Track completion time
                if hasattr(robot, 'last_task_start'):
                    completion_time = time.time() - robot.last_task_start
                    self.learning_stats['completion_times'].append(completion_time)
            
            # Store experience with priority
            priority = min((td_error + 1e-6) ** self.priority_alpha, self.max_priority)
            self.dqn.remember(old_state, action_idx, reward, new_state, priority)
            
            # Train the network
            self.dqn.train_step(self.priority_beta)
            
            # Update statistics
            self.learning_stats['episode_rewards'].append(reward)
            if hasattr(self.dqn, 'get_training_stats'):
                training_stats = self.dqn.get_training_stats()
                if training_stats:
                    self.learning_stats.update(training_stats)
                    
        except Exception as e:
            print(f"Warning: Error during update: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace for debugging
    
    def get_available_tasks(self, robot):
        """Get list of available tasks that can be assigned"""
        return [task for task in self.game.tasks 
                if self.game.grid[task.y][task.x].value == CellType.TASK.value]
    
    def get_reward(self, robot, old_state, action, new_state):
        """Enhanced reward function with better learning signals"""
        if action is None:
            return -10
        
        reward = 0
        robot_idx = self.game.robots.index(robot)
        
        # Individual rewards
        if isinstance(action, Task):
            # Base reward scaled by priority and urgency
            waiting_time = action.get_waiting_time()
            urgency_factor = min(waiting_time / 10.0, 2.0)  # Cap at 2x multiplier
            reward += action.priority * 20 * urgency_factor
            
            # Distance and congestion consideration
            dist = robot.manhattan_distance((robot.x, robot.y), action.get_position())
            reward -= dist * 2  # Distance penalty
            
            path = self.game.astar.find_path((robot.x, robot.y), action.get_position())
            if path:
                # Path congestion
                congestion = 0
                for px, py in path:
                    for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                        check_x, check_y = px + dx, py + dy
                        if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                            if self.game.grid[check_y][check_x] in [CellType.ROBOT, CellType.OBSTACLE]:
                                congestion += 1
                reward -= congestion * 5
                
                # Team coordination bonus
                other_robots_paths = []
                for other_robot in self.game.robots:
                    if other_robot != robot and other_robot.target:
                        other_path = self.game.astar.find_path(
                            (other_robot.x, other_robot.y),
                            other_robot.target.get_position()
                        )
                        if other_path:
                            other_robots_paths.append(other_path)
                
                # Reward for choosing paths that don't interfere with others
                path_conflicts = 0
                for other_path in other_robots_paths:
                    path_set = set((x, y) for x, y in path)
                    other_set = set((x, y) for x, y in other_path)
                    path_conflicts += len(path_set.intersection(other_set))
                reward -= path_conflicts * 10
                
                # Reward for task completion progress
                if len(path) < dist:  # If we're getting closer to the goal
                    reward += 5
            else:
                reward -= 50  # Heavy penalty for unreachable tasks
            
            # Global efficiency consideration
            if robot.target:
                active_robots = len([r for r in self.game.robots if r.target])
                team_factor = active_robots / len(self.game.robots)
                reward += team_factor * 10
            
            # Completion bonus with team consideration
            if robot.target and robot.manhattan_distance((robot.x, robot.y), robot.target.get_position()) == 0:
                reward += 100 * robot.target.priority
                # Extra bonus if team is performing well
                team_completion_rate = self.game.total_tasks_completed / max(1, time.time() - self.game.start_time)
                reward += team_completion_rate * 20
        
        # Track reward for this episode
        self.learning_stats['episode_rewards'].append(reward)
        return reward
    
    def _calculate_path_congestion(self, path):
        """Helper method to calculate path congestion"""
        if not path:
            return 0
            
        congestion = 0
        for px, py in path:
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                check_x, check_y = px + dx, py + dy
                if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                    if self.game.grid[check_y][check_x] in [CellType.ROBOT, CellType.OBSTACLE]:
                        congestion += 1
        return congestion
        
    def get_learning_stats(self):
        """Get learning statistics for monitoring"""
        if not self.learning_stats['episode_rewards']:
            return None
            
        return {
            'avg_reward': sum(self.learning_stats['episode_rewards'][-100:]) / len(self.learning_stats['episode_rewards'][-100:]),
            'avg_q_value': sum(self.learning_stats['q_value_history'][-100:]) / len(self.learning_stats['q_value_history'][-100:]),
            'avg_bid': sum(self.learning_stats['bid_history'][-100:]) / len(self.learning_stats['bid_history'][-100:]),
            'total_episodes': self.episode_count
        }