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
        self.learning_stats = {
            'episode_rewards': [],
            'q_value_history': [],
            'bid_history': []
        }
        
        # Calculate state and action sizes
        self.state_size = (
            2 +  # Robot position (x, y)
            5 * (MAX_TASKS - 1) +  # Other robots' info (fixed size for max possible robots)
            GRID_SIZE * GRID_SIZE * 4 +  # Task info (x, y, priority, waiting_time)
            GRID_SIZE * GRID_SIZE  # Obstacle info
        )
        self.action_size = GRID_SIZE * GRID_SIZE  # Possible task positions
        
        # Initialize centralized DQN
        self.dqn = CentralizedDQN(
            state_size=self.state_size,
            action_size=self.action_size,
            num_agents=len(game.robots)
        )
        
        # Store previous states and actions for learning
        self.prev_states = [None] * len(game.robots)
        self.prev_actions = [None] * len(game.robots)
        
    def get_state(self, robot):
        """Get state tensor for a robot"""
        robot_idx = self.game.robots.index(robot)
        state = []
        
        # Robot's normalized position
        state.extend([robot.x/GRID_SIZE, robot.y/GRID_SIZE])
        
        # Other robots' info (pad to fixed size)
        other_robots_info = []
        for i, other_robot in enumerate(self.game.robots):
            if i != robot_idx and len(other_robots_info) < 5 * (MAX_TASKS - 1):
                other_robots_info.extend([
                    other_robot.x/GRID_SIZE,
                    other_robot.y/GRID_SIZE,
                    other_robot.target.x/GRID_SIZE if other_robot.target else 0,
                    other_robot.target.y/GRID_SIZE if other_robot.target else 0,
                    other_robot.target.priority/3 if other_robot.target else 0
                ])
        
        # Pad with zeros if needed
        padding_needed = 5 * (MAX_TASKS - 1) - len(other_robots_info)
        if padding_needed > 0:
            other_robots_info.extend([0] * padding_needed)
        
        state.extend(other_robots_info)
        
        # Task info (fixed size grid representation)
        task_info = np.zeros(GRID_SIZE * GRID_SIZE * 4)
        for task in self.game.tasks:
            idx = (task.y * GRID_SIZE + task.x) * 4
            task_info[idx:idx+4] = [
                task.x/GRID_SIZE,
                task.y/GRID_SIZE,
                task.priority/3,
                min(task.get_waiting_time()/10.0, 1.0)
            ]
        state.extend(task_info)
        
        # Obstacle info (fixed size grid)
        obstacle_info = np.zeros(GRID_SIZE * GRID_SIZE)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.game.grid[y][x] == CellType.OBSTACLE:
                    obstacle_info[y * GRID_SIZE + x] = 1
        state.extend(obstacle_info)
        
        # Ensure state has correct size
        assert len(state) == self.state_size, f"State size mismatch: {len(state)} != {self.state_size}"
        
        return np.array(state, dtype=np.float32)
    
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
    
    def choose_action(self, robot):
        """Enhanced action selection with better exploration and exploitation"""
        # Get states for all robots
        states = [self.get_state(r) for r in self.game.robots]
        robot_idx = self.game.robots.index(robot)
        
        # Reinitialize DQN if number of robots has changed
        if len(self.game.robots) != self.dqn.num_agents:
            self.dqn = CentralizedDQN(
                state_size=self.state_size,
                action_size=self.action_size,
                num_agents=len(self.game.robots)
            )
            self.prev_states = [None] * len(self.game.robots)
            self.prev_actions = [None] * len(self.game.robots)
        
        # Get Q-values for all actions
        try:
            q_values = self.dqn.get_q_values(states[robot_idx])
            self.learning_stats['q_value_history'].append(float(q_values.max()))
            
            # Calculate bid values for available tasks
            available_tasks = self.get_available_tasks(robot)
            if not available_tasks:
                return None
            
            bids = []
            for task in available_tasks:
                task_idx = task.y * GRID_SIZE + task.x
                q_value = float(q_values[task_idx])
                
                # Calculate bid using Q-value and heuristics
                distance = robot.manhattan_distance((robot.x, robot.y), task.get_position())
                path = self.game.astar.find_path((robot.x, robot.y), task.get_position())
                congestion = self._calculate_path_congestion(path) if path else float('inf')
                
                bid_value = (
                    q_value * 2.0 +  # Learning component
                    task.priority * 30.0 +  # Priority bonus
                    (1.0 - distance/GRID_SIZE) * 20.0 +  # Distance factor
                    (1.0 - congestion/len(path) if path else 0) * 10.0 +  # Congestion factor
                    task.get_waiting_time() * 5.0  # Waiting time bonus
                )
                bids.append((task, bid_value))
                
            # Track bid history
            self.learning_stats['bid_history'].append(max(b[1] for b in bids) if bids else 0)
            
            # Choose task with highest bid
            chosen_task, _ = max(bids, key=lambda x: x[1])
            action_idx = chosen_task.y * GRID_SIZE + chosen_task.x
            
            # Store state and action for learning
            self.prev_states[robot_idx] = states[robot_idx]
            self.prev_actions[robot_idx] = action_idx
            
            return chosen_task
            
        except Exception as e:
            print(f"Warning: Error during action selection: {e}")
            if available_tasks:
                return random.choice(available_tasks)
            return None
            
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
    
    def update(self, robot, old_state, action, reward, new_state):
        """Update DQN with experience"""
        robot_idx = self.game.robots.index(robot)
        
        # Collect experiences from all robots
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        # Calculate expected state size
        expected_state_size = (
            2 +  # Robot position
            5 * (len(self.game.robots) - 1) +  # Other robots' info
            GRID_SIZE * GRID_SIZE * 4 +  # Task info
            GRID_SIZE * GRID_SIZE  # Obstacle info
        )
        
        for i, r in enumerate(self.game.robots):
            if self.prev_states[i] is not None:
                # Ensure state has correct shape
                state = self.prev_states[i]
                if isinstance(state, torch.Tensor):
                    state = state.numpy()
                if len(state) != expected_state_size:
                    # Skip this experience if state size doesn't match
                    continue
                states.append(state)
                
                # Store action
                actions.append(self.prev_actions[i])
                rewards.append(reward if i == robot_idx else 0)
                
                # Get and validate next state
                next_state = self.get_state(r)
                if isinstance(next_state, torch.Tensor):
                    next_state = next_state.numpy()
                if len(next_state) != expected_state_size:
                    # Skip this experience if next state size doesn't match
                    continue
                next_states.append(next_state)
                
                dones.append(False)
        
        if states:  # Only update if we have valid experiences
            try:
                # Convert lists to numpy arrays with explicit shapes
                states = np.array(states, dtype=np.float32).reshape(-1, expected_state_size)
                actions = np.array(actions, dtype=np.int64)
                rewards = np.array(rewards, dtype=np.float32)
                next_states = np.array(next_states, dtype=np.float32).reshape(-1, expected_state_size)
                dones = np.array(dones, dtype=np.float32)
                
                self.dqn.remember(states, actions, rewards, next_states, dones)
                self.dqn.train_step()
            except (ValueError, RuntimeError) as e:
                print(f"Warning: Skipping update due to shape mismatch: {e}")
            
        # Count episode for metrics
        if action == robot.target:
            self.episode_count += 1