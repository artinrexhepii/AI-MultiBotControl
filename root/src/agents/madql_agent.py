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
        self.total_tasks_completed = 0
        
        # Initialize learning statistics
        self.learning_stats = {
            'episode_rewards': [],
            'bid_success_rate': {},
            'bid_history': [],
            'bid_values': [],
            'completion_times': []
        }
        
        # Streamlined core metrics
        self.metrics = {
            'episode_rewards': [],      # Core learning performance
            'completion_times': [],     # Task completion efficiency
            'team_throughput': {        # Team collaboration efficiency
                'window_size': 100,     # Rolling window size
                'tasks_completed': 0,   # Total tasks in current window
                'time_elapsed': 0.0,    # Time elapsed in current window
                'window_start': time.time(),
                'history': []           # List of (tasks/second) measurements
            },
            'agent_performance': {}     # Per-agent performance tracking
        }
        
        # Initialize bid-specific parameters
        self.bid_range = 10  # Number of discrete bid values
        self.min_bid = 0.0
        self.max_bid = 1.0
        self.bid_increment = (self.max_bid - self.min_bid) / self.bid_range
        
        # Initialize tracking
        self.prev_states = {}  # Changed to dict for per-robot tracking
        self.prev_actions = {}  # Changed to dict for per-robot tracking
        self.prev_bids = {}  # Track previous bids for each robot
        
        # Initialize agent performance tracking - moved after DQN initialization
        
        # Initialize learned weights for different factors
        self.factor_weights = {
            'q_value': 1.0,
            'priority': 1.0,
            'distance': 1.0,
            'congestion': 1.0,
            'waiting_time': 1.0
        }
        
        # Experience prioritization parameters
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.max_priority = 1.0
        
        # Calculate state size with local and global components
        self.local_state_size = (
            2 +  # Robot position (x, y)
            4 * 9 +  # 3x3 grid around robot (4 features per cell)
            5 +  # Local task features
            3   # Bid-specific features (last bid, success rate, competition)
        )
        self.global_state_size = (
            3 +  # Global metrics (task count, avg priority, congestion)
            5 * (MAX_TASKS - 1)  # Other robots' compressed info
        )
        self.state_size = self.local_state_size + self.global_state_size
        self.action_size = self.bid_range  # Action space now represents bid values
        
        # Agent pool management
        self.max_pool_size = MAX_TASKS * 2  # Maximum number of potential agents
        self.active_agents = set()  # Currently active agent indices
        self.agent_pool = {}  # Pool of agent networks
        self.agent_stats = {}  # Track statistics per agent
        
        # Initialize agent pool
        self.dqn = CentralizedDQN(
            state_size=self.state_size,
            action_size=self.action_size,
            num_agents=self.max_pool_size  # Initialize with maximum possible agents
        )
        
        # Initialize agent statistics for all robots
        for robot in game.robots:
            self.agent_stats[robot.id] = {
                'last_active': time.time(),
                'total_tasks': 0,
                'avg_reward': 0.0,
                'experience_count': 0
            }
            self.metrics['agent_performance'][robot.id] = {
                'tasks_completed': 0,
                'avg_completion_time': 0.0,
                'success_rate': 0.0,
                'last_update': time.time()
            }
        
        # Initialize experience buffers per agent
        self.agent_experiences = {robot.id: [] for robot in game.robots}
        self.max_experiences_per_agent = 10000
        
        # Add collaboration-specific tracking
        self.shared_info = {
            'obstacle_updates': [],  # Recent obstacle movements/positions
            'task_priorities': {},   # Shared task priority assessments
            'path_conflicts': set(), # Areas with potential path conflicts
            'subgoal_assignments': {} # Track subgoal assignments for shared tasks
        }
        
        # Collaboration parameters
        self.info_share_interval = 1.0  # Share info every second
        self.last_info_share = 0
        self.max_shared_memory = 100  # Limit memory of shared information
        
        # Dynamic learning parameters
        self.learning_params = {
            'learning_rate': {
                'value': LEARNING_RATE,
                'min': 0.0001,
                'max': 0.01,
                'decay': 0.995,
                'history': []
            },
            'epsilon': {
                'value': EPSILON,
                'min': 0.01,
                'max': 1.0,
                'decay': 0.997,
                'history': []
            },
            'discount': {
                'value': DISCOUNT_FACTOR,
                'min': 0.8,
                'max': 0.99,
                'adjust_window': 100,
                'history': []
            }
        }
        
        # Performance tracking for parameter tuning
        self.performance_window = {
            'size': 100,
            'rewards': [],
            'success_rate': [],
            'avg_completion_time': [],
            'last_adjustment': time.time(),
            'adjustment_interval': 60.0  # Adjust parameters every 60 seconds
        }
        
    def get_state(self, robot):
        """Enhanced state representation with bid-specific features"""
        robot_idx = self.game.robots.index(robot)
        state = []
        
        # Local state components
        state.extend([robot.x/GRID_SIZE, robot.y/GRID_SIZE])
        
        # 3x3 grid around robot
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                x, y = robot.x + dx, robot.y + dy
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    cell = self.game.grid.get_cell(x, y)
                    state.extend([
                        1.0 if cell == CellType.ROBOT else 0.0,
                        1.0 if cell == CellType.OBSTACLE else 0.0,
                        1.0 if cell in [CellType.TASK, CellType.TARGET] else 0.0,
                        self._get_cell_priority(x, y) / 3.0
                    ])
                else:
                    state.extend([0.0, 0.0, 0.0, 0.0])
        
        # Local task features
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
        
        # Bid-specific features
        last_bid = self.prev_bids.get(robot.id, 0.0)
        bid_success_rate = self.get_bid_success_rate(robot)
        competition_level = self.get_competition_level(robot)
        state.extend([
            last_bid / self.max_bid,
            bid_success_rate,
            competition_level
        ])
        
        # Global state components
        state.extend([
            len(self.game.grid.tasks)/MAX_TASKS,
            sum(t.priority for t in self.game.grid.tasks)/(3 * MAX_TASKS),
            self._calculate_global_congestion()/10.0
        ])
        
        # Other robots' info
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
    
    def get_bid_success_rate(self, robot):
        """Calculate robot's bid success rate"""
        if robot.id not in self.learning_stats['bid_success_rate']:
            self.learning_stats['bid_success_rate'][robot.id] = []
        
        history = self.learning_stats['bid_success_rate'][robot.id]
        if not history:
            return 0.5  # Default to neutral value
        return sum(history[-100:]) / len(history[-100:])  # Last 100 bids
    
    def update_bid_success_rate(self, robot, success):
        """Update bid success rate for a robot"""
        if robot.id not in self.learning_stats['bid_success_rate']:
            self.learning_stats['bid_success_rate'][robot.id] = []
        self.learning_stats['bid_success_rate'][robot.id].append(1.0 if success else 0.0)
        # Keep only last 100 bids
        if len(self.learning_stats['bid_success_rate'][robot.id]) > 100:
            self.learning_stats['bid_success_rate'][robot.id] = self.learning_stats['bid_success_rate'][robot.id][-100:]
    
    def get_competition_level(self, robot):
        """Calculate competition level for tasks"""
        if not self.game.grid.tasks:
            return 0.0
        unassigned_robots = len([r for r in self.game.robots if not r.target])
        return min(1.0, unassigned_robots / len(self.game.grid.tasks))
    
    def calculate_bid(self, robot, available_tasks=None):
        """Calculate bid value for available tasks"""
        if available_tasks is None:
            available_tasks = self.get_available_tasks(robot)
            
        if not available_tasks:
            print("No available tasks for bidding")  # Debug print
            return None, 0.0
        
        try:
            # Calculate task scores considering multiple factors
            task_scores = []
            for task in available_tasks:
                score = self._calculate_task_score(task, robot)
                task_scores.append((task, score))
                print(f"Task ({task.x}, {task.y}) P{task.priority} score: {score}")  # Debug print
            
            # Sort tasks by score
            task_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Choose best task and calculate bid
            chosen_task = task_scores[0][0] if task_scores else None
            if chosen_task:
                # Calculate base bid from task score
                max_score = max(score for _, score in task_scores)
                base_bid = task_scores[0][1] / max_score if max_score > 0 else 0
                
                # Add priority factor
                priority_factor = chosen_task.priority / 3.0
                
                # Add distance factor (closer is better)
                distance = robot.manhattan_distance((robot.x, robot.y), chosen_task.get_position())
                max_distance = GRID_SIZE * 2
                distance_factor = 1.0 - (distance / max_distance)
                
                # Calculate final bid
                bid_value = (
                    0.4 * base_bid +  # Base task score
                    0.4 * priority_factor +  # Priority contribution
                    0.2 * distance_factor  # Distance contribution
                )
                
                # Ensure bid is within valid range
                bid_value = max(self.min_bid, min(self.max_bid, bid_value))
                
                print(f"Final bid for task ({chosen_task.x}, {chosen_task.y}): {bid_value}")  # Debug print
                
                return chosen_task, bid_value
                
            print("No suitable task found for bidding")  # Debug print
            return None, 0.0
            
        except Exception as e:
            print(f"Error in calculate_bid: {e}")  # Debug print
            import traceback
            traceback.print_exc()
            return None, 0.0
    
    def get_reward(self, robot, old_state, action, new_state):
        """Calculate reward for the given action"""
        if action is None:
            return -1.0  # Small penalty for no action
            
        reward = 0.0
        
        # Initialize metrics for new robots if needed
        if robot.id not in self.metrics['agent_performance']:
            self.metrics['agent_performance'][robot.id] = {
                'tasks_completed': 0,
                'avg_completion_time': 0.0,
                'success_rate': 0.0,
                'last_update': time.time()
            }
        
        if isinstance(action, Task):
            # Base reward scaled by priority
            reward += 10.0 * action.priority  # 10/20/30 for priorities 1/2/3
            
            # Completion time efficiency bonus
            if hasattr(robot, 'last_task_start'):
                completion_time = time.time() - robot.last_task_start
                time_efficiency = max(0, 1.0 - (completion_time / 20.0))  # Cap at 20 seconds
                reward += 5.0 * time_efficiency  # Up to 5 points for quick completion
            
            # Team completion rate bonus
            team_tasks_completed = sum(1 for r in self.game.robots if hasattr(r, 'completed_tasks'))
            if team_tasks_completed > 0:
                team_efficiency = robot.completed_tasks / team_tasks_completed
                reward += 5.0 * team_efficiency  # Up to 5 points for team contribution
            
            # Distance optimization
            current_dist = robot.manhattan_distance((robot.x, robot.y), action.get_position())
            if hasattr(robot, 'prev_distance'):
                distance_change = robot.prev_distance - current_dist
                reward += distance_change * 0.5  # Small continuous reward/penalty
            robot.prev_distance = current_dist
            
            # Path efficiency penalty
            path = self.game.astar.find_path((robot.x, robot.y), action.get_position())
            if path:
                path_efficiency = current_dist / len(path)
                if path_efficiency < 0.8:  # If path is significantly longer than direct distance
                    reward -= 2.0 * (1.0 - path_efficiency)  # Up to -2 points
        
        # Normalize reward
        reward = max(-10.0, min(50.0, reward))
        
        # Update learning statistics
        self.learning_stats['episode_rewards'].append(reward)
        
        return reward
    
    def _calculate_path_conflicts(self, robot, path):
        """Calculate number of path conflicts with other robots"""
        conflicts = 0
        for other_robot in self.game.robots:
            if other_robot != robot and other_robot.target:
                other_path = self.game.astar.find_path(
                    (other_robot.x, other_robot.y),
                    other_robot.target.get_position()
                )
                if other_path:
                    path_set = set((x, y) for x, y in path)
                    other_set = set((x, y) for x, y in other_path)
                    conflicts += len(path_set.intersection(other_set))
        return conflicts
    
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
    
    def choose_action(self, robot):
        """Enhanced action selection with dynamic agent handling"""
        # Ensure robot's agent is active
        if robot.id not in self.active_agents:
            self._update_active_agents()
        
        # Get available tasks
        available_tasks = self.get_available_tasks(robot)
        if not available_tasks:
            return None
        
        try:
            # Use centralized bidding
            chosen_task, _ = self.calculate_bid(robot, available_tasks)
            
            if chosen_task:
                action_idx = chosen_task.y * GRID_SIZE + chosen_task.x
                self.prev_states[robot.id] = self.get_state(robot)
                self.prev_actions[robot.id] = action_idx
            
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
        """Enhanced update with streamlined metrics"""
        robot_idx = robot.id
        
        # Ensure robot's agent is active
        if robot_idx not in self.active_agents:
            self._update_active_agents()
        
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
                current_q = self.dqn.get_q_values(old_state)
                if not isinstance(current_q, torch.Tensor):
                    current_q = torch.tensor(current_q)
                
                max_idx = current_q.numel() - 1
                action_idx = max(0, min(action_idx, max_idx))
                
                try:
                    current_q_value = float(current_q[action_idx])
                except Exception as e:
                    print(f"Warning: Error accessing Q-value: {e}")
                    current_q_value = 0.0
                
                next_q = self.dqn.get_q_values(new_state)
                if not isinstance(next_q, torch.Tensor):
                    next_q = torch.tensor(next_q)
                next_q_value = float(next_q.max())
                
                td_error = abs(reward + self.dqn.gamma * next_q_value - current_q_value)
            
            # Store experience for the specific agent
            experience = (old_state, action_idx, reward, new_state)
            self.agent_experiences[robot_idx].append(experience)
            
            # Limit experience buffer size
            if len(self.agent_experiences[robot_idx]) > self.max_experiences_per_agent:
                self.agent_experiences[robot_idx].pop(0)
            
            # Update agent statistics
            self.agent_stats[robot_idx]['experience_count'] += 1
            self.agent_stats[robot_idx]['avg_reward'] = (
                self.agent_stats[robot_idx]['avg_reward'] * 0.95 + reward * 0.05
            )
            
            # Store experience with priority
            priority = min((td_error + 1e-6) ** self.priority_alpha, self.max_priority)
            self.dqn.remember(old_state, action_idx, reward, new_state, priority)
            
            # Update task statistics and metrics if applicable
            if isinstance(action, Task) and robot.target == action:
                self.agent_stats[robot_idx]['total_tasks'] += 1
                self.total_tasks_completed += 1
                
                completion_time = None
                if hasattr(robot, 'last_task_start'):
                    completion_time = time.time() - robot.last_task_start
                
                # Update core metrics
                self.update_metrics(robot, reward, completion_time)
            else:
                # Update metrics without completion time
                self.update_metrics(robot, reward)
            
            # Train the network focusing on active agents
            self._train_active_agents()
            
            # Update learning statistics
            self.learning_stats['episode_rewards'].append(reward)
            if hasattr(self.dqn, 'get_training_stats'):
                training_stats = self.dqn.get_training_stats()
                if training_stats:
                    self.learning_stats.update(training_stats)
            
        except Exception as e:
            print(f"Warning: Error during update: {e}")
            import traceback
            traceback.print_exc()

    def _train_active_agents(self):
        """Train the network focusing on active agents"""
        # Only train if we have enough experiences
        if not self.active_agents:
            return
        
        try:
            # Prioritize experiences from active agents
            active_experiences = []
            for agent_id in self.active_agents:
                if agent_id in self.agent_experiences:
                    active_experiences.extend(self.agent_experiences[agent_id][-100:])
            
            if active_experiences:
                # Sample and train on experiences
                sample_size = min(32, len(active_experiences))
                batch = random.sample(active_experiences, sample_size)
                
                # Unzip the batch
                states, actions, rewards, next_states = zip(*batch)
                
                # Convert to tensors
                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(np.array(actions))
                rewards = torch.FloatTensor(np.array(rewards))
                next_states = torch.FloatTensor(np.array(next_states))
                
                # Train the network
                self.dqn.train_step(self.priority_beta)
        
        except Exception as e:
            print(f"Warning: Error during training: {e}")
            import traceback
            traceback.print_exc()

    def get_available_tasks(self, robot):
        """Get list of available tasks that can be assigned"""
        return [task for task in self.game.grid.tasks 
                if self.game.grid.get_cell(task.x, task.y).value == CellType.TASK.value]
    
    def _calculate_path_congestion(self, path):
        """Helper method to calculate path congestion"""
        if not path:
            return 0
            
        congestion = 0
        for px, py in path:
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                check_x, check_y = px + dx, py + dy
                if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                    cell = self.game.grid.get_cell(check_x, check_y)
                    if cell in [CellType.ROBOT, CellType.OBSTACLE]:
                        congestion += 1
        return congestion
        
    def get_learning_stats(self):
        """Get streamlined learning statistics for monitoring"""
        return self.get_performance_summary()

    def update_metrics(self, robot, reward, completion_time=None):
        """Update metrics and adjust learning parameters"""
        current_time = time.time()
        
        # 1. Episode Rewards (keep last 1000 rewards)
        self.metrics['episode_rewards'].append(reward)
        if len(self.metrics['episode_rewards']) > 1000:
            self.metrics['episode_rewards'] = self.metrics['episode_rewards'][-1000:]
        
        # 2. Task Completion Time
        if completion_time is not None:
            self.metrics['completion_times'].append(completion_time)
            if len(self.metrics['completion_times']) > 1000:
                self.metrics['completion_times'] = self.metrics['completion_times'][-1000:]
        
        # 3. Team Throughput
        window = self.metrics['team_throughput']
        window['tasks_completed'] += 1
        window['time_elapsed'] = current_time - window['window_start']
        
        # Update throughput every window_size tasks or 60 seconds
        if (window['tasks_completed'] >= window['window_size'] or 
            window['time_elapsed'] >= 60.0):
            throughput = window['tasks_completed'] / max(window['time_elapsed'], 1e-6)
            window['history'].append(throughput)
            if len(window['history']) > 10:  # Keep last 10 windows
                window['history'] = window['history'][-10:]
            
            # Reset window
            window['tasks_completed'] = 0
            window['time_elapsed'] = 0.0
            window['window_start'] = current_time
        
        # 4. Agent Performance
        if robot.id in self.metrics['agent_performance']:
            agent_stats = self.metrics['agent_performance'][robot.id]
            agent_stats['tasks_completed'] += 1
            
            # Update average completion time
            if completion_time is not None:
                agent_stats['avg_completion_time'] = (
                    0.95 * agent_stats['avg_completion_time'] +
                    0.05 * completion_time  # Exponential moving average
                )
            
            # Update success rate (based on positive rewards)
            time_diff = current_time - agent_stats['last_update']
            if time_diff > 0:
                success = 1.0 if reward > 0 else 0.0
                agent_stats['success_rate'] = (
                    0.95 * agent_stats['success_rate'] +
                    0.05 * success  # Exponential moving average
                )
            
            agent_stats['last_update'] = current_time
        
        # Update performance window for parameter tuning
        self.performance_window['rewards'].append(reward)
        
        # Calculate success rate based on positive rewards
        success = 1.0 if reward > 0 else 0.0
        self.performance_window['success_rate'].append(success)
        
        if completion_time is not None:
            self.performance_window['avg_completion_time'].append(completion_time)
        
        # Maintain window size
        for key in ['rewards', 'success_rate', 'avg_completion_time']:
            if len(self.performance_window[key]) > self.performance_window['size']:
                self.performance_window[key] = self.performance_window[key][-self.performance_window['size']:]
        
        # Adjust learning parameters
        self._adjust_learning_parameters()

    def get_performance_summary(self):
        """Enhanced performance summary including learning parameters"""
        current_time = time.time()
        
        try:
            summary = {
                # Overall team performance
                'team_performance': {
                    'total_tasks_completed': self.total_tasks_completed,
                    'avg_reward': sum(self.metrics['episode_rewards'][-100:]) / 100 if self.metrics['episode_rewards'] else 0,
                    'avg_completion_time': sum(self.metrics['completion_times'][-100:]) / 100 if self.metrics['completion_times'] else 0,
                    'current_throughput': self.metrics['team_throughput']['tasks_completed'] / 
                                        max(current_time - self.metrics['team_throughput']['window_start'], 1e-6),
                    'avg_throughput': sum(self.metrics['team_throughput']['history']) / 
                                    len(self.metrics['team_throughput']['history']) if self.metrics['team_throughput']['history'] else 0
                },
                
                # Per-agent performance
                'agent_performance': {
                    robot_id: {
                        'tasks_completed': stats['tasks_completed'],
                        'avg_completion_time': stats['avg_completion_time'],
                        'success_rate': stats['success_rate']
                    }
                    for robot_id, stats in self.metrics['agent_performance'].items()
                }
            }
            
            # Add learning parameter information
            summary['learning_parameters'] = {
                name: {
                    'current_value': param['value'],
                    'trend': self._calculate_trend(param['history']) if param['history'] else 0
                }
                for name, param in self.learning_params.items()
            }
            
            # Add performance trends
            summary['performance_trends'] = {
                'reward_trend': self._calculate_trend(self.performance_window['rewards']),
                'success_trend': self._calculate_trend(self.performance_window['success_rate']),
                'completion_trend': self._calculate_trend(self.performance_window['avg_completion_time'])
            }
            
            return summary
            
        except Exception as e:
            print(f"Warning: Error generating performance summary: {e}")
            return None

    def share_information(self, robot):
        """Share and process information with other robots"""
        current_time = time.time()
        if current_time - self.last_info_share < self.info_share_interval:
            return
            
        self.last_info_share = current_time
        robot_pos = (robot.x, robot.y)
        
        # 1. Share obstacle information
        visible_obstacles = self._get_visible_obstacles(robot)
        self.shared_info['obstacle_updates'] = (
            [(pos, current_time) for pos, _ in self.shared_info['obstacle_updates'][-self.max_shared_memory:]] +
            [(pos, current_time) for pos in visible_obstacles]
        )
        
        # 2. Update shared task priorities
        if robot.target:
            # Calculate task value considering team context
            task_value = self._calculate_team_task_value(robot.target, robot)
            self.shared_info['task_priorities'][robot.target.get_position()] = {
                'value': task_value,
                'assigned_to': robot.id,
                'timestamp': current_time
            }
        
        # 3. Share path information
        if robot.path:
            # Add robot's planned path to shared information
            path_area = set((x, y) for x, y in robot.path)
            self.shared_info['path_conflicts'].update(path_area)
            
            # Clean up old path conflicts
            self._cleanup_path_conflicts()
        
        # 4. Process shared information for decision making
        self._process_shared_information(robot)

    def _get_visible_obstacles(self, robot):
        """Get obstacles visible to the robot within its observation range"""
        visible = set()
        obs_range = 3  # Observable range (3x3 grid)
        
        for dy in range(-obs_range, obs_range + 1):
            for dx in range(-obs_range, obs_range + 1):
                x, y = robot.x + dx, robot.y + dy
                if (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and
                    self.game.grid.get_cell(x, y) == CellType.OBSTACLE):
                    visible.add((x, y))
        
        return visible

    def _calculate_team_task_value(self, task, robot):
        """Calculate task value considering team context"""
        value = task.priority * 10.0  # Base value from priority
        
        # Adjust for team positioning
        team_positions = [(r.x, r.y) for r in self.game.robots if r != robot]
        avg_team_distance = 0
        if team_positions:
            distances = [robot.manhattan_distance(pos, task.get_position()) for pos in team_positions]
            avg_team_distance = sum(distances) / len(distances)
            # Increase value if task is far from team's current focus
            value *= (1.0 + avg_team_distance / GRID_SIZE)
        
        # Adjust for task urgency
        waiting_time = task.get_waiting_time()
        value *= (1.0 + min(waiting_time / 10.0, 1.0))
        
        return value

    def _process_shared_information(self, robot):
        """Process shared information for better decision making"""
        current_time = time.time()
        
        # 1. Update path planning based on shared obstacle information
        recent_obstacles = {pos for pos, t in self.shared_info['obstacle_updates']
                          if current_time - t < 5.0}  # Consider obstacles reported in last 5 seconds
        
        # 2. Consider team task assignments
        if robot.target:
            target_pos = robot.target.get_position()
            if target_pos in self.shared_info['task_priorities']:
                task_info = self.shared_info['task_priorities'][target_pos]
                if (task_info['assigned_to'] != robot.id and 
                    task_info['value'] > self._calculate_team_task_value(robot.target, robot)):
                    # Consider yielding task to other robot
                    robot.target = None
                    robot.path = []
        
        # 3. Avoid path conflicts
        if robot.path:
            conflict_found = False
            for pos in robot.path:
                if pos in self.shared_info['path_conflicts']:
                    conflict_found = True
                    break
            if conflict_found:
                # Recalculate path avoiding conflicts
                if robot.target:
                    new_path = self.game.astar.find_path(
                        (robot.x, robot.y),
                        robot.target.get_position(),
                        blocked_positions=self.shared_info['path_conflicts']
                    )
                    if new_path:
                        robot.path = new_path

    def _cleanup_path_conflicts(self):
        """Remove old path conflicts"""
        # Keep only conflicts that are still relevant
        active_paths = set()
        for robot in self.game.robots:
            if robot.path:
                active_paths.update((x, y) for x, y in robot.path)
        self.shared_info['path_conflicts'] = self.shared_info['path_conflicts'].intersection(active_paths)

    def _adjust_bid_with_shared_info(self, bid_value, robot, available_tasks):
        """Adjust bid value based on shared information"""
        adjustment = 1.0
        
        # 1. Adjust for team task distribution
        team_positions = [(r.x, r.y) for r in self.game.robots if r != robot]
        if team_positions and available_tasks:
            # Calculate average distance to team
            avg_team_distances = []
            for task in available_tasks:
                task_pos = task.get_position()
                team_distances = [robot.manhattan_distance(pos, task_pos) for pos in team_positions]
                avg_team_distances.append(sum(team_distances) / len(team_distances))
            
            # Prefer tasks that are farther from team's current positions
            max_team_dist = max(avg_team_distances)
            if max_team_dist > 0:
                team_distribution_factor = max(avg_team_distances) / max_team_dist
                adjustment *= (1.0 + 0.2 * team_distribution_factor)  # Up to 20% boost
        
        # 2. Adjust for path conflicts
        if robot.path:
            conflict_count = sum(1 for pos in robot.path if pos in self.shared_info['path_conflicts'])
            if conflict_count > 0:
                adjustment *= (1.0 - 0.1 * conflict_count)  # Reduce bid by 10% per conflict
        
        # 3. Adjust for task priority sharing
        if robot.target:
            target_pos = robot.target.get_position()
            if target_pos in self.shared_info['task_priorities']:
                other_value = self.shared_info['task_priorities'][target_pos]['value']
                own_value = self._calculate_team_task_value(robot.target, robot)
                if other_value > own_value:
                    adjustment *= 0.8  # Reduce bid if others value the task more
        
        return bid_value * adjustment

    def _choose_task_with_collaboration(self, available_tasks, bid_value, robot):
        """Choose task considering team collaboration"""
        if not available_tasks:
            return None
            
        task_scores = []
        for task in available_tasks:
            score = task.priority * bid_value
            
            # Distance factor
            distance = robot.manhattan_distance((robot.x, robot.y), task.get_position())
            score /= (1 + distance)
            
            # Team positioning factor
            team_value = self._calculate_team_task_value(task, robot)
            score *= team_value
            
            # Path conflict avoidance
            path = self.game.astar.find_path((robot.x, robot.y), task.get_position())
            if path:
                conflicts = sum(1 for pos in path if pos in self.shared_info['path_conflicts'])
                score *= (1.0 / (1.0 + conflicts))
            
            task_scores.append((task, score))
        
        # Choose task with highest score
        return max(task_scores, key=lambda x: x[1])[0]

    def _update_active_agents(self):
        """Update the set of active agents based on current robots"""
        current_robot_ids = {robot.id for robot in self.game.robots}
        
        # Initialize stats for new robots
        for robot_id in current_robot_ids:
            if robot_id not in self.agent_stats:
                self.agent_stats[robot_id] = {
                    'last_active': time.time(),
                    'total_tasks': 0,
                    'avg_reward': 0.0,
                    'experience_count': 0
                }
                self.metrics['agent_performance'][robot_id] = {
                    'tasks_completed': 0,
                    'avg_completion_time': 0.0,
                    'success_rate': 0.0,
                    'last_update': time.time()
                }
                self.agent_experiences[robot_id] = []
        
        # Deactivate agents for removed robots
        agents_to_deactivate = self.active_agents - current_robot_ids
        for agent_id in agents_to_deactivate:
            if agent_id in self.agent_stats:  # Check if agent exists before updating
                self.agent_stats[agent_id]['last_active'] = time.time()
                self._preserve_agent_experience(agent_id)
        
        # Activate agents for new robots
        for robot_id in current_robot_ids:
            if robot_id not in self.active_agents:
                self._activate_agent(robot_id)
        
        self.active_agents = current_robot_ids

    def _activate_agent(self, agent_id):
        """Activate an agent, potentially reusing previous experience"""
        if agent_id < self.max_pool_size:
            # If agent was previously active, restore its experience
            if self.agent_stats[agent_id]['last_active'] is not None:
                self._restore_agent_experience(agent_id)
            
            self.agent_stats[agent_id]['last_active'] = time.time()
            self.active_agents.add(agent_id)

    def _preserve_agent_experience(self, agent_id):
        """Preserve important experiences when deactivating an agent"""
        if agent_id in self.agent_experiences:
            # Sort experiences by absolute reward
            sorted_exp = sorted(self.agent_experiences[agent_id], 
                              key=lambda x: abs(x[2]),  # x[2] is the reward
                              reverse=True)
            
            # Keep top experiences
            keep_count = min(1000, len(sorted_exp))
            self.agent_experiences[agent_id] = sorted_exp[:keep_count]

    def _restore_agent_experience(self, agent_id):
        """Restore preserved experiences when reactivating an agent"""
        if agent_id in self.agent_experiences and self.agent_experiences[agent_id]:
            # Gradually reintroduce experiences to the DQN
            for exp in self.agent_experiences[agent_id]:
                self.dqn.remember(*exp, priority=1.0)

    def _adjust_learning_parameters(self):
        """Dynamically adjust learning parameters based on performance"""
        current_time = time.time()
        if current_time - self.performance_window['last_adjustment'] < self.performance_window['adjustment_interval']:
            return
            
        try:
            # Calculate performance metrics
            recent_rewards = self.performance_window['rewards'][-self.performance_window['size']:]
            recent_success = self.performance_window['success_rate'][-self.performance_window['size']:]
            recent_completion = self.performance_window['avg_completion_time'][-self.performance_window['size']:]
            
            if not (recent_rewards and recent_success and recent_completion):
                return
                
            # Calculate trend indicators
            reward_trend = self._calculate_trend(recent_rewards)
            success_trend = self._calculate_trend(recent_success)
            completion_trend = self._calculate_trend(recent_completion)
            
            # Adjust learning rate
            if reward_trend < 0:  # If performance is declining
                self.learning_params['learning_rate']['value'] *= self.learning_params['learning_rate']['decay']
            elif reward_trend > 0 and success_trend > 0:  # If performance is improving
                self.learning_params['learning_rate']['value'] = min(
                    self.learning_params['learning_rate']['value'] / self.learning_params['learning_rate']['decay'],
                    self.learning_params['learning_rate']['max']
                )
            
            # Adjust epsilon based on success rate
            avg_success = sum(recent_success) / len(recent_success)
            if avg_success > 0.8:  # High success rate, reduce exploration
                self.learning_params['epsilon']['value'] *= self.learning_params['epsilon']['decay']
            elif avg_success < 0.4:  # Low success rate, increase exploration
                self.learning_params['epsilon']['value'] = min(
                    self.learning_params['epsilon']['value'] / self.learning_params['epsilon']['decay'],
                    self.learning_params['epsilon']['max']
                )
            
            # Adjust discount factor based on completion time trend
            if completion_trend < 0:  # Improving completion times
                self.learning_params['discount']['value'] = min(
                    self.learning_params['discount']['value'] * 1.001,
                    self.learning_params['discount']['max']
                )
            elif completion_trend > 0:  # Worsening completion times
                self.learning_params['discount']['value'] *= 0.999
            
            # Ensure parameters stay within bounds
            for param in self.learning_params.values():
                param['value'] = max(param['min'], min(param['max'], param['value']))
                param['history'].append(param['value'])
                if len(param['history']) > self.performance_window['size']:
                    param['history'] = param['history'][-self.performance_window['size']:]
            
            # Update last adjustment time
            self.performance_window['last_adjustment'] = current_time
            
            # Update DQN parameters
            self.dqn.learning_rate = self.learning_params['learning_rate']['value']
            self.dqn.gamma = self.learning_params['discount']['value']
            
        except Exception as e:
            print(f"Warning: Error adjusting learning parameters: {e}")

    def _calculate_trend(self, values, window_size=20):
        """Calculate trend direction using linear regression"""
        if len(values) < window_size:
            return 0
            
        recent_values = values[-window_size:]
        x = np.arange(window_size)
        y = np.array(recent_values)
        
        try:
            slope, _ = np.polyfit(x, y, 1)
            return slope
        except:
            return 0

    def _calculate_task_score(self, task, robot):
        """Calculate comprehensive task score for allocation"""
        score = 0.0
        
        try:
            # Check if path exists to task
            path = self.game.astar.find_path(
                (robot.x, robot.y),
                task.get_position(),
                robot=robot
            )
            
            if not path:
                print(f"No path found from Robot {robot.id} to task at {task.get_position()}")
                return 0.0  # Task is not accessible
            
            # 1. Base Priority Score (25% weight)
            priority_score = task.priority * 10.0
            score += 0.25 * priority_score
            
            # 2. Distance Factor (20% weight)
            distance = len(path)  # Use actual path length instead of Manhattan distance
            max_distance = GRID_SIZE * 2  # Maximum possible distance
            distance_score = (max_distance - distance) / max_distance * 10.0
            score += 0.2 * distance_score
            
            # 3. Path Efficiency (15% weight)
            direct_distance = robot.manhattan_distance((robot.x, robot.y), task.get_position())
            path_efficiency = direct_distance / distance if distance > 0 else 0
            path_score = 10.0 * path_efficiency
            score += 0.15 * path_score
            
            # 4. Urgency Factor (20% weight)
            waiting_time = task.get_waiting_time()
            urgency_score = min(waiting_time / 10.0, 1.0) * 10.0
            score += 0.2 * urgency_score
            
            # 5. Deadlock Avoidance (20% weight)
            deadlock_score = 10.0
            # Check for potential deadlocks along the path
            for pos in path:
                # Count nearby robots
                nearby_robots = 0
                for other_robot in self.game.robots:
                    if other_robot != robot:
                        dist = robot.manhattan_distance(pos, (other_robot.x, other_robot.y))
                        if dist < 2:  # Close proximity
                            nearby_robots += 1
                            # Check if other robot is also moving
                            if not other_robot.path:
                                deadlock_score -= 2.0  # Penalize paths near stationary robots
                
                # Heavy penalty for congested areas
                if nearby_robots > 1:
                    deadlock_score -= nearby_robots * 1.5
            
            deadlock_score = max(0.0, deadlock_score)  # Ensure non-negative
            score += 0.2 * deadlock_score
            
            print(f"Task ({task.x}, {task.y}) scores - Priority: {priority_score}, Distance: {distance_score}, "
                  f"Path: {path_score}, Urgency: {urgency_score}, Deadlock: {deadlock_score}")  # Debug print
            
            return score
            
        except Exception as e:
            print(f"Error in _calculate_task_score: {e}")  # Debug print
            return 0.0