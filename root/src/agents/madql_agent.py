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
        self.learning_stats = {
            'episode_rewards': [],
            'q_value_history': [],
            'bid_history': [],
            'task_distribution': {},
            'completion_times': [],
            'congestion_levels': [],
            'bid_success_rate': [],  # Track successful vs unsuccessful bids
            'bid_values': []  # Track actual bid values chosen
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
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.max_priority = 1.0
        
        # Bid-specific parameters
        self.bid_range = 10  # Number of discrete bid values
        self.min_bid = 0.0
        self.max_bid = 1.0
        self.bid_increment = (self.max_bid - self.min_bid) / self.bid_range
        
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
        
        # Initialize tracking for all potential agents
        for i in range(self.max_pool_size):
            self.agent_stats[i] = {
                'last_active': None,
                'total_tasks': 0,
                'avg_reward': 0.0,
                'experience_count': 0
            }
        
        # Initialize active agents based on current robots
        self._update_active_agents()
        
        # Maintain experience buffers per agent
        self.agent_experiences = {i: [] for i in range(self.max_pool_size)}
        self.max_experiences_per_agent = 10000
        
        self.prev_states = [None] * len(game.robots)
        self.prev_actions = [None] * len(game.robots)
        self.prev_bids = {}  # Track previous bids for each robot
        
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
        
    def get_state(self, robot):
        """Enhanced state representation with bid-specific features"""
        robot_idx = self.game.robots.index(robot)
        state = []
        
        # Local state components (existing code remains the same until local task features)
        state.extend([robot.x/GRID_SIZE, robot.y/GRID_SIZE])
        
        # 3x3 grid around robot
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                x, y = robot.x + dx, robot.y + dy
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    cell = self.game.grid[y][x]
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
        
        # Global state components (existing code)
        state.extend([
            len(self.game.tasks)/MAX_TASKS,
            sum(t.priority for t in self.game.tasks)/(3 * MAX_TASKS),
            self._calculate_global_congestion()/10.0
        ])
        
        # Other robots' info (existing code)
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
            return 0.5  # Default to neutral value
        history = self.learning_stats['bid_success_rate'][robot.id]
        if not history:
            return 0.5
        return sum(history[-100:]) / len(history[-100:])  # Last 100 bids
    
    def get_competition_level(self, robot):
        """Calculate competition level for tasks"""
        if not self.game.tasks:
            return 0.0
        unassigned_robots = len([r for r in self.game.robots if not r.target])
        return min(1.0, unassigned_robots / len(self.game.tasks))
    
    def calculate_bid(self, robot, available_tasks=None):
        """Enhanced bid calculation with collaboration awareness"""
        if available_tasks is None:
            available_tasks = self.get_available_tasks(robot)
            
        if not available_tasks:
            return None, 0.0
            
        try:
            # Share and process team information before bidding
            self.share_information(robot)
            
            # Get current state
            state = self.get_state(robot)
            
            # Get Q-values for bid actions
            q_values = self.dqn.get_q_values(state)
            
            # Use epsilon-greedy for bid selection
            if random.random() < EPSILON:
                bid_action = random.randrange(self.action_size)
            else:
                bid_action = torch.argmax(q_values).item()
            
            # Convert action to bid value
            bid_value = self.min_bid + bid_action * self.bid_increment
            
            # Adjust bid based on shared information
            bid_value = self._adjust_bid_with_shared_info(bid_value, robot, available_tasks)
            
            # Store bid for learning
            self.prev_bids[robot.id] = bid_value
            
            # Choose task considering team context
            chosen_task = self._choose_task_with_collaboration(available_tasks, bid_value, robot)
            
            # Track bid history
            self.learning_stats['bid_history'].append(bid_value)
            self.learning_stats['bid_values'].append(bid_value)
            
            return chosen_task, bid_value
            
        except Exception as e:
            print(f"Warning: Error calculating bid: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0
    
    def get_reward(self, robot, old_state, action, new_state):
        """Simplified reward function focusing on key performance metrics and team collaboration"""
        if action is None:
            return -1.0  # Small penalty for no action
        
        reward = 0.0
        robot_idx = self.game.robots.index(robot)
        
        if isinstance(action, Task):
            # 1. Task Completion Reward (Primary Component)
            if robot.target == action:  # Successful task assignment
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
            
            # 2. Distance Optimization (Secondary Component)
            current_dist = robot.manhattan_distance((robot.x, robot.y), action.get_position())
            if hasattr(robot, 'prev_distance'):
                # Reward getting closer, penalize getting further
                distance_change = robot.prev_distance - current_dist
                reward += distance_change * 0.5  # Small continuous reward/penalty
            robot.prev_distance = current_dist
            
            # 3. Path Efficiency (Penalty Component)
            path = self.game.astar.find_path((robot.x, robot.y), action.get_position())
            if path:
                # Penalize path length relative to manhattan distance
                path_efficiency = current_dist / len(path)
                if path_efficiency < 0.8:  # If path is significantly longer than direct distance
                    reward -= 2.0 * (1.0 - path_efficiency)  # Up to -2 points
                
                # Penalize congestion
                congestion = self._calculate_path_congestion(path)
                if congestion > 0:
                    congestion_ratio = congestion / len(path)
                    reward -= 3.0 * congestion_ratio  # Up to -3 points for heavy congestion
            else:
                reward -= 5.0  # Significant penalty for no valid path
            
            # 4. Team Collaboration (Bonus/Penalty Component)
            # Calculate team dispersion (reward even distribution of robots)
            robot_positions = [(r.x, r.y) for r in self.game.robots]
            avg_robot_distance = 0
            if len(robot_positions) > 1:
                distances = []
                for i, pos1 in enumerate(robot_positions):
                    for pos2 in robot_positions[i+1:]:
                        distances.append(abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))
                avg_robot_distance = sum(distances) / len(distances)
                dispersion_factor = min(1.0, avg_robot_distance / (GRID_SIZE / 2))
                reward += 2.0 * dispersion_factor  # Up to 2 points for good dispersion
            
            # Penalize task conflicts (multiple robots targeting same area)
            conflict_count = 0
            target_area = set((x, y) for x in range(max(0, action.x-1), min(GRID_SIZE, action.x+2))
                                   for y in range(max(0, action.y-1), min(GRID_SIZE, action.y+2)))
            for other_robot in self.game.robots:
                if other_robot != robot and other_robot.target:
                    other_target = other_robot.target
                    other_area = set((x, y) for x in range(max(0, other_target.x-1), min(GRID_SIZE, other_target.x+2))
                                          for y in range(max(0, other_target.y-1), min(GRID_SIZE, other_target.y+2)))
                    if target_area & other_area:  # If areas overlap
                        conflict_count += 1
            
            if conflict_count > 0:
                reward -= 2.0 * conflict_count  # -2 points per conflict
        
        # Normalize reward to a reasonable range (-10 to +50)
        reward = max(-10.0, min(50.0, reward))
        
        # Track reward for this episode
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
        """Enhanced update handling dynamic agents"""
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
            
            # Update task statistics if applicable
            if isinstance(action, Task) and robot.target == action:
                self.agent_stats[robot_idx]['total_tasks'] += 1
                self.total_tasks_completed += 1
                
                if hasattr(robot, 'last_task_start'):
                    completion_time = time.time() - robot.last_task_start
                    self.learning_stats['completion_times'].append(completion_time)
            
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
        return [task for task in self.game.tasks 
                if self.game.grid[task.y][task.x].value == CellType.TASK.value]
    
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
                    self.game.grid[y][x] == CellType.OBSTACLE):
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
        
        # Deactivate agents for removed robots
        agents_to_deactivate = self.active_agents - current_robot_ids
        for agent_id in agents_to_deactivate:
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