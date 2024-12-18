import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from core.constants import GRID_SIZE
from core.entities import CellType
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # Split state processing into local and global pathways
        self.local_net = nn.Sequential(
            nn.Linear(state_size // 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.global_net = nn.Sequential(
            nn.Linear(state_size - (state_size // 2), 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Combine local and global features
        self.combine_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, action_size)
        )
        
        # Layer normalization instead of batch normalization
        self.local_norm = nn.LayerNorm(64)
        self.global_norm = nn.LayerNorm(64)
        self.combine_norm = nn.LayerNorm(128)
        
    def forward(self, x):
        # Split input into local and global components
        local_size = x.shape[1] // 2
        local_input = x[:, :local_size]
        global_input = x[:, local_size:]
        
        # Process local and global features
        local_features = self.local_net(local_input)
        global_features = self.global_net(global_input)
        
        # Apply layer normalization
        local_features = self.local_norm(local_features)
        global_features = self.global_norm(global_features)
        
        # Combine features
        combined = torch.cat([local_features, global_features], dim=1)
        combined = self.combine_norm(combined)
        
        return self.combine_net(combined)

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def push(self, state, action, reward, next_state, priority):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, beta):
        if len(self.buffer) == 0:
            return None
            
        # Calculate sampling probabilities
        probs = self.priorities[:len(self.buffer)] ** beta
        probs /= probs.sum()
        
        # Sample indices and calculate importance weights
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get batch
        batch = [self.buffer[idx] for idx in indices]
        states = torch.FloatTensor(np.array([x[0] for x in batch]))
        actions = torch.LongTensor(np.array([x[1] for x in batch]))
        rewards = torch.FloatTensor(np.array([x[2] for x in batch]))
        next_states = torch.FloatTensor(np.array([x[3] for x in batch]))
        
        return (states, actions, rewards, next_states), weights, indices
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.buffer)

class CentralizedDQN:
    def __init__(self, state_size, action_size, num_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.memory = PrioritizedReplayBuffer(10000)
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.tau = 0.001
        
        # Initialize performance metrics
        self.total_tasks_completed = 0
        self.total_rewards = 0
        self.episode_count = 0
        
        # Initialize networks
        self.policy_nets = [DQN(state_size, action_size) for _ in range(num_agents)]
        self.target_nets = [DQN(state_size, action_size) for _ in range(num_agents)]
        self.critic_net = DQN(state_size, action_size)
        
        # Initialize optimizers with gradient clipping
        self.optimizers = [
            optim.Adam(net.parameters(), lr=self.learning_rate)
            for net in self.policy_nets
        ]
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.learning_rate)
        
        # Initialize networks properly
        self._initialize_networks()
        
        # Training statistics
        self.training_stats = {
            'policy_losses': [],
            'critic_losses': [],
            'avg_q_values': [],
            'epsilon_history': []
        }
        
        # Set networks to evaluation mode initially
        self._set_eval_mode()
    
    def _set_eval_mode(self):
        """Set all networks to evaluation mode"""
        for net in self.policy_nets + self.target_nets + [self.critic_net]:
            net.eval()
    
    def _set_train_mode(self):
        """Set all networks to training mode"""
        for net in self.policy_nets + self.target_nets + [self.critic_net]:
            net.train()
    
    def train_step(self, beta):
        """Enhanced training step with prioritized experience replay"""
        if len(self.memory) < self.batch_size:
            return
            
        try:
            # Set networks to training mode
            self._set_train_mode()
            
            # Sample batch with importance sampling
            batch, weights, indices = self.memory.sample(self.batch_size, beta)
            if batch is None:
                return
                
            states, actions, rewards, next_states = batch
            
            # Ensure proper tensor shapes
            states = torch.FloatTensor(np.array(states))  # [batch_size, state_size]
            next_states = torch.FloatTensor(np.array(next_states))  # [batch_size, state_size]
            actions = torch.LongTensor(np.array(actions)).view(-1, 1)  # [batch_size, 1]
            rewards = torch.FloatTensor(np.array(rewards)).view(-1, 1)  # [batch_size, 1]
            weights = torch.FloatTensor(weights).view(-1, 1)  # [batch_size, 1]
            
            total_policy_loss = 0
            total_q_value = 0
            new_priorities = []
            
            # Train individual policy networks
            for i in range(self.num_agents):
                try:
                    # Current Q values
                    current_q = self.policy_nets[i](states)  # [batch_size, action_size]
                    current_q = current_q.gather(1, actions)  # [batch_size, 1]
                    
                    # Next Q values with Double DQN
                    with torch.no_grad():
                        next_q_policy = self.policy_nets[i](next_states)  # [batch_size, action_size]
                        next_actions = next_q_policy.max(1)[1].view(-1, 1)  # [batch_size, 1]
                        next_q = self.target_nets[i](next_states)  # [batch_size, action_size]
                        next_q = next_q.gather(1, next_actions)  # [batch_size, 1]
                    
                    # Compute target Q values
                    target_q = rewards + self.gamma * next_q  # [batch_size, 1]
                    
                    # Compute weighted Huber loss
                    loss = F.smooth_l1_loss(current_q, target_q, reduction='none')  # [batch_size, 1]
                    weighted_loss = (loss * weights).mean()
                    
                    total_policy_loss += weighted_loss.item()
                    total_q_value += current_q.mean().item()
                    
                    # Store new priorities
                    new_priorities.extend(loss.detach().cpu().numpy().flatten())
                    
                    # Optimize policy network
                    self.optimizers[i].zero_grad()
                    weighted_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_nets[i].parameters(), 1.0)
                    self.optimizers[i].step()
                    
                    # Soft update target network
                    self.soft_update(self.target_nets[i], self.policy_nets[i])
                except Exception as e:
                    print(f"Warning: Error training policy network {i}: {e}")
                    continue
            
            # Update priorities in replay buffer
            if new_priorities:
                self.memory.update_priorities(indices, np.array(new_priorities))
            
            try:
                # Train centralized critic
                critic_q = self.critic_net(states)  # [batch_size, action_size]
                critic_q = critic_q.gather(1, actions)  # [batch_size, 1]
                
                with torch.no_grad():
                    next_critic_q = self.critic_net(next_states)  # [batch_size, action_size]
                    next_critic_actions = next_critic_q.max(1)[1].view(-1, 1)  # [batch_size, 1]
                    next_critic_value = next_critic_q.gather(1, next_critic_actions)  # [batch_size, 1]
                    critic_target = rewards + self.gamma * next_critic_value  # [batch_size, 1]
                
                # Compute weighted critic loss
                critic_loss = F.smooth_l1_loss(critic_q, critic_target, reduction='none')  # [batch_size, 1]
                weighted_critic_loss = (critic_loss * weights).mean()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                weighted_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 1.0)
                self.critic_optimizer.step()
                
                # Update statistics
                self.training_stats['critic_losses'].append(weighted_critic_loss.item())
            except Exception as e:
                print(f"Warning: Error training critic network: {e}")
            
            # Update statistics
            if total_policy_loss > 0:
                self.training_stats['policy_losses'].append(total_policy_loss / self.num_agents)
                self.training_stats['avg_q_values'].append(total_q_value / self.num_agents)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.training_stats['epsilon_history'].append(self.epsilon)
            
        except Exception as e:
            print(f"Warning: Error during training step: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Set networks back to evaluation mode
            self._set_eval_mode()
    
    def get_q_values(self, state):
        """Get Q-values for all actions given a state"""
        try:
            with torch.no_grad():
                # Ensure state is a 2D tensor [batch_size, state_size]
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state)
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                
                # Get Q-values and ensure output is 1D
                q_values = self.policy_nets[0](state)
                if q_values.dim() == 2 and q_values.size(0) == 1:
                    q_values = q_values.squeeze(0)
                return q_values
        except Exception as e:
            print(f"Warning: Error getting Q-values: {e}")
            return torch.zeros(self.action_size)
    
    def soft_update(self, target_net, policy_net):
        """Soft update model parameters"""
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def get_training_stats(self):
        """Get training statistics"""
        if not self.training_stats['policy_losses']:
            return None
            
        window = 100  # Moving average window
        return {
            'avg_policy_loss': sum(self.training_stats['policy_losses'][-window:]) / min(window, len(self.training_stats['policy_losses'])),
            'avg_critic_loss': sum(self.training_stats['critic_losses'][-window:]) / min(window, len(self.training_stats['critic_losses'])),
            'avg_q_value': sum(self.training_stats['avg_q_values'][-window:]) / min(window, len(self.training_stats['avg_q_values'])),
            'current_epsilon': self.epsilon
        }
    
    def get_actions(self, states):
        """Get actions for all agents using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return [random.randrange(self.action_size) for _ in range(self.num_agents)]
        
        try:
            states_tensor = torch.FloatTensor(np.array(states))
            actions = []
            
            with torch.no_grad():
                for i in range(self.num_agents):
                    q_values = self.policy_nets[i](states_tensor[i].unsqueeze(0))
                    actions.append(q_values.argmax().item())
            
            return actions
        except Exception as e:
            print(f"Warning: Error during action selection: {e}")
            return [random.randrange(self.action_size) for _ in range(self.num_agents)]
    
    def get_state_tensor(self, game, robot_idx):
        """Convert game state to tensor for a specific robot"""
        try:
            robot = game.robots[robot_idx]
            state = []
            
            # Robot's normalized position
            state.extend([float(robot.x)/GRID_SIZE, float(robot.y)/GRID_SIZE])
            
            # Other robots' positions and targets
            for i, other_robot in enumerate(game.robots):
                if i != robot_idx:
                    state.extend([
                        float(other_robot.x)/GRID_SIZE,
                        float(other_robot.y)/GRID_SIZE
                    ])
                    if other_robot.target:
                        try:
                            state.extend([
                                float(other_robot.target.x)/GRID_SIZE,
                                float(other_robot.target.y)/GRID_SIZE,
                                float(other_robot.target.priority)/3.0
                            ])
                        except (AttributeError, TypeError, ValueError) as e:
                            print(f"Warning: Error processing robot target: {e}")
                            state.extend([0.0, 0.0, 0.0])
                    else:
                        state.extend([0.0, 0.0, 0.0])
            
            # Available tasks information
            task_info = np.zeros(GRID_SIZE * GRID_SIZE * 4, dtype=np.float32)  # x, y, priority, waiting_time
            for task in game.tasks:
                try:
                    x = int(float(task.x))
                    y = int(float(task.y))
                    priority = float(task.priority)
                    waiting_time = float(task.get_waiting_time())
                    
                    idx = (y * GRID_SIZE + x) * 4
                    if 0 <= idx < len(task_info) - 3:  # Ensure we have space for all 4 values
                        task_info[idx:idx+4] = [
                            float(x)/GRID_SIZE,
                            float(y)/GRID_SIZE,
                            priority/3.0,
                            min(waiting_time/10.0, 1.0)
                        ]
                except (AttributeError, TypeError, ValueError, IndexError) as e:
                    print(f"Warning: Error processing task in state tensor: {e}")
                    continue
            
            state.extend(task_info.tolist())
            
            # Obstacle information
            obstacle_info = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    if game.grid[y][x] == CellType.OBSTACLE:
                        idx = y * GRID_SIZE + x
                        if 0 <= idx < len(obstacle_info):
                            obstacle_info[idx] = 1.0
            
            state.extend(obstacle_info.tolist())
            
            # Convert to tensor and ensure correct shape
            state_tensor = torch.FloatTensor(state)
            if state_tensor.dim() == 0:
                state_tensor = state_tensor.unsqueeze(0)
            
            return state_tensor
            
        except Exception as e:
            print(f"Warning: Error creating state tensor: {e}")
            import traceback
            traceback.print_exc()  # Print the full stack trace for debugging
            # Return a zero tensor of the correct size as fallback
            return torch.zeros(self.state_size)
    
    def save(self, path):
        """Save models"""
        save_dict = {
            'policy_nets': [net.state_dict() for net in self.policy_nets],
            'target_nets': [net.state_dict() for net in self.target_nets],
            'critic_net': self.critic_net.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(save_dict, path)
    
    def load(self, path):
        """Load models"""
        checkpoint = torch.load(path)
        for i, state_dict in enumerate(checkpoint['policy_nets']):
            self.policy_nets[i].load_state_dict(state_dict)
        for i, state_dict in enumerate(checkpoint['target_nets']):
            self.target_nets[i].load_state_dict(state_dict)
        self.critic_net.load_state_dict(checkpoint['critic_net'])
        self.epsilon = checkpoint['epsilon'] 
    
    def _initialize_networks(self):
        """Initialize network weights properly"""
        for net in self.policy_nets + self.target_nets + [self.critic_net]:
            for layer in net.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def remember(self, state, action, reward, next_state, priority):
        """Store experience in memory"""
        try:
            # Convert action to index if it's a Task object
            if hasattr(action, 'x') and hasattr(action, 'y'):
                action_idx = action.y * GRID_SIZE + action.x
            else:
                action_idx = int(action)  # Ensure action is an integer
            
            # Ensure state and next_state are numpy arrays
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            if isinstance(next_state, torch.Tensor):
                next_state = next_state.cpu().numpy()
            
            self.memory.push(state, action_idx, reward, next_state, priority)
        except Exception as e:
            print(f"Warning: Error storing experience: {e}")
            import traceback
            traceback.print_exc()