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
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        self.global_net = nn.Sequential(
            nn.Linear(state_size - (state_size // 2), 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        # Combine local and global features
        self.combine_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, action_size)
        )
        
    def forward(self, x):
        # Split input into local and global components
        local_size = x.shape[1] // 2
        local_input = x[:, :local_size]
        global_input = x[:, local_size:]
        
        # Process local and global features
        local_features = self.local_net(local_input)
        global_features = self.global_net(global_input)
        
        # Combine features
        combined = torch.cat([local_features, global_features], dim=1)
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
        
    def _initialize_networks(self):
        """Initialize network weights properly"""
        for net in self.policy_nets + self.target_nets + [self.critic_net]:
            for layer in net.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
    
    def remember(self, state, action, reward, next_state, priority):
        """Store experience in prioritized replay buffer"""
        self.memory.push(state, action, reward, next_state, priority)
    
    def train_step(self, beta):
        """Enhanced training step with prioritized experience replay"""
        if len(self.memory) < self.batch_size:
            return
            
        try:
            # Sample batch with importance sampling
            batch, weights, indices = self.memory.sample(self.batch_size, beta)
            if batch is None:
                return
                
            states, actions, rewards, next_states = batch
            weights = torch.FloatTensor(weights)
            
            total_policy_loss = 0
            total_q_value = 0
            new_priorities = []
            
            # Train individual policy networks
            for i in range(self.num_agents):
                # Current Q values
                current_q = self.policy_nets[i](states)
                current_q = current_q.gather(1, actions.unsqueeze(1))
                
                # Next Q values with Double DQN
                with torch.no_grad():
                    next_actions = self.policy_nets[i](next_states).max(1)[1].unsqueeze(1)
                    next_q = self.target_nets[i](next_states)
                    next_q = next_q.gather(1, next_actions)
                
                # Compute target Q values
                target_q = rewards.unsqueeze(1) + self.gamma * next_q
                
                # Compute weighted Huber loss
                loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
                weighted_loss = (loss * weights.unsqueeze(1)).mean()
                
                total_policy_loss += weighted_loss.item()
                total_q_value += current_q.mean().item()
                
                # Store new priorities
                new_priorities.extend(loss.detach().abs().cpu().numpy())
                
                # Optimize policy network
                self.optimizers[i].zero_grad()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_nets[i].parameters(), 1.0)
                self.optimizers[i].step()
                
                # Soft update target network
                self.soft_update(self.target_nets[i], self.policy_nets[i])
            
            # Update priorities in replay buffer
            self.memory.update_priorities(indices, new_priorities)
            
            # Train centralized critic
            critic_q = self.critic_net(states)
            critic_q = critic_q.gather(1, actions.sum(1, keepdim=True))
            
            with torch.no_grad():
                next_critic_q = self.critic_net(next_states)
                next_critic_actions = next_critic_q.max(1)[1].unsqueeze(1)
                next_critic_value = next_critic_q.gather(1, next_critic_actions)
                critic_target = rewards.sum(1, keepdim=True) + \
                               self.gamma * next_critic_value
            
            # Compute weighted critic loss
            critic_loss = F.smooth_l1_loss(critic_q, critic_target, reduction='none')
            weighted_critic_loss = (critic_loss * weights.unsqueeze(1)).mean()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            weighted_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 1.0)
            self.critic_optimizer.step()
            
            # Update statistics
            self.training_stats['policy_losses'].append(total_policy_loss / self.num_agents)
            self.training_stats['critic_losses'].append(weighted_critic_loss.item())
            self.training_stats['avg_q_values'].append(total_q_value / self.num_agents)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.training_stats['epsilon_history'].append(self.epsilon)
            
        except Exception as e:
            print(f"Warning: Error during training step: {e}")
    
    def get_q_values(self, state):
        """Get Q-values for all actions given a state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return self.policy_nets[0](state_tensor).squeeze()
    
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
        robot = game.robots[robot_idx]
        state = []
        
        # Robot's normalized position
        state.extend([robot.x/GRID_SIZE, robot.y/GRID_SIZE])
        
        # Other robots' positions and targets
        for i, other_robot in enumerate(game.robots):
            if i != robot_idx:
                state.extend([
                    other_robot.x/GRID_SIZE,
                    other_robot.y/GRID_SIZE
                ])
                if other_robot.target:
                    state.extend([
                        other_robot.target.x/GRID_SIZE,
                        other_robot.target.y/GRID_SIZE,
                        other_robot.target.priority/3
                    ])
                else:
                    state.extend([0, 0, 0])
        
        # Available tasks information
        task_info = np.zeros(GRID_SIZE * GRID_SIZE * 4)  # x, y, priority, waiting_time
        for task in game.tasks:
            idx = (task.y * GRID_SIZE + task.x) * 4
            task_info[idx:idx+4] = [
                task.x/GRID_SIZE,
                task.y/GRID_SIZE,
                task.priority/3,
                min(task.get_waiting_time()/10.0, 1.0)
            ]
        state.extend(task_info)
        
        # Obstacle information
        obstacle_info = np.zeros(GRID_SIZE * GRID_SIZE)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if game.grid[y][x] == CellType.OBSTACLE:
                    obstacle_info[y * GRID_SIZE + x] = 1
        state.extend(obstacle_info)
        
        return torch.FloatTensor(state)
    
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