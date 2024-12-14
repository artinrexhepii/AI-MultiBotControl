import random
import numpy as np
import pickle

# Define the Environment Class
class Environment:
    def __init__(self, grid_size=10, num_robots=2):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.robots = []
        self.robot_paths = []  # Predefined paths
        self.initialize_robots()

    def initialize_robots(self):
        """Initialize robots with random start positions and predefined paths."""
        for i in range(self.num_robots):
            path = self.generate_random_path()
            
            self.robot_paths.append(path)
            self.robots.append(Robot(path, i + 1))  # Assign priority based on index
        

    def generate_random_path(self):
        """Generate a realistic random path for robots."""
        start = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        path = [start]
        for _ in range(random.randint(5, 10)):  # Path length of 5-10 steps
            next_pos = (max(0, min(self.grid_size - 1, path[-1][0] + random.choice([-1, 0, 1]))),
                        max(0, min(self.grid_size - 1, path[-1][1] + random.choice([-1, 0, 1]))))
            path.append(next_pos)
        
        return path
    

    def get_state(self):
        """Represent the state as a tuple of robot positions."""
        positions = tuple(robot.get_current_position() for robot in self.robots)
        
        return positions


    def check_collisions(self):
        """Check if robots have collided at the current step."""
        current_positions = {}
        collisions = []

        for i, robot in enumerate(self.robots):
            current_pos = robot.get_current_position()
            
            if current_pos in current_positions:
                # Record collision between robots
                collisions.append((i, current_positions[current_pos]))
            else:
                current_positions[current_pos] = i  # Mark this position as occupied

        return collisions

    def step(self, actions):
        """Take a step in the environment based on robot actions."""
        for i, action in enumerate(actions):
            if action == 0:  # Move
                self.robots[i].move()
                
            elif action == 1:  # Wait
                pass  # Robot remains in place

        collisions = self.check_collisions()
        state = self.get_state()
        return state, collisions


# Define the Robot Class
class Robot:
    def __init__(self, path, priority):
        self.path = path
        self.current_step = 0
        self.priority = priority

    def get_current_position(self):
        """Get the robot's current position."""
        return self.path[self.current_step]

    def has_next_step(self):
        """Check if the robot has a next step."""
        return self.current_step + 1 < len(self.path)

    def move(self):
        """Move the robot to the next step."""
        if self.has_next_step():
            self.current_step += 1
            


# Define the Q-learning Model
class QLearningModel:
    def __init__(self, num_actions=2, num_states=1000, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.num_actions = num_actions  # Move or Wait
        self.num_states = num_states  # State space (can vary based on complexity)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state_index):
        """Epsilon-greedy action selection with exploration decay."""
        if random.uniform(0, 1) < self.epsilon:  # Exploration
            return random.choice([0, 1])  # Randomly choose between moving or waiting
        else:  # Exploitation
            return np.argmax(self.q_table[state_index])  # Exploit the best action

    def update_q_table(self, state_index, action, reward, next_state_index):
        """Update the Q-table using the Q-learning formula."""
        best_next_action = np.max(self.q_table[next_state_index])
        self.q_table[state_index][action] += self.learning_rate * (
            reward + self.discount_factor * best_next_action - self.q_table[state_index][action]
        )
        

    def save_q_table(self, filename="q_table.pkl"):
        """Save the Q-table to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename="q_table.pkl"):
        """Load the Q-table from a file."""
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)


def train(episodes):
    
    state_to_index = {}  # Mapping from states to indices
    index_counter = 0

    def get_state_index(state):
        nonlocal index_counter
        if state not in state_to_index:
            state_to_index[state] = index_counter
            index_counter += 1
        return state_to_index[state]

    for episode in range(episodes):
        environment = Environment(grid_size=10, num_robots=3)
        model = QLearningModel(num_actions=2, num_states=1000)
        state = environment.get_state()
        state_index = get_state_index(state)
        total_reward = 0

        # Decay epsilon over time to encourage more exploitation
        model.epsilon = max(0.01, model.epsilon * 0.995)  # Decay epsilon to a minimum of 0.01

        for step in range(10):  # Limit to 100 steps per episode
            # Choose actions for each robot based on the current state
            actions = [model.choose_action(state_index) for _ in environment.robots]
            
            # Update environment by performing the chosen actions
            next_state, collisions = environment.step(actions)
            next_state_index = get_state_index(next_state)

            # Calculate reward based on collisions
            reward = 0
            if collisions:
                # Penalize collisions, giving higher penalties to higher-priority robots
                for colliding_robot_1, colliding_robot_2 in collisions:
                    if environment.robots[colliding_robot_1].priority > environment.robots[colliding_robot_2].priority:
                        reward -= 15  # Higher penalty for higher-priority robot collision
                    else:
                        reward -= 10  # Penalty for a collision
            else:
                reward = 1  # Positive reward for no collision

            # Update the Q-table for each robot based on the actions
            for i, action in enumerate(actions):
                model.update_q_table(state_index, action, reward, next_state_index)

            # Set the new state for the next iteration
            state = next_state
            state_index = next_state_index
            total_reward += reward

        # Optionally print the reward for each episode
        print(f"Episode {episode + 1}: Total reward = {total_reward}")
        


    model.save_q_table("q_table.pkl")  # Save trained Q-table after training



# Main function for Training
def main_train():
    
    train(episodes=100)


# Run the training phase
if __name__ == "__main__":
    main_train()