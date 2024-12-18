import random
import time
from src.core.constants import GRID_SIZE, MAX_TASKS, TASK_GEN_CHANCE
from src.core.entities import CellType, Task

class Grid:
    def __init__(self, game=None):
        self.game = game
        self.grid = [[CellType.EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.tasks = []
        self.moving_obstacles = []
        
    def get_cell(self, x, y):
        """Get cell type at given coordinates"""
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            return self.grid[y][x]
        return None
        
    def set_cell(self, x, y, cell_type):
        """Set cell type at given coordinates"""
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            self.grid[y][x] = cell_type
            return True
        return False
        
    def find_empty_cells(self):
        """Find all empty cells in the grid"""
        return [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
                if self.grid[y][x] == CellType.EMPTY]
                
    def add_robot(self, x, y, robot):
        """Add robot to grid"""
        if self.get_cell(x, y) == CellType.EMPTY:
            self.set_cell(x, y, CellType.ROBOT)
            robot.x, robot.y = x, y
            return True
        return False
        
    def move_robot(self, robot, new_x, new_y):
        """Move robot to new position"""
        # Check bounds
        if not (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE):
            print(f"Move failed: Position ({new_x}, {new_y}) out of bounds")  # Debug print
            return False
        
        # Get cell types
        current_cell = self.get_cell(robot.x, robot.y)
        target_cell = self.get_cell(new_x, new_y)
        
        print(f"Moving robot from ({robot.x}, {robot.y}) [{current_cell}] to ({new_x}, {new_y}) [{target_cell}]")  # Debug print
        
        # Allow movement to empty cells, tasks, and targets
        if target_cell in [CellType.EMPTY, CellType.TASK, CellType.TARGET]:
            # Check for other robots in target position
            if self.game:
                for other_robot in self.game.robots:
                    if other_robot != robot and (other_robot.x, other_robot.y) == (new_x, new_y):
                        print(f"Move failed: Position ({new_x}, {new_y}) occupied by another robot")  # Debug print
                        return False
            
            # Clear old position
            self.set_cell(robot.x, robot.y, CellType.EMPTY)
            
            # Update robot position
            robot.x, robot.y = new_x, new_y
            
            # Set new position to robot
            self.set_cell(new_x, new_y, CellType.ROBOT)
            
            print(f"Move successful: Robot now at ({robot.x}, {robot.y})")  # Debug print
            return True
        else:
            print(f"Move failed: Target cell type {target_cell} not allowed")  # Debug print
            return False
        
    def add_task(self, x, y, priority=None):
        """Add task to grid"""
        if self.get_cell(x, y) == CellType.EMPTY:
            # Generate random priority if not specified
            if priority is None:
                priority = random.randint(1, 3)
                
            # Create and add task
            task = Task(x, y, priority)
            self.tasks.append(task)
            self.set_cell(x, y, CellType.TASK)
            return task
        return None
        
    def remove_task(self, task):
        """Remove task from grid"""
        if task in self.tasks:
            # Check if there's a robot at this position
            robot_at_position = False
            if self.game:
                for robot in self.game.robots:
                    if (robot.x, robot.y) == (task.x, task.y):
                        robot_at_position = True
                        break
            
            # Only set cell to empty if no robot is there
            if not robot_at_position:
                self.set_cell(task.x, task.y, CellType.EMPTY)
            
            # Remove task from tracking
            self.tasks.remove(task)
            print(f"Removed task at ({task.x}, {task.y}), robot_at_position: {robot_at_position}")  # Debug print
            return True
        return False
        
    def mark_task_assigned(self, task):
        """Mark task as assigned"""
        if task in self.tasks:
            self.set_cell(task.x, task.y, CellType.TARGET)
            return True
        return False
        
    def generate_random_task(self):
        """Generate random task with improved placement"""
        # Check if we should generate a task
        if len(self.tasks) >= MAX_TASKS or random.random() > TASK_GEN_CHANCE:
            return None
        
        # Find empty cells that are not too close to other tasks
        empty_cells = []
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.grid[y][x] == CellType.EMPTY:
                    # Check distance to other tasks
                    min_distance = float('inf')
                    for task in self.tasks:
                        dist = abs(x - task.x) + abs(y - task.y)
                        min_distance = min(min_distance, dist)
                    
                    if min_distance > 2 or min_distance == float('inf'):  # Allow first task or maintain spacing
                        empty_cells.append((x, y))
        
        if empty_cells:
            x, y = random.choice(empty_cells)
            # Higher chance for lower priority tasks
            priority = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
            task = self.add_task(x, y, priority)
            if task:
                print(f"Generated task at ({x}, {y}) with priority {priority}")  # Debug print
                return task
        return None
        
    def add_obstacle(self, x, y, moving=False):
        """Add obstacle to grid"""
        if self.get_cell(x, y) == CellType.EMPTY:
            self.set_cell(x, y, CellType.OBSTACLE)
            if moving:
                direction = random.choice([(0,1), (1,0), (0,-1), (-1,0)])
                self.moving_obstacles.append({
                    'x': x, 'y': y,
                    'dx': direction[0], 'dy': direction[1],
                    'last_move': time.time()
                })
            return True
        return False
        
    def update_moving_obstacles(self, current_time, move_delay):
        """Update positions of moving obstacles"""
        for obstacle in self.moving_obstacles[:]:
            if current_time - obstacle['last_move'] < move_delay:
                continue
                
            new_x = obstacle['x'] + obstacle['dx']
            new_y = obstacle['y'] + obstacle['dy']
            
            if (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and 
                self.grid[new_y][new_x] == CellType.EMPTY):
                # Update grid
                self.grid[obstacle['y']][obstacle['x']] = CellType.EMPTY
                self.grid[new_y][new_x] = CellType.OBSTACLE
                obstacle['x'] = new_x
                obstacle['y'] = new_y
                obstacle['last_move'] = current_time
            else:
                # Change direction if blocked
                obstacle['dx'], obstacle['dy'] = random.choice([(0,1), (1,0), (0,-1), (-1,0)])
                
    def clear(self):
        """Clear the grid"""
        self.grid = [[CellType.EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.tasks = []
        self.moving_obstacles = []
        
    def is_valid_path(self, path):
        """Check if path is valid (no obstacles)"""
        if not path:
            return False
        return all(self.get_cell(x, y) != CellType.OBSTACLE for x, y in path)
        
    def get_neighbors(self, x, y):
        """Get valid neighboring cells"""
        neighbors = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and
                self.grid[new_y][new_x] != CellType.OBSTACLE):
                neighbors.append((new_x, new_y))
        return neighbors 

    def get_task_at(self, x, y):
        """Get task at given position"""
        for task in self.tasks:
            if task.x == x and task.y == y:
                return task
        return None