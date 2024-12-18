import random
from src.core.constants import GRID_SIZE, MAX_TASKS, TASK_GEN_CHANCE
from src.core.entities import CellType, Task

class Grid:
    def __init__(self):
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
        if self.get_cell(new_x, new_y) == CellType.EMPTY:
            self.set_cell(robot.x, robot.y, CellType.EMPTY)
            self.set_cell(new_x, new_y, CellType.ROBOT)
            robot.x, robot.y = new_x, new_y
            return True
        return False
        
    def add_task(self, x, y, priority=None):
        """Add task to grid"""
        if self.get_cell(x, y) == CellType.EMPTY:
            if priority is None:
                priority = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
            task = Task(x, y, priority)
            self.grid[y][x] = CellType.TASK
            self.tasks.append(task)
            return task
        return None
        
    def remove_task(self, task):
        """Remove task from grid"""
        if task in self.tasks:
            self.grid[task.y][task.x] = CellType.EMPTY
            self.tasks.remove(task)
            return True
        return False
        
    def mark_task_assigned(self, task):
        """Mark task as assigned (target)"""
        if task in self.tasks:
            self.grid[task.y][task.x] = CellType.TARGET
            return True
        return False
        
    def generate_random_task(self):
        """Generate random task if conditions are met"""
        if len(self.tasks) < MAX_TASKS and random.random() < TASK_GEN_CHANCE:
            empty_cells = self.find_empty_cells()
            if empty_cells:
                x, y = random.choice(empty_cells)
                return self.add_task(x, y)
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