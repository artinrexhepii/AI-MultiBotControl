import pygame
import random
import time
import numpy as np
from src.core.constants import *
from src.core.entities import Robot, CellType, Task
from src.agents.astar import AStar
from src.ui.button import Button
from src.utils.metrics import PerformanceMetrics
from src.agents.RL.predict import predict_action  # Adjust path if necessary


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
        pygame.display.set_caption("Multi-Robot Control System")
        
        # Adjust grid size
        self.grid = [[CellType.EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.current_tool = None
        self.robots = []
        self.tasks = []
        self.moving_obstacles = []
        self.dynamic_tasks_enabled = True
        self.end_simulation = False
        self.start_time = None
        self.total_tasks_completed = 0
        self.auction_interval = 2.0
        self.last_auction_time = 0
        
        self.astar = AStar(self)
        self.clock = pygame.time.Clock()
        
        # Create buttons
        self.buttons = {
            'robot': Button(TOTAL_WIDTH - MENU_WIDTH + 20, 50, 160, 40, "Robot"),
            'obstacle': Button(TOTAL_WIDTH - MENU_WIDTH + 20, 100, 160, 40, "Obstacle"),
            'task': Button(TOTAL_WIDTH - MENU_WIDTH + 20, 150, 160, 40, "Task"),
            'random': Button(TOTAL_WIDTH - MENU_WIDTH + 20, 200, 160, 40, "Random Generate"),
            'play': Button(TOTAL_WIDTH - MENU_WIDTH + 20, 250, 160, 40, "Play"),
            'end': Button(TOTAL_WIDTH - MENU_WIDTH + 20, 300, 160, 40, "End")
        }
        
        self.running = True
        self.simulation_running = False
        self.performance_metrics = None
        self.status_messages = []
        self.max_messages = 8
        self.robot_counter = 0

    def add_status_message(self, message):
        self.status_messages.insert(0, message)
        if len(self.status_messages) > self.max_messages:
            self.status_messages.pop()

    def handle_click(self, pos):
        # Handle menu clicks
        for name, button in self.buttons.items():
            if button.rect.collidepoint(pos):
                if name == 'random':
                    self.generate_random()
                elif name == 'play':
                    self.simulation_running = not self.simulation_running
                    if self.simulation_running and self.start_time is None:
                        self.start_time = time.time()
                        for robot in self.robots:
                            robot.start_time = time.time()
                        # Reallocate tasks when simulation starts
                        self.reallocate_all_tasks()
                elif name == 'end':
                    self.end_simulation = True
                    self.dynamic_tasks_enabled = False
                else:
                    # Deselect all buttons except the clicked one
                    for btn in self.buttons.values():
                        btn.selected = False
                    button.selected = True
                    self.current_tool = name
                return

        # Handle grid clicks
        if pos[0] < WINDOW_SIZE:
            grid_x = pos[0] // CELL_SIZE
            grid_y = pos[1] // CELL_SIZE
            
            if self.current_tool == 'robot':
                if self.grid[grid_y][grid_x] == CellType.EMPTY:
                    self.grid[grid_y][grid_x] = CellType.ROBOT
                    
                    # Create a new robot object and assign properties
                    new_robot = Robot(grid_x, grid_y, self.robot_counter)
                    
                    # Increment robot counter and assign it to the robot
                    self.robot_counter += 1
                    
                    # Add new robot to the robots list
                    self.robots.append(new_robot)
                    
                    # Add status message for robot placement
                    self.add_status_message(f"Robot {new_robot.id} placed at ({grid_x}, {grid_y})")
                    
                    # Reallocate tasks if the simulation is running
                    if self.simulation_running:
                        self.reallocate_all_tasks()


            elif self.current_tool == 'obstacle':
                if self.grid[grid_y][grid_x] not in [CellType.ROBOT, CellType.TARGET]:
                    self.grid[grid_y][grid_x] = CellType.OBSTACLE
            elif self.current_tool == 'task':
                if self.grid[grid_y][grid_x] == CellType.EMPTY:
                    # Create task with random priority
                    priority = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
                    new_task = Task(grid_x, grid_y, priority)
                    self.grid[grid_y][grid_x] = CellType.TASK
                    self.tasks.append(new_task)
                    self.add_status_message(f"Created P{priority} task at ({grid_x}, {grid_y})")
                    if self.simulation_running:
                        self.auction_tasks()

    def reallocate_all_tasks(self):
        """Reallocate all tasks among all robots for optimal distribution"""
        # Clear all current assignments
        for robot in self.robots:
            if robot.target:
                target = robot.target
                # Create a new task with the same priority
                new_task = Task(target.x, target.y, target.priority)
                self.tasks.append(new_task)
                self.grid[target.y][target.x] = CellType.TASK
            robot.target = None
            robot.path = []
        
        # Create a list of all tasks (both assigned and unassigned)
        all_tasks = []
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.grid[y][x] in [CellType.TASK, CellType.TARGET]:
                    # For existing tasks without priority info, assign random priority
                    priority = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
                    new_task = Task(x, y, priority)
                    all_tasks.append(new_task)
                    self.grid[y][x] = CellType.TASK
        
        self.tasks = all_tasks
        self.add_status_message("Reallocating all tasks for optimal distribution")
        self.auction_tasks()

    

    def generate_random_task(self):
        if len(self.tasks) < MAX_TASKS and random.random() < TASK_GEN_CHANCE:
            empty_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
                          if self.grid[y][x] == CellType.EMPTY]
            if empty_cells:
                x, y = random.choice(empty_cells)
                # Assign random priority with weighted probability
                priority = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
                task = Task(x, y, priority)
                self.grid[y][x] = CellType.TASK
                self.tasks.append(task)
                self.add_status_message(f"Generated new task at ({x}, {y}) with priority {priority}")

    def generate_moving_obstacle(self):
        """Generate a moving obstacle with random direction"""
        if len(self.moving_obstacles) < MAX_MOVING_OBSTACLES and random.random() < OBSTACLE_GEN_CHANCE:
            # Find empty spot
            empty_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
                          if self.grid[y][x] == CellType.EMPTY]
            if empty_cells:
                x, y = random.choice(empty_cells)
                direction = random.choice([(0,1), (1,0), (0,-1), (-1,0)])
                self.moving_obstacles.append({
                    'x': x, 'y': y,
                    'dx': direction[0], 'dy': direction[1],
                    'last_move': time.time()
                })
                self.grid[y][x] = CellType.OBSTACLE
                self.add_status_message(f"Added moving obstacle at ({x}, {y})")

    def update_moving_obstacles(self, current_time):
        """Update positions of moving obstacles"""
        for obstacle in self.moving_obstacles[:]:
            if current_time - obstacle['last_move'] < OBSTACLE_MOVE_DELAY:
                continue

            # Calculate new position
            new_x = obstacle['x'] + obstacle['dx']
            new_y = obstacle['y'] + obstacle['dy']

            # Check if new position is valid
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

    def auction_tasks(self):
        """Market-based task allocation using auction mechanism."""
        if not self.tasks:
            return

        unassigned_robots = [robot for robot in self.robots if not robot.target]
        if not unassigned_robots:
            return

        # Ensure task positions are always 5
        task_positions = [task.get_position() for task in self.tasks]
        if len(task_positions) > 5:
            task_positions = task_positions[:5]  # Take the first 5
        elif len(task_positions) < 5:
            while len(task_positions) < 5:
                task_positions.append(task_positions[-1])  # Repeat the last position

        # Calculate bids for each robot-task pair
        bids = []
        for robot in unassigned_robots:
            # Get the predicted task for the robot using the predict function
            predicted_task_index = predict_action(robot.position, task_positions)  # Use the imported function

            # Ensure the predicted index is within the valid range
            if predicted_task_index < 0 or predicted_task_index >= len(self.tasks):
                continue  # Skip this iteration if the index is invalid

            predicted_task = self.tasks[predicted_task_index]
            print(f"Robot {robot.id} predicted task at {predicted_task.get_position()}")

            # Correctly calculate Manhattan distance
            manhattan_distance = robot.manhattan_distance(
                robot.position,  # Robot's current position as a tuple (x, y)
                predicted_task.get_position()  # Task's position as a tuple (x, y)
            )

            # Avoid division by zero
            bid_value = 1 / manhattan_distance if manhattan_distance != 0 else float('inf')

            # Append the bid
            bids.append((robot, predicted_task, bid_value))

        # Sort bids by value (higher bid wins)
        bids.sort(key=lambda x: x[2], reverse=True)

        # Assign tasks to highest bidders
        assigned_tasks = set()
        assigned_robots = set()

        for robot, task, bid_value in bids:
            if (robot not in assigned_robots and
                    task not in assigned_tasks and
                    task in self.tasks):  # Check if task still available

                # Assign task
                self.grid[task.y][task.x] = CellType.TARGET
                robot.set_target(task)
                self.tasks.remove(task)
                assigned_tasks.add(task)
                assigned_robots.add(robot)

                # Print successful bid message
                print(f"Robot {robot.id} won P{task.priority} task with bid {bid_value:.2f}")
                self.add_status_message(
                    f"Auction: Robot {robot.id} won P{task.priority} task with bid {bid_value:.2f}"
                )

    def update_simulation(self):
        if not self.simulation_running:
            return
            
        current_time = time.time()
        
        # Update dynamic environment
        if self.dynamic_tasks_enabled:
            self.generate_random_task()
            self.generate_moving_obstacle()
        
        self.update_moving_obstacles(current_time)
        
        # Update waiting times for all robots
        for robot in self.robots:
            robot.update_waiting(robot.waiting, current_time)
            
        # Run auction-based task allocation
        self.auction_tasks()
        
        # Sort robots by priority of their tasks and waiting time
        active_robots = [r for r in self.robots if r.target]
        active_robots.sort(key=lambda r: (
            r.target.priority if r.target else 0,
            -r.waiting_time,  # Negative so longer waiting time gets priority
            r.manhattan_distance((r.x, r.y), r.target.get_position()) if r.target else float('inf')
        ), reverse=True)
            
        # Process all robots, not just active ones
        for robot in self.robots:
            if current_time - robot.last_move_time < MOVE_DELAY:
                continue  # Skip if not enough time has passed since last move
                
            # Check if path is still valid
            if robot.path:
                path_invalid = False
                for pos in robot.path:
                    if self.grid[pos[1]][pos[0]] == CellType.OBSTACLE:
                        path_invalid = True
                        break
                if path_invalid:
                    robot.path = []
                    self.add_status_message(f"Robot {robot.id}: Replanning due to obstacle")

            if not robot.path:
                if robot.target:
                    # Create a temporary grid with other robots marked as obstacles
                    temp_grid = [row[:] for row in self.grid]  # Create a deep copy
                    for other_robot in self.robots:
                        if other_robot != robot:
                            temp_grid[other_robot.y][other_robot.x] = CellType.OBSTACLE
                    
                    # Use the temporary grid for pathfinding
                    self.astar.grid = temp_grid
                    robot.path = self.astar.find_path(
                        (robot.x, robot.y),
                        robot.target.get_position()
                    )
                    # Restore original grid in A*
                    self.astar.grid = self.grid
                    
                    if robot.path:
                        path_length = len(robot.path)
                        self.add_status_message(
                            f"Robot {robot.id}: Found path to P{robot.target.priority} task, length {path_length}"
                        )
                    else:
                        self.add_status_message(
                            f"Robot {robot.id}: No path to P{robot.target.priority} task"
                        )
                        # If no path found, clear target and try another task
                        robot.target = None
                        continue
                    robot.path.pop(0)  # Remove current position

            if robot.path:
                next_pos = robot.path[0]
                
                # Check for collision
                collision = False
                for other_robot in self.robots:
                    if other_robot != robot:
                        # Check current position and next planned position
                        if ((other_robot.x, other_robot.y) == next_pos or
                            (other_robot.path and other_robot.path[0] == next_pos) or
                            # Check if robots are facing each other
                            (other_robot.path and 
                             (other_robot.x, other_robot.y) == robot.path[-1] and
                             (robot.x, robot.y) == other_robot.path[-1])):
                            
                            collision = True
                            # Clear path to force rerouting
                            robot.path = []
                            self.add_status_message(f"Robot {robot.id}: Rerouting to avoid collision with Robot {other_robot.id}")
                            break

                if not collision:
                    # Update grid and robot position
                    self.grid[robot.y][robot.x] = CellType.EMPTY
                    robot.x, robot.y = next_pos
                    self.grid[robot.y][robot.x] = CellType.ROBOT
                    robot.path.pop(0)
                    robot.last_move_time = current_time
                    robot.total_distance += 1
                    robot.waiting = False
                    
                    if robot.target and (robot.x, robot.y) == robot.target.get_position():
                        completed_priority = robot.target.priority
                        robot.target = None
                        robot.completed_tasks += 1
                        self.total_tasks_completed += 1
                        self.add_status_message(
                            f"Robot {robot.id}: Completed P{completed_priority} task! Total: {robot.completed_tasks}"
                        )

        
        # Check if simulation should end
        if self.end_simulation and not any(robot.target for robot in self.robots):
            self.simulation_running = False
            self.performance_metrics = PerformanceMetrics.calculate_metrics(self)

    def generate_random(self):
        # Clear the grid and robots
        self.grid = [[CellType.EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.robots = []
        self.robot_counter = 0
        
        # Add random robots (2-3)
        num_robots = random.randint(2, 3)
        for _ in range(num_robots):
            while True:
                x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
                if self.grid[y][x] == CellType.EMPTY:
                    self.grid[y][x] = CellType.ROBOT
                    new_robot = Robot(x, y, id=self.robot_counter + 1)
                    self.robot_counter += 1
                    self.robots.append(new_robot)
                    break

        
        # Add random obstacles (5-8)
        num_obstacles = random.randint(5, 8)
        for _ in range(num_obstacles):
            while True:
                x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
                if self.grid[y][x] == CellType.EMPTY:
                    self.grid[y][x] = CellType.OBSTACLE
                    break

    def draw(self):
        self.screen.fill(WHITE)
        
        # Draw grid (adjusted for smaller grid size)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, WHITE, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1)
                
                if self.grid[y][x] == CellType.ROBOT:
                    self.screen.blit(robot_image, (x * CELL_SIZE, y * CELL_SIZE))
                elif self.grid[y][x] == CellType.OBSTACLE:
                    self.screen.blit(obstacle_image, (x * CELL_SIZE, y * CELL_SIZE))

                elif self.grid[y][x] == CellType.TARGET:
                    pygame.draw.rect(self.screen, GREEN,
                                   (x * CELL_SIZE + 5, y * CELL_SIZE + 5,
                                    CELL_SIZE - 10, CELL_SIZE - 10))
                elif self.grid[y][x] == CellType.TASK:
                    pygame.draw.rect(self.screen, PURPLE,
                                   (x * CELL_SIZE + 5, y * CELL_SIZE + 5,
                                    CELL_SIZE - 10, CELL_SIZE - 10))
        
        # Draw paths for robots
        for robot in self.robots:
            if robot.path:
                points = [(p[0] * CELL_SIZE + CELL_SIZE//2, 
                          p[1] * CELL_SIZE + CELL_SIZE//2) for p in [(robot.x, robot.y)] + robot.path]
                pygame.draw.lines(self.screen, YELLOW, False, points, 2)
        
        # Draw menu background
        pygame.draw.rect(self.screen, WHITE, (WINDOW_SIZE, 0, MENU_WIDTH, WINDOW_SIZE))
        pygame.draw.line(self.screen, BLACK, (WINDOW_SIZE, 0), (WINDOW_SIZE, WINDOW_SIZE), 2)
        
        # Draw buttons
        for button in self.buttons.values():
            button.draw(self.screen)
        
        # Draw performance metrics if available
        if self.performance_metrics:
            font = pygame.font.Font(None, 24)
            metrics_text = [
                f"Time: {self.performance_metrics['total_time']:.1f}s",
                f"Tasks: {self.performance_metrics['total_tasks']}",
                f"Distance: {self.performance_metrics['total_distance']}",
                f"Time Saved: {self.performance_metrics['time_saved']:.1f}s",
                f"Distance Saved: {self.performance_metrics['distance_saved']:.1f}",
                f"Tasks/s: {self.performance_metrics['tasks_per_second']:.2f}"
            ]
            
            y_offset = 350
            for text in metrics_text:
                text_surface = font.render(text, True, BLACK)
                self.screen.blit(text_surface, (WINDOW_SIZE + 20, y_offset))
                y_offset += 25
        
        # Draw status messages at the bottom
        if self.simulation_running or self.performance_metrics:
            font = pygame.font.Font(None, 24)
            y_offset = GRID_SIZE * CELL_SIZE + 10  # Adjust for new grid size

            # Draw status panel background for logs
            status_panel = pygame.Rect(10, GRID_SIZE * CELL_SIZE, TOTAL_WIDTH - 20, LOG_HEIGHT - 10)
            pygame.draw.rect(self.screen, (240, 240, 240), status_panel, border_radius=10)
            pygame.draw.rect(self.screen, BLACK, status_panel, 2, border_radius=10)

            # Add shadow effect
            shadow_panel = pygame.Rect(12, GRID_SIZE * CELL_SIZE + 2, TOTAL_WIDTH - 20, LOG_HEIGHT - 10)
            pygame.draw.rect(self.screen, (200, 200, 200), shadow_panel, border_radius=10)

            # Draw title
            title = font.render("Status Log:", True, BLACK)
            self.screen.blit(title, (20, y_offset + 10))

            # Start logs below the title
            y_offset += 35

            # Draw messages within log area
            for message in self.status_messages[-self.max_messages:]:
                words = message.split()
                line = []
                for word in words:
                    if font.size(' '.join(line + [word]))[0] <= TOTAL_WIDTH - 40:
                        line.append(word)
                    else:
                        text_surface = font.render(' '.join(line), True, BLACK)
                        self.screen.blit(text_surface, (20, y_offset))
                        y_offset += 20

                        # Stop drawing if log exceeds the panel height
                        if y_offset > GRID_SIZE * CELL_SIZE + LOG_HEIGHT - 20:
                            break

                        line = [word]
                text_surface = font.render(' '.join(line), True, BLACK)
                self.screen.blit(text_surface, (20, y_offset))
                y_offset += 20

                # Stop drawing if log exceeds the panel height
                if y_offset > GRID_SIZE * CELL_SIZE + LOG_HEIGHT - 20:
                    break

        
        pygame.display.flip()

    def run(self):
        while self.running:
            self.clock.tick(10)  # Limit to 10 FPS for better visualization
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
            
            if self.simulation_running:
                self.update_simulation()
            
            self.draw()

        pygame.quit() 