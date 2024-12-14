import pygame
import random
import time
from src.core.constants import *
from src.core.entities import Robot, CellType, Task
from src.agents.madql_agent import MADQLAgent
from src.agents.astar import AStar
from src.ui.button import Button
from src.utils.metrics import PerformanceMetrics

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((TOTAL_WIDTH, WINDOW_SIZE))
        pygame.display.set_caption("Multi-Robot Control System")
        
        self.grid = [[CellType.EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.current_tool = None
        self.robots = []
        self.tasks = []
        self.dynamic_tasks_enabled = True
        self.end_simulation = False
        self.start_time = None
        self.total_tasks_completed = 0
        
        self.madql = MADQLAgent(self)
        self.astar = AStar(self)
        self.clock = pygame.time.Clock()
        
        # Create buttons
        self.buttons = {
            'robot': Button(WINDOW_SIZE + 20, 50, 160, 40, "Robot"),
            'obstacle': Button(WINDOW_SIZE + 20, 100, 160, 40, "Obstacle"),
            'task': Button(WINDOW_SIZE + 20, 150, 160, 40, "Task"),
            'random': Button(WINDOW_SIZE + 20, 200, 160, 40, "Random Generate"),
            'play': Button(WINDOW_SIZE + 20, 250, 160, 40, "Play"),
            'end': Button(WINDOW_SIZE + 20, 300, 160, 40, "End")
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
                    new_robot = Robot(grid_x, grid_y)
                    self.robot_counter += 1
                    new_robot.id = self.robot_counter
                    self.robots.append(new_robot)
                    self.add_status_message(f"Robot {new_robot.id} placed at ({grid_x}, {grid_y})")
                    # Reallocate tasks when new robot is added
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
                        self.assign_tasks()

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
        self.assign_tasks()

    def assign_tasks(self):
        """Assign tasks to robots using MADQL"""
        unassigned_robots = [robot for robot in self.robots if not robot.target]
        
        for robot in unassigned_robots:
            # Get current state
            old_state = self.madql.get_state(robot)
            
            # Choose task using MADQL
            chosen_task = self.madql.choose_action(robot)
            
            if chosen_task:
                # Mark task as assigned
                self.grid[chosen_task.y][chosen_task.x] = CellType.TARGET
                robot.set_target(chosen_task)
                self.tasks.remove(chosen_task)
                
                # Get new state and reward
                new_state = self.madql.get_state(robot)
                reward = self.madql.get_reward(robot, old_state, chosen_task, new_state)
                
                # Update Q-values
                self.madql.update(robot, old_state, chosen_task, reward, new_state)
                
                self.add_status_message(
                    f"Robot {robot.id} assigned to P{chosen_task.priority} task at ({chosen_task.x}, {chosen_task.y}) [R: {reward:.1f}]"
                )

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

    def update_simulation(self):
        if not self.simulation_running:
            return
            
        current_time = time.time()
        
        # Generate new tasks if enabled
        if self.dynamic_tasks_enabled:
            self.generate_random_task()
        
        # Update waiting times for all robots
        for robot in self.robots:
            robot.update_waiting(robot.waiting, current_time)
            
        # Assign available tasks using MADQL
        self.assign_tasks()
            
        # First, reset waiting status for all robots at the start of each update
        for robot in self.robots:
            robot.waiting = False
            
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
                    robot.path = self.astar.find_path(
                        (robot.x, robot.y),
                        robot.target.get_position()
                    )
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
                colliding_robot = None
                for other_robot in self.robots:
                    if other_robot != robot:
                        # Check current position
                        if (other_robot.x, other_robot.y) == next_pos:
                            collision = True
                            colliding_robot = other_robot
                            break
                        # Check if other robot is planning to move to our next position
                        elif (other_robot.path and 
                              other_robot.path[0] == next_pos):
                            # Consider task priorities in collision resolution
                            if robot.target and other_robot.target:
                                if robot.target.priority < other_robot.target.priority:
                                    collision = True
                                    colliding_robot = other_robot
                                    break
                                elif robot.target.priority == other_robot.target.priority:
                                    # If same priority, consider waiting time and distance
                                    if robot.waiting_time < other_robot.waiting_time:
                                        collision = True
                                        colliding_robot = other_robot
                                        break
                                    elif robot.waiting_time == other_robot.waiting_time:
                                        # If same waiting time, let robot closer to target proceed
                                        if (other_robot.target and
                                            other_robot.manhattan_distance((other_robot.x, other_robot.y), other_robot.target.get_position()) < 
                                            robot.manhattan_distance((robot.x, robot.y), robot.target.get_position())):
                                            collision = True
                                            colliding_robot = other_robot
                                            break

                if not collision:
                    # Get old state before moving
                    old_state = self.madql.get_state(robot)
                    
                    # Update grid and robot position
                    self.grid[robot.y][robot.x] = CellType.EMPTY
                    old_pos = (robot.x, robot.y)
                    robot.x, robot.y = next_pos
                    self.grid[robot.y][robot.x] = CellType.ROBOT
                    robot.path.pop(0)
                    robot.last_move_time = current_time
                    robot.total_distance += 1
                    robot.waiting = False
                    
                    # Get new state and update Q-values
                    new_state = self.madql.get_state(robot)
                    reward = self.madql.get_reward(robot, old_state, next_pos, new_state)
                    self.madql.update(robot, old_state, next_pos, reward, new_state)
                    
                    # Check if reached target
                    if robot.target and (robot.x, robot.y) == robot.target.get_position():
                        completed_priority = robot.target.priority
                        robot.target = None
                        robot.completed_tasks += 1
                        self.total_tasks_completed += 1
                        self.add_status_message(
                            f"Robot {robot.id}: Completed P{completed_priority} task! Total: {robot.completed_tasks}"
                        )
                else:
                    robot.waiting = True
                    if colliding_robot:
                        # If both robots are waiting too long, force one to find alternative path
                        if (robot.waiting and colliding_robot.waiting and 
                            current_time - robot.last_move_time > MOVE_DELAY * 3):
                            # Robot with lower priority task should find alternative
                            if (robot.target and colliding_robot.target and 
                                robot.target.priority <= colliding_robot.target.priority):
                                robot.path = []  # Force path recalculation
                                self.add_status_message(
                                    f"Robot {robot.id}: Finding alternative path (lower priority)"
                                )
                            elif robot.waiting_time > colliding_robot.waiting_time:
                                robot.path = []  # Force path recalculation
                                self.add_status_message(
                                    f"Robot {robot.id}: Finding alternative path (waited longer)"
                                )
                        else:
                            self.add_status_message(
                                f"Robot {robot.id}: Waiting for Robot {colliding_robot.id} to move"
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
                    new_robot = Robot(x, y)
                    self.robot_counter += 1
                    new_robot.id = self.robot_counter
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
        
        # Draw grid
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, WHITE, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1)
                
                if self.grid[y][x] == CellType.ROBOT:
                    pygame.draw.circle(self.screen, BLUE, 
                                    (x * CELL_SIZE + CELL_SIZE//2, 
                                     y * CELL_SIZE + CELL_SIZE//2), 
                                    CELL_SIZE//3)
                elif self.grid[y][x] == CellType.OBSTACLE:
                    pygame.draw.rect(self.screen, RED, 
                                   (x * CELL_SIZE + 5, y * CELL_SIZE + 5, 
                                    CELL_SIZE - 10, CELL_SIZE - 10))
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
        
        # Draw status messages
        if self.simulation_running or self.performance_metrics:
            font = pygame.font.Font(None, 24)
            y_offset = 400 if self.performance_metrics else 350
            
            # Draw status panel background
            status_panel = pygame.Rect(WINDOW_SIZE + 10, y_offset, MENU_WIDTH - 20, 200)
            pygame.draw.rect(self.screen, (240, 240, 240), status_panel)
            pygame.draw.rect(self.screen, BLACK, status_panel, 2)
            
            # Draw title
            title = font.render("Status Log:", True, BLACK)
            self.screen.blit(title, (WINDOW_SIZE + 20, y_offset + 10))
            
            # Draw messages
            y_offset += 35
            for message in self.status_messages:
                # Word wrap messages
                words = message.split()
                lines = []
                line = []
                for word in words:
                    if font.size(' '.join(line + [word]))[0] <= MENU_WIDTH - 40:
                        line.append(word)
                    else:
                        lines.append(' '.join(line))
                        line = [word]
                lines.append(' '.join(line))
                
                for line in lines:
                    text_surface = font.render(line, True, BLACK)
                    self.screen.blit(text_surface, (WINDOW_SIZE + 20, y_offset))
                    y_offset += 20
                    if y_offset > WINDOW_SIZE - 20:  # Prevent drawing outside window
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