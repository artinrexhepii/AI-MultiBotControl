import pygame
import random
import time
from src.core.constants import *
from src.core.entities import Robot, Task
from src.agents.madql_agent import MADQLAgent
from src.agents.astar import AStar
from src.core.grid import Grid
from src.ui.renderer import Renderer
from src.utils.metrics import PerformanceMetrics

class Game:
    def __init__(self):
        pygame.init()
        
        # Initialize core components
        self.grid = Grid()
        self.renderer = Renderer(WINDOW_SIZE, MENU_WIDTH)
        self.madql = MADQLAgent(self)
        self.astar = AStar(self)
        
        # Game state
        self.robots = []
        self.robot_counter = 0
        self.simulation_running = False
        self.dynamic_tasks_enabled = True
        self.end_simulation = False
        self.start_time = None
        self.total_tasks_completed = 0
        
        # Timing and updates
        self.clock = pygame.time.Clock()
        self.auction_interval = 2.0
        self.last_auction_time = 0
        
        # Status messages
        self.status_messages = []
        self.max_messages = 8
        
        self.running = True

    def add_status_message(self, message):
        """Add status message to log"""
        self.status_messages.insert(0, message)
        if len(self.status_messages) > self.max_messages:
            self.status_messages.pop()

    def handle_click(self, pos):
        """Handle mouse click events"""
        # Check for button clicks
        button_clicked = self.renderer.handle_button_click(pos)
        if button_clicked:
            self._handle_button_action(button_clicked)
            return
            
        # Handle grid clicks
        grid_pos = self.renderer.get_grid_position(pos)
        if grid_pos and hasattr(self, 'current_tool'):
            x, y = grid_pos
            self._handle_grid_click(x, y)

    def _handle_button_action(self, action):
        """Handle button click actions"""
        if action == 'random':
            self._generate_random_scenario()
        elif action == 'play':
            self._toggle_simulation()
        elif action == 'end':
            self.end_simulation = True
            self.dynamic_tasks_enabled = False
        else:
            # Set current tool
            self.current_tool = action
            # Update button states
            for name, button in self.renderer.buttons.items():
                button.selected = (name == action)

    def _handle_grid_click(self, x, y):
        """Handle clicks on the grid"""
        if self.current_tool == 'robot':
            if self.grid.add_robot(x, y, Robot(x, y)):
                self.robot_counter += 1
                new_robot = self.robots[-1]
                new_robot.id = self.robot_counter
                self.add_status_message(f"Robot {new_robot.id} placed at ({x}, {y})")
                if self.simulation_running:
                    self.reallocate_all_tasks()
        elif self.current_tool == 'obstacle':
            self.grid.add_obstacle(x, y)
        elif self.current_tool == 'task':
            task = self.grid.add_task(x, y)
            if task:
                self.add_status_message(f"Created P{task.priority} task at ({x}, {y})")
                if self.simulation_running:
                    self.assign_tasks()

    def _generate_random_scenario(self):
        """Generate random scenario"""
        self.grid.clear()
        self.robots = []
        self.robot_counter = 0
        
        # Add random robots (2-3)
        num_robots = random.randint(2, 3)
        empty_cells = self.grid.find_empty_cells()
        for _ in range(num_robots):
            if empty_cells:
                x, y = random.choice(empty_cells)
                empty_cells.remove((x, y))
                new_robot = Robot(x, y)
                if self.grid.add_robot(x, y, new_robot):
                    self.robot_counter += 1
                    new_robot.id = self.robot_counter
                    self.robots.append(new_robot)
        
        # Add random obstacles (5-8)
        num_obstacles = random.randint(5, 8)
        empty_cells = self.grid.find_empty_cells()
        for _ in range(num_obstacles):
            if empty_cells:
                x, y = random.choice(empty_cells)
                empty_cells.remove((x, y))
                self.grid.add_obstacle(x, y)

    def _toggle_simulation(self):
        """Toggle simulation state"""
        self.simulation_running = not self.simulation_running
        if self.simulation_running and self.start_time is None:
            self.start_time = time.time()
            for robot in self.robots:
                robot.start_time = time.time()
            self.reallocate_all_tasks()

    def update_simulation(self):
        """Update simulation state"""
        if not self.simulation_running:
            return
            
        current_time = time.time()
        
        # Update dynamic environment
        if self.dynamic_tasks_enabled:
            task = self.grid.generate_random_task()
            if task:
                self.add_status_message(f"Generated new task at ({task.x}, {task.y}) with priority {task.priority}")
            
            if random.random() < OBSTACLE_GEN_CHANCE and len(self.grid.moving_obstacles) < MAX_MOVING_OBSTACLES:
                empty_cells = self.grid.find_empty_cells()
                if empty_cells:
                    x, y = random.choice(empty_cells)
                    if self.grid.add_obstacle(x, y, moving=True):
                        self.add_status_message(f"Added moving obstacle at ({x}, {y})")
        
        # Update moving obstacles
        self.grid.update_moving_obstacles(current_time, MOVE_DELAY)
        
        # Update robot states
        self._update_robots(current_time)
        
        # Run task allocation
        if current_time - self.last_auction_time >= self.auction_interval:
            self.auction_tasks()
            self.last_auction_time = current_time

    def _update_robots(self, current_time):
        """Update robot positions and states"""
        # Update waiting times
        for robot in self.robots:
            robot.update_waiting(robot.waiting, current_time)
        
        # Sort robots by priority
        active_robots = [r for r in self.robots if r.target]
        active_robots.sort(key=lambda r: (
            r.target.priority if r.target else 0,
            -r.waiting_time,
            r.manhattan_distance((r.x, r.y), r.target.get_position()) if r.target else float('inf')
        ), reverse=True)
        
        # Track reserved positions
        reserved_positions = set()
        
        # Update robot positions
        for robot in active_robots:
            if current_time - robot.last_move_time < MOVE_DELAY:
                continue
                
            if not robot.path or not self.grid.is_valid_path(robot.path):
                if robot.target:
                    # Find new path avoiding obstacles and other robots
                    blocked = reserved_positions | {(r.x, r.y) for r in self.robots if r != robot}
                    robot.path = self.astar.find_path(
                        (robot.x, robot.y),
                        robot.target.get_position(),
                        blocked
                    )
                    
                    if robot.path:
                        self.add_status_message(
                            f"Robot {robot.id}: Found path to P{robot.target.priority} task, length {len(robot.path)}"
                        )
                        robot.path.pop(0)  # Remove current position
                    else:
                        robot.waiting = True
                        robot.waiting_time += MOVE_DELAY
                        self.add_status_message(
                            f"Robot {robot.id}: Waiting for path to P{robot.target.priority} task"
                        )
                        continue
            
            if robot.path:
                next_pos = robot.path[0]
                if next_pos not in reserved_positions and self.grid.move_robot(robot, *next_pos):
                    reserved_positions.add(next_pos)
                    robot.path.pop(0)
                    robot.last_move_time = current_time
                    robot.total_distance += 1
                    robot.waiting = False
                    robot.waiting_time = 0
                    
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
                    robot.waiting_time += MOVE_DELAY

    def run(self):
        """Main game loop"""
        while self.running:
            self.clock.tick(10)  # Limit to 10 FPS
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
            
            if self.simulation_running:
                self.update_simulation()
            
            # Draw current state
            self.renderer.draw_grid(self.grid, self.robots)
            self.renderer.draw_metrics(PerformanceMetrics.calculate_metrics(self))
            self.renderer.draw_status_messages(self.status_messages)
            self.renderer.update_display()
        
        pygame.quit() 