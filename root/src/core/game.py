import pygame
import random
import time
from src.core.constants import *
from src.core.entities import Robot, Task, CellType
from src.agents.madql_agent import MADQLAgent
from src.agents.astar import AStar
from src.core.grid import Grid
from src.ui.renderer import Renderer
from src.utils.metrics import PerformanceMetrics

class Game:
    def __init__(self):
        pygame.init()
        
        # Game state
        self.robots = []
        self.robot_counter = 0
        self.simulation_running = False
        self.dynamic_tasks_enabled = True
        self.end_simulation = False
        self.start_time = None
        self.total_tasks_completed = 0
        
        # Initialize core components
        self.grid = Grid(self)
        self.renderer = Renderer(WINDOW_SIZE, MENU_WIDTH)
        self.astar = AStar(self)
        self.madql = MADQLAgent(self)
        
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
            new_robot = Robot(x, y)
            self.robot_counter += 1
            new_robot.id = self.robot_counter
            if self.grid.add_robot(x, y, new_robot):
                self.robots.append(new_robot)
                self.add_status_message(f"Robot {new_robot.id} placed at ({x}, {y})")
                if self.simulation_running:
                    self.auction_tasks()
        elif self.current_tool == 'obstacle':
            self.grid.add_obstacle(x, y)
        elif self.current_tool == 'task':
            task = self.grid.add_task(x, y)
            if task:
                self.add_status_message(f"Created P{task.priority} task at ({x}, {task.y})")
                if self.simulation_running:
                    self.auction_tasks()

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
        if self.simulation_running:
            if self.start_time is None:
                self.start_time = time.time()
                self.last_auction_time = time.time()  # Initialize auction timer
                for robot in self.robots:
                    robot.start_time = time.time()
            # Force immediate task allocation when simulation starts
            self.auction_tasks()

    def reallocate_all_tasks(self):
        """Reallocate all tasks among all robots for optimal distribution"""
        # Clear all current assignments
        for robot in self.robots:
            if robot.target:
                target_pos = (robot.target.x, robot.target.y)
                # Create a new task with the same priority
                self.grid.add_task(target_pos[0], target_pos[1], robot.target.priority)
            robot.target = None
            robot.path = []
        
        # Get all tasks
        tasks = self.grid.tasks.copy()
        
        # Clear task assignments
        for task in tasks:
            self.grid.mark_task_assigned(task)
        
        self.add_status_message("Reallocating all tasks for optimal distribution")
        self.assign_tasks()

    def assign_tasks(self):
        """Assign tasks to robots using MADQL"""
        # Get unassigned robots and available tasks
        unassigned_robots = [robot for robot in self.robots if not robot.target]
        available_tasks = [task for task in self.grid.tasks if not task.assigned]
        
        if not unassigned_robots or not available_tasks:
            return
            
        print(f"Assigning tasks to {len(unassigned_robots)} robots, {len(available_tasks)} tasks available")
        
        for robot in unassigned_robots:
            # Get current state
            old_state = self.madql.get_state(robot)
            
            # Choose task using MADQL
            chosen_task = self.madql.choose_action(robot)
            
            if chosen_task:
                # Mark task as assigned
                self.grid.mark_task_assigned(chosen_task)
                robot.set_target(chosen_task)
                
                # Find path to the task
                path = self.astar.find_path(
                    (robot.x, robot.y),
                    chosen_task.get_position(),
                    robot=robot
                )
                
                if path:
                    robot.path = path
                    print(f"Robot {robot.id} assigned to task at ({chosen_task.x}, {chosen_task.y}) with path length {len(path)}")
                else:
                    print(f"No path found for Robot {robot.id} to assigned task")
                    robot.target = None
                    continue
                
                # Get new state and reward
                new_state = self.madql.get_state(robot)
                reward = self.madql.get_reward(robot, old_state, chosen_task, new_state)
                
                # Update Q-values
                self.madql.update(robot, old_state, chosen_task, reward, new_state)
                
                self.add_status_message(
                    f"Robot {robot.id} assigned to P{chosen_task.priority} task at ({chosen_task.x}, {chosen_task.y}) [R: {reward:.1f}]"
                )

    def auction_tasks(self):
        """Auction available tasks to robots"""
        # Get unassigned robots and available tasks
        unassigned_robots = [robot for robot in self.robots if not robot.target]
        available_tasks = [task for task in self.grid.tasks if not task.assigned]
        
        if not unassigned_robots or not available_tasks:
            return
        
        print(f"Auction starting with {len(unassigned_robots)} robots and {len(available_tasks)} tasks")
        
        # Initialize bid tracking
        all_bids = []
        
        # Collect bids from all unassigned robots for all tasks
        for robot in unassigned_robots:
            for task in available_tasks:
                chosen_task, bid_value = self.madql.calculate_bid(robot, [task])
                if chosen_task:
                    all_bids.append((robot, chosen_task, bid_value))
                    print(f"Robot {robot.id} bid {bid_value:.2f} for task at ({task.x}, {task.y})")
                else:
                    print(f"Robot {robot.id} could not bid for task at ({task.x}, {task.y})")
        
        if not all_bids:
            print("No valid bids received")
            return
        
        # Sort bids by value (highest first)
        all_bids.sort(key=lambda x: x[2], reverse=True)
        
        # Track assignments
        assigned_tasks = set()
        assigned_robots = set()
        
        # Assign tasks based on highest bids
        for robot, task, bid_value in all_bids:
            if (robot not in assigned_robots and 
                task not in assigned_tasks and 
                task in self.grid.tasks):  # Ensure task still exists
                
                print(f"Assigning task at ({task.x}, {task.y}) to Robot {robot.id}")
                
                # Assign task to robot
                robot.target = task
                task.assigned = True
                self.grid.mark_task_assigned(task)
                
                # Find path to task
                path = self.astar.find_path(
                    (robot.x, robot.y),
                    task.get_position(),
                    robot=robot
                )
                
                if path:
                    robot.path = path[1:]  # Skip current position
                    print(f"Found path of length {len(path)} for Robot {robot.id}")
                    
                    # Update tracking
                    assigned_tasks.add(task)
                    assigned_robots.add(robot)
                    robot.last_task_start = time.time()
                else:
                    print(f"No path found for Robot {robot.id}")
                    robot.target = None

    def update_simulation(self):
        """Update simulation state"""
        if not self.simulation_running:
            return
        
        current_time = time.time()
        
        # Update dynamic environment
        if self.dynamic_tasks_enabled:
            task = self.grid.generate_random_task()
            if task:
                print(f"Generated new task at ({task.x}, {task.y}) with priority {task.priority}")  # Debug print
                self.add_status_message(f"Generated new P{task.priority} task at ({task.x}, {task.y})")
                # Try to allocate new task immediately
                self.auction_tasks()
        
        # Update moving obstacles
        self.grid.update_moving_obstacles(current_time, MOVE_DELAY)
        
        # Update robot states
        self._update_robots()
        
        # Run task allocation more frequently
        if current_time - self.last_auction_time >= self.auction_interval:
            self.auction_tasks()

    def _update_robots(self):
        """Update robot positions and handle task completion"""
        current_time = time.time()
        
        # First check for idle robots and assign tasks
        idle_robots = [robot for robot in self.robots if not robot.target]
        if idle_robots:
            print(f"Found {len(idle_robots)} idle robots, running auction")
            # Make sure there are available tasks
            if self.grid.tasks:
                self.auction_tasks()
            else:
                # Generate a new task if none available
                if self.dynamic_tasks_enabled:
                    task = self.grid.generate_random_task()
                    if task:
                        print(f"Generated new task for idle robots at ({task.x}, {task.y})")
                        self.auction_tasks()
        
        # Then update robot movements
        for robot in self.robots:
            # Skip robots without targets
            if not robot.target:
                continue
            
            # If robot has no path, try to find one
            if not robot.path:
                path = self.astar.find_path(
                    (robot.x, robot.y),
                    robot.target.get_position(),
                    robot=robot
                )
                if path and len(path) > 1:  # Make sure path has more than just current position
                    print(f"Found path for Robot {robot.id} to target at ({robot.target.x}, {robot.target.y})")
                    robot.path = path[1:]  # Skip current position
                else:
                    print(f"No valid path found for Robot {robot.id} to target")
                    robot.target = None
                    continue
            
            # Check if enough time has passed since last move
            if current_time - robot.last_move_time < MOVE_DELAY:
                continue
            
            # Try to move along path
            if robot.path:
                next_pos = robot.path[0]
                success = self.grid.move_robot(robot, next_pos[0], next_pos[1])
                
                if success:
                    robot.last_move_time = current_time
                    robot.x, robot.y = next_pos
                    robot.path.pop(0)
                    
                    # Check if robot has reached its target task
                    if robot.target and (robot.x, robot.y) == (robot.target.x, robot.target.y):
                        completed_task = robot.target
                        
                        # Remove the task
                        if completed_task in self.grid.tasks:
                            self.grid.tasks.remove(completed_task)
                            print(f"Robot {robot.id} completed task at ({completed_task.x}, {completed_task.y})")
                            
                            # Update robot's completion status
                            robot.completed_tasks += 1
                            self.total_tasks_completed += 1
                            
                            # Notify other robots if their target task was completed
                            for other_robot in self.robots:
                                if other_robot != robot and other_robot.target == completed_task:
                                    print(f"Robot {other_robot.id}'s target task was completed by Robot {robot.id}")
                                    other_robot.target = None
                                    other_robot.path = []
                            
                            # Clear robot's target and path but keep its position
                            robot.target = None
                            robot.path = []
                            
                            # Generate a new task if needed
                            if len(self.grid.tasks) < MAX_TASKS and self.dynamic_tasks_enabled:
                                task = self.grid.generate_random_task()
                                if task:
                                    print(f"Generated replacement task at ({task.x}, {task.y})")
                            
                            # Run auction for all robots without tasks
                            self.auction_tasks()
                else:
                    # If move failed, clear path and try to find a new one next update
                    print(f"Robot {robot.id} failed to move, clearing path")
                    robot.path = []

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