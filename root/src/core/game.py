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
        unassigned_robots = [robot for robot in self.robots if not robot.target]
        
        for robot in unassigned_robots:
            # Get current state
            old_state = self.madql.get_state(robot)
            
            # Choose task using MADQL
            chosen_task = self.madql.choose_action(robot)
            
            if chosen_task:
                # Mark task as assigned
                self.grid.mark_task_assigned(chosen_task)
                robot.set_target(chosen_task)
                self.grid.tasks.remove(chosen_task)
                
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
        current_time = time.time()
        if current_time - self.last_auction_time < self.auction_interval:
            return
        
        self.last_auction_time = current_time
        
        # Get unassigned robots and available tasks
        unassigned_robots = [robot for robot in self.robots if not robot.target]
        available_tasks = [task for task in self.grid.tasks if not task.assigned]
        
        if not unassigned_robots or not available_tasks:
            return
        
        print(f"Auction starting with {len(unassigned_robots)} robots and {len(available_tasks)} tasks")  # Debug print
        
        # Initialize bid tracking
        all_bids = []
        
        # Collect bids from all unassigned robots for all tasks
        for robot in unassigned_robots:
            for task in available_tasks:
                chosen_task, bid_value = self.madql.calculate_bid(robot, [task])
                if chosen_task:
                    all_bids.append((robot, chosen_task, bid_value))
                    print(f"Robot {robot.id} bid {bid_value:.2f} for task at ({task.x}, {task.y})")  # Debug print
                    self.add_status_message(f"Robot {robot.id} bid {bid_value:.2f} for task at ({task.x}, {task.y})")
        
        if not all_bids:
            print("No valid bids received")  # Debug print
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
                
                print(f"Assigning task at ({task.x}, {task.y}) to Robot {robot.id}")  # Debug print
                
                # Assign task to robot
                robot.set_target(task)
                task.assigned = True
                self.grid.mark_task_assigned(task)
                
                # Update tracking
                assigned_tasks.add(task)
                assigned_robots.add(robot)
                
                # Update metrics
                robot.last_task_start = current_time
                
                self.add_status_message(
                    f"Robot {robot.id} assigned to task at ({task.x}, {task.y}) with bid {bid_value:.2f}"
                )
                
                # Find initial path
                path = self.astar.find_path(
                    (robot.x, robot.y),
                    task.get_position(),
                    robot=robot
                )
                if path:
                    robot.path = path
                    if len(path) > 1:
                        robot.path.pop(0)  # Remove current position
                    print(f"Found path of length {len(path)} for Robot {robot.id}")  # Debug print
                else:
                    print(f"No path found for Robot {robot.id}")  # Debug print

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
        self._update_robots(current_time)
        
        # Run task allocation more frequently
        if current_time - self.last_auction_time >= self.auction_interval:
            self.auction_tasks()

    def _update_robots(self, current_time):
        """Update robot positions and states with improved movement handling"""
        # Update waiting times
        for robot in self.robots:
            robot.update_waiting(robot.waiting, current_time)
        
        # Sort robots by priority and efficiency
        active_robots = [r for r in self.robots if r.target]
        active_robots.sort(key=lambda r: (
            r.target.priority if r.target else 0,
            -r.waiting_time,
            r.manhattan_distance((r.x, r.y), r.target.get_position()) if r.target else float('inf'),
            -r.completed_tasks  # Give preference to more efficient robots
        ), reverse=True)
        
        # Track reserved positions for this update cycle
        reserved_positions = set()
        moving_robots = set()
        
        # First pass: Check for task completion and update paths
        for robot in active_robots:
            # Check if robot has reached its target
            if robot.target and (robot.x, robot.y) == robot.target.get_position():
                self._complete_task(robot, current_time)
                continue
            
            # Update path if needed
            if not robot.path or not self.grid.is_valid_path(robot.path):
                self._update_robot_path(robot, current_time, reserved_positions)
        
        # Second pass: Move robots
        for robot in active_robots:
            if robot.target and robot.path:
                self._move_robot(robot, current_time, reserved_positions, moving_robots)

    def _complete_task(self, robot, current_time):
        """Handle task completion"""
        completed_priority = robot.target.priority
        
        # Keep the robot's current position
        robot_pos = (robot.x, robot.y)
        
        # Remove task from grid and tracking
        self.grid.remove_task(robot.target)
        
        # Update robot state
        robot.target = None
        robot.path = []
        robot.completed_tasks += 1
        self.total_tasks_completed += 1
        robot.waiting = False
        robot.waiting_time = 0
        robot.last_move_time = current_time
        
        # Ensure robot stays on the grid at its current position
        self.grid.set_cell(robot_pos[0], robot_pos[1], CellType.ROBOT)
        
        # Update performance metrics
        if hasattr(robot, 'last_task_start'):
            completion_time = current_time - robot.last_task_start
            self.madql.metrics['completion_times'].append(completion_time)
        
        self.add_status_message(
            f"Robot {robot.id}: Completed P{completed_priority} task! Total: {robot.completed_tasks}"
        )
        print(f"Robot {robot.id} completed task and remains at position {robot_pos}")  # Debug print

    def _update_robot_path(self, robot, current_time, reserved_positions):
        """Update robot's path with collision avoidance"""
        if not robot.target:
            return
        
        # Get positions to avoid (only consider current robot positions)
        blocked = {(r.x, r.y) for r in self.robots if r != robot}
        
        # Find new path
        path = self.astar.find_path(
            (robot.x, robot.y),
            robot.target.get_position(),
            blocked,
            robot
        )
        
        if path:
            print(f"Found path for Robot {robot.id}: {path}")  # Debug print
            robot.path = path
            if len(path) > 1:  # Only remove current position if path has more than one point
                robot.path.pop(0)
            self.add_status_message(
                f"Robot {robot.id}: Found path to P{robot.target.priority} task, length {len(path)}"
            )
        else:
            print(f"No path found for Robot {robot.id} from ({robot.x}, {robot.y}) to {robot.target.get_position()}")  # Debug print
            robot.waiting = True
            robot.waiting_time += MOVE_DELAY
            self.add_status_message(
                f"Robot {robot.id}: Waiting for path to P{robot.target.priority} task"
            )

    def _move_robot(self, robot, current_time, reserved_positions, moving_robots):
        """Handle robot movement with improved collision avoidance"""
        if current_time - robot.last_move_time < MOVE_DELAY:
            return
        
        if not robot.path:
            return
        
        next_pos = robot.path[0]
        print(f"Robot {robot.id} attempting to move to {next_pos}")  # Debug print
        
        # Check if next position is the target
        is_target = robot.target and next_pos == robot.target.get_position()
        
        # Check if movement is safe
        can_move = True
        
        # Don't move if another robot is already moving to this position
        if next_pos in moving_robots and not is_target:
            print(f"Robot {robot.id} waiting - position {next_pos} reserved by moving robot")  # Debug print
            can_move = False
        
        # Don't move if position is reserved (unless it's the target)
        if next_pos in reserved_positions and not is_target:
            print(f"Robot {robot.id} waiting - position {next_pos} in reserved positions")  # Debug print
            can_move = False
        
        # Check for potential collisions with other robots' paths
        for other_robot in self.robots:
            if other_robot != robot and other_robot.path:
                if other_robot.path[0] == next_pos:
                    # If other robot has higher priority, wait
                    if (other_robot.target and other_robot.target.priority > robot.target.priority):
                        print(f"Robot {robot.id} waiting - collision with higher priority Robot {other_robot.id}")  # Debug print
                        can_move = False
                        break
        
        # Try to move
        if can_move:
            # Get current cell type before moving
            current_cell = self.grid.get_cell(robot.x, robot.y)
            next_cell = self.grid.get_cell(*next_pos)
            
            print(f"Robot {robot.id} moving from ({robot.x}, {robot.y}) to {next_pos}")  # Debug print
            print(f"Current cell: {current_cell}, Next cell: {next_cell}")  # Debug print
            
            if self.grid.move_robot(robot, *next_pos):
                print(f"Robot {robot.id} moved successfully to {next_pos}")  # Debug print
                moving_robots.add(next_pos)
                reserved_positions.add(next_pos)
                robot.path.pop(0)
                robot.last_move_time = current_time
                robot.total_distance += 1
                robot.waiting = False
                robot.waiting_time = 0
            else:
                print(f"Robot {robot.id} failed to move to {next_pos}")  # Debug print
                robot.waiting = True
                robot.waiting_time += MOVE_DELAY
        else:
            print(f"Robot {robot.id} cannot move - movement not safe")  # Debug print
            robot.waiting = True
            robot.waiting_time += MOVE_DELAY
            
            # If robot has been waiting too long, try to find alternative path
            if robot.waiting_time > MOVE_DELAY * 5:
                print(f"Robot {robot.id} waited too long, finding new path")  # Debug print
                self._update_robot_path(robot, current_time, reserved_positions)

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