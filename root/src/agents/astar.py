import heapq
from src.core.constants import GRID_SIZE, MOVE_DELAY
from src.core.entities import CellType
import time

class AStar:
    def __init__(self, game):
        self.game = game
        self.time_window = 5  # Time window for temporal planning
        self.safety_distance = 2  # Minimum distance between robots
        
    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
    def get_neighbors(self, pos, blocked_positions=None, robot=None, is_target_cell=False, ignore_tasks=False, ignore_robots=False):
        """Get valid neighboring positions with improved collision avoidance"""
        if blocked_positions is None:
            blocked_positions = set()
            
        neighbors = []
        x, y = pos
        
        print(f"Getting neighbors for position ({x}, {y})")
        
        # Only consider 4 cardinal directions (no diagonals)
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            new_x, new_y = x + dx, y + dy
            new_pos = (new_x, new_y)
            
            # Check grid boundaries
            if not (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE):
                continue
                
            cell = self.game.grid.get_cell(new_x, new_y)
            print(f"  Checking neighbor ({new_x}, {new_y}): {cell}")
            
            # Handle different cell types
            if cell == CellType.OBSTACLE:
                print(f"  Skipping obstacle at ({new_x}, {new_y})")
                continue
                
            if cell == CellType.ROBOT and not ignore_robots:
                # Only allow if it's the target cell AND the robot there is about to move
                if is_target_cell:
                    robot_at_pos = None
                    for r in self.game.robots:
                        if (r.x, r.y) == new_pos:
                            robot_at_pos = r
                            break
                    
                    if robot_at_pos and robot_at_pos.path and len(robot_at_pos.path) > 1:
                        # Verify the robot will move before we get there
                        time_to_move = len(robot_at_pos.path) * MOVE_DELAY
                        if time_to_move < 1.0:  # If robot will move within 1 second
                            print(f"  Adding robot position with high cost at ({new_x}, {new_y})")
                            neighbors.append((new_pos, 5.0))  # High cost to prefer other paths
                    else:
                        print(f"  Skipping robot position at ({new_x}, {new_y})")
                continue
            
            # Always allow task cells as valid neighbors
            if cell in [CellType.TASK, CellType.TARGET]:
                print(f"  Adding task position at ({new_x}, {new_y})")
                neighbors.append((new_pos, 1.0))
                continue
            
            # Check blocked positions
            if new_pos in blocked_positions and not is_target_cell:
                print(f"  Position ({new_x}, {new_y}) is blocked")
                continue
            
            # Check for nearby robots if not ignoring them
            if not ignore_robots:
                too_close = False
                for other_robot in self.game.robots:
                    if other_robot != robot:
                        dist = self.manhattan_distance(new_pos, (other_robot.x, other_robot.y))
                        if dist < 2:  # Minimum separation of 2 cells
                            too_close = True
                            print(f"  Position ({new_x}, {new_y}) is too close to robot {other_robot.id}")
                            break
                
                if too_close and not is_target_cell:
                    continue
            
            # Add position with movement cost
            base_cost = 1.0
            
            # Add congestion cost if not ignoring robots
            if not ignore_robots:
                congestion_cost = 0.0
                for other_robot in self.game.robots:
                    if other_robot != robot:
                        dist = self.manhattan_distance(new_pos, (other_robot.x, other_robot.y))
                        if dist < self.safety_distance:
                            congestion_cost += (self.safety_distance - dist) * 0.5
                        
                        # Add cost for moving towards other robot's path
                        if other_robot.path:
                            for i, path_pos in enumerate(other_robot.path[:5]):  # Check next 5 steps
                                if path_pos == new_pos:
                                    congestion_cost += 2.0 / (i + 1)  # Higher cost for immediate conflicts
                
                total_cost = base_cost + congestion_cost
            else:
                total_cost = base_cost
            
            print(f"  Adding position ({new_x}, {new_y}) with cost {total_cost:.2f}")
            neighbors.append((new_pos, total_cost))
        
        print(f"Found {len(neighbors)} valid neighbors")
        return neighbors
        
    def _is_position_safe(self, pos, robot, is_target_cell):
        """Enhanced safety check for a position"""
        if not robot:
            return True
            
        x, y = pos
        
        # Check for immediate collisions with other robots
        for other_robot in self.game.robots:
            if other_robot == robot:
                continue
                
            # Check current position
            if (other_robot.x, other_robot.y) == pos and not is_target_cell:
                return False
                
            # Check safety distance (only for non-target cells)
            if not is_target_cell:
                dist = self.manhattan_distance((other_robot.x, other_robot.y), pos)
                if dist < self.safety_distance:
                    return False
        
        return True
        
    def find_path(self, start, goal, blocked_positions=None, robot=None, ignore_robots=False, ignore_tasks=False):
        """Find path using A* algorithm with improved collision avoidance"""
        if blocked_positions is None:
            blocked_positions = set()
            
        # Check if start or goal is blocked
        start_cell = self.game.grid.get_cell(*start)
        goal_cell = self.game.grid.get_cell(*goal)
        
        print(f"Path finding from {start} to {goal}")
        print(f"Start cell type: {start_cell}, Goal cell type: {goal_cell}")
        
        if start_cell == CellType.OBSTACLE:
            print(f"Path finding failed: Start position is blocked")
            return None
            
        # Initialize data structures
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        # Keep track of explored alternatives
        alternatives = []
        max_alternatives = 3
        explored_count = 0
        
        # Search for path
        while frontier:
            current = heapq.heappop(frontier)[1]
            explored_count += 1
            
            if current == goal:
                # Found a path, store it as an alternative
                path = self._reconstruct_path(came_from, start, goal)
                if path:
                    alternatives.append(path)
                    print(f"Found path of length {len(path)}")
                    if len(alternatives) >= max_alternatives:
                        break
                continue
            
            # Check if current position is the goal
            is_target_cell = (current == goal)
            
            # Get valid neighbors (only cardinal directions)
            neighbors = self.get_neighbors(current, blocked_positions, robot if not ignore_robots else None, is_target_cell, ignore_tasks, ignore_robots)
            if not neighbors:
                print(f"No valid neighbors found for position {current}")
            
            for next_pos, move_cost in neighbors:
                # Calculate new cost including movement cost
                new_cost = cost_so_far[current] + move_cost
                
                # Add costs for various factors if not ignoring robots
                if not ignore_robots:
                    new_cost += self._calculate_position_cost(next_pos, robot, goal)
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.manhattan_distance(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        print(f"Explored {explored_count} positions")
        
        # Choose the best alternative path
        if alternatives:
            # Sort alternatives by length and congestion
            scored_paths = []
            for path in alternatives:
                congestion = sum(self._calculate_path_congestion(path, robot if not ignore_robots else None))
                score = len(path) + (congestion * 2 if not ignore_robots else 0)  # Weight congestion more heavily
                scored_paths.append((score, path))
            
            # Return the path with the lowest score
            best_path = min(scored_paths, key=lambda x: x[0])[1]
            print(f"Selected best path of length {len(best_path)}")
            return best_path
        
        print(f"No path found from {start} to {goal}")
        return None
        
    def _reconstruct_path(self, came_from, start, goal):
        """Reconstruct path from came_from dict"""
        if goal not in came_from:
            return None
            
        current = goal
        path = []
        while current is not None:
            path.append(current)
            current = came_from.get(current)
            
        path.reverse()
        return path if path[0] == start else None
        
    def _calculate_path_congestion(self, path, robot):
        """Calculate congestion along a path"""
        congestion_values = []
        for pos in path:
            congestion = 0
            for other_robot in self.game.robots:
                if other_robot != robot:
                    dist = self.manhattan_distance(pos, (other_robot.x, other_robot.y))
                    if dist < self.safety_distance:
                        congestion += (self.safety_distance - dist)
            congestion_values.append(congestion)
        return congestion_values
        
    def _calculate_position_cost(self, pos, robot, goal):
        """Calculate additional cost factors for a position"""
        cost = 0
        x, y = pos
        
        # Base distance cost
        distance_to_goal = self.manhattan_distance(pos, goal)
        cost += distance_to_goal * 0.1  # Small cost for distance
        
        # Cost for proximity to other robots
        for other_robot in self.game.robots:
            if other_robot != robot:
                dist = self.manhattan_distance(pos, (other_robot.x, other_robot.y))
                if dist < self.safety_distance:
                    cost += (self.safety_distance - dist)
        
        return cost
        
    def _post_process_path(self, path, robot):
        """Post-process path for smoothness and safety"""
        if not path or len(path) < 3:
            return path
            
        # Smooth path by removing unnecessary waypoints
        smoothed_path = [path[0]]
        for i in range(1, len(path) - 1):
            prev_pos = smoothed_path[-1]
            current_pos = path[i]
            next_pos = path[i + 1]
            
            # Check if we can skip the current waypoint
            if self._is_safe_move(prev_pos, next_pos, robot):
                continue
            smoothed_path.append(current_pos)
        smoothed_path.append(path[-1])
        
        # Ensure minimum distance between waypoints
        final_path = [smoothed_path[0]]
        for i in range(1, len(smoothed_path)):
            current_pos = smoothed_path[i]
            if self.manhattan_distance(final_path[-1], current_pos) >= 1:
                final_path.append(current_pos)
        
        return final_path
        
    def _is_safe_move(self, start, end, robot):
        """Check if moving directly between two points is safe"""
        # Check if points are too far for direct movement
        if self.manhattan_distance(start, end) > 2:  # Allow diagonal movement
            return False
            
        # Check for obstacles
        cell = self.game.grid.get_cell(*end)
        if cell == CellType.OBSTACLE:
            return False
            
        # Check for other robots
        for other_robot in self.game.robots:
            if other_robot != robot:
                if (other_robot.x, other_robot.y) == end:
                    return False
                if other_robot.path and end == other_robot.path[0]:
                    return False
                # Check if the move crosses another robot's path
                if other_robot.path:
                    for path_pos in other_robot.path[:self.time_window]:
                        if self.manhattan_distance(end, path_pos) < self.safety_distance:
                            return False
        
        return True