import heapq
from src.core.entities import CellType

class AStar:
    def __init__(self, grid):
        self.grid = grid
        
    def heuristic(self, a, b):
        """Manhattan distance heuristic"""
        return abs(b[0] - a[0]) + abs(b[1] - a[1])
        
    def is_valid_position(self, pos, ignore_robots=False, ignore_tasks=False):
        """Check if a position is valid for pathfinding"""
        x, y = pos
        if not (0 <= x < self.grid.width and 0 <= y < self.grid.height):
            return False
            
        cell = self.grid.get_cell(x, y)
        
        # Always consider obstacles as invalid
        if cell == CellType.OBSTACLE:
            return False
            
        # Optionally ignore robots as obstacles
        if cell == CellType.ROBOT and not ignore_robots:
            return False
            
        # Always allow task cells as valid destinations
        if cell in [CellType.TASK, CellType.TARGET]:
            return True
            
        return True
        
    def find_path(self, start, goal, ignore_robots=False, ignore_tasks=False):
        """Find path from start to goal using A* algorithm"""
        if not self.is_valid_position(start, ignore_robots, ignore_tasks) or \
           not self.is_valid_position(goal, ignore_robots, ignore_tasks):
            return None
            
        start = tuple(start)
        goal = tuple(goal)
        
        # Initialize data structures
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {start: None}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                return self.reconstruct_path(came_from, current)
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4-directional movement
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.is_valid_position(neighbor, ignore_robots, ignore_tasks):
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
        
    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from map"""
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        return path[::-1]  # Reverse path to get start->goal order 