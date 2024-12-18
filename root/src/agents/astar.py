import heapq
from src.core.constants import GRID_SIZE
from src.core.entities import CellType

class AStar:
    def __init__(self, game):
        self.game = game
    
    def heuristic(self, a, b):
        return abs(b[0] - a[0]) + abs(b[1] - a[1])
    
    def get_neighbors(self, pos, blocked_positions=None):
        x, y = pos
        neighbors = []
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            new_x, new_y = x + dx, y + dy
            new_pos = (new_x, new_y)
            if (0 <= new_x < GRID_SIZE and 
                0 <= new_y < GRID_SIZE and 
                self.game.grid[new_y][new_x] != CellType.OBSTACLE and
                (blocked_positions is None or new_pos not in blocked_positions)):
                neighbors.append(new_pos)
        return neighbors
    
    def find_path(self, start, goal, blocked_positions=None):
        if blocked_positions is None:
            blocked_positions = set()
            
        # If goal is blocked, no path is possible
        if goal in blocked_positions:
            return []
            
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
                
            for next_pos in self.get_neighbors(current, blocked_positions):
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        # Reconstruct path
        if goal not in came_from:  # No path found
            return []
            
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        
        # Verify path is valid
        return path if path and path[0] == start else [] 