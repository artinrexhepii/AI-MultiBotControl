import heapq
from src.core.constants import GRID_SIZE
from src.core.entities import CellType

class AStar:
    def __init__(self, game):
        self.game = game
    
    def heuristic(self, a, b):
        return abs(b[0] - a[0]) + abs(b[1] - a[1])
    
    def get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            new_x, new_y = x + dx, y + dy
            # Check grid boundaries and static obstacles
            if (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and 
                self.game.grid[new_y][new_x] != CellType.OBSTACLE):
                
                # Check for other robots and treat them as obstacles
                position_blocked = False
                for robot in self.game.robots:
                    if (robot.x == new_x and robot.y == new_y) or \
                       (robot.path and robot.path[0] == (new_x, new_y)):  # Also check next planned position
                        position_blocked = True
                        break
                
                if not position_blocked:
                    neighbors.append((new_x, new_y))
        return neighbors
    
    def find_path(self, start, goal):
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
                
            for next_pos in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        # Reconstruct path
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        return path if path[0] == start else [] 