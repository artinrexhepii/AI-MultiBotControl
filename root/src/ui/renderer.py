import pygame
from src.core.constants import *
from src.core.entities import CellType
from src.ui.button import Button

class Renderer:
    def __init__(self, screen_size, menu_width):
        self.screen = pygame.display.set_mode((screen_size + menu_width, screen_size))
        pygame.display.set_caption("Multi-Robot Control System")
        
        # Initialize buttons
        self.buttons = {
            'robot': Button(screen_size + 20, 50, 160, 40, "Robot"),
            'obstacle': Button(screen_size + 20, 100, 160, 40, "Obstacle"),
            'task': Button(screen_size + 20, 150, 160, 40, "Task"),
            'random': Button(screen_size + 20, 200, 160, 40, "Random Generate"),
            'play': Button(screen_size + 20, 250, 160, 40, "Play"),
            'end': Button(screen_size + 20, 300, 160, 40, "End")
        }
        
        self.screen_size = screen_size
        self.menu_width = menu_width
        self.cell_size = screen_size // GRID_SIZE
        
    def draw_grid(self, grid, robots):
        """Draw the grid and all entities"""
        self.screen.fill(WHITE)
        
        # Draw grid cells
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                 self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, WHITE, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1)
                
                cell = grid.get_cell(x, y)
                if cell == CellType.ROBOT:
                    self._draw_robot(x, y)
                elif cell == CellType.OBSTACLE:
                    self._draw_obstacle(x, y)
                elif cell == CellType.TARGET:
                    self._draw_target(x, y)
                elif cell == CellType.TASK:
                    self._draw_task(x, y)
        
        # Draw robot paths
        for robot in robots:
            if robot.path:
                self._draw_path(robot)
        
        # Draw menu
        self._draw_menu()
        
    def _draw_robot(self, x, y):
        """Draw robot at grid position"""
        center = (x * self.cell_size + self.cell_size//2, 
                 y * self.cell_size + self.cell_size//2)
        pygame.draw.circle(self.screen, BLUE, center, self.cell_size//3)
        
    def _draw_obstacle(self, x, y):
        """Draw obstacle at grid position"""
        rect = pygame.Rect(x * self.cell_size + 5, y * self.cell_size + 5,
                          self.cell_size - 10, self.cell_size - 10)
        pygame.draw.rect(self.screen, RED, rect)
        
    def _draw_target(self, x, y):
        """Draw target at grid position"""
        rect = pygame.Rect(x * self.cell_size + 5, y * self.cell_size + 5,
                          self.cell_size - 10, self.cell_size - 10)
        pygame.draw.rect(self.screen, GREEN, rect)
        
    def _draw_task(self, x, y):
        """Draw task at grid position"""
        rect = pygame.Rect(x * self.cell_size + 5, y * self.cell_size + 5,
                          self.cell_size - 10, self.cell_size - 10)
        pygame.draw.rect(self.screen, PURPLE, rect)
        
    def _draw_path(self, robot):
        """Draw robot's planned path"""
        points = [(p[0] * self.cell_size + self.cell_size//2,
                  p[1] * self.cell_size + self.cell_size//2)
                 for p in [(robot.x, robot.y)] + robot.path]
        pygame.draw.lines(self.screen, YELLOW, False, points, 2)
        
    def _draw_menu(self):
        """Draw menu background and buttons"""
        # Draw menu background
        pygame.draw.rect(self.screen, WHITE, 
                        (self.screen_size, 0, self.menu_width, self.screen_size))
        pygame.draw.line(self.screen, BLACK, 
                        (self.screen_size, 0), 
                        (self.screen_size, self.screen_size), 2)
        
        # Draw buttons
        for button in self.buttons.values():
            button.draw(self.screen)
            
    def draw_metrics(self, metrics, y_offset=350):
        """Draw performance metrics"""
        if not metrics:
            return
            
        font = pygame.font.Font(None, 24)
        metrics_text = [
            f"Time: {metrics['total_time']:.1f}s",
            f"Tasks: {metrics['total_tasks']}",
            f"Distance: {metrics['total_distance']}",
            f"Time Saved: {metrics['time_saved']:.1f}s",
            f"Tasks/s: {metrics['tasks_per_second']:.2f}"
        ]
        
        for text in metrics_text:
            text_surface = font.render(text, True, BLACK)
            self.screen.blit(text_surface, (self.screen_size + 20, y_offset))
            y_offset += 25
            
    def draw_status_messages(self, messages, y_offset=500):
        """Draw status messages"""
        if not messages:
            return
            
        font = pygame.font.Font(None, 24)
        
        # Draw status panel background
        status_panel = pygame.Rect(self.screen_size + 10, y_offset, 
                                 self.menu_width - 20, 200)
        pygame.draw.rect(self.screen, (240, 240, 240), status_panel)
        pygame.draw.rect(self.screen, BLACK, status_panel, 2)
        
        # Draw title
        title = font.render("Status Log:", True, BLACK)
        self.screen.blit(title, (self.screen_size + 20, y_offset + 10))
        
        # Draw messages with word wrap
        y_offset += 35
        for message in messages:
            words = message.split()
            lines = []
            line = []
            for word in words:
                if font.size(' '.join(line + [word]))[0] <= self.menu_width - 40:
                    line.append(word)
                else:
                    lines.append(' '.join(line))
                    line = [word]
            lines.append(' '.join(line))
            
            for line in lines:
                text_surface = font.render(line, True, BLACK)
                self.screen.blit(text_surface, (self.screen_size + 20, y_offset))
                y_offset += 20
                if y_offset > self.screen_size - 20:
                    break
                    
    def update_display(self):
        """Update the display"""
        pygame.display.flip()
        
    def get_grid_position(self, screen_pos):
        """Convert screen coordinates to grid position"""
        x, y = screen_pos
        if x < self.screen_size:
            return (x // self.cell_size, y // self.cell_size)
        return None
        
    def handle_button_click(self, pos):
        """Check if a button was clicked and return its name"""
        for name, button in self.buttons.items():
            if button.rect.collidepoint(pos):
                return name
        return None 