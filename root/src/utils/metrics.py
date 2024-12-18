import time

class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(game):
        """Calculate core performance metrics"""
        if not game.start_time:
            return None
            
        current_time = time.time()
        total_time = current_time - game.start_time
        
        # Calculate total distance and tasks
        total_distance = sum(robot.total_distance for robot in game.robots)
        total_tasks = game.total_tasks_completed
        
        # Calculate time saved through collaboration
        time_saved = 0
        if total_tasks > 0:
            # Estimate time that would be needed without collaboration
            single_robot_time = total_time * len(game.robots)
            time_saved = single_robot_time - total_time
        
        # Calculate tasks per second
        tasks_per_second = total_tasks / total_time if total_time > 0 else 0
        
        return {
            'total_time': total_time,
            'total_tasks': total_tasks,
            'total_distance': total_distance,
            'time_saved': time_saved,
            'tasks_per_second': tasks_per_second
        }

    @staticmethod
    def format_metrics(metrics):
        """Format metrics for display"""
        lines = [
            f"Time: {metrics['total_time']:.1f}s",
            f"Tasks: {metrics['total_tasks']} ({metrics['tasks_per_second']:.2f}/s)",
            f"Distance: {metrics['total_distance']} ({metrics['total_distance'] / metrics['total_tasks']:.1f}/task)",
            f"Time Saved: {metrics['time_saved']:.1f}s"
        ]
        
        return lines