import time
import numpy as np
from src.core.constants import GRID_SIZE

class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(game):
        end_time = time.time()
        metrics = {}
        
        # Basic timing metrics
        total_time = end_time - game.start_time
        metrics['total_time'] = total_time
        
        # Task completion metrics
        metrics['total_tasks'] = sum(robot.completed_tasks for robot in game.robots)
        metrics['tasks_per_second'] = metrics['total_tasks'] / total_time if total_time > 0 else 0
        
        # Priority-based metrics
        priority_completion_times = {1: [], 2: [], 3: []}
        for robot in game.robots:
            if hasattr(robot, 'task_completion_times'):
                for priority, time_taken in robot.task_completion_times:
                    priority_completion_times[priority].append(time_taken)
        
        for priority in [1, 2, 3]:
            times = priority_completion_times[priority]
            if times:
                metrics[f'p{priority}_avg_time'] = np.mean(times)
                metrics[f'p{priority}_tasks'] = len(times)
            else:
                metrics[f'p{priority}_avg_time'] = 0
                metrics[f'p{priority}_tasks'] = 0
        
        # Distance and efficiency metrics
        total_distance = sum(robot.total_distance for robot in game.robots)
        metrics['total_distance'] = total_distance
        metrics['distance_per_task'] = total_distance / metrics['total_tasks'] if metrics['total_tasks'] > 0 else 0
        
        # Collision and avoidance metrics
        total_replans = sum(getattr(robot, 'replan_count', 0) for robot in game.robots)
        total_waits = sum(getattr(robot, 'wait_count', 0) for robot in game.robots)
        metrics['total_replans'] = total_replans
        metrics['total_waits'] = total_waits
        metrics['collision_avoidance_rate'] = (
            total_replans + total_waits) / total_distance if total_distance > 0 else 0
        
        # Theoretical vs actual performance
        theoretical_min_distance = 0
        theoretical_min_time = 0
        
        for robot in game.robots:
            completed_tasks = robot.completed_tasks
            if completed_tasks > 0:
                avg_task_distance = robot.total_distance / completed_tasks
                theoretical_min_distance += avg_task_distance * 0.7  # Assuming 30% optimization possible
                theoretical_min_time += (end_time - robot.start_time) * 0.7
        
        metrics['time_saved'] = theoretical_min_time - total_time if theoretical_min_time > total_time else 0
        metrics['distance_saved'] = theoretical_min_distance - total_distance if theoretical_min_distance > total_distance else 0
        
        # Efficiency scores
        metrics['overall_efficiency'] = (
            (metrics['tasks_per_second'] * 50) +  # Weight task completion rate
            (metrics['collision_avoidance_rate'] * 30) +  # Weight collision avoidance
            ((1 - metrics['distance_per_task']/GRID_SIZE) * 20)  # Weight path efficiency
        ) / 100  # Normalize to 0-1 scale
        
        return metrics

    @staticmethod
    def format_metrics(metrics):
        """Format metrics for display"""
        return [
            f"Time: {metrics['total_time']:.1f}s",
            f"Tasks: {metrics['total_tasks']} ({metrics['tasks_per_second']:.2f}/s)",
            f"Distance: {metrics['total_distance']} ({metrics['distance_per_task']:.1f}/task)",
            f"Collisions Avoided: {metrics['total_replans'] + metrics['total_waits']}",
            f"Priority 1 Tasks: {metrics['p1_tasks']} ({metrics['p1_avg_time']:.1f}s avg)",
            f"Priority 2 Tasks: {metrics['p2_tasks']} ({metrics['p2_avg_time']:.1f}s avg)",
            f"Priority 3 Tasks: {metrics['p3_tasks']} ({metrics['p3_avg_time']:.1f}s avg)",
            f"Overall Efficiency: {metrics['overall_efficiency']:.2f}"
        ] 