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
        
        # Learning metrics
        if hasattr(game.madql, 'get_learning_stats'):
            learning_stats = game.madql.get_learning_stats()
            if learning_stats:
                metrics.update({
                    'avg_reward': learning_stats['avg_reward'],
                    'avg_q_value': learning_stats['avg_q_value'],
                    'avg_bid': learning_stats['avg_bid']
                })
        
        if hasattr(game.madql.dqn, 'get_training_stats'):
            training_stats = game.madql.dqn.get_training_stats()
            if training_stats:
                metrics.update({
                    'policy_loss': training_stats['avg_policy_loss'],
                    'critic_loss': training_stats['avg_critic_loss'],
                    'dqn_q_value': training_stats['avg_q_value'],
                    'epsilon': training_stats['current_epsilon']
                })
        
        # Collision and avoidance metrics
        total_replans = sum(getattr(robot, 'replan_count', 0) for robot in game.robots)
        total_waits = sum(getattr(robot, 'wait_count', 0) for robot in game.robots)
        metrics['total_replans'] = total_replans
        metrics['total_waits'] = total_waits
        metrics['collision_avoidance_rate'] = (
            total_replans + total_waits) / total_distance if total_distance > 0 else 0
        
        # Team coordination metrics
        active_robots = len([r for r in game.robots if r.target])
        metrics['team_coordination'] = active_robots / len(game.robots)
        
        # Path efficiency
        optimal_distances = []
        actual_distances = []
        for robot in game.robots:
            if hasattr(robot, 'completed_task_distances'):
                for optimal, actual in robot.completed_task_distances:
                    optimal_distances.append(optimal)
                    actual_distances.append(actual)
        
        if optimal_distances:
            metrics['path_efficiency'] = sum(optimal_distances) / sum(actual_distances)
        else:
            metrics['path_efficiency'] = 0
        
        # Learning progress
        if hasattr(game.madql, 'episode_count'):
            metrics['training_episodes'] = game.madql.episode_count
        
        # Overall efficiency score
        metrics['overall_efficiency'] = (
            (metrics['tasks_per_second'] * 0.3) +  # Task completion rate
            (metrics['path_efficiency'] * 0.2) +   # Path efficiency
            (metrics['team_coordination'] * 0.2) + # Team coordination
            ((1 - metrics['collision_avoidance_rate']) * 0.2) + # Collision avoidance
            (min(1.0, metrics['total_tasks'] / 100) * 0.1)  # Progress
        )
        
        return metrics

    @staticmethod
    def format_metrics(metrics):
        """Format metrics for display"""
        lines = [
            f"Time: {metrics['total_time']:.1f}s",
            f"Tasks: {metrics['total_tasks']} ({metrics['tasks_per_second']:.2f}/s)",
            f"Distance: {metrics['total_distance']} ({metrics['distance_per_task']:.1f}/task)",
            f"Path Efficiency: {metrics['path_efficiency']:.2f}",
            f"Team Coordination: {metrics['team_coordination']:.2f}",
            f"Collisions Avoided: {metrics['total_replans'] + metrics['total_waits']}",
            f"Overall Efficiency: {metrics['overall_efficiency']:.2f}"
        ]
        
        # Add learning metrics if available
        if 'avg_reward' in metrics:
            lines.extend([
                f"Avg Reward: {metrics['avg_reward']:.2f}",
                f"Avg Q-Value: {metrics['avg_q_value']:.2f}",
                f"Avg Bid: {metrics['avg_bid']:.2f}"
            ])
            
        if 'policy_loss' in metrics:
            lines.extend([
                f"Policy Loss: {metrics['policy_loss']:.3f}",
                f"Critic Loss: {metrics['critic_loss']:.3f}",
                f"Epsilon: {metrics['epsilon']:.3f}"
            ])
            
        return lines