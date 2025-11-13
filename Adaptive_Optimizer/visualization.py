"""
Visualization and Metrics System for Metro Adaptive Optimization.

This module provides comprehensive plotting and metrics visualization for the
adaptive metro optimization system, including real-time monitoring charts,
fitness score evolution, Z-score tracking, and performance comparisons.

Created: November 2025
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from real_time_monitor import MonitoringResult
from adaptive_ga import AdaptiveOptimizationResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    figure_size: Tuple[int, int] = (15, 10)
    dpi: int = 100
    save_plots: bool = True
    output_directory: str = "plots"
    plot_format: str = "png"
    color_scheme: str = "viridis"


class MetroVisualizationSystem:
    """
    Comprehensive visualization system for metro optimization analytics.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the visualization system.
        
        Parameters
        ----------
        config : Optional[VisualizationConfig]
            Configuration for visualization settings
        """
        self.config = config or VisualizationConfig()
        
        # Create output directory if needed
        import os
        if self.config.save_plots and not os.path.exists(self.config.output_directory):
            os.makedirs(self.config.output_directory)
        
        logger.info("Visualization system initialized")
    
    def plot_monitoring_dashboard(self, monitoring_history: List[MonitoringResult],
                                title: str = "Real-time Monitoring Dashboard",
                                z_threshold: float = 2.0) -> plt.Figure:
        """
        Create a comprehensive monitoring dashboard.
        
        Parameters
        ----------
        monitoring_history : List[MonitoringResult]
            History of monitoring results
        title : str
            Dashboard title
            
        Returns
        -------
        plt.Figure
            Dashboard figure
        """
        if not monitoring_history:
            logger.warning("No monitoring data available for dashboard")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=self.config.figure_size, dpi=self.config.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Extract data
        timestamps = [r.timestamp for r in monitoring_history]
        agg_z_scores = [r.aggregated_z_score for r in monitoring_history]
        threshold_flags = [r.threshold_exceeded for r in monitoring_history]
        
        # 1. Aggregated Z-score over time
        axes[0, 0].plot(timestamps, agg_z_scores, 'b-', linewidth=2, label='Aggregated Z-score')
        axes[0, 0].axhline(y=z_threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold ({z_threshold})')
        axes[0, 0].fill_between(timestamps, agg_z_scores, 0, alpha=0.3)
        
        # Highlight threshold exceedances
        for i, (t, z, exceeded) in enumerate(zip(timestamps, agg_z_scores, threshold_flags)):
            if exceeded:
                axes[0, 0].scatter(t, z, color='red', s=50, zorder=5)
        
        axes[0, 0].set_title('Aggregated Z-score Evolution')
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('Z-score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Station-wise Z-score heatmap
        if len(monitoring_history) > 1:
            station_names = list(monitoring_history[0].z_scores.keys())
            z_score_matrix = np.array([[r.z_scores.get(station, 0) for station in station_names] 
                                     for r in monitoring_history])
            
            im = axes[0, 1].imshow(z_score_matrix.T, aspect='auto', cmap='RdBu_r', 
                                 vmin=-3, vmax=3, interpolation='nearest')
            axes[0, 1].set_title('Station Z-scores Heatmap')
            axes[0, 1].set_xlabel('Time Steps')
            axes[0, 1].set_ylabel('Stations')
            axes[0, 1].set_yticks(range(len(station_names)))
            axes[0, 1].set_yticklabels(station_names, fontsize=8)
            plt.colorbar(im, ax=axes[0, 1], label='Z-score')
        
        # 3. Threshold exceedance frequency
        exceedance_counts = {}
        for result in monitoring_history:
            for station in result.stations_above_threshold:
                exceedance_counts[station] = exceedance_counts.get(station, 0) + 1
        
        if exceedance_counts:
            stations = list(exceedance_counts.keys())
            counts = list(exceedance_counts.values())
            axes[0, 2].bar(stations, counts, color='coral')
            axes[0, 2].set_title('Station Threshold Exceedances')
            axes[0, 2].set_xlabel('Stations')
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Passenger load trends (example for top 3 stations)
        station_names = list(monitoring_history[0].station_loads.keys())
        top_stations = station_names[:3]  # Show top 3 for clarity
        
        for station in top_stations:
            loads = [r.station_loads.get(station, 0) for r in monitoring_history]
            axes[1, 0].plot(timestamps, loads, linewidth=2, label=station, marker='o', markersize=3)
        
        axes[1, 0].set_title('Passenger Load Trends (Top 3 Stations)')
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('Passenger Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Z-score distribution histogram
        all_z_scores = []
        for result in monitoring_history:
            all_z_scores.extend(result.z_scores.values())
        
        if all_z_scores:
            axes[1, 1].hist(all_z_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 1].axvline(x=z_threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({z_threshold})')
            axes[1, 1].axvline(x=-z_threshold, color='red', linestyle='--', alpha=0.7)
            axes[1, 1].set_title('Z-score Distribution')
            axes[1, 1].set_xlabel('Z-score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. System health summary
        total_points = len(monitoring_history)
        exceedance_rate = sum(threshold_flags) / total_points if total_points > 0 else 0
        avg_z_score = np.mean(agg_z_scores) if agg_z_scores else 0
        max_z_score = max(agg_z_scores) if agg_z_scores else 0
        
        health_text = f"""System Health Summary:
        
Total Monitoring Points: {total_points}
Threshold Exceedance Rate: {exceedance_rate:.1%}
Average Aggregated Z-score: {avg_z_score:.2f}
Maximum Aggregated Z-score: {max_z_score:.2f}
Most Problematic Stations: {', '.join(list(exceedance_counts.keys())[:3])}
        """
        
        axes[1, 2].text(0.05, 0.95, health_text, transform=axes[1, 2].transAxes, 
                        verticalalignment='top', fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 2].set_title('System Health Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            filename = f"{self.config.output_directory}/monitoring_dashboard.{self.config.plot_format}"
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Monitoring dashboard saved to {filename}")
        
        return fig
    
    def plot_optimization_results(self, adaptation_history: List[AdaptiveOptimizationResult],
                                title: str = "Adaptive Optimization Results", 
                                z_threshold: float = 2.0) -> plt.Figure:
        """
        Plot optimization results and performance metrics.
        
        Parameters
        ----------
        adaptation_history : List[AdaptiveOptimizationResult]
            History of adaptive optimizations
        title : str
            Plot title
            
        Returns
        -------
        plt.Figure
            Optimization results figure
        """
        if not adaptation_history:
            logger.warning("No adaptation data available for plotting")
            return None
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16), dpi=self.config.dpi)
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        
        # Extract data
        optimization_times = [r.optimization_time for r in adaptation_history]
        original_fitness = [r.original_fitness for r in adaptation_history]
        adapted_fitness = [r.adapted_fitness for r in adaptation_history]
        improvements = [r.improvement_percentage for r in adaptation_history]
        computation_times = [r.total_optimization_time_seconds for r in adaptation_history]
        
        # Extract waiting time data (NEW)
        original_waiting_times = [getattr(r, 'original_avg_waiting_time', 0) for r in adaptation_history]
        adapted_waiting_times = [getattr(r, 'adapted_avg_waiting_time', 0) for r in adaptation_history]
        waiting_time_improvements = [getattr(r, 'waiting_time_improvement_percentage', 0) for r in adaptation_history]
        
        # 1. Fitness improvement over time
        axes[0, 0].plot(optimization_times, original_fitness, 'ro-', linewidth=2, 
                       label='Original Fitness', markersize=6)
        axes[0, 0].plot(optimization_times, adapted_fitness, 'go-', linewidth=2, 
                       label='Adapted Fitness', markersize=6)
        axes[0, 0].set_title('Fitness Evolution', fontsize=12, pad=15)
        axes[0, 0].set_xlabel('Time (minutes)', fontsize=10)
        axes[0, 0].set_ylabel('Fitness Score', fontsize=10)
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Waiting time evolution (NEW)
        axes[0, 1].plot(optimization_times, original_waiting_times, 'bo-', linewidth=2, 
                       label='Original', markersize=6)
        axes[0, 1].plot(optimization_times, adapted_waiting_times, 'mo-', linewidth=2, 
                       label='Adapted', markersize=6)
        axes[0, 1].set_title('Waiting Time Evolution', fontsize=12, pad=15)
        axes[0, 1].set_xlabel('Time (minutes)', fontsize=10)
        axes[0, 1].set_ylabel('Avg Wait Time (min)', fontsize=10)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Fitness improvement percentage
        axes[0, 2].bar(range(len(improvements)), improvements, 
                      color=['green' if x > 0 else 'red' for x in improvements], alpha=0.7)
        axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0, 2].set_title('Fitness Improvement %', fontsize=12, pad=15)
        axes[0, 2].set_xlabel('Adaptation #', fontsize=10)
        axes[0, 2].set_ylabel('Improvement (%)', fontsize=10)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Waiting time improvement percentage (NEW)
        axes[1, 0].bar(range(len(waiting_time_improvements)), waiting_time_improvements, 
                      color=['green' if x > 0 else 'red' for x in waiting_time_improvements], alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].set_title('Wait Time Improvement %', fontsize=12, pad=15)
        axes[1, 0].set_xlabel('Adaptation #', fontsize=10)
        axes[1, 0].set_ylabel('Improvement (%)', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Computation time analysis
        axes[1, 1].plot(optimization_times, computation_times, 'co-', linewidth=2, markersize=6)
        axes[1, 1].set_title('Computation Time', fontsize=12, pad=15)
        axes[1, 1].set_xlabel('Time (minutes)', fontsize=10)
        axes[1, 1].set_ylabel('Comp. Time (sec)', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Headway evolution (show first and last few trains)
        if adaptation_history:
            first_original = adaptation_history[0].original_headways[:4]
            last_adapted = adaptation_history[-1].adapted_headways[:4]
            
            x_pos = np.arange(len(first_original))
            width = 0.35
            
            axes[1, 2].bar(x_pos - width/2, first_original, width, 
                          label='Original', color='lightcoral', alpha=0.8)
            axes[1, 2].bar(x_pos + width/2, last_adapted, width, 
                          label='Adapted', color='lightgreen', alpha=0.8)
            axes[1, 2].set_title('Headway Comparison', fontsize=12, pad=15)
            axes[1, 2].set_xlabel('Train Number', fontsize=10)
            axes[1, 2].set_ylabel('Headway (minutes)', fontsize=10)
            axes[1, 2].set_xticks(x_pos)
            axes[1, 2].set_xticklabels([f'T{i+1}' for i in range(len(first_original))], fontsize=9)
            axes[1, 2].legend(fontsize=9)
            axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Z-score triggers analysis
        all_trigger_z_scores = []
        trigger_stations = []
        
        for result in adaptation_history:
            for station, z_score in result.z_scores_at_trigger.items():
                if abs(z_score) > z_threshold:  # Above threshold
                    all_trigger_z_scores.append(abs(z_score))
                    trigger_stations.append(station)
        
        if all_trigger_z_scores:
            # Count station triggers
            from collections import Counter
            station_counts = Counter(trigger_stations)
            
            stations = list(station_counts.keys())[:5]  # Top 5
            counts = [station_counts[s] for s in stations]
            
            axes[2, 0].bar(stations, counts, color='orange', alpha=0.7)
            axes[2, 0].set_title('Adaptation Triggers', fontsize=12, pad=15)
            axes[2, 0].set_xlabel('Stations', fontsize=10)
            axes[2, 0].set_ylabel('Trigger Count', fontsize=10)
            axes[2, 0].tick_params(axis='x', rotation=45, labelsize=9)
            axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Waiting time vs Fitness correlation (NEW)
        if original_waiting_times and adapted_waiting_times:
            axes[2, 1].scatter(original_fitness, original_waiting_times, 
                             color='red', alpha=0.6, s=60, label='Original')
            axes[2, 1].scatter(adapted_fitness, adapted_waiting_times, 
                             color='green', alpha=0.6, s=60, label='Adapted')
            axes[2, 1].set_title('Wait Time vs Fitness', fontsize=12, pad=15)
            axes[2, 1].set_xlabel('Fitness Score', fontsize=10)
            axes[2, 1].set_ylabel('Wait Time (min)', fontsize=10)
            axes[2, 1].legend(fontsize=9)
            axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Performance summary
        total_adaptations = len(adaptation_history)
        avg_improvement = np.mean(improvements) if improvements else 0
        avg_computation_time = np.mean(computation_times) if computation_times else 0
        successful_adaptations = sum(1 for x in improvements if x > 0)
        
        # Calculate waiting time summary (NEW)
        avg_waiting_improvement = np.mean(waiting_time_improvements) if waiting_time_improvements else 0
        successful_waiting_adaptations = sum(1 for x in waiting_time_improvements if x > 0)
        
        summary_text = f"""Performance Summary

Adaptations: {total_adaptations}
Success Rate: {successful_adaptations}/{total_adaptations} ({successful_adaptations/total_adaptations:.1%})
Wait Success: {successful_waiting_adaptations}/{total_adaptations} ({successful_waiting_adaptations/total_adaptations:.1%})

Avg Improvements:
• Fitness: {avg_improvement:.1f}%
• Wait Time: {avg_waiting_improvement:.1f}%

Best Results:
• Fitness: {max(improvements):.1f}%
• Wait Time: {max(waiting_time_improvements):.1f}%

Avg Comp Time: {avg_computation_time:.1f}s
        """
        
        axes[2, 2].text(0.05, 0.95, summary_text, transform=axes[2, 2].transAxes, 
                        verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[2, 2].set_title('Performance Summary', fontsize=12, pad=15)
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            filename = f"{self.config.output_directory}/optimization_results.{self.config.plot_format}"
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Optimization results saved to {filename}")
        
        return fig
    
    def plot_fitness_evolution(self, fitness_histories: List[List[float]], 
                             labels: List[str],
                             title: str = "Fitness Evolution Comparison") -> plt.Figure:
        """
        Plot fitness evolution for multiple optimization runs.
        
        Parameters
        ----------
        fitness_histories : List[List[float]]
            List of fitness evolution histories
        labels : List[str]
            Labels for each history
        title : str
            Plot title
            
        Returns
        -------
        plt.Figure
            Fitness evolution figure
        """
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.config.dpi)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(fitness_histories)))
        
        for history, label, color in zip(fitness_histories, labels, colors):
            generations = range(1, len(history) + 1)
            ax.plot(generations, history, 'o-', linewidth=2, markersize=4, 
                   label=label, color=color, alpha=0.8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add trend lines
        for history, color in zip(fitness_histories, colors):
            z = np.polyfit(range(len(history)), history, 1)
            p = np.poly1d(z)
            ax.plot(range(len(history)), p(range(len(history))), 
                   linestyle='--', color=color, alpha=0.5)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            filename = f"{self.config.output_directory}/fitness_evolution.{self.config.plot_format}"
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Fitness evolution plot saved to {filename}")
        
        return fig
    
    def plot_comparison_analysis(self, static_results: Dict, adaptive_results: Dict,
                               title: str = "Static vs Adaptive Optimization Comparison") -> plt.Figure:
        """
        Create a comprehensive comparison between static and adaptive optimization.
        
        Parameters
        ----------
        static_results : Dict
            Results from static optimization
        adaptive_results : Dict
            Results from adaptive optimization
        title : str
            Plot title
            
        Returns
        -------
        plt.Figure
            Comparison figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size, dpi=self.config.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Fitness comparison
        methods = ['Static', 'Adaptive']
        fitness_scores = [
            static_results.get('fitness', 0),
            adaptive_results.get('avg_fitness', 0)
        ]
        
        bars1 = axes[0, 0].bar(methods, fitness_scores, color=['coral', 'lightgreen'], alpha=0.7)
        axes[0, 0].set_title('Fitness Score Comparison')
        axes[0, 0].set_ylabel('Fitness Score')
        
        # Add value labels on bars
        for bar, value in zip(bars1, fitness_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(fitness_scores), 
                           f'{value:.2f}', ha='center', va='bottom')
        
        # 2. Waiting time comparison
        waiting_times = [
            static_results.get('avg_waiting_time', 0),
            adaptive_results.get('avg_waiting_time', 0)
        ]
        
        bars2 = axes[0, 1].bar(methods, waiting_times, color=['coral', 'lightgreen'], alpha=0.7)
        axes[0, 1].set_title('Average Waiting Time Comparison')
        axes[0, 1].set_ylabel('Waiting Time (minutes)')
        
        for bar, value in zip(bars2, waiting_times):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(waiting_times), 
                           f'{value:.2f}', ha='center', va='bottom')
        
        # 3. Passenger service comparison
        passengers_served = [
            static_results.get('total_passengers_served', 0),
            adaptive_results.get('avg_passengers_served', 0)
        ]
        
        passengers_left = [
            static_results.get('total_passengers_left', 0),
            adaptive_results.get('avg_passengers_left', 0)
        ]
        
        x_pos = np.arange(len(methods))
        width = 0.35
        
        bars3a = axes[1, 0].bar(x_pos - width/2, passengers_served, width, 
                               label='Served', color='lightblue', alpha=0.8)
        bars3b = axes[1, 0].bar(x_pos + width/2, passengers_left, width, 
                               label='Left Behind', color='lightcoral', alpha=0.8)
        
        axes[1, 0].set_title('Passenger Service Comparison')
        axes[1, 0].set_ylabel('Number of Passengers')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(methods)
        axes[1, 0].legend()
        
        # 4. Performance summary table
        summary_data = {
            'Metric': ['Fitness Score', 'Avg Waiting Time', 'Passengers Served', 'Passengers Left', 'Adaptations'],
            'Static': [
                f"{static_results.get('fitness', 0):.2f}",
                f"{static_results.get('avg_waiting_time', 0):.2f}",
                f"{static_results.get('total_passengers_served', 0)}",
                f"{static_results.get('total_passengers_left', 0)}",
                "0"
            ],
            'Adaptive': [
                f"{adaptive_results.get('avg_fitness', 0):.2f}",
                f"{adaptive_results.get('avg_waiting_time', 0):.2f}",
                f"{adaptive_results.get('avg_passengers_served', 0):.0f}",
                f"{adaptive_results.get('avg_passengers_left', 0):.0f}",
                f"{adaptive_results.get('total_adaptations', 0)}"
            ],
            'Improvement': [
                f"{((static_results.get('fitness', 1) - adaptive_results.get('avg_fitness', 1)) / static_results.get('fitness', 1) * 100):.1f}%",
                f"{((static_results.get('avg_waiting_time', 1) - adaptive_results.get('avg_waiting_time', 1)) / static_results.get('avg_waiting_time', 1) * 100):.1f}%",
                f"{((adaptive_results.get('avg_passengers_served', 0) - static_results.get('total_passengers_served', 0)) / max(static_results.get('total_passengers_served', 1), 1) * 100):.1f}%",
                f"{((static_results.get('total_passengers_left', 1) - adaptive_results.get('avg_passengers_left', 1)) / max(static_results.get('total_passengers_left', 1), 1) * 100):.1f}%",
                "N/A"
            ]
        }
        
        # Create table
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        table = axes[1, 1].table(cellText=[list(row) for row in zip(summary_data['Static'], 
                                                                   summary_data['Adaptive'], 
                                                                   summary_data['Improvement'])],
                               rowLabels=summary_data['Metric'],
                               colLabels=['Static', 'Adaptive', 'Improvement'],
                               cellLoc='center',
                               loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        axes[1, 1].set_title('Performance Summary')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            filename = f"{self.config.output_directory}/comparison_analysis.{self.config.plot_format}"
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Comparison analysis saved to {filename}")
        
        return fig
    
    def create_animated_monitoring_plot(self, monitoring_history: List[MonitoringResult],
                                      save_gif: bool = True, 
                                      z_threshold: float = 2.0) -> Optional[FuncAnimation]:
        """
        Create an animated plot of real-time monitoring data.
        
        Parameters
        ----------
        monitoring_history : List[MonitoringResult]
            History of monitoring results
        save_gif : bool
            Whether to save animation as GIF
            
        Returns
        -------
        Optional[FuncAnimation]
            Animation object if created successfully
        """
        if len(monitoring_history) < 2:
            logger.warning("Need at least 2 monitoring points for animation")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=self.config.dpi)
        fig.suptitle('Real-time Metro Monitoring Animation', fontsize=14, fontweight='bold')
        
        timestamps = [r.timestamp for r in monitoring_history]
        agg_z_scores = [r.aggregated_z_score for r in monitoring_history]
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Plot up to current frame
            current_timestamps = timestamps[:frame+1]
            current_z_scores = agg_z_scores[:frame+1]
            
            # Z-score evolution
            ax1.plot(current_timestamps, current_z_scores, 'b-', linewidth=2)
            ax1.axhline(y=z_threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold ({z_threshold})')
            ax1.set_title(f'Aggregated Z-score Evolution (Time: {timestamps[frame]:.1f} min)')
            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Z-score')
            ax1.set_ylim(0, max(max(agg_z_scores), 3))
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Current station status
            current_result = monitoring_history[frame]
            stations = list(current_result.z_scores.keys())
            z_values = list(current_result.z_scores.values())
            
            colors = ['red' if abs(z) > z_threshold else 'orange' if abs(z) > z_threshold/2 else 'green' 
                     for z in z_values]
            
            ax2.bar(range(len(stations)), z_values, color=colors, alpha=0.7)
            ax2.axhline(y=z_threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold ({z_threshold})')
            ax2.axhline(y=-z_threshold, color='r', linestyle='--', alpha=0.7)
            ax2.set_title(f'Station Z-scores at Time {timestamps[frame]:.1f} min')
            ax2.set_xlabel('Stations')
            ax2.set_ylabel('Z-score')
            ax2.set_xticks(range(len(stations)))
            ax2.set_xticklabels(stations, rotation=45, fontsize=8)
            ax2.set_ylim(-3, 3)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
        
        anim = FuncAnimation(fig, animate, frames=len(monitoring_history), 
                           interval=500, repeat=True, blit=False)
        
        if save_gif and self.config.save_plots:
            filename = f"{self.config.output_directory}/monitoring_animation.gif"
            try:
                anim.save(filename, writer='pillow', fps=2)
                logger.info(f"Animation saved to {filename}")
            except Exception as e:
                logger.warning(f"Could not save animation: {e}")
        
        return anim
    
    def generate_comprehensive_report(self, monitoring_history: List[MonitoringResult],
                                    adaptation_history: List[AdaptiveOptimizationResult],
                                    static_results: Dict,
                                    adaptive_summary: Dict,
                                    z_threshold: float = 2.0) -> None:
        """
        Generate a comprehensive report with all visualizations.
        
        Parameters
        ----------
        monitoring_history : List[MonitoringResult]
            History of monitoring results
        adaptation_history : List[AdaptiveOptimizationResult]
            History of adaptive optimizations
        static_results : Dict
            Results from static optimization
        adaptive_summary : Dict
            Summary of adaptive optimization performance
        z_threshold : float
            Z-score threshold value for visualization
        """
        logger.info("Generating comprehensive visualization report...")
        
        # 1. Monitoring Dashboard
        self.plot_monitoring_dashboard(monitoring_history, 
                                     "Real-time Monitoring Dashboard - Full Report",
                                     z_threshold)
        
        # 2. Optimization Results
        if adaptation_history:
            self.plot_optimization_results(adaptation_history, 
                                         "Adaptive Optimization Results - Full Report",
                                         z_threshold)
        
        # 3. Comparison Analysis
        self.plot_comparison_analysis(static_results, adaptive_summary,
                                    "Static vs Adaptive Optimization - Full Report")
        
        # 4. Animated monitoring (if data available)
        if len(monitoring_history) >= 5:  # Only create if enough data points
            self.create_animated_monitoring_plot(monitoring_history, save_gif=True, z_threshold=z_threshold)
        
        logger.info(f"Comprehensive report generated and saved to {self.config.output_directory}/")
        
        # Show all plots
        plt.show()


if __name__ == "__main__":
    # Example usage for testing
    from real_time_monitor import MonitoringResult
    from adaptive_ga import AdaptiveOptimizationResult
    
    # Create sample data for testing
    sample_monitoring = []
    for i in range(20):
        sample_monitoring.append(MonitoringResult(
            timestamp=i * 5.0,
            station_loads={f'S{j:02d}': 50 + 20 * np.sin(i/5) + 10 * np.random.random() 
                          for j in range(1, 17)},
            z_scores={f'S{j:02d}': 2 * np.sin(i/3) + 0.5 * np.random.random() - 0.25 
                     for j in range(1, 17)},
            aggregated_z_score=2.0 + 0.5 * np.sin(i/4) + 0.2 * np.random.random(),
            threshold_exceeded=(i % 7 == 0),
            stations_above_threshold=[f'S{j:02d}' for j in range(1, 4)] if i % 7 == 0 else []
        ))
    
    # Initialize visualization system
    viz = MetroVisualizationSystem()
    
    # Test monitoring dashboard
    print("Creating monitoring dashboard...")
    fig1 = viz.plot_monitoring_dashboard(sample_monitoring, "Test Monitoring Dashboard")
    
    # Test static vs adaptive comparison
    static_results = {
        'fitness': 250.5,
        'avg_waiting_time': 8.3,
        'total_passengers_served': 1450,
        'total_passengers_left': 120
    }
    
    adaptive_summary = {
        'avg_fitness': 220.3,
        'avg_waiting_time': 7.1,
        'avg_passengers_served': 1520,
        'avg_passengers_left': 95,
        'total_adaptations': 3
    }
    
    print("Creating comparison analysis...")
    fig2 = viz.plot_comparison_analysis(static_results, adaptive_summary, 
                                       "Test Static vs Adaptive Comparison")
    
    plt.show()
    
    print(f"Test plots saved to {viz.config.output_directory}/")