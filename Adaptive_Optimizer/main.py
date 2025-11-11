"""
Main Application for Metro Adaptive Optimization System.

This is the main orchestration module that integrates all components:
- Historical data processing
- Initial genetic algorithm optimization
- Real-time monitoring with Z-score analysis
- Adaptive optimization when thresholds are exceeded
- Comprehensive visualization and reporting

Created: November 2025
"""

import sys
import os
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import json
import numpy as np

# Add the Simple Optimizer directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../Simple Optimizer'))

from data_processor import MetroDataProcessor
from real_time_monitor import RealTimeMonitor, MonitoringResult
from adaptive_ga import AdaptiveGeneticAlgorithm, AdaptiveOptimizationResult
from visualization import MetroVisualizationSystem, VisualizationConfig

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('metro_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MetroOptimizationConfig:
    """Configuration for the metro optimization system."""
    
    def __init__(self):
        # Data settings
        self.csv_file_path = "../data/simulation_log.csv"
        self.sample_size = 10000  # Use subset for proof of concept
        
        # Monitoring settings
        self.z_score_threshold = 2.0  # Threshold for triggering adaptive optimization
        self.monitoring_interval_minutes = 5.0
        self.noise_level = 0.2  # Noise in simulated real-time data
        
        # Simulation settings
        self.total_simulation_time_hours = 8.0  # 8 hours of operation
        self.adaptation_duration_minutes = 30.0  # Duration of demand adaptation
        
        # GA settings
        self.ga_params = {
            'alpha': 3.5,      # Weight for waiting time
            'beta': 0.25,      # Weight for leftover passengers
            'gamma': 50.0,     # Weight for number of trains
            'pop_size': 25,
            'generations': 30,
            'headway_min': 3,
            'headway_max': 10,
            'num_trains': 8,
            'mutation_rate': 0.15,
            'crossover_rate': 0.6
        }
        
        # Adaptive GA settings (faster for real-time response)
        self.adaptive_ga_params = {
            'alpha': 3.5,
            'beta': 0.25,
            'gamma': 50.0,
            'pop_size': 20,     # Smaller for faster optimization
            'generations': 20,   # Fewer generations for speed
            'headway_min': 3,
            'headway_max': 10,
            'num_trains': 8,
            'mutation_rate': 0.2,   # Higher mutation for exploration
            'crossover_rate': 0.6
        }
        
        # Simulator settings
        self.sim_params = {
            'n_stations': 16,
            'capacity': 1200,
            'initial_queues': None
        }
        
        # Output settings
        self.output_directory = "optimization_results"
        self.save_detailed_logs = True
        self.generate_visualizations = True


class MetroAdaptiveOptimizationSystem:
    """
    Main orchestration system for metro adaptive optimization.
    
    Integrates all components and manages the full optimization lifecycle:
    1. Data preprocessing and historical analysis
    2. Initial optimization with genetic algorithm
    3. Real-time monitoring and Z-score analysis
    4. Adaptive re-optimization when needed
    5. Comprehensive reporting and visualization
    """
    
    def __init__(self, config: Optional[MetroOptimizationConfig] = None):
        """
        Initialize the metro optimization system.
        
        Parameters
        ----------
        config : Optional[MetroOptimizationConfig]
            System configuration
        """
        self.config = config or MetroOptimizationConfig()
        
        # Initialize components
        self.data_processor = None
        self.monitor = None
        self.adaptive_ga = None
        self.visualization_system = None
        
        # System state
        self.monitoring_history = []
        self.adaptation_history = []
        self.current_headways = None
        self.initial_optimization_result = None
        self.system_start_time = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_adaptations': 0,
            'total_monitoring_points': 0,
            'avg_z_score': 0.0,
            'threshold_exceedances': 0,
            'total_optimization_time': 0.0
        }
        
        # Create output directory
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        logger.info("Metro Adaptive Optimization System initialized")
    
    def initialize_system(self) -> bool:
        """
        Initialize all system components.
        
        Returns
        -------
        bool
            True if initialization successful
        """
        try:
            logger.info("Initializing system components...")
            
            # 1. Initialize data processor
            logger.info("Loading and processing historical data...")
            self.data_processor = MetroDataProcessor(self.config.csv_file_path)
            self.data_processor.load_and_process_data(sample_size=self.config.sample_size)
            
            # Process historical patterns
            self.data_processor.calculate_hourly_arrival_rates()
            self.data_processor.calculate_od_matrix()
            self.data_processor.calculate_baseline_demand()
            
            # Add travel times to sim params
            self.config.sim_params['travel_times'] = self.data_processor.get_travel_times()
            
            logger.info(f"Data processing completed. Processed {len(self.data_processor.data)} trips.")
            
            # 2. Initialize real-time monitor
            self.monitor = RealTimeMonitor(
                data_processor=self.data_processor,
                z_score_threshold=self.config.z_score_threshold,
                monitoring_interval_minutes=self.config.monitoring_interval_minutes,
                noise_level=self.config.noise_level
            )
            
            # 3. Initialize adaptive GA
            self.adaptive_ga = AdaptiveGeneticAlgorithm(
                data_processor=self.data_processor,
                sim_params=self.config.sim_params,
                ga_params=self.config.adaptive_ga_params,
                adaptation_duration_minutes=self.config.adaptation_duration_minutes
            )
            
            # 4. Initialize visualization system
            if self.config.generate_visualizations:
                viz_config = VisualizationConfig(
                    output_directory=os.path.join(self.config.output_directory, "plots"),
                    save_plots=True
                )
                self.visualization_system = MetroVisualizationSystem(viz_config)
            
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def run_initial_optimization(self, current_time_minutes: float = 0.0) -> bool:
        """
        Run initial optimization based on historical data.
        
        Parameters
        ----------
        current_time_minutes : float
            Starting time for optimization
            
        Returns
        -------
        bool
            True if optimization successful
        """
        try:
            logger.info("Starting initial optimization...")
            start_time = time.time()
            
            # Run initial optimization with full GA parameters
            initial_ga = AdaptiveGeneticAlgorithm(
                data_processor=self.data_processor,
                sim_params=self.config.sim_params,
                ga_params=self.config.ga_params,  # Use full GA params for initial optimization
                adaptation_duration_minutes=self.config.adaptation_duration_minutes
            )
            
            self.current_headways, self.initial_optimization_result = initial_ga.initial_optimization(current_time_minutes)
            
            optimization_time = time.time() - start_time
            self.performance_metrics['total_optimization_time'] += optimization_time
            
            logger.info(f"Initial optimization completed in {optimization_time:.2f} seconds")
            logger.info(f"Initial headways: {self.current_headways}")
            logger.info(f"Initial fitness: {self.initial_optimization_result.fitness:.2f}")
            logger.info(f"Initial avg waiting time: {self.initial_optimization_result.avg_waiting_time:.2f} min")
            logger.info(f"Passengers served: {self.initial_optimization_result.total_passengers_served}")
            logger.info(f"Passengers left: {self.initial_optimization_result.total_passengers_left}")
            
            # Update adaptive GA with initial headways
            self.adaptive_ga.current_headways = self.current_headways
            
            return True
            
        except Exception as e:
            logger.error(f"Initial optimization failed: {e}")
            return False
    
    def run_monitoring_and_adaptation_loop(self) -> bool:
        """
        Run the main monitoring and adaptation loop.
        
        Returns
        -------
        bool
            True if loop completed successfully
        """
        try:
            logger.info("Starting monitoring and adaptation loop...")
            
            total_simulation_minutes = self.config.total_simulation_time_hours * 60
            current_time = 0.0
            
            self.system_start_time = datetime.now()
            
            while current_time <= total_simulation_minutes:
                # Perform monitoring step
                monitoring_result = self.monitor.monitor_step(current_time)
                self.monitoring_history.append(monitoring_result)
                
                self.performance_metrics['total_monitoring_points'] += 1
                self.performance_metrics['avg_z_score'] = np.mean([r.aggregated_z_score for r in self.monitoring_history])
                
                if monitoring_result.threshold_exceeded:
                    self.performance_metrics['threshold_exceedances'] += 1
                    
                    logger.info(f"Time {current_time:.1f}: Threshold exceeded - Z-score: {monitoring_result.aggregated_z_score:.2f}")
                    
                    # Check if adaptive optimization should be triggered
                    if self.monitor.should_trigger_adaptive_optimization(monitoring_result):
                        logger.info(f"Triggering adaptive optimization at time {current_time:.1f} minutes")
                        
                        start_time = time.time()
                        adaptation_result = self.adaptive_ga.adaptive_optimization(monitoring_result)
                        adaptation_time = time.time() - start_time
                        
                        self.adaptation_history.append(adaptation_result)
                        self.current_headways = adaptation_result.adapted_headways
                        
                        self.performance_metrics['total_adaptations'] += 1
                        self.performance_metrics['total_optimization_time'] += adaptation_time
                        
                        logger.info(f"Adaptive optimization completed in {adaptation_time:.2f} seconds")
                        logger.info(f"Improvement: {adaptation_result.improvement_percentage:.1f}%")
                
                # Progress update every 30 minutes
                if current_time % 30.0 == 0 and current_time > 0:
                    progress_pct = (current_time / total_simulation_minutes) * 100
                    logger.info(f"Simulation progress: {progress_pct:.1f}% ({current_time:.0f}/{total_simulation_minutes:.0f} min)")
                
                current_time += self.config.monitoring_interval_minutes
            
            logger.info("Monitoring and adaptation loop completed")
            return True
            
        except Exception as e:
            logger.error(f"Monitoring and adaptation loop failed: {e}")
            return False
    
    def generate_final_report(self) -> Dict:
        """
        Generate comprehensive final report.
        
        Returns
        -------
        Dict
            Final system performance report
        """
        logger.info("Generating final report...")
        
        # Calculate final performance metrics
        final_metrics = self._calculate_final_metrics()
        
        # Generate visualizations if enabled
        if self.config.generate_visualizations and self.visualization_system:
            self._generate_visualizations(final_metrics)
        
        # Save detailed logs if enabled
        if self.config.save_detailed_logs:
            self._save_detailed_logs(final_metrics)
        
        # Print summary to console
        self._print_summary_report(final_metrics)
        
        return final_metrics
    
    def _calculate_final_metrics(self) -> Dict:
        """
        Calculate comprehensive final performance metrics.
        
        Returns
        -------
        Dict
            Final performance metrics
        """
        # Static optimization metrics (baseline)
        static_metrics = {
            'fitness': self.initial_optimization_result.fitness if self.initial_optimization_result else 0,
            'avg_waiting_time': self.initial_optimization_result.avg_waiting_time if self.initial_optimization_result else 0,
            'total_passengers_served': self.initial_optimization_result.total_passengers_served if self.initial_optimization_result else 0,
            'total_passengers_left': self.initial_optimization_result.total_passengers_left if self.initial_optimization_result else 0
        }
        
        # Adaptive optimization metrics
        adaptive_metrics = {
            'total_adaptations': len(self.adaptation_history),
            'avg_fitness': np.mean([r.adapted_fitness for r in self.adaptation_history]) if self.adaptation_history else static_metrics['fitness'],
            'avg_improvement_percentage': np.mean([r.improvement_percentage for r in self.adaptation_history]) if self.adaptation_history else 0,
            'total_adaptation_time_seconds': sum(r.total_optimization_time_seconds for r in self.adaptation_history),
            'avg_adaptation_time_seconds': np.mean([r.total_optimization_time_seconds for r in self.adaptation_history]) if self.adaptation_history else 0
        }
        
        # Monitoring metrics
        monitoring_metrics = {
            'total_monitoring_points': len(self.monitoring_history),
            'threshold_exceedances': sum(1 for r in self.monitoring_history if r.threshold_exceeded),
            'avg_aggregated_z_score': np.mean([r.aggregated_z_score for r in self.monitoring_history]) if self.monitoring_history else 0,
            'max_aggregated_z_score': max(r.aggregated_z_score for r in self.monitoring_history) if self.monitoring_history else 0,
            'exceedance_rate': sum(1 for r in self.monitoring_history if r.threshold_exceeded) / len(self.monitoring_history) if self.monitoring_history else 0
        }
        
        # System performance
        system_metrics = {
            'total_simulation_time_hours': self.config.total_simulation_time_hours,
            'total_optimization_time_seconds': self.performance_metrics['total_optimization_time'],
            'optimization_efficiency': adaptive_metrics['total_adaptations'] / max(1, monitoring_metrics['threshold_exceedances']),
            'system_responsiveness': adaptive_metrics['avg_adaptation_time_seconds'] if adaptive_metrics['avg_adaptation_time_seconds'] > 0 else float('inf')
        }
        
        # Calculate overall improvement
        if static_metrics['fitness'] > 0 and adaptive_metrics['avg_fitness'] < static_metrics['fitness']:
            overall_improvement = ((static_metrics['fitness'] - adaptive_metrics['avg_fitness']) / static_metrics['fitness']) * 100
        else:
            overall_improvement = 0.0
        
        return {
            'static_optimization': static_metrics,
            'adaptive_optimization': adaptive_metrics,
            'monitoring_performance': monitoring_metrics,
            'system_performance': system_metrics,
            'overall_improvement_percentage': overall_improvement,
            'final_headways': self.current_headways,
            'simulation_timestamp': datetime.now().isoformat()
        }
    
    def _generate_visualizations(self, final_metrics: Dict) -> None:
        """
        Generate all visualization outputs.
        
        Parameters
        ----------
        final_metrics : Dict
            Final performance metrics
        """
        try:
            logger.info("Generating visualizations...")
            
            # Generate comprehensive report with all plots
            self.visualization_system.generate_comprehensive_report(
                monitoring_history=self.monitoring_history,
                adaptation_history=self.adaptation_history,
                static_results=final_metrics['static_optimization'],
                adaptive_summary=final_metrics['adaptive_optimization']
            )
            
            logger.info(f"Visualizations saved to {self.visualization_system.config.output_directory}")
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
    
    def _save_detailed_logs(self, final_metrics: Dict) -> None:
        """
        Save detailed logs and data.
        
        Parameters
        ----------
        final_metrics : Dict
            Final performance metrics
        """
        try:
            # Save final metrics as JSON
            metrics_file = os.path.join(self.config.output_directory, "final_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(final_metrics, f, indent=2, default=str)
            
            # Save monitoring history
            monitoring_file = os.path.join(self.config.output_directory, "monitoring_history.json")
            monitoring_data = [
                {
                    'timestamp': r.timestamp,
                    'aggregated_z_score': r.aggregated_z_score,
                    'threshold_exceeded': r.threshold_exceeded,
                    'stations_above_threshold': r.stations_above_threshold,
                    'z_scores': r.z_scores
                }
                for r in self.monitoring_history
            ]
            
            with open(monitoring_file, 'w') as f:
                json.dump(monitoring_data, f, indent=2)
            
            # Save adaptation history
            if self.adaptation_history:
                adaptation_file = os.path.join(self.config.output_directory, "adaptation_history.json")
                adaptation_data = [
                    {
                        'optimization_time': r.optimization_time,
                        'trigger_reason': r.trigger_reason,
                        'original_fitness': r.original_fitness,
                        'adapted_fitness': r.adapted_fitness,
                        'improvement_percentage': r.improvement_percentage,
                        'adapted_headways': r.adapted_headways,
                        'scaling_factors': r.scaling_factors
                    }
                    for r in self.adaptation_history
                ]
                
                with open(adaptation_file, 'w') as f:
                    json.dump(adaptation_data, f, indent=2)
            
            logger.info(f"Detailed logs saved to {self.config.output_directory}")
            
        except Exception as e:
            logger.error(f"Failed to save detailed logs: {e}")
    
    def _print_summary_report(self, final_metrics: Dict) -> None:
        """
        Print summary report to console.
        
        Parameters
        ----------
        final_metrics : Dict
            Final performance metrics
        """
        print("\n" + "="*80)
        print("METRO ADAPTIVE OPTIMIZATION SYSTEM - FINAL REPORT")
        print("="*80)
        
        print(f"\nSIMULATION PARAMETERS:")
        print(f"  Total simulation time: {self.config.total_simulation_time_hours} hours")
        print(f"  Monitoring interval: {self.config.monitoring_interval_minutes} minutes")
        print(f"  Z-score threshold: {self.config.z_score_threshold}")
        print(f"  Data sample size: {self.config.sample_size} trips")
        
        static = final_metrics['static_optimization']
        adaptive = final_metrics['adaptive_optimization']
        monitoring = final_metrics['monitoring_performance']
        
        print(f"\nOPTIMIZATION RESULTS:")
        print(f"  Initial (Static) Fitness: {static['fitness']:.2f}")
        print(f"  Final (Adaptive) Fitness: {adaptive['avg_fitness']:.2f}")
        print(f"  Overall Improvement: {final_metrics['overall_improvement_percentage']:.1f}%")
        
        print(f"\nPASSENGER SERVICE:")
        print(f"  Initial Passengers Served: {static['total_passengers_served']}")
        print(f"  Initial Passengers Left: {static['total_passengers_left']}")
        print(f"  Initial Avg Waiting Time: {static['avg_waiting_time']:.2f} minutes")
        
        print(f"\nADAPTIVE OPTIMIZATION:")
        print(f"  Total Adaptations: {adaptive['total_adaptations']}")
        print(f"  Average Improvement per Adaptation: {adaptive['avg_improvement_percentage']:.1f}%")
        print(f"  Average Adaptation Time: {adaptive['avg_adaptation_time_seconds']:.2f} seconds")
        
        print(f"\nMONITORING PERFORMANCE:")
        print(f"  Total Monitoring Points: {monitoring['total_monitoring_points']}")
        print(f"  Threshold Exceedances: {monitoring['threshold_exceedances']}")
        print(f"  Exceedance Rate: {monitoring['exceedance_rate']:.1%}")
        print(f"  Average Z-score: {monitoring['avg_aggregated_z_score']:.2f}")
        print(f"  Maximum Z-score: {monitoring['max_aggregated_z_score']:.2f}")
        
        print(f"\nFINAL HEADWAY SCHEDULE:")
        print(f"  {final_metrics['final_headways']}")
        
        print(f"\nOUTPUT LOCATIONS:")
        print(f"  Results directory: {self.config.output_directory}")
        if self.config.generate_visualizations:
            print(f"  Plots directory: {os.path.join(self.config.output_directory, 'plots')}")
        
        print("\n" + "="*80)
    
    def run_complete_simulation(self) -> bool:
        """
        Run the complete simulation from start to finish.
        
        Returns
        -------
        bool
            True if simulation completed successfully
        """
        logger.info("Starting complete metro optimization simulation...")
        
        # Step 1: Initialize system
        if not self.initialize_system():
            return False
        
        # Step 2: Run initial optimization
        if not self.run_initial_optimization():
            return False
        
        # Step 3: Run monitoring and adaptation loop
        if not self.run_monitoring_and_adaptation_loop():
            return False
        
        # Step 4: Generate final report
        final_metrics = self.generate_final_report()
        
        logger.info("Complete simulation finished successfully")
        return True


def main():
    """
    Main entry point for the metro optimization application.
    """
    parser = argparse.ArgumentParser(description="Metro Adaptive Optimization System")
    parser.add_argument("--sample-size", type=int, default=10000, 
                      help="Number of data rows to process (default: 10000)")
    parser.add_argument("--simulation-hours", type=float, default=8.0,
                      help="Simulation duration in hours (default: 8.0)")
    parser.add_argument("--z-threshold", type=float, default=2.0,
                      help="Z-score threshold for triggering adaptation (default: 2.0)")
    parser.add_argument("--output-dir", type=str, default="optimization_results",
                      help="Output directory for results (default: optimization_results)")
    parser.add_argument("--no-viz", action="store_true",
                      help="Disable visualization generation")
    
    args = parser.parse_args()
    
    # Create configuration
    config = MetroOptimizationConfig()
    config.sample_size = args.sample_size
    config.total_simulation_time_hours = args.simulation_hours
    config.z_score_threshold = args.z_threshold
    config.output_directory = args.output_dir
    config.generate_visualizations = not args.no_viz
    
    # Run simulation
    system = MetroAdaptiveOptimizationSystem(config)
    
    try:
        success = system.run_complete_simulation()
        if success:
            print("\n✓ Simulation completed successfully!")
            print(f"Check {config.output_directory}/ for detailed results.")
            return 0
        else:
            print("\n✗ Simulation failed. Check logs for details.")
            return 1
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"Simulation failed with unexpected error: {e}")
        print(f"\n✗ Simulation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())