"""
Adaptive Genetic Algorithm for Metro Headway Optimization.

This module extends the basic genetic algorithm to handle dynamic scenarios where
passenger demand patterns change in real-time. It adapts optimization based on
Z-score thresholds and scales demand for specific time windows while maintaining
historical patterns for the rest of the day.

Created: November 2025
"""

import sys
import os

# Add the Simple Optimizer directory to the path to import existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../Simple Optimizer'))

import numpy as np
import random
import copy
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from Simulator import HeadwaySimulator, SimulatorResult
from GA import GeneticAlgorithm
from data_processor import MetroDataProcessor
from real_time_monitor import RealTimeMonitor, MonitoringResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AdaptiveOptimizationResult:
    """Container for adaptive optimization results."""
    optimization_time: float
    trigger_reason: str
    z_scores_at_trigger: Dict[str, float]
    scaling_factors: Dict[str, float]
    original_headways: List[float]
    adapted_headways: List[float]
    original_fitness: float
    adapted_fitness: float
    improvement_percentage: float
    adaptation_period_minutes: float
    total_optimization_time_seconds: float
    # NEW: Add passenger serving data
    original_passengers_served: int = 0
    adapted_passengers_served: int = 0
    passengers_improvement_percentage: float = 0.0
    # NEW: Add waiting time data
    original_avg_waiting_time: float = 0.0
    adapted_avg_waiting_time: float = 0.0
    waiting_time_improvement_percentage: float = 0.0


class AdaptiveLambdaFunction:
    """
    Adaptive lambda function that changes demand based on time windows and scaling factors.
    """
    
    def __init__(self, base_lambda_func: Callable, station_index: int, 
                 adaptation_start_time: float, adaptation_duration: float,
                 scaling_factor: float):
        """
        Initialize adaptive lambda function.
        
        Parameters
        ----------
        base_lambda_func : Callable
            Original lambda function from historical data
        station_index : int
            Index of the station this lambda applies to
        adaptation_start_time : float
            Time when adaptation starts (minutes)
        adaptation_duration : float
            Duration of adaptation period (minutes)
        scaling_factor : float
            Factor to scale demand during adaptation period
        """
        self.base_lambda_func = base_lambda_func
        self.station_index = station_index
        self.adaptation_start_time = adaptation_start_time
        self.adaptation_end_time = adaptation_start_time + adaptation_duration
        self.scaling_factor = scaling_factor
    
    def __call__(self, t: float) -> float:
        """
        Calculate arrival rate at time t.
        
        Parameters
        ----------
        t : float
            Time in minutes
            
        Returns
        -------
        float
            Arrival rate (passengers/minute)
        """
        base_rate = self.base_lambda_func(t)
        
        # Apply scaling only during adaptation period
        if self.adaptation_start_time <= t <= self.adaptation_end_time:
            return base_rate * self.scaling_factor
        else:
            return base_rate


class AdaptiveGeneticAlgorithm:
    """
    Adaptive genetic algorithm for real-time metro optimization.
    
    Extends the basic GA to handle dynamic passenger demand changes
    and optimize headways based on real-time monitoring data.
    """
    
    def __init__(self, data_processor: MetroDataProcessor,
                 sim_params: Dict,
                 ga_params: Dict,
                 adaptation_duration_minutes: float = 30.0):
        """
        Initialize the adaptive genetic algorithm.
        
        Parameters
        ----------
        data_processor : MetroDataProcessor
            Processed historical data
        sim_params : Dict
            Simulator parameters (n_stations, capacity, travel_times, etc.)
        ga_params : Dict
            GA parameters (pop_size, generations, mutation_rate, etc.)
        adaptation_duration_minutes : float
            Duration for which demand adaptation applies (default 30 minutes)
        """
        self.data_processor = data_processor
        self.adaptation_duration = adaptation_duration_minutes
        
        # Initialize simulator
        self.simulator = HeadwaySimulator(
            n_stations=sim_params.get('n_stations', 16),
            capacity=sim_params.get('capacity', 1200),
            travel_times=sim_params.get('travel_times', data_processor.get_travel_times()),
            p_dest=data_processor.od_matrix,
            initial_queues=sim_params.get('initial_queues', None)
        )
        
        # Store GA parameters
        self.ga_params = ga_params
        
        # Current optimization state
        self.current_headways = None
        self.baseline_lambda_funcs = None
        self.adaptation_history = []
        
        logger.info("Adaptive Genetic Algorithm initialized")
        logger.info(f"Adaptation duration: {adaptation_duration_minutes} minutes")
    
    def initial_optimization(self, current_time_minutes: float = 0.0) -> Tuple[List[float], SimulatorResult]:
        """
        Perform initial optimization based on historical data.
        
        Parameters
        ----------
        current_time_minutes : float
            Current time for lambda function generation
            
        Returns
        -------
        Tuple[List[float], SimulatorResult]
            (optimized_headways, simulation_result)
        """
        logger.info("Starting initial optimization with historical data")
        
        # Get lambda functions based on historical patterns
        self.baseline_lambda_funcs = self.data_processor.get_lambda_functions(current_time_minutes)
        
        # Create and run GA
        logger.info(f"Creating GA with parameters: pop_size={self.ga_params.get('pop_size', 25)}, generations={self.ga_params.get('generations', 30)}")
        logger.info(f"GA bounds: headway_min={self.ga_params.get('headway_min', 5)}, headway_max={self.ga_params.get('headway_max', 15)}")
        
        ga = GeneticAlgorithm(
            sim=self.simulator,
            lambdas=self.baseline_lambda_funcs,
            alpha=self.ga_params.get('alpha', 3.5),
            beta=self.ga_params.get('beta', 0.25),
            gamma=self.ga_params.get('gamma', 50.0),
            pop_size=self.ga_params.get('pop_size', 25),
            generations=self.ga_params.get('generations', 30),
            headway_min=self.ga_params.get('headway_min', 5),
            headway_max=self.ga_params.get('headway_max', 15),
            num_trains=self.ga_params.get('num_trains', 8),
            mutation_rate=self.ga_params.get('mutation_rate', 0.15),
            crossover_rate=self.ga_params.get('crossover_rate', 0.6)
        )
        
        start_time = datetime.now()
        logger.info("Running initial genetic algorithm optimization...")
        best_headways, best_fitness, fitness_history, best_result = ga.run()
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"GA optimization completed with {len(fitness_history)} generations")
        self.current_headways = best_headways
        
        logger.info(f"Initial optimization completed in {optimization_time:.2f} seconds")
        logger.info(f"Best fitness: {best_fitness:.2f}")
        logger.info(f"Best headways: {best_headways}")
        
        return best_headways, best_result
    
    def adaptive_optimization(self, monitoring_result: MonitoringResult) -> AdaptiveOptimizationResult:
        """
        Perform adaptive optimization based on real-time monitoring data.
        
        Parameters
        ----------
        monitoring_result : MonitoringResult
            Current monitoring result that triggered adaptation
            
        Returns
        -------
        AdaptiveOptimizationResult
            Results of the adaptive optimization
        """
        start_time = datetime.now()
        current_time = monitoring_result.timestamp
        
        logger.info(f"Starting adaptive optimization at time {current_time:.1f} minutes")
        
        # Calculate scaling factors based on Z-scores
        scaling_factors = self._calculate_scaling_factors(monitoring_result.z_scores)
        
        # Create adaptive lambda functions
        adaptive_lambdas = self._create_adaptive_lambda_functions(
            current_time, scaling_factors
        )
        
        # Store original headways and fitness for comparison
        original_headways = self.current_headways.copy() if self.current_headways else None
        original_result = None
        if original_headways:
            original_result = self.simulator.simulate(
                original_headways, adaptive_lambdas, 
                weights=(self.ga_params.get('alpha', 3.5), 
                        self.ga_params.get('beta', 0.25), 
                        self.ga_params.get('gamma', 50.0))
            )
        
        # Run adaptive GA with modified lambda functions
        logger.info(f"Creating adaptive GA with parameters: pop_size={self.ga_params.get('pop_size', 20)}, generations={self.ga_params.get('generations', 20)}")
        logger.info(f"Adaptive GA bounds: headway_min={self.ga_params.get('headway_min', 5)}, headway_max={self.ga_params.get('headway_max', 15)}")
        
        ga = GeneticAlgorithm(
            sim=self.simulator,
            lambdas=adaptive_lambdas,
            alpha=self.ga_params.get('alpha', 3.5),
            beta=self.ga_params.get('beta', 0.25),
            gamma=self.ga_params.get('gamma', 50.0),
            pop_size=self.ga_params.get('pop_size', 20),  # Slightly smaller for faster adaptation
            generations=self.ga_params.get('generations', 20),  # Fewer generations for faster response
            headway_min=self.ga_params.get('headway_min', 5),
            headway_max=self.ga_params.get('headway_max', 15),
            num_trains=self.ga_params.get('num_trains', 8),
            mutation_rate=self.ga_params.get('mutation_rate', 0.2),  # Slightly higher mutation for exploration
            crossover_rate=self.ga_params.get('crossover_rate', 0.6)
        )
        
        # Run optimization
        logger.info("Running adaptive genetic algorithm optimization...")
        adapted_headways, adapted_fitness, fitness_history, adapted_result = ga.run()
        logger.info(f"Adaptive GA optimization completed with {len(fitness_history)} generations")
        
        # Update current headways
        self.current_headways = adapted_headways
        
        optimization_time_seconds = (datetime.now() - start_time).total_seconds()
        
        # Calculate improvement
        original_fitness = original_result.fitness if original_result else float('inf')
        improvement_pct = ((original_fitness - adapted_fitness) / original_fitness * 100 
                          if original_fitness != 0 and original_fitness != float('inf') else 0.0)
        
        # Calculate passenger serving metrics
        original_passengers = original_result.total_passengers_served if original_result else 0
        adapted_passengers = adapted_result.total_passengers_served if adapted_result else 0
        passengers_improvement_pct = ((adapted_passengers - original_passengers) / max(original_passengers, 1) * 100
                                    if original_passengers > 0 else 0.0)
        
        # Calculate waiting time metrics
        original_waiting_time = original_result.avg_waiting_time if original_result else 0.0
        adapted_waiting_time = adapted_result.avg_waiting_time if adapted_result else 0.0
        waiting_time_improvement_pct = ((original_waiting_time - adapted_waiting_time) / max(original_waiting_time, 0.01) * 100
                                      if original_waiting_time > 0 else 0.0)
        
        # Create result object
        result = AdaptiveOptimizationResult(
            optimization_time=current_time,
            trigger_reason=f"Z-score threshold exceeded: {monitoring_result.aggregated_z_score:.2f}",
            z_scores_at_trigger=monitoring_result.z_scores.copy(),
            scaling_factors=scaling_factors.copy(),
            original_headways=original_headways or [],
            adapted_headways=adapted_headways,
            original_fitness=original_fitness,
            adapted_fitness=adapted_fitness,
            improvement_percentage=improvement_pct,
            adaptation_period_minutes=self.adaptation_duration,
            total_optimization_time_seconds=optimization_time_seconds,
            # NEW: Include passenger data
            original_passengers_served=original_passengers,
            adapted_passengers_served=adapted_passengers,
            passengers_improvement_percentage=passengers_improvement_pct,
            # NEW: Include waiting time data
            original_avg_waiting_time=original_waiting_time,
            adapted_avg_waiting_time=adapted_waiting_time,
            waiting_time_improvement_percentage=waiting_time_improvement_pct
        )
        
        self.adaptation_history.append(result)
        
        logger.info(f"Adaptive optimization completed in {optimization_time_seconds:.2f} seconds")
        logger.info(f"Original fitness: {original_fitness:.2f}, Adapted fitness: {adapted_fitness:.2f}")
        logger.info(f"Fitness improvement: {improvement_pct:.1f}%")
        logger.info(f"Original passengers served: {original_passengers}, Adapted passengers served: {adapted_passengers}")
        logger.info(f"Passengers improvement: {passengers_improvement_pct:.1f}%")
        logger.info(f"Original waiting time: {original_waiting_time:.2f} min, Adapted waiting time: {adapted_waiting_time:.2f} min")
        logger.info(f"Waiting time improvement: {waiting_time_improvement_pct:.1f}%")
        logger.info(f"Adapted headways: {adapted_headways}")
        
        return result
    
    def _calculate_scaling_factors(self, z_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate demand scaling factors based on Z-scores.
        
        Parameters
        ----------
        z_scores : Dict[str, float]
            Z-scores for each station
            
        Returns
        -------
        Dict[str, float]
            Scaling factors for demand adjustment
        """
        scaling_factors = {}
        
        for station, z_score in z_scores.items():
            # Convert Z-score to scaling factor with more aggressive scaling
            # for optimization purposes
            
            if abs(z_score) <= 1.0:
                # Small deviation
                factor = 1.0 + (z_score * 0.15)
            elif abs(z_score) <= 2.0:
                # Medium deviation
                factor = 1.0 + (z_score * 0.25)
            elif abs(z_score) <= 3.0:
                # Large deviation
                factor = 1.0 + (z_score * 0.4)
            else:
                # Very large deviation
                factor = 1.0 + (z_score * 0.5)
            
            # Ensure reasonable bounds
            factor = max(0.1, min(4.0, factor))
            scaling_factors[station] = factor
        
        return scaling_factors
    
    def _create_adaptive_lambda_functions(self, current_time: float, 
                                        scaling_factors: Dict[str, float]) -> List[Callable]:
        """
        Create adaptive lambda functions with scaling factors.
        
        Parameters
        ----------
        current_time : float
            Current simulation time
        scaling_factors : Dict[str, float]
            Scaling factors by station
            
        Returns
        -------
        List[Callable]
            List of adaptive lambda functions
        """
        if not self.baseline_lambda_funcs:
            # Fallback to current time-based lambdas
            self.baseline_lambda_funcs = self.data_processor.get_lambda_functions(current_time)
        
        adaptive_lambdas = []
        
        for i, station in enumerate(self.data_processor.station_ids):
            base_lambda = self.baseline_lambda_funcs[i]
            scaling_factor = scaling_factors.get(station, 1.0)
            
            # Create adaptive lambda function
            adaptive_lambda = AdaptiveLambdaFunction(
                base_lambda_func=base_lambda,
                station_index=i,
                adaptation_start_time=current_time,
                adaptation_duration=self.adaptation_duration,
                scaling_factor=scaling_factor
            )
            
            adaptive_lambdas.append(adaptive_lambda)
        
        return adaptive_lambdas
    
    def simulate_with_current_headways(self, lambda_funcs: Optional[List[Callable]] = None) -> SimulatorResult:
        """
        Simulate the system with current headway schedule.
        
        Parameters
        ----------
        lambda_funcs : Optional[List[Callable]]
            Lambda functions to use. If None, uses baseline lambdas.
            
        Returns
        -------
        SimulatorResult
            Simulation result
        """
        if not self.current_headways:
            raise ValueError("No current headways available. Run initial optimization first.")
        
        lambdas = lambda_funcs or self.baseline_lambda_funcs
        if not lambdas:
            raise ValueError("No lambda functions available")
        
        return self.simulator.simulate(
            self.current_headways, 
            lambdas,
            weights=(self.ga_params.get('alpha', 3.5), 
                    self.ga_params.get('beta', 0.25), 
                    self.ga_params.get('gamma', 50.0))
        )
    
    def get_adaptation_summary(self) -> Dict:
        """
        Get summary of all adaptations performed.
        
        Returns
        -------
        Dict
            Summary statistics of adaptations
        """
        if not self.adaptation_history:
            return {"total_adaptations": 0}
        
        total_adaptations = len(self.adaptation_history)
        avg_improvement = np.mean([r.improvement_percentage for r in self.adaptation_history])
        avg_optimization_time = np.mean([r.total_optimization_time_seconds for r in self.adaptation_history])
        
        # Find stations that triggered adaptations most frequently
        all_trigger_stations = []
        for result in self.adaptation_history:
            trigger_stations = [
                station for station, z_score in result.z_scores_at_trigger.items()
                if abs(z_score) > 2.0  # Assuming threshold of 2.0
            ]
            all_trigger_stations.extend(trigger_stations)
        
        from collections import Counter
        station_trigger_counts = Counter(all_trigger_stations)
        
        summary = {
            "total_adaptations": total_adaptations,
            "average_improvement_percentage": avg_improvement,
            "average_optimization_time_seconds": avg_optimization_time,
            "most_problematic_stations": station_trigger_counts.most_common(5),
            "total_adaptation_time_minutes": sum(r.adaptation_period_minutes for r in self.adaptation_history),
            "adaptation_times": [r.optimization_time for r in self.adaptation_history]
        }
        
        return summary
    
    def reset_adaptation_state(self):
        """Reset adaptation state for new simulation."""
        self.current_headways = None
        self.baseline_lambda_funcs = None
        self.adaptation_history = []
        
        logger.info("Adaptive GA state reset")


if __name__ == "__main__":
    # Example usage for testing
    from data_processor import MetroDataProcessor
    from real_time_monitor import RealTimeMonitor
    
    # Initialize components
    processor = MetroDataProcessor("../data/simulation_log.csv")
    processor.load_and_process_data(sample_size=5000)
    processor.calculate_hourly_arrival_rates()
    processor.calculate_od_matrix()
    processor.calculate_baseline_demand()
    
    # Simulator parameters
    sim_params = {
        'n_stations': 16,
        'capacity': 1200,
        'travel_times': processor.get_travel_times()
    }
    
    # GA parameters
    ga_params = {
        'alpha': 3.5,
        'beta': 0.25,
        'gamma': 50.0,
        'pop_size': 20,
        'generations': 15,  # Reduced for testing
        'headway_min': 5,
        'headway_max': 15,
        'num_trains': 8,
        'mutation_rate': 0.15,
        'crossover_rate': 0.6
    }
    
    # Initialize adaptive GA
    adaptive_ga = AdaptiveGeneticAlgorithm(processor, sim_params, ga_params)
    
    # Test initial optimization
    print("Running initial optimization...")
    initial_headways, initial_result = adaptive_ga.initial_optimization()
    
    print(f"Initial optimization results:")
    print(f"Headways: {initial_headways}")
    print(f"Fitness: {initial_result.fitness:.2f}")
    print(f"Average waiting time: {initial_result.avg_waiting_time:.2f}")
    print(f"Passengers served: {initial_result.total_passengers_served}")
    
    # Simulate a monitoring result that triggers adaptation
    from real_time_monitor import MonitoringResult
    
    # Create a fake monitoring result with high Z-scores
    fake_z_scores = {
        'S01UN': 2.5,
        'S06M10': 1.8,
        'S16MJ': -2.1
    }
    for station in processor.station_ids:
        if station not in fake_z_scores:
            fake_z_scores[station] = 0.0
    
    fake_monitoring_result = MonitoringResult(
        timestamp=60.0,  # 1 hour into simulation
        station_loads={station: 100.0 for station in processor.station_ids},
        z_scores=fake_z_scores,
        aggregated_z_score=2.2,
        threshold_exceeded=True,
        stations_above_threshold=['S01UN', 'S16MJ']
    )
    
    # Test adaptive optimization
    print(f"\nRunning adaptive optimization...")
    adaptation_result = adaptive_ga.adaptive_optimization(fake_monitoring_result)
    
    print(f"Adaptive optimization results:")
    print(f"Original headways: {adaptation_result.original_headways}")
    print(f"Adapted headways: {adaptation_result.adapted_headways}")
    print(f"Original fitness: {adaptation_result.original_fitness:.2f}")
    print(f"Adapted fitness: {adaptation_result.adapted_fitness:.2f}")
    print(f"Improvement: {adaptation_result.improvement_percentage:.1f}%")
    print(f"Optimization time: {adaptation_result.total_optimization_time_seconds:.2f} seconds")
    
    # Get summary
    summary = adaptive_ga.get_adaptation_summary()
    print(f"\nAdaptation summary:")
    print(f"Total adaptations: {summary['total_adaptations']}")
    print(f"Average improvement: {summary['average_improvement_percentage']:.1f}%")