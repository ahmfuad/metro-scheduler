"""
Real-time monitoring system for Metro Adaptive Optimization.

This module simulates real-time passenger load monitoring across metro stations
and calculates Z-scores to detect deviations from historical patterns. When 
significant deviations are detected, it triggers adaptive optimization.

Created: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
import random
from dataclasses import dataclass
from data_processor import MetroDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MonitoringResult:
    """Container for monitoring results."""
    timestamp: float
    station_loads: Dict[str, float]
    z_scores: Dict[str, float]
    aggregated_z_score: float
    threshold_exceeded: bool
    stations_above_threshold: List[str]


class RealTimeMonitor:
    """
    Real-time monitoring system for metro passenger loads.
    
    Simulates real-time data collection and analysis for detecting
    significant deviations from historical passenger patterns.
    """
    
    def __init__(self, data_processor: MetroDataProcessor, 
                 z_score_threshold: float = 2.0,
                 monitoring_interval_minutes: float = 5.0,
                 noise_level: float = 0.15):
        """
        Initialize the real-time monitoring system.
        
        Parameters
        ----------
        data_processor : MetroDataProcessor
            Preprocessed historical data
        z_score_threshold : float
            Z-score threshold for triggering adaptive optimization
        monitoring_interval_minutes : float
            Interval between monitoring checks in minutes
        noise_level : float
            Level of noise to add to simulated real-time data (0.0-1.0)
        """
        self.data_processor = data_processor
        self.z_score_threshold = z_score_threshold
        self.monitoring_interval = monitoring_interval_minutes
        self.noise_level = noise_level
        
        # Current simulation state
        self.current_time = 0.0  # Minutes from start of day
        self.monitoring_history = []
        self.last_trigger_time = None
        
        # Station configurations
        self.station_ids = data_processor.station_ids
        self.n_stations = len(self.station_ids)
        
        # Real-time state simulation
        self.current_passenger_counts = {station: 0.0 for station in self.station_ids}
        self.passenger_accumulation = {station: 0.0 for station in self.station_ids}
        
        logger.info(f"Real-time monitor initialized with {self.n_stations} stations")
        logger.info(f"Z-score threshold: {z_score_threshold}")
        logger.info(f"Monitoring interval: {monitoring_interval_minutes} minutes")
    
    def simulate_real_time_passenger_data(self, current_time_minutes: float) -> Dict[str, float]:
        """
        Simulate real-time passenger count data with realistic variations.
        
        Parameters
        ----------
        current_time_minutes : float
            Current simulation time in minutes
            
        Returns
        -------
        Dict[str, float]
            Current passenger counts by station
        """
        passenger_counts = {}
        current_hour = int(current_time_minutes / 60) % 24
        
        for station in self.station_ids:
            # Get historical baseline
            if station in self.data_processor.hourly_arrival_rates:
                base_rate = self.data_processor.hourly_arrival_rates[station].get(current_hour, 0.0)
            else:
                base_rate = 0.0
            
            # Calculate expected passengers in monitoring interval
            expected_passengers = base_rate * self.monitoring_interval
            
            # Add realistic variations
            # 1. Random noise
            noise_factor = 1.0 + random.uniform(-self.noise_level, self.noise_level)
            
            # 2. Time-of-day variations (rush hours, etc.)
            time_factor = self._get_time_factor(current_hour)
            
            # 3. Occasional spikes or drops (special events, delays, etc.)
            event_factor = self._get_event_factor(current_time_minutes, station)
            
            # Calculate current passenger count
            simulated_count = expected_passengers * noise_factor * time_factor * event_factor
            simulated_count = max(0, simulated_count)  # Ensure non-negative
            
            passenger_counts[station] = simulated_count
        
        return passenger_counts
    
    def _get_time_factor(self, hour: int) -> float:
        """
        Get time-of-day factor for passenger demand variation.
        
        Parameters
        ----------
        hour : int
            Hour of the day (0-23)
            
        Returns
        -------
        float
            Multiplication factor for demand
        """
        # Simulate rush hour patterns
        if 7 <= hour <= 9:  # Morning rush
            return 1.5
        elif 17 <= hour <= 19:  # Evening rush
            return 1.4
        elif 12 <= hour <= 13:  # Lunch time
            return 1.2
        elif 22 <= hour or hour <= 5:  # Late night/early morning
            return 0.3
        else:  # Regular hours
            return 1.0
    
    def _get_event_factor(self, current_time_minutes: float, station: str) -> float:
        """
        Get event-based factor to simulate special conditions.
        
        Parameters
        ----------
        current_time_minutes : float
            Current time in minutes
        station : str
            Station identifier
            
        Returns
        -------
        float
            Event-based multiplication factor
        """
        # Simulate occasional events that cause demand spikes/drops
        # This is where you might integrate real-time external data sources
        
        # Simulate a random event every ~2 hours on average
        event_probability = 0.001  # Low probability per minute
        
        if random.random() < event_probability:
            # Random event: could be positive (festival) or negative (delay)
            if random.random() < 0.7:  # 70% chance of positive event
                return random.uniform(1.5, 3.0)  # Demand spike
            else:
                return random.uniform(0.2, 0.7)   # Demand drop
        
        return 1.0  # No event
    
    def calculate_current_z_scores(self, current_time_minutes: float,
                                  passenger_counts: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate Z-scores comparing current passenger counts to historical patterns.
        
        Parameters
        ----------
        current_time_minutes : float
            Current time in minutes
        passenger_counts : Dict[str, float]
            Current passenger counts by station
            
        Returns
        -------
        Dict[str, float]
            Z-scores for each station
        """
        return self.data_processor.calculate_z_score(
            passenger_counts, current_time_minutes, 
            time_window_minutes=int(self.monitoring_interval)
        )
    
    def check_threshold_exceeded(self, z_scores: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Check if Z-score threshold is exceeded at any station.
        
        Parameters
        ----------
        z_scores : Dict[str, float]
            Z-scores for each station
            
        Returns
        -------
        Tuple[bool, List[str]]
            (threshold_exceeded, list_of_stations_above_threshold)
        """
        stations_above_threshold = []
        
        for station, z_score in z_scores.items():
            if abs(z_score) > self.z_score_threshold:
                stations_above_threshold.append(station)
        
        threshold_exceeded = len(stations_above_threshold) > 0
        
        return threshold_exceeded, stations_above_threshold
    
    def monitor_step(self, current_time_minutes: float) -> MonitoringResult:
        """
        Perform one monitoring step.
        
        Parameters
        ----------
        current_time_minutes : float
            Current simulation time in minutes
            
        Returns
        -------
        MonitoringResult
            Results of this monitoring step
        """
        # Simulate real-time passenger data
        passenger_counts = self.simulate_real_time_passenger_data(current_time_minutes)
        
        # Calculate Z-scores
        z_scores = self.calculate_current_z_scores(current_time_minutes, passenger_counts)
        
        # Calculate aggregated Z-score
        aggregated_z_score = self.data_processor.get_aggregated_z_score(z_scores)
        
        # Check threshold
        threshold_exceeded, stations_above_threshold = self.check_threshold_exceeded(z_scores)
        
        # Create result
        result = MonitoringResult(
            timestamp=current_time_minutes,
            station_loads=passenger_counts,
            z_scores=z_scores,
            aggregated_z_score=aggregated_z_score,
            threshold_exceeded=threshold_exceeded,
            stations_above_threshold=stations_above_threshold
        )
        
        # Store in history
        self.monitoring_history.append(result)
        
        # Log if threshold exceeded
        if threshold_exceeded:
            logger.warning(f"Time {current_time_minutes:.1f}: Z-score threshold exceeded!")
            logger.warning(f"Aggregated Z-score: {aggregated_z_score:.2f}")
            logger.warning(f"Stations above threshold: {stations_above_threshold}")
            self.last_trigger_time = current_time_minutes
        
        return result
    
    def should_trigger_adaptive_optimization(self, current_result: MonitoringResult) -> bool:
        """
        Determine if adaptive optimization should be triggered.
        
        Parameters
        ----------
        current_result : MonitoringResult
            Current monitoring result
            
        Returns
        -------
        bool
            True if adaptive optimization should be triggered
        """
        # Don't trigger too frequently (wait at least 30 minutes between triggers)
        if (self.last_trigger_time is not None and 
            current_result.timestamp - self.last_trigger_time < 30.0):
            return False
        
        return current_result.threshold_exceeded
    
    def get_demand_scaling_factors(self, z_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate demand scaling factors based on Z-scores for adaptive optimization.
        
        Parameters
        ----------
        z_scores : Dict[str, float]
            Current Z-scores by station
            
        Returns
        -------
        Dict[str, float]
            Scaling factors for demand adjustment (1.0 = no change)
        """
        scaling_factors = {}
        
        for station, z_score in z_scores.items():
            # Convert Z-score to scaling factor
            # Positive Z-score = higher than expected demand = scale up
            # Negative Z-score = lower than expected demand = scale down
            
            if abs(z_score) <= 1.0:
                # Small deviation - minimal adjustment
                factor = 1.0 + (z_score * 0.1)
            elif abs(z_score) <= 2.0:
                # Medium deviation - moderate adjustment
                factor = 1.0 + (z_score * 0.2)
            else:
                # Large deviation - significant adjustment
                factor = 1.0 + (z_score * 0.3)
            
            # Ensure factor is reasonable (between 0.2 and 3.0)
            factor = max(0.2, min(3.0, factor))
            
            scaling_factors[station] = factor
        
        return scaling_factors
    
    def get_monitoring_summary(self, time_window_minutes: float = 60.0) -> Dict:
        """
        Get summary of monitoring results for the specified time window.
        
        Parameters
        ----------
        time_window_minutes : float
            Time window for summary calculation
            
        Returns
        -------
        Dict
            Summary statistics
        """
        if not self.monitoring_history:
            return {}
        
        # Filter recent history
        current_time = self.monitoring_history[-1].timestamp
        recent_results = [
            r for r in self.monitoring_history 
            if r.timestamp >= current_time - time_window_minutes
        ]
        
        if not recent_results:
            return {}
        
        # Calculate summary statistics
        threshold_exceedances = sum(1 for r in recent_results if r.threshold_exceeded)
        avg_aggregated_z_score = np.mean([r.aggregated_z_score for r in recent_results])
        max_aggregated_z_score = max(r.aggregated_z_score for r in recent_results)
        
        # Find most problematic stations
        station_exceedance_counts = {}
        for result in recent_results:
            for station in result.stations_above_threshold:
                station_exceedance_counts[station] = station_exceedance_counts.get(station, 0) + 1
        
        summary = {
            'time_window_minutes': time_window_minutes,
            'monitoring_points': len(recent_results),
            'threshold_exceedances': threshold_exceedances,
            'exceedance_rate': threshold_exceedances / len(recent_results),
            'avg_aggregated_z_score': avg_aggregated_z_score,
            'max_aggregated_z_score': max_aggregated_z_score,
            'most_problematic_stations': sorted(
                station_exceedance_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }
        
        return summary
    
    def reset_monitoring_state(self):
        """Reset monitoring state for new simulation."""
        self.current_time = 0.0
        self.monitoring_history = []
        self.last_trigger_time = None
        self.current_passenger_counts = {station: 0.0 for station in self.station_ids}
        self.passenger_accumulation = {station: 0.0 for station in self.station_ids}
        
        logger.info("Monitoring state reset")


if __name__ == "__main__":
    # Example usage for testing
    from data_processor import MetroDataProcessor
    
    # Initialize data processor
    processor = MetroDataProcessor("../data/simulation_log.csv")
    processor.load_and_process_data(sample_size=5000)
    processor.calculate_hourly_arrival_rates()
    processor.calculate_baseline_demand()
    
    # Initialize monitor
    monitor = RealTimeMonitor(processor, z_score_threshold=1.5)
    
    # Simulate monitoring for 2 hours
    simulation_time = 0.0
    monitoring_interval = 5.0  # 5 minutes
    total_simulation_time = 120.0  # 2 hours
    
    print("Starting real-time monitoring simulation...")
    print(f"Z-score threshold: {monitor.z_score_threshold}")
    print("-" * 60)
    
    adaptive_triggers = 0
    
    while simulation_time <= total_simulation_time:
        result = monitor.monitor_step(simulation_time)
        
        # Check if adaptive optimization should be triggered
        if monitor.should_trigger_adaptive_optimization(result):
            adaptive_triggers += 1
            scaling_factors = monitor.get_demand_scaling_factors(result.z_scores)
            print(f"\n*** ADAPTIVE OPTIMIZATION TRIGGERED at {simulation_time:.1f} min ***")
            print(f"Aggregated Z-score: {result.aggregated_z_score:.2f}")
            print(f"Affected stations: {result.stations_above_threshold}")
            print(f"Suggested scaling factors: {scaling_factors}")
            print("-" * 60)
        
        # Print periodic updates
        if simulation_time % 30.0 == 0:  # Every 30 minutes
            print(f"Time {simulation_time:.0f} min - Agg Z-score: {result.aggregated_z_score:.2f}")
        
        simulation_time += monitoring_interval
    
    # Print final summary
    summary = monitor.get_monitoring_summary(total_simulation_time)
    print(f"\nMonitoring Summary:")
    print(f"Total monitoring points: {summary.get('monitoring_points', 0)}")
    print(f"Threshold exceedances: {summary.get('threshold_exceedances', 0)}")
    print(f"Adaptive optimization triggers: {adaptive_triggers}")
    print(f"Average aggregated Z-score: {summary.get('avg_aggregated_z_score', 0):.2f}")
    print(f"Maximum aggregated Z-score: {summary.get('max_aggregated_z_score', 0):.2f}")