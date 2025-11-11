"""
Data processing module for Metro Adaptive Optimization System.

This module processes historical passenger data from CSV files and extracts:
- Temporal passenger arrival patterns for each station
- Origin-Destination (OD) matrices
- Baseline demand patterns for comparison with real-time data
- Historical statistics for Z-score calculations

Created: November 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetroDataProcessor:
    """
    Processes historical metro passenger data to extract patterns for optimization.
    """

    def __init__(self, csv_file_path: str):
        """
        Initialize the data processor with a CSV file path.

        Parameters
        ----------
        csv_file_path : str
            Path to the CSV file containing passenger data
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.station_ids = [
            'S01UN', 'S02UC', 'S03US', 'S04PL', 'S05M11', 'S06M10', 
            'S07KP', 'S08SP', 'S09AG', 'S10BS', 'S11FG', 'S12KB', 
            'S13SB', 'S14DU', 'S15BS', 'S16MJ'
        ]
        self.n_stations = len(self.station_ids)
        self.station_to_index = {station: i for i, station in enumerate(self.station_ids)}
        
        # Historical patterns
        self.hourly_arrival_rates = {}
        self.od_matrix = None
        self.baseline_demand = {}
        self.demand_std = {}
        self.hourly_predictions = {}  # Store predictions for each hour and station
        self.prediction_weights = {    # Basic model weights for predictions
            'historical_weight': 0.7,  # Weight for historical data
            'trend_weight': 0.2,       # Weight for recent trend
            'seasonal_weight': 0.1     # Weight for seasonal adjustments
        }
        
    def load_and_process_data(self, sample_size: int = None) -> None:
        """
        Load data from CSV and perform initial processing.
        
        Parameters
        ----------
        sample_size : int, optional
            If provided, only load this many rows for faster processing
        """
        logger.info(f"Loading data from {self.csv_file_path}")
        
        if sample_size:
            # Load only a sample for proof of concept
            self.data = pd.read_csv(self.csv_file_path, nrows=sample_size)
            logger.info(f"Loaded {sample_size} rows for proof of concept")
        else:
            self.data = pd.read_csv(self.csv_file_path)
            logger.info(f"Loaded {len(self.data)} rows")
        
        # Convert time columns to appropriate format
        self.data['Entry'] = pd.to_numeric(self.data['Entry'])
        self.data['Exit'] = pd.to_numeric(self.data['Exit'])
        
        # Extract hour from entry time (assuming entry time is in minutes from start of day)
        self.data['entry_hour'] = (self.data['Entry'] / 60).astype(int)
        
        logger.info("Data processing completed")
    
    def calculate_hourly_arrival_rates(self) -> Dict[str, Dict[int, float]]:
        """
        Calculate hourly passenger arrival rates for each station.
        
        Returns
        -------
        Dict[str, Dict[int, float]]
            Dictionary with station_id as key and hour:rate mapping as value
        """
        logger.info("Calculating hourly arrival rates")
        
        hourly_rates = {}
        
        # Group by origin station and entry hour
        for station in self.station_ids:
            station_data = self.data[self.data['Origin'] == station]
            if len(station_data) == 0:
                # If no data for this origin, assume all passengers board from S01UN
                if station == 'S01UN':
                    station_data = self.data.copy()
                else:
                    continue
            
            hourly_counts = station_data.groupby('entry_hour').size()
            
            # Convert counts to rates (passengers per minute)
            # Assuming each hour has 60 minutes
            hourly_rates_dict = {}
            for hour in range(24):  # 24 hours in a day
                count = hourly_counts.get(hour, 0)
                rate_per_minute = count / 60.0  # Convert to per-minute rate
                hourly_rates_dict[hour] = rate_per_minute
            
            hourly_rates[station] = hourly_rates_dict
        
        self.hourly_arrival_rates = hourly_rates
        logger.info("Hourly arrival rates calculated")
        return hourly_rates
    
    def calculate_od_matrix(self) -> np.ndarray:
        """
        Calculate Origin-Destination probability matrix.
        
        Returns
        -------
        np.ndarray
            OD matrix where element [i][j] is probability of passenger
            boarding at station i going to station j
        """
        logger.info("Calculating OD matrix")
        
        # Initialize OD matrix
        od_matrix = np.zeros((self.n_stations, self.n_stations))
        
        # Count trips for each OD pair
        for _, row in self.data.iterrows():
            origin = row['Origin']
            dest = row['Dest']
            
            if origin in self.station_to_index and dest in self.station_to_index:
                i = self.station_to_index[origin]
                j = self.station_to_index[dest]
                od_matrix[i][j] += 1
        
        # Normalize to get probabilities
        for i in range(self.n_stations):
            row_sum = np.sum(od_matrix[i, :])
            if row_sum > 0:
                od_matrix[i, :] = od_matrix[i, :] / row_sum
        
        self.od_matrix = od_matrix
        logger.info("OD matrix calculated")
        return od_matrix
    
    def calculate_baseline_demand(self, time_window_minutes: int = 60) -> Dict[str, Dict[int, Tuple[float, float]]]:
        """
        Calculate baseline demand patterns with statistics for Z-score calculations.
        
        Parameters
        ----------
        time_window_minutes : int
            Time window for demand aggregation in minutes
            
        Returns
        -------
        Dict[str, Dict[int, Tuple[float, float]]]
            Dictionary with station as key and time_window:(mean, std) as value
        """
        logger.info("Calculating baseline demand patterns")
        
        baseline_demand = {}
        demand_std = {}
        
        # Calculate demand for each station in time windows
        for station in self.station_ids:
            station_data = self.data[self.data['Origin'] == station]
            if len(station_data) == 0 and station == 'S01UN':
                station_data = self.data.copy()
            elif len(station_data) == 0:
                continue
            
            # Group by time windows
            station_data['time_window'] = (station_data['Entry'] / time_window_minutes).astype(int)
            window_counts = station_data.groupby('time_window').size()
            
            # Calculate mean and std for each time window across days (simulated)
            demand_by_window = {}
            std_by_window = {}
            
            for window_id in window_counts.index:
                count = window_counts[window_id]
                # For simplicity, we'll use the count as mean and estimate std
                # In a real system, you'd have multiple days of data
                demand_by_window[window_id] = float(count)
                # Estimate std as 20% of mean (adjust based on real data patterns)
                std_by_window[window_id] = max(1.0, float(count) * 0.2)
            
            baseline_demand[station] = demand_by_window
            demand_std[station] = std_by_window
        
        self.baseline_demand = baseline_demand
        self.demand_std = demand_std
        
        logger.info("Baseline demand patterns calculated")
    
    def generate_hourly_predictions(self, simulation_hours: int = 24) -> Dict:
        """
        Generate hourly predictions for each station using basic model weights.
        
        Parameters
        ----------
        simulation_hours : int
            Number of hours to predict (default: 24)
            
        Returns
        -------
        Dict
            Hourly predictions by station and hour
        """
        logger.info("Generating hourly predictions with basic model weights...")
        
        predictions = {}
        
        for station in self.station_ids:
            predictions[station] = {}
            
            # Get or generate base rate for this station
            if station in self.hourly_arrival_rates:
                # Use actual historical data if available
                station_rates = self.hourly_arrival_rates[station]
                if station_rates:
                    base_rate = np.mean(list(station_rates.values()))
                else:
                    base_rate = 2.0  # Default fallback
            else:
                # Generate synthetic base rate for stations without historical data
                station_index = self.station_ids.index(station)
                # Different stations have different characteristics
                base_rates = [3.2, 2.8, 4.1, 3.9, 2.3, 3.7, 4.3, 2.9, 3.4, 3.1, 3.8, 2.5, 4.2, 3.3, 2.9, 3.6]
                base_rate = base_rates[station_index % len(base_rates)]
                logger.debug(f"Generated synthetic base rate for {station}: {base_rate}")
            
            for hour in range(simulation_hours):
                current_hour = hour % 24
                
                # Get hour-specific historical rate or use base rate with time-of-day variation
                if (station in self.hourly_arrival_rates and 
                    current_hour in self.hourly_arrival_rates[station]):
                    historical_rate = self.hourly_arrival_rates[station][current_hour]
                else:
                    # Apply time-of-day factor to base rate
                    time_factor = self._get_time_of_day_factor(current_hour)
                    historical_rate = base_rate * time_factor
                
                # Apply basic prediction model with weights
                # 1. Historical component (70%)
                historical_component = historical_rate * self.prediction_weights['historical_weight']
                
                # 2. Trend component (20%) - simulate slight upward trend
                trend_factor = 1.0 + (0.02 * (hour / 24))  # 2% daily growth simulation
                trend_component = historical_rate * trend_factor * self.prediction_weights['trend_weight']
                
                # 3. Seasonal component (10%) - time of day adjustments
                seasonal_factor = self._get_seasonal_factor(current_hour)
                seasonal_component = historical_rate * seasonal_factor * self.prediction_weights['seasonal_weight']
                
                # Combine components
                predicted_rate = historical_component + trend_component + seasonal_component
                
                # Add small random variation for realism
                variation = np.random.normal(0, 0.05 * predicted_rate)
                predicted_rate = max(0.1, predicted_rate + variation)  # Minimum 0.1 passengers per minute
                
                predictions[station][hour] = predicted_rate
        
        self.hourly_predictions = predictions
        
        # Save predictions to file
        self._save_predictions_to_file()
        
        logger.info(f"Hourly predictions generated for {len(self.station_ids)} stations over {simulation_hours} hours")
        return predictions
    
    def _get_seasonal_factor(self, hour: int) -> float:
        """
        Get seasonal adjustment factor for time of day.
        
        Parameters
        ----------
        hour : int
            Hour of the day (0-23)
            
        Returns
        -------
        float
            Seasonal adjustment factor
        """
        # Rush hour patterns
        if 7 <= hour <= 9:      # Morning rush
            return 1.3
        elif 17 <= hour <= 19:  # Evening rush  
            return 1.2
        elif 12 <= hour <= 13:  # Lunch time
            return 1.1
        elif 0 <= hour <= 5:    # Early morning
            return 0.3
        elif 22 <= hour <= 23:  # Late night
            return 0.5
        else:                   # Normal hours
            return 1.0
    
    def _get_time_of_day_factor(self, hour: int) -> float:
        """
        Get time-of-day factor for passenger demand variation.
        Similar to seasonal factor but for base rate calculation.
        
        Parameters
        ----------
        hour : int
            Hour of the day (0-23)
            
        Returns
        -------
        float
            Time-of-day adjustment factor
        """
        # Rush hour patterns for base rates
        if 7 <= hour <= 9:      # Morning rush
            return 1.4
        elif 17 <= hour <= 19:  # Evening rush  
            return 1.3
        elif 12 <= hour <= 13:  # Lunch time
            return 1.1
        elif 0 <= hour <= 5:    # Early morning
            return 0.3
        elif 22 <= hour <= 23:  # Late night
            return 0.4
        else:                   # Normal hours
            return 1.0
    
    def _save_predictions_to_file(self):
        """Save hourly predictions to CSV files."""
        try:
            import os
            import pandas as pd
            
            # Create predictions directory
            predictions_dir = "predictions"
            os.makedirs(predictions_dir, exist_ok=True)
            
            # Convert predictions to DataFrame
            data = []
            for station in self.station_ids:
                for hour, prediction in self.hourly_predictions[station].items():
                    data.append({
                        'Station': station,
                        'Hour': hour,
                        'Predicted_Rate': prediction,
                        'Predicted_Passengers_Per_Hour': prediction * 60  # Convert to passengers per hour
                    })
            
            df = pd.DataFrame(data)
            
            # Save to CSV
            df.to_csv(os.path.join(predictions_dir, "hourly_predictions.csv"), index=False)
            
            # Also save summary by station
            summary_data = []
            for station in self.station_ids:
                station_predictions = [self.hourly_predictions[station][h] for h in range(24)]
                summary_data.append({
                    'Station': station,
                    'Average_Daily_Rate': np.mean(station_predictions),
                    'Peak_Rate': np.max(station_predictions),
                    'Peak_Hour': np.argmax(station_predictions),
                    'Off_Peak_Rate': np.min(station_predictions),
                    'Off_Peak_Hour': np.argmin(station_predictions)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(predictions_dir, "prediction_summary.csv"), index=False)
            
            logger.info(f"Predictions saved to {predictions_dir}/hourly_predictions.csv")
            logger.info(f"Prediction summary saved to {predictions_dir}/prediction_summary.csv")
            
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
    
    def get_lambda_functions(self, current_time_minutes: float = 0) -> List[Callable]:
        """
        Generate lambda functions for each station based on current time.
        
        Parameters
        ----------
        current_time_minutes : float
            Current time in minutes from start of day
            
        Returns
        -------
        List[Callable]
            List of lambda functions for each station
        """
        lambda_functions = []
        current_hour = int(current_time_minutes / 60) % 24
        
        for station in self.station_ids:
            if station in self.hourly_arrival_rates:
                base_rate = self.hourly_arrival_rates[station].get(current_hour, 0.0)
            else:
                base_rate = 0.0
            
            # Create lambda function that returns constant rate
            lambda_func = self._create_lambda_function(base_rate)
            lambda_functions.append(lambda_func)
        
        return lambda_functions
    
    def _create_lambda_function(self, base_rate: float) -> Callable:
        """
        Create a lambda function that returns the base rate.
        
        Parameters
        ----------
        base_rate : float
            Base arrival rate
            
        Returns
        -------
        Callable
            Lambda function
        """
        return lambda t: base_rate
    
    def get_travel_times(self) -> List[float]:
        """
        Get inter-station travel times (simplified).
        
        Returns
        -------
        List[float]
            List of travel times between consecutive stations
        """
        # Simplified travel times (2 minutes between stations as per current code)
        return [2.0] * (self.n_stations - 1)
    
    def calculate_z_score(self, current_demand: Dict[str, float], 
                         current_time_minutes: float,
                         time_window_minutes: int = 60) -> Dict[str, float]:
        """
        Calculate Z-score for current demand vs predicted patterns for each station.
        
        Parameters
        ----------
        current_demand : Dict[str, float]
            Current passenger counts by station
        current_time_minutes : float
            Current time in minutes
        time_window_minutes : int
            Time window for comparison
            
        Returns
        -------
        Dict[str, float]
            Z-scores for each station
        """
        z_scores = {}
        current_hour = int(current_time_minutes / 60) % 24
        
        for station in self.station_ids:
            current_count = current_demand.get(station, 0.0)
            
            # Get predicted value for this station and hour
            if (station in self.hourly_predictions and 
                current_hour in self.hourly_predictions[station]):
                
                predicted_rate = self.hourly_predictions[station][current_hour]
                predicted_count = predicted_rate * (time_window_minutes / 60)  # Scale to time window
                
                # Calculate standard deviation from historical data or use default
                if (station in self.demand_std and 
                    current_hour in self.demand_std[station]):
                    std_dev = self.demand_std[station][current_hour]
                else:
                    # Use 20% of predicted value as default std dev
                    std_dev = max(0.2 * predicted_count, 1.0)
                
                # Calculate Z-score comparing current vs predicted
                if std_dev > 0:
                    z_score = (current_count - predicted_count) / std_dev
                else:
                    z_score = 0.0
                
                z_scores[station] = z_score
                
                logger.debug(f"Station {station}: current={current_count:.1f}, "
                           f"predicted={predicted_count:.1f}, z_score={z_score:.2f}")
            else:
                # Fallback if no prediction available
                z_scores[station] = 0.0
                logger.debug(f"Station {station}: No prediction available, z_score=0.0")
        
        return z_scores
    
    def get_aggregated_z_score(self, z_scores: Dict[str, float]) -> float:
        """
        Calculate aggregated Z-score across all stations.
        
        Parameters
        ----------
        z_scores : Dict[str, float]
            Z-scores for individual stations
            
        Returns
        -------
        float
            Aggregated Z-score (RMS)
        """
        if not z_scores:
            return 0.0
        
        # Calculate RMS (Root Mean Square) of Z-scores
        sum_squares = sum(z**2 for z in z_scores.values())
        rms_z_score = np.sqrt(sum_squares / len(z_scores))
        
        return rms_z_score
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics of the processed data.
        
        Returns
        -------
        Dict
            Summary statistics
        """
        if self.data is None:
            return {}
        
        total_trips = len(self.data)
        unique_origins = self.data['Origin'].nunique()
        unique_destinations = self.data['Dest'].nunique()
        time_span_hours = (self.data['Entry'].max() - self.data['Entry'].min()) / 60
        
        stats = {
            'total_trips': total_trips,
            'unique_origins': unique_origins,
            'unique_destinations': unique_destinations,
            'time_span_hours': time_span_hours,
            'avg_trips_per_hour': total_trips / max(1, time_span_hours),
            'stations_configured': len(self.station_ids)
        }
        
        return stats


if __name__ == "__main__":
    # Example usage for testing
    processor = MetroDataProcessor("../data/simulation_log.csv")
    
    # Load a sample for proof of concept
    processor.load_and_process_data(sample_size=5000)
    
    # Calculate patterns
    hourly_rates = processor.calculate_hourly_arrival_rates()
    od_matrix = processor.calculate_od_matrix()
    baseline_demand = processor.calculate_baseline_demand()
    
    # Get summary
    stats = processor.get_summary_statistics()
    
    print("Data Processing Summary:")
    print(f"Total trips processed: {stats['total_trips']}")
    print(f"Time span: {stats['time_span_hours']:.1f} hours")
    print(f"Average trips per hour: {stats['avg_trips_per_hour']:.1f}")
    
    print(f"\nHourly arrival rates for S01UN:")
    if 'S01UN' in hourly_rates:
        for hour, rate in hourly_rates['S01UN'].items():
            if rate > 0:
                print(f"Hour {hour}: {rate:.2f} passengers/min")