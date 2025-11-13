import pandas as pd
import numpy as np
import os

class HourlyLoadPredictor:
    """
    A simple predictor for hourly passenger load at metro stations.
    """
    def __init__(self, data_path=None):
        """
        Initializes the predictor and loads the historical data.

        Args:
            data_path (str, optional): Path to the simulation log data. 
                                       Defaults to '../data/simulation_log.csv'.
        """
        if data_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(script_dir, '..', 'data', 'simulation_log_nwq.csv')
            
        self.data_path = data_path
        self.hourly_avg = self._calculate_hourly_averages()

    def _calculate_hourly_averages(self):
        """
        Calculates the average hourly entries and exits for each station.
        """
        try:
            df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
            return None

        df['entry_hour'] = (df['Entry'] / 60).astype(int)
        df['exit_hour'] = (df['Exit'] / 60).astype(int)

        stations = pd.unique(df[['Origin', 'Dest']].values.ravel('K'))
        
        hourly_load = {station: {'entries': np.zeros(24), 'exits': np.zeros(24)} for station in stations}

        entry_counts = df.groupby(['Origin', 'entry_hour']).size().unstack(fill_value=0)
        for station in entry_counts.index:
            if station in hourly_load:
                for hour in entry_counts.columns:
                    if 0 <= hour < 24:
                        hourly_load[station]['entries'][hour] = entry_counts.loc[station, hour]

        exit_counts = df.groupby(['Dest', 'exit_hour']).size().unstack(fill_value=0)
        for station in exit_counts.index:
            if station in hourly_load:
                for hour in exit_counts.columns:
                    if 0 <= hour < 24:
                        hourly_load[station]['exits'][hour] = exit_counts.loc[station, hour]
        
        return hourly_load

    def predict(self, station, hour):
        """
        Predicts the number of entries and exits for a given station and hour.

        Args:
            station (str): The station code (e.g., 'S01UN').
            hour (int): The hour of the day (0-23).

        Returns:
            dict: A dictionary with predicted 'entries' and 'exits', or None if prediction is not available.
        """
        if not self.hourly_avg or station not in self.hourly_avg:
            return None
        
        if not 0 <= hour <= 23:
            raise ValueError("Hour must be between 0 and 23.")

        return {
            'entries': self.hourly_avg[station]['entries'][hour],
            'exits': self.hourly_avg[station]['exits'][hour]
        }

    def get_all_stations(self):
        """Returns a list of all available stations."""
        if not self.hourly_avg:
            return []
        return sorted(list(self.hourly_avg.keys()))

if __name__ == '__main__':
    predictor = HourlyLoadPredictor()
    
    if predictor.hourly_avg:
        stations_to_predict = predictor.get_all_stations()
        hours_to_predict = [8, 12, 18] # Example hours

        print("--- Hourly Load Predictions ---")
        for station in stations_to_predict:
            print(f"\nStation: {station}")
            for hour in hours_to_predict:
                prediction = predictor.predict(station, hour)
                if prediction:
                    print(f"  Hour {hour}: Predicted Entries = {prediction['entries']:.0f}, Predicted Exits = {prediction['exits']:.0f}")
