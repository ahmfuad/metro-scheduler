import pandas as pd
import numpy as np
import os

def analyze_hourly_load(file_path=None):
    """
    Analyzes the hourly passenger load (entries and exits) for each station
    from the simulation log data.

    Args:
        file_path (str): The path to the simulation log CSV file.
    """
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, '..', 'data', 'simulation_log_nwq.csv')
        

    print(f"Reading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return

    # Convert minutes to hours
    df['entry_hour'] = (df['Entry'] / 60).astype(int)
    df['exit_hour'] = (df['Exit'] / 60).astype(int)

    # Get all unique station names
    stations = pd.unique(df[['Origin', 'Dest']].values.ravel('K'))
    
    # Initialize results dictionary
    hourly_load = {station: {'entries': np.zeros(24), 'exits': np.zeros(24)} for station in stations}

    # Calculate entries for each station and hour
    entry_counts = df.groupby(['Origin', 'entry_hour']).size().unstack(fill_value=0)
    for station in entry_counts.index:
        if station in hourly_load:
            for hour in entry_counts.columns:
                if 0 <= hour < 24:
                    hourly_load[station]['entries'][hour] = entry_counts.loc[station, hour]

    # Calculate exits for each station and hour
    exit_counts = df.groupby(['Dest', 'exit_hour']).size().unstack(fill_value=0)
    for station in exit_counts.index:
        if station in hourly_load:
            for hour in exit_counts.columns:
                if 0 <= hour < 24:
                    hourly_load[station]['exits'][hour] = exit_counts.loc[station, hour]
    
    print("\n--- Hourly Passenger Load ---")
    for station in sorted(hourly_load.keys()):
        print(f"\nStation: {station}")
        station_df = pd.DataFrame({
            'Hour': range(24),
            'Entries': hourly_load[station]['entries'],
            'Exits': hourly_load[station]['exits']
        })
        print(station_df.to_string(index=False))
        station_df.to_csv(f'hourly_load_{station}.csv', index=False)


if __name__ == '__main__':
    analyze_hourly_load()
