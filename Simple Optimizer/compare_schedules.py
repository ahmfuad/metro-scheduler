import sys
import os
import json
import numpy as np
import pandas as pd

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Simulator import HeadwaySimulator
from Adaptive_Optimizer.hourly_load_predictor import HourlyLoadPredictor

def run_full_day_simulation(headway_schedule, predictor, sim_params):
    """
    Runs a continuous 24-hour simulation for a given headway schedule.
    
    Args:
        headway_schedule (dict or list): A dict of {hour: headway_list} for dynamic 
                                         schedules, or a single list for static schedules.
        predictor (HourlyLoadPredictor): The passenger load predictor.
        sim_params (dict): A dictionary of simulation parameters.

    Returns:
        dict: A dictionary containing the aggregated results for the full day.
    """
    all_stations = predictor.get_all_stations()
    daily_results = []
    
    # Initial queues for the start of the day are zero
    current_queues = [0] * sim_params['n_stations']

    for hour in range(15):
        # Determine the headway for the current hour
        is_dynamic = isinstance(headway_schedule, dict)

        if is_dynamic:
            # DYNAMIC: Truncate the schedule to fit within a 60-minute window
            H_original = headway_schedule.get(str(hour), headway_schedule.get(hour, [10] * 8))
            
            cumulative_sum = 0
            last_valid_index = -1
            for i, h in enumerate(H_original):
                if cumulative_sum + h < 60:
                    cumulative_sum += h
                    last_valid_index = i
                else:
                    break
            
            # Take headways up to the last one that fits, then add one more
            H_truncated = H_original[:last_valid_index + 1]
            
            if cumulative_sum < 60:
                # Add one final train to depart at exactly the 60-minute mark
                final_headway = 60 - cumulative_sum
                H = H_truncated + [final_headway]
            else:
                H = H_truncated

        else:
            # STATIC: Use the full schedule without truncation
            H = headway_schedule

        # If the schedule for the hour is empty, skip
        if not H:
            print(f"\nHour {hour}: Headway schedule is empty. Skipping.")
            continue

        # Get arrival rates for the current hour
        def constant_lambda(rate):
            return lambda t: float(rate)

        lambdas_this_hour = [
            constant_lambda(predictor.predict(st, hour)['entries'] / 60.0)
            if predictor.predict(st, hour) else constant_lambda(0)
            for st in all_stations
        ]


        print(f"\nHour {hour}: Using headway {H} with lambdas {[l(0) for l in lambdas_this_hour]}")
        # Initialize simulator with the queues from the previous hour
        sim = HeadwaySimulator(
            n_stations=sim_params['n_stations'],
            capacity=sim_params['capacity'],
            travel_times=sim_params['travel_times'],
            p_dest=sim_params['p_dest'],
            initial_queues=current_queues
        )

        # Run simulation for the hour
        result = sim.simulate(H, lambdas_this_hour, weights=sim_params['weights'])
        
        # Store results for this hour, including the actual headway used
        daily_results.append({
            'hour': hour,
            'headway_schedule': H,
            'avg_waiting_time': result.avg_waiting_time,
            'served': result.total_passengers_served,
            'leftover': result.total_passengers_left,
            'arrived': result.total_passengers_arrived,
            'fitness': result.fitness,
        })
        
        # Update current_queues for the next hour's simulation
        # The final queue state is in the last entry of the time series
        current_queues = result.queue_time_series[-1]
    
    # Aggregate results over the full day
    df = pd.DataFrame(daily_results)
    print(df.to_string(index=False))
    total_served = df['served'].sum()
    # The final leftover is the queue at the end of the last hour
    total_leftover = df.iloc[-1]['leftover']
    total_arrived = df['arrived'].sum()
    
    # Calculate the weighted average of waiting time by the number of passengers who arrived in that hour
    # We filter out hours where no one arrived to avoid issues with zero weights.
    valid_hours = df[df['arrived'] > 0]
    if not valid_hours.empty:
        weighted_avg_wait = np.average(valid_hours['avg_waiting_time'], weights=valid_hours['arrived'])
    else:
        weighted_avg_wait = 0
    
    return {
        'Total Passengers Served': total_served,
        'Total Passengers Leftover': total_leftover,
        'Overall Average Waiting Time (min)': weighted_avg_wait,
        'Total Arrived': total_arrived,
        'Fitness': df['fitness'].sum()
    }, daily_results # Return both summary and detailed results


def main():
    # --- Load Predictions and Schedules ---
    print("Loading data...")
    predictor = HourlyLoadPredictor()
    if not predictor.hourly_avg:
        print("Could not load prediction data. Exiting.")
        return

    # Load dynamic headways
    headway_file = os.path.join(current_dir, 'hourly_headways.json')
    if not os.path.exists(headway_file):
        print(f"Error: '{headway_file}' not found.")
        print("Please run 'hourly_optimizer_driver.py' first to generate the schedule.")
        return
        
    with open(headway_file, 'r') as f:
        dynamic_headways = json.load(f)

    # Define static headway
    static_headway = [5] * 21

    # --- Simulation Parameters ---
    sim_params = {
        'n_stations': 16,
        'capacity': 2184,
        'travel_times': [2] * 15,
        'weights': (3.5, 0.25, 50.0), # alpha, beta, gamma from Driver.py
        'p_dest': np.array([
            [0.000, 0.003, 0.003, 0.056, 0.077, 0.159, 0.033, 0.041,
            0.092, 0.018, 0.054, 0.076, 0.058, 0.062, 0.131, 0.136],
            [0.000, 0.000, 0.002, 0.089, 0.082, 0.189, 0.028, 0.040,
            0.085, 0.024, 0.072, 0.086, 0.046, 0.059, 0.088, 0.110],
            [0.000, 0.000, 0.000, 0.121, 0.146, 0.209, 0.053, 0.050,
            0.078, 0.020, 0.057, 0.068, 0.029, 0.045, 0.058, 0.065],
            [0.000, 0.000, 0.000, 0.000, 0.008, 0.080, 0.061, 0.057,
            0.129, 0.026, 0.095, 0.131, 0.067, 0.078, 0.143, 0.124],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.022, 0.053, 0.085,
            0.104, 0.024, 0.101, 0.122, 0.096, 0.086, 0.160, 0.146],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.010, 0.032,
            0.076, 0.027, 0.105, 0.144, 0.087, 0.106, 0.206, 0.208],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.007,
            0.059, 0.040, 0.131, 0.175, 0.117, 0.102, 0.206, 0.165],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.023, 0.030, 0.157, 0.200, 0.113, 0.113, 0.185, 0.179],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.008, 0.057, 0.152, 0.105, 0.127, 0.278, 0.274],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.033, 0.114, 0.152, 0.197, 0.202, 0.302],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.045, 0.101, 0.192, 0.257, 0.405],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.060, 0.161, 0.359, 0.420],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.034, 0.226, 0.740],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.156, 0.844],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
        ])
    }

    # --- Run Simulations ---
    print("Running 24-hour simulation for STATIC headway...")
    static_summary, _ = run_full_day_simulation(static_headway, predictor, sim_params)

    print("Running 24-hour simulation for DYNAMIC headway...")
    dynamic_summary, _ = run_full_day_simulation(dynamic_headways, predictor, sim_params)

    # --- Display Comparison ---
    comparison_df = pd.DataFrame([static_summary, dynamic_summary], 
                                 index=['Static Schedule', 'Dynamic Schedule'])
    
    print("\n\n--- Full Day Simulation Comparison ---")
    print(comparison_df.to_string())


if __name__ == "__main__":
    main()
