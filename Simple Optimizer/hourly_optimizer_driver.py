import sys
import os
import numpy as np
import json

# Add the parent directory of 'Simple Optimizer' to the Python path
# to allow imports from 'Adaptive_Optimizer'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Simulator import HeadwaySimulator
from GA import GeneticAlgorithm
from Adaptive_Optimizer.hourly_load_predictor import HourlyLoadPredictor


def main():
    sum_total_pass = 0

    """
    Runs a genetic algorithm to find the optimal headway for each hour of the day
    based on predicted hourly passenger loads.
    """
    # --- Load Predictions ---
    print("Loading hourly passenger load predictions...")
    predictor = HourlyLoadPredictor()
    if not predictor.hourly_avg:
        print("Could not load prediction data. Exiting.")
        return

    all_stations = predictor.get_all_stations()
    
    # --- Simulation and GA Parameters (from Driver.py) ---
    n_stations = 16
    train_capacity = 2184
    travel_times = [2] * (n_stations - 1)
    p_matrix = np.array([
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

    # --- Hourly Simulation Loop ---
    hourly_headways = {}

    for hour in range(24):
        print(f"\n--- Optimizing for Hour: {hour}:00 ---")

        # 1. Get lambdas for the current hour from the predictor
        # The predictor gives total entries for the hour. We need rate per minute.
        def constant_lambda(rate):
            return lambda t: float(rate)

        lambdas_this_hour = []
        for station in all_stations:
            prediction = predictor.predict(station, hour)
            if prediction:
                # Convert hourly entry count to per-minute rate
                entry_rate = prediction['entries'] / 60.0
                lambdas_this_hour.append(constant_lambda(entry_rate))
            else:
                # Default to 0 if no prediction
                lambdas_this_hour.append(constant_lambda(0))
        
        print("Lambdas for this hour (passengers/minute):",
              [round(lambdas_this_hour[i](0), 2) for i in range(len(lambdas_this_hour))])
        # If no passengers are predicted for this hour, we can use a default headway
        # or skip optimization. For now, let's assume a default.
        total_passengers_this_hour = sum(predictor.predict(station, hour)['entries'] for station in all_stations if predictor.predict(station, hour))
        
        sum_total_pass += total_passengers_this_hour
        
        print("Passengers predicted this hour:", total_passengers_this_hour)
        if total_passengers_this_hour == 0:
            print("No passengers predicted for this hour. Using default headway.")
            # Using a large headway as a default when there's no traffic
            hourly_headways[hour] = [10] * 8 
            continue
        
        # 2. Initialize Simulator for this hour
        sim = HeadwaySimulator(
            n_stations=n_stations,
            capacity=train_capacity,
            travel_times=travel_times,
            p_dest=p_matrix
        )

        # 3. Initialize and run GA for this hour
        ga = GeneticAlgorithm(
            sim=sim,
            lambdas=lambdas_this_hour,
            alpha=3.5,
            beta=0.25,
            gamma=50.0,
            pop_size=25,
            generations=30,
            headway_min=3,
            headway_max=10,
            num_trains=21,
            mutation_rate=0.15,
            crossover_rate=0.6
        )

        best_H, best_J, _, _ = ga.run()
        hourly_headways[hour] = best_H
        print(f"Optimal Headway for hour {hour}: {best_H} (Fitness: {best_J:.2f})")

    # --- Final Results ---
    print("\n\n--- Optimal Hourly Headway Schedules ---")
    for hour, H in hourly_headways.items():
        print(f"Hour {hour:02d}:00 - {hour+1:02d}:00 | Headway: {H}")

    # Save results to a file
    output_path = os.path.join(current_dir, 'hourly_headways.json')
    with open(output_path, 'w') as f:
        json.dump(hourly_headways, f, indent=4)
    print(f"\nSaved hourly headways to {output_path}")
    print("\nTotal passengers predicted over the day:", sum_total_pass)

if __name__ == "__main__":
    main()
