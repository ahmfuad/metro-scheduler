import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Add parent directories to Python path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Simulator import HeadwaySimulator
# from Constrained_GA import ConstrainedGeneticAlgorithm # <-- Use the new Constrained GA
from GA import GeneticAlgorithm # <-- Reverted to the original GA
from Adaptive_Optimizer.hourly_load_predictor import HourlyLoadPredictor
from compare_schedules import run_full_day_simulation

def generate_dynamic_schedule(num_trains, predictor, ga_params, sim_params):
    """
    Runs the GA for each hour to generate a full-day dynamic schedule.
    """
    print(f"  Generating dynamic schedule for fleet size: {num_trains}...")
    all_stations = predictor.get_all_stations()
    hourly_headways = {}

    # Operational hours (e.g., 15 hours from 6 AM to 9 PM)
    for hour in range(15): 
        
        def constant_lambda(rate):
            return lambda t: float(rate)

        # Use the predictor for the current hour
        total_passengers_this_hour = sum(
            predictor.predict(station, hour)['entries'] 
            for station in all_stations
        )
        
        # If no passengers are predicted, use a default even headway
        if total_passengers_this_hour == 0:
            # Create a valid schedule that sums to 60
            base_headway = 60 / num_trains
            hourly_headways[hour] = [base_headway] * num_trains
            print(f"Hour {hour}: No passengers predicted. Using default headway: {hourly_headways[hour]}")
            continue

        lambdas_this_hour = [
            constant_lambda(predictor.predict(st, hour)['entries'] / 60.0)
            for st in all_stations
        ]

        sim = HeadwaySimulator(
            n_stations=sim_params['n_stations'],
            capacity=sim_params['capacity'],
            travel_times=sim_params['travel_times'],
            p_dest=sim_params['p_dest']
        )

        # The GA needs to know how many trains to schedule for the hour
        ga_params['num_trains'] = num_trains
        ga = GeneticAlgorithm(sim=sim, lambdas=lambdas_this_hour, **ga_params)
        
        print(f"  Running GA for hour {hour}...")
        best_H, _, _, _ = ga.run()
        
        hourly_headways[hour] = best_H

    return hourly_headways

def run_experiment(fleet_sizes, predictor, ga_params, sim_params):
    """
    Runs the full experiment for a range of fleet sizes using realistic simulation.
    """
    summary_results = []
    detailed_results = []

    for k in fleet_sizes:
        print(f"\n--- Running Experiment for Fleet Size: {k} ---")

        ga_params['headway_min'] = min(20, 60 // k) #
        
        # 1. Generate dynamic schedule using GA
        dynamic_schedule = generate_dynamic_schedule(k, predictor, ga_params, sim_params)
        
        # 2. Define a simple static schedule for comparison
        static_schedule = [math.floor(60/k)] * k
        
        # 3. Run full-day simulation for both
        print(f"  Simulating STATIC schedule for fleet size {k}...")
        static_summary, static_details = run_full_day_simulation(static_schedule, predictor, sim_params)
        
        print(f"  Simulating DYNAMIC schedule for fleet size {k}...")
        dynamic_summary, dynamic_details = run_full_day_simulation(dynamic_schedule, predictor, sim_params)
        
        # 4. Store summary results for plotting
        summary_results.append({
            'fleet_size': k,
            'static_fitness': static_summary['Fitness'],
            'dynamic_fitness': dynamic_summary['Fitness'],
            'static_served': static_summary['Total Passengers Served'],
            'dynamic_served': dynamic_summary['Total Passengers Served'],
            'static_avg_wait': static_summary['Overall Average Waiting Time (min)'],
            'dynamic_avg_wait': dynamic_summary['Overall Average Waiting Time (min)'],
        })

        # 5. Store detailed hourly results for CSV export
        for result in static_details:
            result['fleet_size'] = k
            result['schedule_type'] = 'static'
        detailed_results.extend(static_details)

        for result in dynamic_details:
            result['fleet_size'] = k
            result['schedule_type'] = 'dynamic'
        detailed_results.extend(dynamic_details)
        
    return pd.DataFrame(summary_results), pd.DataFrame(detailed_results)

def plot_results(df):
    """
    Plots the comparison results and saves them to files.
    """
    print("\nPlotting results...")
    
    # Plot 1: Fitness (Overall Cost) Comparison
    plt.figure(figsize=(12, 7))
    plt.plot(df['fleet_size'], df['static_fitness'], marker='o', linestyle='--', label='Static Schedule Fitness')
    plt.plot(df['fleet_size'], df['dynamic_fitness'], marker='s', linestyle='-', label='Dynamic Schedule Fitness')
    plt.title('Overall Fitness (Cost) vs. Fleet Size', fontsize=16)
    plt.xlabel('Fleet Size (Number of Trains per Hour)', fontsize=12)
    plt.ylabel('Total Fitness (Lower is Better)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.xticks(df['fleet_size'])
    fitness_plot_path = os.path.join(current_dir, 'fleet_size_vs_fitness.jpg')
    plt.savefig(fitness_plot_path)
    print(f"Saved fitness plot to {fitness_plot_path}")
    plt.close() # Close the figure to free memory

    # Plot 2: Passengers Served Comparison
    plt.figure(figsize=(12, 7))
    plt.plot(df['fleet_size'], df['static_served'], marker='o', linestyle='--', label='Static Schedule Served')
    plt.plot(df['fleet_size'], df['dynamic_served'], marker='s', linestyle='-', label='Dynamic Schedule Served')
    plt.title('Total Passengers Served vs. Fleet Size', fontsize=16)
    plt.xlabel('Fleet Size (Number of Trains per Hour)', fontsize=12)
    plt.ylabel('Total Passengers Served', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.xticks(df['fleet_size'])
    served_plot_path = os.path.join(current_dir, 'fleet_size_vs_served.jpg')
    plt.savefig(served_plot_path)
    print(f"Saved served passengers plot to {served_plot_path}")
    plt.close() # Close the figure

    # Plot 3: Average Waiting Time Comparison
    plt.figure(figsize=(12, 7))
    plt.plot(df['fleet_size'], df['static_avg_wait'], marker='o', linestyle='--', label='Static Schedule Avg Wait Time')
    plt.plot(df['fleet_size'], df['dynamic_avg_wait'], marker='s', linestyle='-', label='Dynamic Schedule Avg Wait Time')
    plt.title('Overall Average Waiting Time vs. Fleet Size', fontsize=16)
    plt.xlabel('Fleet Size (Number of Trains per Hour)', fontsize=12)
    plt.ylabel('Average Waiting Time (minutes)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.xticks(df['fleet_size'])
    wait_plot_path = os.path.join(current_dir, 'fleet_size_vs_wait_time.jpg')
    plt.savefig(wait_plot_path)
    print(f"Saved average waiting time plot to {wait_plot_path}")
    plt.close() # Close the figure
    
    print("\nPlots generated. To view them, check the files in the 'Simple Optimizer' directory.")


def main():
    """
    Main function to set up and run the experiment.
    """
    # --- Parameters ---
    fleet_sizes_to_test = range(5, 21)
    
    sim_params = {
        'n_stations': 16,
        'capacity': 2184,
        'travel_times': [2] * 15,
        'weights': (3.5, 0.25, 1680),
        'p_dest': np.array([
            [0.000, 0.003, 0.003, 0.056, 0.077, 0.159, 0.033, 0.041, 0.092, 0.018, 0.054, 0.076, 0.058, 0.062, 0.131, 0.136],
            [0.000, 0.000, 0.002, 0.089, 0.082, 0.189, 0.028, 0.040, 0.085, 0.024, 0.072, 0.086, 0.046, 0.059, 0.088, 0.110],
            [0.000, 0.000, 0.000, 0.121, 0.146, 0.209, 0.053, 0.050, 0.078, 0.020, 0.057, 0.068, 0.029, 0.045, 0.058, 0.065],
            [0.000, 0.000, 0.000, 0.000, 0.008, 0.080, 0.061, 0.057, 0.129, 0.026, 0.095, 0.131, 0.067, 0.078, 0.143, 0.124],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.022, 0.053, 0.085, 0.104, 0.024, 0.101, 0.122, 0.096, 0.086, 0.160, 0.146],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.010, 0.032, 0.076, 0.027, 0.105, 0.144, 0.087, 0.106, 0.206, 0.208],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.007, 0.059, 0.040, 0.131, 0.175, 0.117, 0.102, 0.206, 0.165],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.023, 0.030, 0.157, 0.200, 0.113, 0.113, 0.185, 0.179],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.008, 0.057, 0.152, 0.105, 0.127, 0.278, 0.274],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.033, 0.114, 0.152, 0.197, 0.202, 0.302],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.045, 0.101, 0.192, 0.257, 0.405],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.060, 0.161, 0.359, 0.420],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.034, 0.226, 0.740],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.156, 0.844],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
        ])
    }
    
    ga_params = {
        'alpha': sim_params['weights'][0],
        'beta': sim_params['weights'][1],
        'gamma': sim_params['weights'][2],
        'pop_size': 20, 
        'generations': 20,
        'headway_min': 3,
        'headway_max': 15,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        # 'max_total_headway': 60 # This parameter is for the Constrained GA
    }

    # --- Load Data and Run Experiment ---
    print("Loading passenger prediction data...")
    data_file = os.path.join(parent_dir, 'data', 'simulation_log.csv')
    predictor = HourlyLoadPredictor(data_path=data_file)
    if not predictor.hourly_avg:
        print("Could not load prediction data. Exiting.")
        return

    summary_df, details_df = run_experiment(fleet_sizes_to_test, predictor, ga_params, sim_params)
    
    print("\n--- Final Experiment Summary ---")
    print(summary_df.to_string(index=False))

    # Save detailed results to CSV
    details_csv_path = os.path.join(current_dir, 'fleet_size_hourly_details.csv')
    details_df.to_csv(details_csv_path, index=False)
    print(f"\nSaved detailed hourly results to {details_csv_path}")
    
    plot_results(summary_df)

if __name__ == "__main__":
    # Note: This script can take a long time to run due to the nested loops
    # of fleet sizes, hours, and GA generations.
    main()
