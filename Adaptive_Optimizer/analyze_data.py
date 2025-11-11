#!/usr/bin/env python3

import pandas as pd
import json
import numpy as np

print("=== ANALYZING PREDICTIONS AND CURRENT PASSENGER DATA ===")

# Load predictions data
try:
    predictions_df = pd.read_csv('predictions/hourly_predictions.csv')
    print("✓ Predictions data loaded successfully")
    print(f"Predictions shape: {predictions_df.shape}")
    print(f"Columns: {predictions_df.columns.tolist()}")
    print("\nFirst few prediction entries:")
    print(predictions_df.head(10))
    
    print(f"\n=== PREDICTIONS BY STATION (HOURLY) ===")
    for station in sorted(predictions_df['Station'].unique()):
        station_data = predictions_df[predictions_df['Station'] == station]
        print(f"\n{station}:")
        print("  Hour | Pred Rate | Pred/Hour")
        for _, row in station_data.head(24).iterrows():
            print(f"  {row['Hour']:2d}   | {row['Predicted_Rate']:8.2f} | {row['Predicted_Passengers_Per_Hour']:8.1f}")
            
except Exception as e:
    print(f"❌ Error loading predictions: {e}")

# Load monitoring history to see actual passenger data
try:
    with open('test_fixed/monitoring_history.json', 'r') as f:
        monitoring_data = json.load(f)
    
    print(f"\n=== ACTUAL PASSENGER DATA FROM MONITORING ===")
    print(f"Total monitoring entries: {len(monitoring_data)}")
    
    # Show first few entries with detailed passenger data
    print("\nFirst 10 monitoring entries:")
    print("Time | Station Data (Station: Current Passengers)")
    
    for i, entry in enumerate(monitoring_data[:10]):
        time = entry['time_minutes']
        hour = int(time // 60)
        passenger_data = entry['current_demand']
        
        print(f"\n{time:5.1f} min (H{hour:2d}) |", end="")
        stations_with_data = [(k, v) for k, v in passenger_data.items() if v > 0]
        if stations_with_data:
            for j, (station, count) in enumerate(stations_with_data[:5]):  # Show first 5 for brevity
                print(f" {station}: {count:.1f}", end="")
            if len(stations_with_data) > 5:
                print(f" (+ {len(stations_with_data)-5} more)", end="")
        else:
            print(" No passenger data", end="")
    
    # Analyze passenger data by hour and station
    print(f"\n\n=== PASSENGER DATA ANALYSIS BY HOUR ===")
    
    hourly_data = {}
    for entry in monitoring_data:
        hour = int(entry['time_minutes'] // 60)
        if hour not in hourly_data:
            hourly_data[hour] = {}
        
        for station, count in entry['current_demand'].items():
            if station not in hourly_data[hour]:
                hourly_data[hour][station] = []
            hourly_data[hour][station].append(count)
    
    # Show average passenger data per hour per station
    print("\nAverage passengers per station per hour:")
    print("Hour |", end="")
    stations = ['S01UN', 'S02UC', 'S03US', 'S04PL', 'S05M11', 'S06M10', 'S07KP', 'S08SP']
    for station in stations:
        print(f" {station:6}", end="")
    print()
    
    for hour in sorted(hourly_data.keys())[:12]:  # Show first 12 hours
        print(f"{hour:4d} |", end="")
        for station in stations:
            if station in hourly_data[hour]:
                avg_passengers = np.mean(hourly_data[hour][station])
                print(f" {avg_passengers:6.1f}", end="")
            else:
                print(f" {0.0:6.1f}", end="")
        print()
        
except Exception as e:
    print(f"❌ Error loading monitoring data: {e}")

print(f"\n=== CHECKING ADAPTATION RESULTS ===")
try:
    with open('test_fixed/adaptation_history.json', 'r') as f:
        adaptation_data = json.load(f)
    
    print(f"Total adaptations: {len(adaptation_data)}")
    
    if adaptation_data:
        print("\nAdaptation details:")
        for i, adaptation in enumerate(adaptation_data):
            print(f"\nAdaptation {i+1}:")
            print(f"  Time: {adaptation['trigger_time']:.1f} minutes")
            print(f"  Trigger stations: {adaptation['trigger_stations']}")
            print(f"  Before fitness: {adaptation['before_fitness']:.2f}")
            print(f"  After fitness: {adaptation['after_fitness']:.2f}")
            print(f"  Improvement: {adaptation['improvement']:.1f}%")
            print(f"  Before headways: {adaptation['before_headways']}")
            print(f"  After headways: {adaptation['after_headways']}")
            
            # Check if passengers served data is available
            if 'before_passengers_served' in adaptation:
                print(f"  Before passengers served: {adaptation['before_passengers_served']}")
                print(f"  After passengers served: {adaptation['after_passengers_served']}")
            else:
                print("  ❌ Passengers served data missing!")
    else:
        print("No adaptations found in history")
        
except Exception as e:
    print(f"❌ Error loading adaptation data: {e}")