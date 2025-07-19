import pandas as pd
import matplotlib.pyplot as plt

def simulate_and_plot(csv_file, num_stations, simulation_time, headway, cost_per_train, cost_per_waiting_minute):
    df = pd.read_csv(csv_file)

    # Prepare arrival data structure
    arrivals = [[[0 for _ in range(simulation_time)] for _ in range(2)] for _ in range(num_stations)]
    for _, row in df.iterrows():
        station = int(row['station_id'])
        platform = int(row['platform'])
        minute = int(row['minute'])
        count = int(row['passengers_arriving'])

        if station < num_stations and minute < simulation_time:
            arrivals[station][platform][minute] += count

    # Initialize variables
    platforms = [[0 for _ in range(2)] for _ in range(num_stations)]
    total_boarded = 0
    total_train_arrivals = 0
    passenger_cost_over_time = []
    operator_cost_over_time = []
    passengers_waiting_over_time = []

    total_waiting_minutes = 0
    cumulative_passenger_cost = 0
    cumulative_operator_cost = 0

    for t in range(simulation_time):
        current_waiting = 0

        for station in range(num_stations):
            for platform in [0, 1]:
                # Add new arrivals
                platforms[station][platform] += arrivals[station][platform][t]
                current_waiting += platforms[station][platform]

                # Train arrives
                if t % headway == 0:
                    total_boarded += platforms[station][platform]
                    platforms[station][platform] = 0
                    total_train_arrivals += 1

        # Waiting cost accumulates
        total_waiting_minutes += current_waiting
        cumulative_passenger_cost += current_waiting * cost_per_waiting_minute
        if t % headway == 0:
            cumulative_operator_cost += cost_per_train

        # Collect data for plotting
        passengers_waiting_over_time.append(current_waiting)
        passenger_cost_over_time.append(cumulative_passenger_cost)
        operator_cost_over_time.append(cumulative_operator_cost)

    net_cost = cumulative_operator_cost + cumulative_passenger_cost

    # Plotting
    time = list(range(simulation_time))
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, passengers_waiting_over_time, label="Waiting Passengers", color='blue')
    plt.ylabel("Passengers")
    plt.title("Passengers Waiting Over Time")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, passenger_cost_over_time, label="Passenger Cost", color='red')
    plt.ylabel("Cost (Currency)")
    plt.title("Cumulative Passenger Cost")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time, operator_cost_over_time, label="Operator Cost", color='green')
    plt.xlabel("Time (minutes)")
    plt.ylabel("Cost (Currency)")
    plt.title("Cumulative Operator Cost")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Final summary
    print("✅ Simulation Summary:")
    print(f"Total Passengers Boarded: {total_boarded}")
    print(f"Total Passenger Waiting Cost: {cumulative_passenger_cost}")
    print(f"Total Operator Cost: {cumulative_operator_cost}")
    print(f"Total Net Cost: {net_cost}")

# Example usage
if __name__ == "__main__":

    monthly_income = 29700
    week_hours = 48.8
    VoT = ((monthly_income/4.33)/(week_hours * 60)) * 1.5
    operational_cost = 23300000
    arrival_cost = 7664.5

    simulate_and_plot(
        csv_file="C:\\Users\\awsaf\OneDrive - BUET\\Projects\\IUT Poster Presentation\\metro-scheduler\\Simulator\\passenger_input.csv",
        num_stations=2,
        simulation_time=30,
        headway=2,
        cost_per_train=arrival_cost,
        cost_per_waiting_minute=VoT
    )
