import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
from tabulate import tabulate  # For pretty printing tables
from deap import base, creator, tools, algorithms
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import copy


# Simulation Parameters and Data
SIMULATION_TIME = 60  # Total minutes from 7:00 to 21:00
TRAIN_CAPACITY = 2184  # Train capacity
MAX_HEADWAY = 5  # Max headway in minutes
MIN_HEADWAY = .1   # Min headway in minutes
TRAVEL_TIME_BETWEEN_STATIONS = 2  # minutes between stations
DWELL_TIME = 1  # minutes at each station
MAX_STATION_CAPACITY = 1e9  # Max passengers at a station

# Cost Parameters
VALUE_OF_TIME = 3.5  # BDT per minute
COST_PER_TRAIN_DISPATCH = 277000  # BDT per train dispatch
COST_PER_TRAIN_TRAVEL_MIN = 26500  # BDT per train per minute
PENALTY_WAITING = None  # Penalty per passenger left waiting at end of day
PENALTY_LEFTOVER = None  # Penalty per passenger left unboarded at end of day
EPOCH_SIZE = 5  # minutes per epoch for dynamic scheduling


class MRTSimulator:
    """
    A class to simulate MRT Line 6 operations for Dhaka, supporting static and dynamic scheduling.
    Handles OD matrix, passenger arrivals, departure generation, and minute-by-minute simulation.
    """
    
    def __init__(self, monthly_boarding=None, T=SIMULATION_TIME, C=TRAIN_CAPACITY, h_min=MIN_HEADWAY, h_max=MAX_HEADWAY, d_i=TRAVEL_TIME_BETWEEN_STATIONS, dwell_time=DWELL_TIME, 
                 Q_threshold=MAX_STATION_CAPACITY, vot=VALUE_OF_TIME, ct=COST_PER_TRAIN_DISPATCH, cd=COST_PER_TRAIN_TRAVEL_MIN, ww=None, wo=None, wp=1, gamma=100, epoch_size=EPOCH_SIZE):
        """
        Initialize the simulator.
        """
        self.stations = [
            'S01UN', 'S02UC', 'S03US', 'S04PL', 'S05M11', 'S06M10',
            'S07KP', 'S08SP', 'S09AG', 'S10BS', 'S11FG', 'S12KB',
            'S13SB', 'S14DU', 'S15BS', 'S16MJ'
        ]
        self.n_stations = len(self.stations)
        self.station_to_idx = {s: i for i, s in enumerate(self.stations)}
        
        # Parse OD matrix from report (monthly totals for OD pairs)
        row_lists_15 = [
            [2614, 3037, 53646, 73547, 151650, 31849, 38966, 87927, 17074, 51458, 71943, 54814, 59100, 124996, 129478],  # S01 to others
            [2969, 436, 20366, 18739, 43402, 6491, 9154, 19617, 5563, 16584, 19693, 10565, 13516, 20292, 25264],  # S02
            [2916, 509, 9962, 11964, 17116, 4310, 4096, 6435, 1668, 4692, 5576, 2407, 3716, 4793, 5312],  # S03
            [57564, 20518, 9289, 4016, 40037, 30485, 28453, 64101, 13027, 47142, 65538, 33534, 38943, 71497, 61751],  # S04
            [83913, 21074, 12003, 4380, 12204, 28951, 46666, 57070, 13099, 55348, 66701, 52291, 46920, 87489, 80082],  # S05
            [143726, 38428, 12369, 29628, 8758, 9286, 28499, 67369, 23705, 92755, 127172, 76593, 93890, 182115, 183857],  # S06
            [37329, 7941, 3883, 30302, 25984, 11170, 1686, 15182, 10365, 33712, 45013, 29968, 26111, 52877, 42318],  # S07
            [43241, 9830, 4268, 28862, 42404, 35812, 1695, 8196, 10738, 56666, 72141, 40657, 40681, 66749, 64595],  # S08
            [86770, 18388, 5063, 55044, 48258, 76663, 13452, 8121, 3816, 27329, 73252, 50620, 61135, 134467, 132613],  # S09
            [19011, 6681, 1556, 15426, 15161, 35349, 11831, 11520, 3694, 4343, 15101, 20197, 26216, 26852, 40084],  # S10
            [50574, 15974, 3916, 41363, 47465, 98554, 30103, 54173, 30598, 4863, 17310, 38713, 73446, 98291, 155142],  # S11
            [69580, 19413, 4578, 59207, 58015, 137840, 37952, 62915, 69492, 13947, 11557, 19873, 53785, 119543, 140129],  # S12
            [58401, 11123, 2151, 31138, 47247, 86685, 27723, 37785, 54448, 23193, 36606, 22141, 4757, 31337, 102505],  # S13
            [61059, 14102, 2985, 37057, 42198, 101944, 23023, 38020, 62311, 26032, 65659, 53077, 4360, 11158, 60329],  # S14
            [106143, 18859, 4016, 55950, 68090, 177188, 42021, 55432, 127587, 27037, 80565, 113438, 26901, 12013, 18701],  # S15
            [143459, 29421, 5132, 62738, 82818, 215612, 40877, 68244, 145448, 40804, 144861, 146226, 92737, 57968, 16106]   # S16
        ]
        
        self.od_matrix = np.zeros((self.n_stations, self.n_stations), dtype=float)
        for i in range(self.n_stations):
            list_data = row_lists_15[i]
            k = 0
            for j in range(i+1, self.n_stations):  # Only downstream
                self.od_matrix[i, j] = list_data[k]
                k += 1
        
        # Monthly boarding totals (row sums)
        if monthly_boarding is None:
            self.monthly_boarding = np.sum(self.od_matrix, axis=1)
        else:
            self.monthly_boarding = monthly_boarding
        
        # Daily boarding (31 days)
        self.daily_boarding = self.monthly_boarding / 31
        self.T = T
        self.lambda_per_station = self.daily_boarding / T  # Constant base rate per min per origin
        
        # Parameters
        self.C = C
        self.h_min = h_min
        self.h_max = h_max
        self.d_i = d_i
        self.dwell_time = dwell_time
        self.total_trip_time = (self.n_stations - 1) * d_i + self.n_stations * self.dwell_time
        self.Q_threshold = Q_threshold
        self.epoch_size = epoch_size  # minutes per epoch, default 5
        
        # Costs
        self.vot = vot
        self.ct = ct
        self.cd = cd
        self.ww = ww if ww is not None else vot
        self.wo = wo if wo is not None else vot
        self.wp = wp
        self.gamma = gamma
        
        # Static schedule periods: (start_str, end_str, headway)
        self.static_periods = [
            ('7:10', '7:30', 10),
            ('7:31', '11:36', 8),
            ('11:37', '14:36', 10),
            ('14:37', '20:20', 8),
            ('20:21', '21:00', 10)
        ]
        
        # Total daily passengers
        self.total_daily_pass = np.sum(self.daily_boarding)
    
    def time_str_to_minutes(self, time_str, start_hour=7):
        """Convert 'H:MM' to minutes from 7:00."""
        h, m = map(int, time_str.split(':'))
        return (h - start_hour) * 60 + m
    
    def get_period_for_minute(self, t):
        """Get headway for minute t (from 0 = 7:00)."""
        for start_str, end_str, h in self.static_periods:
            start_min = self.time_str_to_minutes(start_str)
            end_min = self.time_str_to_minutes(end_str)
            if start_min <= t < end_min:
                return h
        # Default to last period headway
        return self.static_periods[-1][2]
    
    def generate_departure_times(self, headways=None, is_static=True):
        """Generate departure times in minutes from 7:00 for static or dynamic schedule."""
        if is_static:
            departures = []
            current_time = self.time_str_to_minutes(self.static_periods[0][0])
            departures.append(current_time)
            for start_str, end_str, h in self.static_periods:
                start_min = self.time_str_to_minutes(start_str)
                end_min = self.time_str_to_minutes(end_str)
                next_dep = current_time + h
                while next_dep <= end_min:
                    departures.append(next_dep)
                    next_dep += h
                current_time = next_dep
                if current_time < start_min:
                    current_time = start_min
                if current_time not in departures:
                    departures.append(current_time)
                current_time += h
            return sorted(set(departures))
        else:
            # For dynamic: cumulative sum of headways
            return np.cumsum([0] + headways).astype(int)
    
    def generate_passenger_arrivals_csv(self, output_file='passenger_arrivals.csv', stochastic=False, scale_by_headway=True):
        """
        Generate a CSV with passenger arrivals per minute per OD pair.
        Columns: minute, origin, destination, num_passengers
        
        Scales arrival rate by 1/headway if scale_by_headway=True (higher demand in low headway periods, 
        assuming pressure correlates inversely with headway).
        """
        data = []
        avg_h = np.mean([h for _, _, h in self.static_periods])
        
        for t in range(self.T):
            h_period = self.get_period_for_minute(t)
            scale = (avg_h / h_period) if scale_by_headway else 1.0
            
            for i, origin in enumerate(self.stations):
                # Base arrival rate at origin i
                base_arr_rate = self.lambda_per_station[i] * scale
                if stochastic:
                    num_arr_origin = np.random.poisson(base_arr_rate)
                else:
                    num_arr_origin = round(base_arr_rate)  # Integer for simplicity
                
                if num_arr_origin > 0:
                    # Distribute to destinations based on OD proportions
                    total_od_monthly = np.sum(self.od_matrix[i])
                    if total_od_monthly > 0:
                        for j in range(i+1, self.n_stations):  # Only downstream
                            prop = self.od_matrix[i, j] / total_od_monthly
                            num_to_dest = round(num_arr_origin * prop)
                            if num_to_dest > 0:
                                data.append({
                                    'minute': t,
                                    'origin': origin,
                                    'destination': self.stations[j],
                                    'num_passengers': num_to_dest
                                })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Generated {output_file} with {len(df)} rows (total arrivals: {df['num_passengers'].sum():.0f} passengers).")
        print("Sample data:")
        print(df.head(10))
        return df
    
    def load_arrivals_from_csv(self, csv_file='passenger_arrivals.csv'):
        """Load arrivals from CSV and prepare data structure for simulation."""
        df = pd.read_csv(csv_file)
        # Convert station names to indices
        df['origin_idx'] = df['origin'].map(self.station_to_idx)
        df['dest_idx'] = df['destination'].map(self.station_to_idx)
        
        # Select only non-grouping columns to avoid deprecation warning
        select_cols = ['dest_idx', 'num_passengers']
        arrivals_per_min_origin = df.groupby(['minute', 'origin_idx'])[select_cols].apply(
            lambda g: [(row['dest_idx'], row['num_passengers'], g.name[0]) for _, row in g.iterrows()]
        ).to_dict()

        
        # Reorganize into {minute: {origin_idx: groups}}
        arrivals_per_min = {}
        for (minute, origin_idx), groups in arrivals_per_min_origin.items():
            if minute not in arrivals_per_min:
                arrivals_per_min[minute] = {}
            arrivals_per_min[minute][origin_idx] = groups
        
        return arrivals_per_min
    
    def simulate_per_epoch(self, csv_file, departure_times, epoch_size=None):
        """
        Simulate per epoch using CSV arrivals and departure times.
        Epochs are groups of minutes (default self.epoch_size=5).
        
        Returns: dict of epoch_start: {'waiting_cost': float, 'operating_cost': float, 
                                       'travelling_cost': float, 'total_cost': float,
                                       'passengers_boarded': int, 'avg_wait': float,
                                       'trains_dispatched': int, 'dispatch_times': list[int],
                                       'total_arrivals': int, 'arrivals_per_station': dict[station_idx, int]}
        """
        if epoch_size is None:
            epoch_size = self.epoch_size
        
        arrivals_data = self.load_arrivals_from_csv(csv_file)
        num_epochs = (self.T + epoch_size - 1) // epoch_size
        epoch_results = {}
        
        # Queues: dict of station_idx: list of {'arrival_min': int, 'dest_idx': int, 'num': int}
        queues = {i: [] for i in range(self.n_stations)}
        
        # Sort departures
        dep_set = sorted(set(departure_times))
        train_idx = 0
        
        # Cumulative trackers per epoch
        cum_waiting_time = 0
        cum_boarded = 0
        cum_oper_cost = 0
        cum_travel_cost = 0
        cum_trains = 0
        cum_dispatch_times = []
        cum_arrivals = 0
        cum_arrivals_per_station = {i: 0 for i in range(self.n_stations)}
        
        last_epoch_start = 0
        
        for t in range(1, self.T + 1):
            # Add arrivals at minute t to queues and count
            m = t - 1  # arrivals for previous minute affect t
            if m in arrivals_data:
                for origin_i, groups in arrivals_data[m].items():
                    epoch_arr = sum(num for _, num, _ in groups)
                    cum_arrivals += epoch_arr
                    cum_arrivals_per_station[origin_i] += epoch_arr
                    for dest_j, num, arr_min in groups:
                        queues[origin_i].append({'arrival_min': arr_min, 'dest_idx': dest_j, 'num': num})
            
            # Check for train dispatch at t (from station 0)
            dispatched_this_min = False
            dispatch_time = None
            if train_idx < len(dep_set) and dep_set[train_idx] == t:
                dispatched_this_min = True
                dispatch_time = t
                cum_dispatch_times.append(t)
                train_idx += 1
                cum_oper_cost += self.ct  # fixed dispatch cost
                cum_trains += 1
                
                # Simulate train journey
                train = {'occ': 0, 'onboard': []}
                for pos in range(self.n_stations):
                    arr_time_at_pos = t + pos * (self.d_i + self.dwell_time)
                    if arr_time_at_pos > self.T:
                        break
                    
                    # Alight at pos
                    new_onboard = []
                    alighted = 0
                    for onboard_group in train['onboard']:
                        if onboard_group['dest_idx'] == pos:
                            alighted += onboard_group['num']
                        else:
                            new_onboard.append(onboard_group)
                    train['onboard'] = new_onboard
                    train['occ'] -= alighted
                    
                    # Board at pos (FCFS: sort queues by arrival_min)
                    if pos in queues and queues[pos]:
                        queues[pos].sort(key=lambda g: g['arrival_min'])
                        remaining_cap = self.C - train['occ']
                        while remaining_cap > 0 and queues[pos]:
                            group = queues[pos].pop(0)
                            board_num = min(group['num'], remaining_cap)
                            if board_num > 0:
                                boarded_group = {
                                    'origin_idx': pos, 'dest_idx': group['dest_idx'], 
                                    'num': board_num, 'arr_min': group['arrival_min']
                                }
                                wait_time = arr_time_at_pos - group['arrival_min']
                                cum_waiting_time += wait_time * board_num
                                cum_boarded += board_num
                                remaining_cap -= board_num
                                train['occ'] += board_num
                                train['onboard'].append(boarded_group)
                            if group['num'] > board_num:
                                queues[pos].insert(0, {'arrival_min': group['arrival_min'], 
                                                       'dest_idx': group['dest_idx'], 'num': group['num'] - board_num})
                
                # Travel cost for this train
                cum_travel_cost += self.cd * self.total_trip_time
            
            # Epoch boundary
            if t % epoch_size == 0 or t == self.T:
                epoch_end = min(t, self.T)
                epoch_start = last_epoch_start
                epoch_key = epoch_start
                
                # Compute for this epoch
                epoch_waiting_cost = self.ww * cum_waiting_time
                epoch_operating_cost = self.wp * cum_oper_cost
                epoch_travelling_cost = cum_travel_cost
                epoch_total_cost = epoch_waiting_cost + epoch_operating_cost + epoch_travelling_cost
                
                epoch_results[epoch_key] = {
                    'waiting_cost': epoch_waiting_cost,
                    'operating_cost': epoch_operating_cost,
                    'travelling_cost': epoch_travelling_cost,
                    'total_cost': epoch_total_cost,
                    'passengers_boarded': int(cum_boarded),
                    'total_arrivals': int(cum_arrivals),
                    'avg_wait': cum_waiting_time / max(cum_boarded, 1) if cum_boarded > 0 else 0,
                    'trains_dispatched': cum_trains,
                    'dispatch_times': cum_dispatch_times.copy(),
                    'arrivals_per_station': cum_arrivals_per_station.copy()
                }
                
                # Reset cumulatives for next epoch
                cum_waiting_time = 0
                cum_boarded = 0
                cum_oper_cost = 0
                cum_travel_cost = 0
                cum_trains = 0
                cum_dispatch_times = []
                cum_arrivals = 0
                cum_arrivals_per_station = {i: 0 for i in range(self.n_stations)}
                
                last_epoch_start = epoch_end + 1
        
        # Leftovers: add penalty to last epoch
        last_epoch = max(epoch_results.keys())
        leftovers = sum(sum(g['num'] for g in q) for q in queues.values())
        epoch_results[last_epoch]['total_cost'] += self.wo * leftovers
        epoch_results[last_epoch]['waiting_cost'] += self.ww * leftovers * (self.T - last_epoch)  # Approx remaining wait
        
        return epoch_results
    
    def print_epoch_summary(self, epoch_results):
        """
        Print detailed summary of each epoch, including total cost, key factors, dispatch times, etc.
        Also prints grand totals and arrivals per station per epoch.
        """
        epochs = sorted(epoch_results.keys())
        
        # Per-epoch table
        table_data = []
        for e in epochs:
            res = epoch_results[e]
            dispatch_str = ', '.join(map(str, res['dispatch_times'])) if res['dispatch_times'] else 'None'
            table_data.append([
                e,
                res['total_arrivals'],
                res['passengers_boarded'],
                res['trains_dispatched'],
                dispatch_str,
                f"{res['avg_wait']:.2f}",
                f"{res['waiting_cost']:.2f}",
                f"{res['operating_cost']:.2f}",
                f"{res['travelling_cost']:.2f}",
                f"{res['total_cost']:.2f}"
            ])
        
        headers = ['Epoch Start', 'Total Arrivals', 'Passengers Boarded', 'Trains Dispatched', 'Dispatch Times', 
                   'Avg Wait (min)', 'Waiting Cost', 'Operating Cost', 'Travelling Cost', 'Total Cost']
        print("\nPer-Epoch Summary:")
        print(tabulate(table_data, headers=headers, floatfmt=".2f"))
        
        # Arrivals per station per epoch
        print("\nArrivals Per Station Per Epoch:")
        for e in epochs:
            print(f"Epoch {e}:")
            station_arrivals = epoch_results[e]['arrivals_per_station']
            station_data = [[self.stations[i], station_arrivals.get(i, 0)] for i in range(self.n_stations)]
            print(tabulate(station_data, headers=['Station', 'Arrivals'], floatfmt=".0f"))
        
        # Grand totals
        grand_totals = {
            'total_arrivals': sum(res['total_arrivals'] for res in epoch_results.values()),
            'total_boarded': sum(res['passengers_boarded'] for res in epoch_results.values()),
            'total_trains': sum(res['trains_dispatched'] for res in epoch_results.values()),
            'overall_avg_wait': np.mean([res['avg_wait'] for res in epoch_results.values() if res['passengers_boarded'] > 0]),
            'total_waiting_cost': sum(res['waiting_cost'] for res in epoch_results.values()),
            'total_operating_cost': sum(res['operating_cost'] for res in epoch_results.values()),
            'total_travelling_cost': sum(res['travelling_cost'] for res in epoch_results.values()),
            'grand_total_cost': sum(res['total_cost'] for res in epoch_results.values())
        }
        
        print("\nGrand Totals:")
        for key, value in grand_totals.items():
            if isinstance(value, (int, float)):
                print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Grand total arrivals per station
        grand_arrivals_per_station = {i: sum(epoch_results[e]['arrivals_per_station'].get(i, 0) for e in epochs) for i in range(self.n_stations)}
        print("\nGrand Total Arrivals Per Station:")
        grand_station_data = [[self.stations[i], grand_arrivals_per_station[i]] for i in range(self.n_stations)]
        print(tabulate(grand_station_data, headers=['Station', 'Total Arrivals'], floatfmt=".0f"))
    
    def plot_epoch_results(self, epoch_results, save_path='costs_plot.png'):
        """
        Plot cumulative total cost, cumulative operating cost, cumulative waiting cost, and cumulative number of trains dispatched per epoch.
        Uses matplotlib for visualization. Saves the plot to a file.
        """
        epochs = sorted(epoch_results.keys())
        total_costs = [epoch_results[e]['total_cost'] for e in epochs]
        operating_costs = [epoch_results[e]['operating_cost'] for e in epochs]
        waiting_costs = [epoch_results[e]['waiting_cost'] for e in epochs]
        trains = [epoch_results[e]['trains_dispatched'] for e in epochs]
        
        # Calculate cumulative values
        cum_total_costs = np.cumsum(total_costs)
        cum_operating_costs = np.cumsum(operating_costs)
        cum_waiting_costs = np.cumsum(waiting_costs)
        cum_trains = np.cumsum(trains)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot costs on left y-axis
        line1 = ax1.plot(epochs, cum_total_costs, label='Cumulative Total Cost', color='blue', linewidth=2)
        line2 = ax1.plot(epochs, cum_operating_costs, label='Cumulative Operating Cost', color='green', linewidth=2)
        line3 = ax1.plot(epochs, cum_waiting_costs, label='Cumulative Waiting Cost', color='red', linewidth=2)
        ax1.set_xlabel('Epoch Start Minute')
        ax1.set_ylabel('Cumulative Costs (BDT)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.legend(loc='upper left')
        
        # Plot trains on right y-axis
        ax2 = ax1.twinx()
        bars = ax2.bar(epochs, cum_trains, alpha=0.3, color='orange', label='Cumulative Trains Dispatched')
        ax2.set_ylabel('Cumulative Number of Trains Dispatched', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.legend(loc='upper right')
        
        plt.title('Cumulative Costs and Cumulative Train Dispatches per Epoch')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        return save_path
    
    def generate_report_file(self, epoch_results, report_file='simulation_report.md', plot_path='costs_plot.png'):
        """
        Generate a full simulation report in a Markdown file, including all summaries and a reference to the plot.
        """
        with open(report_file, 'w') as f:
            f.write("# MRT Line 6 Simulation Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Per-Epoch Summary
            f.write("## Per-Epoch Summary\n\n")
            epochs = sorted(epoch_results.keys())
            table_data = []
            for e in epochs:
                res = epoch_results[e]
                dispatch_str = ', '.join(map(str, res['dispatch_times'])) if res['dispatch_times'] else 'None'
                table_data.append([
                    e,
                    res['total_arrivals'],
                    res['passengers_boarded'],
                    res['trains_dispatched'],
                    dispatch_str,
                    f"{res['avg_wait']:.2f}",
                    f"{res['waiting_cost']:.2f}",
                    f"{res['operating_cost']:.2f}",
                    f"{res['travelling_cost']:.2f}",
                    f"{res['total_cost']:.2f}"
                ])
            headers = ['Epoch Start', 'Total Arrivals', 'Passengers Boarded', 'Trains Dispatched', 'Dispatch Times', 
                       'Avg Wait (min)', 'Waiting Cost', 'Operating Cost', 'Travelling Cost', 'Total Cost']
            f.write(tabulate(table_data, headers=headers, tablefmt='pipe', floatfmt=".2f"))
            f.write("\n\n")
            
            # Arrivals Per Station Per Epoch
            f.write("## Arrivals Per Station Per Epoch\n\n")
            for e in epochs:
                f.write(f"### Epoch {e}\n\n")
                station_arrivals = epoch_results[e]['arrivals_per_station']
                station_data = [[self.stations[i], station_arrivals.get(i, 0)] for i in range(self.n_stations)]
                f.write(tabulate(station_data, headers=['Station', 'Arrivals'], tablefmt='pipe', floatfmt=".0f"))
                f.write("\n\n")
            
            # Grand Totals
            f.write("## Grand Totals\n\n")
            grand_totals = {
                'total_arrivals': sum(res['total_arrivals'] for res in epoch_results.values()),
                'total_boarded': sum(res['passengers_boarded'] for res in epoch_results.values()),
                'total_trains': sum(res['trains_dispatched'] for res in epoch_results.values()),
                'overall_avg_wait': np.mean([res['avg_wait'] for res in epoch_results.values() if res['passengers_boarded'] > 0]),
                'total_waiting_cost': sum(res['waiting_cost'] for res in epoch_results.values()),
                'total_operating_cost': sum(res['operating_cost'] for res in epoch_results.values()),
                'total_travelling_cost': sum(res['travelling_cost'] for res in epoch_results.values()),
                'grand_total_cost': sum(res['total_cost'] for res in epoch_results.values())
            }
            for key, value in grand_totals.items():
                f.write(f"- {key.replace('_', ' ').title()}: {value:.2f}\n")
            f.write("\n")
            
            # Grand Total Arrivals Per Station
            f.write("## Grand Total Arrivals Per Station\n\n")
            grand_arrivals_per_station = {i: sum(epoch_results[e]['arrivals_per_station'].get(i, 0) for e in epochs) for i in range(self.n_stations)}
            grand_station_data = [[self.stations[i], grand_arrivals_per_station[i]] for i in range(self.n_stations)]
            f.write(tabulate(grand_station_data, headers=['Station', 'Total Arrivals'], tablefmt='pipe', floatfmt=".0f"))
            f.write("\n\n")
            
            # Plot reference
            plot_save = self.plot_epoch_results(epoch_results, plot_path)
            f.write("## Costs and Train Dispatches Plot\n\n")
            f.write(f"![Costs Plot]({plot_path})\n")
        
        print(f"Report generated and saved to {report_file}")

    def optimize_headway_ga(self, current_queues, horizon=30, pop_size=50, generations=20):
        """
        Use Genetic Algorithm (DEAP) to optimize the next headway based on current passenger counts.
        Simulates forward for a horizon to evaluate cost.
        """
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_h", random.randint, self.h_min, self.h_max)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_h, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def eval_cost(individual):
            h = individual[0]
            # Simulate forward for horizon minutes with this headway
            # Simplified: assume dispatch at current t + h, compute projected cost
            # Use a mini-simulation: copy queues, add projected arrivals, dispatch train after h min, compute wait + oper
            projected_queues = {k: copy.deepcopy(v) for k, v in current_queues.items()}
            projected_wait = 0
            projected_boarded = 0
            projected_arrivals = sum(sum(g['num'] for g in v) for v in projected_queues.values())  # Current waiting
            
            for min_ahead in range(1, h + 1):
                # Add projected arrivals (avg rate)
                for i in range(self.n_stations):
                    proj_arr = round(self.lambda_per_station[i])
                    if proj_arr > 0:
                        # Simplified: add to queue as aggregate
                        projected_queues[i].append({'arrival_min': 0, 'dest_idx': 0, 'num': proj_arr})  # Dummy
                # Accumulate wait
                projected_wait += sum(sum(p['num'] for p in v) for v in projected_queues.values())
            
            # Dispatch train after h min, board
            train_occ = 0
            for pos in range(self.n_stations):
                # Simplified alight: assume average
                train_occ *= 0.8  # Approx
                # Board
                q_pos = projected_queues[pos]
                if q_pos:
                    board = min(sum(g['num'] for g in q_pos), self.C - train_occ)
                    projected_boarded += board
                    train_occ += board
                    # Reduce queue (simplified)
                    for g in q_pos:
                        g['num'] = 0
            
            # Cost: wait cost + oper (dispatch + travel)
            cost = self.ww * projected_wait + self.wp * (self.ct + self.cd * self.total_trip_time)
            return cost,
        
        toolbox.register("evaluate", eval_cost)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)  # Changed to cxUniform for single gene (though not ideal)
        toolbox.register("mutate", tools.mutUniformInt, low=self.h_min, up=self.h_max, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        pop = toolbox.population(n=pop_size)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        for gen in range(generations):
            offspring = algorithms.varAnd(pop, toolbox, cxpb=0.0, mutpb=0.1)  # Set cxpb=0 to avoid crossover error
            fitnesses = list(map(toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit
            pop = toolbox.select(offspring, len(pop))
        
        best = tools.selBest(pop, 1)[0]
        return best[0]  # Optimal headway
    
    def simulate_dynamic_with_ga(self, csv_file, epoch_size=None, ga_horizon=30, ga_pop=50, ga_gens=200):
        """
        Simulate dynamic scheduling using GA to optimize next headway at each epoch.
        Compares to static schedule costs.
        Updates dynamic plot with tkinter after each epoch.
        """
        if epoch_size is None:
            epoch_size = self.epoch_size
        
        arrivals_data = self.load_arrivals_from_csv(csv_file)
        num_epochs = (self.T + epoch_size - 1) // epoch_size
        dynamic_results = {}
        static_results = self.simulate_per_epoch(csv_file, self.generate_departure_times(is_static=True))
        
        # Queues for dynamic
        queues = {i: [] for i in range(self.n_stations)}
        
        # Dynamic departures
        dynamic_departures = []
        last_dispatch = -self.h_min  # Start with a dispatch possible at t=0
        
        # Tkinter setup for dynamic plot
        root = tk.Tk()
        root.title("Dynamic MRT Simulation Plot")
        fig = Figure(figsize=(10, 6))
        ax1 = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Cumulative trackers for dynamic
        cum_total_cost_dyn = 0
        cum_total_cost_stat = 0
        epochs_list = []
        cum_costs_dyn = []
        cum_costs_stat = []
        
        for epoch_num in range(num_epochs):
            epoch_start = epoch_num * epoch_size
            epoch_end = min(epoch_start + epoch_size, self.T)
            
            # Add arrivals in this epoch
            cum_arrivals = 0
            cum_arrivals_per_station = {i: 0 for i in range(self.n_stations)}
            for t in range(epoch_start, epoch_end):
                m = t
                if m in arrivals_data:
                    for origin_i, groups in arrivals_data[m].items():
                        epoch_arr = sum(num for _, num, _ in groups)
                        cum_arrivals += epoch_arr
                        cum_arrivals_per_station[origin_i] += epoch_arr
                        for dest_j, num, arr_min in groups:
                            queues[origin_i].append({'arrival_min': arr_min, 'dest_idx': dest_j, 'num': num})
            
            # Use GA to optimize next headway
            next_h = self.optimize_headway_ga(queues, horizon=ga_horizon, pop_size=ga_pop, generations=ga_gens)
            next_dispatch = last_dispatch + next_h
            if next_dispatch > epoch_end:
                next_dispatch = epoch_end
            if next_dispatch <= epoch_start:
                dynamic_departures.append(next_dispatch)
                last_dispatch = next_dispatch
                
                # Dispatch and simulate train
                cum_waiting_time = 0
                cum_boarded = 0
                cum_oper_cost = self.ct
                cum_travel_cost = self.cd * self.total_trip_time
                cum_trains = 1
                cum_dispatch_times = [next_dispatch]
                
                train = {'occ': 0, 'onboard': []}
                for pos in range(self.n_stations):
                    arr_time_at_pos = next_dispatch + pos * (self.d_i + self.dwell_time)
                    if arr_time_at_pos > self.T:
                        break
                    
                    # Alight
                    new_onboard = []
                    alighted = 0
                    for onboard_group in train['onboard']:
                        if onboard_group['dest_idx'] == pos:
                            alighted += onboard_group['num']
                        else:
                            new_onboard.append(onboard_group)
                    train['onboard'] = new_onboard
                    train['occ'] -= alighted
                    
                    # Board
                    if pos in queues and queues[pos]:
                        queues[pos].sort(key=lambda g: g['arrival_min'])
                        remaining_cap = self.C - train['occ']
                        while remaining_cap > 0 and queues[pos]:
                            group = queues[pos].pop(0)
                            board_num = min(group['num'], remaining_cap)
                            if board_num > 0:
                                boarded_group = {
                                    'origin_idx': pos, 'dest_idx': group['dest_idx'], 
                                    'num': board_num, 'arr_min': group['arrival_min']
                                }
                                wait_time = arr_time_at_pos - group['arrival_min']
                                cum_waiting_time += wait_time * board_num
                                cum_boarded += board_num
                                remaining_cap -= board_num
                                train['occ'] += board_num
                                train['onboard'].append(boarded_group)
                            if group['num'] > board_num:
                                queues[pos].insert(0, {'arrival_min': group['arrival_min'], 
                                                       'dest_idx': group['dest_idx'], 'num': group['num'] - board_num})
            else:
                cum_waiting_time = 0
                cum_boarded = 0
                cum_oper_cost = 0
                cum_travel_cost = 0
                cum_trains = 0
                cum_dispatch_times = []
            
            # Compute epoch results for dynamic
            epoch_waiting_cost = self.ww * cum_waiting_time
            epoch_operating_cost = self.wp * cum_oper_cost
            epoch_travelling_cost = cum_travel_cost
            epoch_total_cost = epoch_waiting_cost + epoch_operating_cost + epoch_travelling_cost
            
            dynamic_results[epoch_start] = {
                'waiting_cost': epoch_waiting_cost,
                'operating_cost': epoch_operating_cost,
                'travelling_cost': epoch_travelling_cost,
                'total_cost': epoch_total_cost,
                'passengers_boarded': int(cum_boarded),
                'total_arrivals': int(cum_arrivals),
                'avg_wait': cum_waiting_time / max(cum_boarded, 1) if cum_boarded > 0 else 0,
                'trains_dispatched': cum_trains,
                'dispatch_times': cum_dispatch_times,
                'arrivals_per_station': cum_arrivals_per_station.copy()
            }
            
            # Cumulative cost
            cum_total_cost_dyn += epoch_total_cost
            cum_total_cost_stat += static_results.get(epoch_start, {}).get('total_cost', 0)
            
            epochs_list.append(epoch_start)
            cum_costs_dyn.append(cum_total_cost_dyn)
            cum_costs_stat.append(cum_total_cost_stat)
            
            # Update dynamic plot
            ax1.clear()
            ax1.plot(epochs_list, cum_costs_dyn, label='Dynamic Cumulative Cost', color='blue')
            ax1.plot(epochs_list, cum_costs_stat, label='Static Cumulative Cost', color='red')
            ax1.set_xlabel('Epoch Start Minute')
            ax1.set_ylabel('Cumulative Total Cost (BDT)')
            ax1.legend()
            canvas.draw()
            root.update()
        
        root.destroy()
        return dynamic_results, static_results

# Example usage
if __name__ == "__main__":
    simulator = MRTSimulator(epoch_size=5)
    
    # Generate CSV
    simulator.generate_passenger_arrivals_csv('passenger_arrivals.csv')
    
    # Run dynamic simulation with GA
    dynamic_results, static_results = simulator.simulate_dynamic_with_ga('passenger_arrivals.csv')
    
    # Print and report for dynamic
    simulator.print_epoch_summary(dynamic_results)
    simulator.generate_report_file(dynamic_results, report_file='dynamic_simulation_report.md')
    
    # For static (optional)
    simulator.print_epoch_summary(static_results)
    simulator.generate_report_file(static_results, report_file='static_simulation_report.md')