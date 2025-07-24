import random
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
from datetime import datetime, timedelta
import csv
import os

# Bangladesh Metro Rail Station Names (MRT Line-6)
DHAKA_METRO_STATIONS = [
    "Uttara North", "Uttara Center", "Uttara South", "Pallabi", "Mirpur-11",
    "Mirpur-10", "Kazipara", "Shewrapara", "Agargaon", "Bijoy Sarani",
    "Farmgate", "Karwan Bazar", "Shahbagh", "Dhaka University", "Bangladesh Secretariat",
    "Motijheel", "Kamalapur"
]

# Extended stations for future lines
EXTENDED_STATIONS = DHAKA_METRO_STATIONS + [
    "Basabo", "Mugda Medical", "Rampura", "Badda", "Airport"
]


@dataclass
class BangladeshMetroParameters:
    """Bangladesh Metro Rail specific parameters"""
    # Network Configuration
    num_stops: int = 20  # MRT Line-6 actual stations
    station_names: List[str] = None
    stop_distances: List[float] = None  # km between stations

    # Operational Parameters (Based on Dhaka Metro specifications)
    dwell_time: float = 0.5  # 30 seconds average dwell time
    acc_dec_time: float = 0.33  # 20 seconds acceleration/deceleration
    speed: float = 35  # Average speed including stops (km/h)
    max_speed: float = 100  # Maximum design speed (km/h)

    # Cost parameters (BDT - Bangladeshi Taka)
    # Based on Dhaka economic conditions and metro fare structure
    c1: float = 8.0  # Passenger waiting cost (BDT/min)
    c2: float = 4.0  # Passenger on-board cost (BDT/min)
    c3: float = 140.0  # Vehicle operation cost (BDT/min)

    # Economic factors
    avg_income_per_hour: float = 120.0  # BDT/hour average income
    metro_fare_base: float = 20.0  # BDT base fare

    # Genetic Algorithm Parameters
    pop_size: int = 80  # Larger population for better solution
    generations: int = 300  # More generations for convergence
    crossover_rate: float = 0.85
    mutation_rate: float = 0.08

    # Dynamic simulation parameters
    threshold_percent_change: float = 0.20  # 20% threshold as requested
    change_detection_interval: int = 5  # minutes
    total_time: int = 14*60  # 14 hours operation (7 AM to 9 PM)

    # Headway constraints (Based on Dhaka Metro capacity)
    headway_min: float = 2.5  # Minimum headway during peak
    headway_max: float = 12.0  # Maximum headway during off-peak

    # Capacity constraints
    train_capacity: int = 2184  # Passengers per train (6-car configuration)
    max_occupancy_ratio: float = 0.85  # Maximum comfortable occupancy

    def __post_init__(self):
        if self.station_names is None:
            self.station_names = EXTENDED_STATIONS[:self.num_stops]

        if self.stop_distances is None:
            # Actual approximate distances between Dhaka Metro stations (km)
            self.stop_distances = [
                                      1.2, 1.0, 1.5, 1.3, 0.8, 1.1, 0.9, 1.4, 1.0, 1.2,
                                      0.8, 1.0, 0.9, 1.1, 1.3, 1.5, 1.0, 0.8, 1.2, 1.4
                                  ][:self.num_stops - 1]

        self.run_times = [d / self.speed * 60 for d in self.stop_distances]


class BangladeshMetroGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bangladesh Metro Rail Scheduling Optimization System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # Set window icon and styling
        self.setup_styling()

        # Initialize parameters
        self.params = BangladeshMetroParameters()
        self.metro_system = None
        self.optimizer = None
        self.simulation_running = False
        self.results_history = []

        self.setup_gui()
        self.load_bangladesh_presets()

    def setup_styling(self):
        """Setup Bangladesh-themed styling"""
        style = ttk.Style()
        style.theme_use('clam')

        # Bangladesh flag colors: Green and Red
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'),
                        foreground='#006A4E', background='#f0f0f0')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'),
                        foreground='#F42A41')
        style.configure('BD.TButton', background='#006A4E', foreground='white')

    def setup_gui(self):
        """Setup the GUI interface with Bangladesh Metro branding"""
        # Header frame with Bangladesh Metro branding
        header_frame = tk.Frame(self.root, bg='#006A4E', height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        title_label = tk.Label(header_frame,
                               text="🚇 বাংলাদেশ মেট্রো রেল সিস্টেম অপটিমাইজার",
                               font=('Arial', 16, 'bold'),
                               fg='white', bg='#006A4E')
        title_label.pack(pady=20)

        subtitle_label = tk.Label(header_frame,
                                  text="Bangladesh Metro Rail Scheduling Optimization System",
                                  font=('Arial', 12),
                                  fg='#FFD700', bg='#006A4E')
        subtitle_label.pack()

        # Main content area
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create main frames
        self.control_frame = ttk.Frame(main_frame)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_control_panel()
        self.setup_plot_area()

    def setup_control_panel(self):
        """Setup control panel with Bangladesh Metro specific parameters"""
        # Title
        title_label = ttk.Label(self.control_frame, text="System Parameters",
                                style='Title.TLabel')
        title_label.pack(pady=10)

        # Create notebook for different parameter categories
        self.notebook = ttk.Notebook(self.control_frame, width=400)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Metro System Tab
        self.metro_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.metro_frame, text="Metro System")
        self.setup_metro_params()

        # Economic Parameters Tab
        self.economic_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.economic_frame, text="Economic")
        self.setup_economic_params()

        # GA Parameters Tab
        self.ga_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ga_frame, text="GA Settings")
        self.setup_ga_params()

        # Simulation Tab
        self.sim_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sim_frame, text="Simulation")
        self.setup_simulation_params()

        # Control buttons
        self.setup_control_buttons()

    def setup_metro_params(self):
        """Setup metro system parameters"""
        # Scrollable frame
        canvas = tk.Canvas(self.metro_frame)
        scrollbar = ttk.Scrollbar(self.metro_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        params = [
            ("Number of Stations:", "num_stops", 20, 10, 25),
            ("Dwell Time (min):", "dwell_time", 0.5, 0.3, 2.0),
            ("Acc/Dec Time (min):", "acc_dec_time", 0.33, 0.2, 1.0),
            ("Average Speed (km/h):", "speed", 35, 25, 60),
            ("Maximum Speed (km/h):", "max_speed", 100, 80, 120),
            ("Train Capacity (passengers):", "train_capacity", 2184, 1500, 3000),
            ("Max Occupancy Ratio:", "max_occupancy_ratio", 0.85, 0.6, 0.95),
            ("Min Headway (min):", "headway_min", 2.5, 1.5, 5.0),
            ("Max Headway (min):", "headway_max", 12.0, 8.0, 20.0),
        ]

        self.metro_vars = {}
        for i, (label, var_name, default, min_val, max_val) in enumerate(params):
            ttk.Label(scrollable_frame, text=label, style='Header.TLabel').grid(
                row=i*2, column=0, sticky="w", padx=5, pady=3)

            var = tk.DoubleVar() if isinstance(default, float) else tk.IntVar()
            var.set(default)
            self.metro_vars[var_name] = var

            scale = ttk.Scale(scrollable_frame, from_=min_val, to=max_val,
                              variable=var, orient=tk.HORIZONTAL, length=180)
            scale.grid(row=i*2, column=1, padx=5, pady=3)

            value_label = ttk.Label(scrollable_frame, text=f"{var.get():.2f}" if isinstance(var.get(), float) else f"{var.get()}",
                                   font=('Arial', 10, 'bold'), foreground='#006A4E', anchor='center')
            value_label.grid(row=i*2+1, column=1, padx=5, pady=(0,8), sticky='n')
            def update_label(var=var, label=value_label):
                label.config(text=f"{var.get():.2f}" if isinstance(var.get(), float) else f"{var.get()}")
            var.trace_add('write', lambda *args, var=var, label=value_label: update_label(var, label))

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def setup_economic_params(self):
        """Setup economic parameters specific to Bangladesh"""
        params = [
            ("Waiting Cost (BDT/min):", "c1", 8.0, 3.0, 20.0),
            ("On-board Cost (BDT/min):", "c2", 4.0, 2.0, 15.0),
            ("Operation Cost (BDT/min):", "c3", 25.0, 10.0, 60.0),
            ("Average Income (BDT/hour):", "avg_income_per_hour", 120.0, 60.0, 300.0),
            ("Base Metro Fare (BDT):", "metro_fare_base", 20.0, 10.0, 50.0),
        ]

        self.economic_vars = {}
        for i, (label, var_name, default, min_val, max_val) in enumerate(params):
            ttk.Label(self.economic_frame, text=label, style='Header.TLabel').grid(
                row=i*2, column=0, sticky="w", padx=5, pady=3)

            var = tk.DoubleVar()
            var.set(default)
            self.economic_vars[var_name] = var

            scale = ttk.Scale(self.economic_frame, from_=min_val, to=max_val,
                              variable=var, orient=tk.HORIZONTAL, length=180)
            scale.grid(row=i*2, column=1, padx=5, pady=3)

            value_label = ttk.Label(self.economic_frame, text=f"{var.get():.2f}",
                                   font=('Arial', 10, 'bold'), foreground='#006A4E', anchor='center')
            value_label.grid(row=i*2+1, column=1, padx=5, pady=(0,8), sticky='n')
            def update_label(var=var, label=value_label):
                label.config(text=f"{var.get():.2f}")
            var.trace_add('write', lambda *args, var=var, label=value_label: update_label(var, label))

    def setup_ga_params(self):
        """Setup GA parameters"""
        params = [
            ("Population Size:", "pop_size", 80, 30, 200),
            ("Generations:", "generations", 300, 100, 500),
            ("Crossover Rate:", "crossover_rate", 0.85, 0.5, 1.0),
            ("Mutation Rate:", "mutation_rate", 0.08, 0.01, 0.2),
        ]

        self.ga_vars = {}
        for i, (label, var_name, default, min_val, max_val) in enumerate(params):
            ttk.Label(self.ga_frame, text=label, style='Header.TLabel').grid(
                row=i*2, column=0, sticky="w", padx=5, pady=3)

            var = tk.DoubleVar() if isinstance(default, float) else tk.IntVar()
            var.set(default)
            self.ga_vars[var_name] = var

            scale = ttk.Scale(self.ga_frame, from_=min_val, to=max_val,
                              variable=var, orient=tk.HORIZONTAL, length=180)
            scale.grid(row=i*2, column=1, padx=5, pady=3)

            value_label = ttk.Label(self.ga_frame, text=f"{var.get():.2f}" if isinstance(var.get(), float) else f"{var.get()}",
                                   font=('Arial', 10, 'bold'), foreground='#006A4E', anchor='center')
            value_label.grid(row=i*2+1, column=1, padx=5, pady=(0,8), sticky='n')
            def update_label(var=var, label=value_label):
                label.config(text=f"{var.get():.2f}" if isinstance(var.get(), float) else f"{var.get()}")
            var.trace_add('write', lambda *args, var=var, label=value_label: update_label(var, label))

    def setup_simulation_params(self):
        """Setup simulation parameters"""
        params = [
            ("Demand Change Threshold (%):", "threshold_percent_change", 0.20, 0.10, 0.50),
            ("Check Interval (min):", "change_detection_interval", 45, 2, 60),
            ("Total Operation Time (min):", "total_time", 1080, 480, 1440),
        ]

        self.sim_vars = {}
        for i, (label, var_name, default, min_val, max_val) in enumerate(params):
            ttk.Label(self.sim_frame, text=label, style='Header.TLabel').grid(
                row=i*2, column=0, sticky="w", padx=5, pady=3)

            var = tk.DoubleVar() if isinstance(default, float) else tk.IntVar()
            var.set(default)
            self.sim_vars[var_name] = var

            scale = ttk.Scale(self.sim_frame, from_=min_val, to=max_val,
                              variable=var, orient=tk.HORIZONTAL, length=180)
            scale.grid(row=i*2, column=1, padx=5, pady=3)

            value_label = ttk.Label(self.sim_frame, text=f"{var.get():.2f}" if isinstance(var.get(), float) else f"{var.get()}",
                                   font=('Arial', 10, 'bold'), foreground='#006A4E', anchor='center')
            value_label.grid(row=i*2+1, column=1, padx=5, pady=(0,8), sticky='n')
            def update_label(var=var, label=value_label):
                label.config(text=f"{var.get():.2f}" if isinstance(var.get(), float) else f"{var.get()}")
            var.trace_add('write', lambda *args, var=var, label=value_label: update_label(var, label))

    def setup_control_buttons(self):
        """Setup control buttons with Bangladesh styling"""
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # Preset buttons
        preset_frame = ttk.LabelFrame(button_frame, text="Preset Configurations")
        preset_frame.pack(fill=tk.X, pady=5)

        presets = [
            ("🌅 Morning Peak", "morning_peak"),
            ("🌆 Evening Peak", "evening_peak"),
            ("🌙 Off-Peak", "off_peak"),
            ("🚀 High Capacity", "high_capacity")
        ]

        for i, (preset_name, preset_key) in enumerate(presets):
            btn = tk.Button(preset_frame, text=preset_name,
                            command=lambda p=preset_key: self.load_preset(p),
                            bg='#006A4E', fg='white', font=('Arial', 9),
                            relief=tk.RAISED, bd=2)
            btn.grid(row=i // 2, column=i % 2, padx=2, pady=2, sticky='ew')

        preset_frame.grid_columnconfigure(0, weight=1)
        preset_frame.grid_columnconfigure(1, weight=1)

        # Control buttons
        control_frame = ttk.LabelFrame(button_frame, text="Simulation Controls")
        control_frame.pack(fill=tk.X, pady=5)

        self.start_btn = tk.Button(control_frame, text="🚀 Start Optimization",
                                   command=self.start_simulation,
                                   bg='#F42A41', fg='white', font=('Arial', 10, 'bold'),
                                   relief=tk.RAISED, bd=3)
        self.start_btn.pack(fill=tk.X, pady=2)

        self.stop_btn = tk.Button(control_frame, text="⏹️ Stop Simulation",
                                  command=self.stop_simulation, state=tk.DISABLED,
                                  bg='#666666', fg='white', font=('Arial', 10))
        self.stop_btn.pack(fill=tk.X, pady=2)

        save_btn = tk.Button(control_frame, text="💾 Save Results",
                             command=self.save_results,
                             bg='#006A4E', fg='white', font=('Arial', 10))
        save_btn.pack(fill=tk.X, pady=2)

        # Progress indicators
        progress_frame = ttk.LabelFrame(button_frame, text="Progress")
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_var = tk.StringVar()
        self.progress_var.set("Ready for optimization")
        progress_label = ttk.Label(progress_frame, textvariable=self.progress_var,
                                   font=('Arial', 9))
        progress_label.pack(pady=2)

        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=2)

        # Current time display
        self.time_var = tk.StringVar()
        self.time_var.set("System Time: Not started")
        time_label = ttk.Label(progress_frame, textvariable=self.time_var,
                               font=('Arial', 8))
        time_label.pack(pady=1)

    def setup_plot_area(self):
        """Setup the plotting area with Bangladesh Metro styling"""
        self.plot_notebook = ttk.Notebook(self.plot_frame)
        self.plot_notebook.pack(fill=tk.BOTH, expand=True)

        # Metro Line Schedule plot
        self.schedule_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.schedule_frame, text="🚇 Metro Schedule")

        self.schedule_fig = Figure(figsize=(12, 8), dpi=100)
        self.schedule_canvas = FigureCanvasTkAgg(self.schedule_fig, self.schedule_frame)
        self.schedule_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Optimization Convergence plot
        self.convergence_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.convergence_frame, text="📈 GA Convergence")

        self.convergence_fig = Figure(figsize=(12, 8), dpi=100)
        self.convergence_canvas = FigureCanvasTkAgg(self.convergence_fig, self.convergence_frame)
        self.convergence_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # System Performance plot
        self.performance_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.performance_frame, text="📊 System Performance")

        self.performance_fig = Figure(figsize=(12, 8), dpi=100)
        self.performance_canvas = FigureCanvasTkAgg(self.performance_fig, self.performance_frame)
        self.performance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Cost Analysis plot
        self.cost_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.cost_frame, text="💰 Cost Analysis")

        self.cost_fig = Figure(figsize=(12, 8), dpi=100)
        self.cost_canvas = FigureCanvasTkAgg(self.cost_fig, self.cost_frame)
        self.cost_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_bangladesh_presets(self):
        """Load Bangladesh Metro specific presets"""
        self.presets = {
            "morning_peak": {
                "name": "Morning Peak (7-10 AM)",
                "num_stops": 20, "dwell_time": 0.5, "acc_dec_time": 0.33, "speed": 32,
                "c1": 12.0, "c2": 6.0, "c3": 30.0, "headway_min": 2.5, "headway_max": 8.0,
                "pop_size": 100, "generations": 350, "crossover_rate": 0.9, "mutation_rate": 0.1,
                "threshold_percent_change": 0.15, "change_detection_interval": 3, "total_time": 180,
                "avg_income_per_hour": 150.0, "metro_fare_base": 25.0
            },
            "evening_peak": {
                "name": "Evening Peak (5-8 PM)",
                "num_stops": 20, "dwell_time": 0.6, "acc_dec_time": 0.4, "speed": 30,
                "c1": 15.0, "c2": 8.0, "c3": 35.0, "headway_min": 2.5, "headway_max": 7.0,
                "pop_size": 120, "generations": 400, "crossover_rate": 0.85, "mutation_rate": 0.12,
                "threshold_percent_change": 0.18, "change_detection_interval": 4, "total_time": 180,
                "avg_income_per_hour": 160.0, "metro_fare_base": 25.0
            },
            "off_peak": {
                "name": "Off-Peak Hours",
                "num_stops": 20, "dwell_time": 0.4, "acc_dec_time": 0.25, "speed": 40,
                "c1": 6.0, "c2": 3.0, "c3": 20.0, "headway_min": 5.0, "headway_max": 15.0,
                "pop_size": 60, "generations": 200, "crossover_rate": 0.8, "mutation_rate": 0.06,
                "threshold_percent_change": 0.25, "change_detection_interval": 8, "total_time": 480,
                "avg_income_per_hour": 100.0, "metro_fare_base": 20.0
            },
            "high_capacity": {
                "name": "High Capacity Operation",
                "num_stops": 25, "dwell_time": 0.7, "acc_dec_time": 0.5, "speed": 35,
                "c1": 10.0, "c2": 5.0, "c3": 40.0, "headway_min": 2.0, "headway_max": 12.0,
                "pop_size": 150, "generations": 500, "crossover_rate": 0.9, "mutation_rate": 0.15,
                "threshold_percent_change": 0.12, "change_detection_interval": 2, "total_time": 1080,
                "avg_income_per_hour": 180.0, "metro_fare_base": 30.0
            }
        }

    def load_preset(self, preset_key):
        """Load a specific preset configuration"""
        if preset_key not in self.presets:
            return

        preset = self.presets[preset_key]

        # Update metro parameters
        for var_name, value in preset.items():
            if var_name == "name":
                continue
            if var_name in self.metro_vars:
                self.metro_vars[var_name].set(value)
            elif var_name in self.economic_vars:
                self.economic_vars[var_name].set(value)
            elif var_name in self.ga_vars:
                self.ga_vars[var_name].set(value)
            elif var_name in self.sim_vars:
                self.sim_vars[var_name].set(value)

        messagebox.showinfo("Preset Loaded",
                            f"{preset['name']} configuration loaded successfully!\n\n"
                            f"Optimized for Bangladesh Metro Rail operations.")

    def update_parameters(self):
        """Update parameters from GUI values"""
        # Metro parameters
        self.params.num_stops = int(self.metro_vars["num_stops"].get())
        self.params.dwell_time = self.metro_vars["dwell_time"].get()
        self.params.acc_dec_time = self.metro_vars["acc_dec_time"].get()
        self.params.speed = self.metro_vars["speed"].get()
        self.params.max_speed = self.metro_vars["max_speed"].get()
        self.params.train_capacity = int(self.metro_vars["train_capacity"].get())
        self.params.max_occupancy_ratio = self.metro_vars["max_occupancy_ratio"].get()
        self.params.headway_min = self.metro_vars["headway_min"].get()
        self.params.headway_max = self.metro_vars["headway_max"].get()

        # Economic parameters
        self.params.c1 = self.economic_vars["c1"].get()
        self.params.c2 = self.economic_vars["c2"].get()
        self.params.c3 = self.economic_vars["c3"].get()
        self.params.avg_income_per_hour = self.economic_vars["avg_income_per_hour"].get()
        self.params.metro_fare_base = self.economic_vars["metro_fare_base"].get()

        # GA parameters
        self.params.pop_size = int(self.ga_vars["pop_size"].get())
        self.params.generations = int(self.ga_vars["generations"].get())
        self.params.crossover_rate = self.ga_vars["crossover_rate"].get()
        self.params.mutation_rate = self.ga_vars["mutation_rate"].get()

        # Simulation parameters
        self.params.threshold_percent_change = self.sim_vars["threshold_percent_change"].get()
        self.params.change_detection_interval = int(self.sim_vars["change_detection_interval"].get())
        self.params.total_time = int(self.sim_vars["total_time"].get())

        # Update derived parameters
        self.params.station_names = EXTENDED_STATIONS[:self.params.num_stops]
        self.params.stop_distances = [
                                         1.2, 1.0, 1.5, 1.3, 0.8, 1.1, 0.9, 1.4, 1.0, 1.2,
                                         0.8, 1.0, 0.9, 1.1, 1.3, 1.5, 1.0, 0.8, 1.2, 1.4,
                                         1.1, 0.9, 1.3, 1.2, 1.0
                                     ][:self.params.num_stops - 1]
        self.params.run_times = [d / self.params.speed * 60 for d in self.params.stop_distances]

    def start_simulation(self):
        """Start the optimization simulation"""
        self.update_parameters()
        self.simulation_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_bar.start()

        # Start simulation in separate thread
        thread = threading.Thread(target=self.run_simulation)
        thread.daemon = True
        thread.start()

    def stop_simulation(self):
        """Stop the simulation"""
        self.simulation_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_bar.stop()
        self.progress_var.set("Simulation stopped by user")

    def run_simulation(self):
        """Run the Bangladesh Metro optimization simulation"""
        try:
            # Initialize system
            self.metro_system = BangladeshMetroSystem(self.params)
            self.optimizer = BangladeshMetroOptimizer(self.metro_system, self.params)
            self.results_history = []

            current_time = 0  # Start at 0 minutes from simulation start (7 AM)
            base_time = 420  # 7 AM in minutes from midnight
            current_rates = self.metro_system.generate_dhaka_demand_pattern(current_time + base_time)
            last_optimization_time = 0

            # Initial optimization
            self.progress_var.set(f"🚀 Initial optimization - {self.format_time(current_time + base_time)}")
            self.time_var.set(f"System Time: {self.format_time(current_time + base_time)}")

            # --- Use CSV headway if available ---
            csv_headway = self.metro_system.get_csv_headway(current_time + base_time, direction=0)
            if csv_headway:
                headway = csv_headway
                interval_minutes = self.params.change_detection_interval
                num_trains = int(np.ceil(interval_minutes / headway))
                schedule_for_plot = ['01'] * num_trains
                zone_pattern, express_pattern = self.optimizer.calculate_bangladesh_patterns(current_rates)
                od_matrix = self.metro_system.generate_dhaka_od_matrix(current_rates)
                # Calculate cost using the same formulas as the optimizer
                chromosome = BangladeshMetroChromosome(self.params, headway, schedule_for_plot)
                waiting_cost = self.optimizer.calculate_waiting_cost(chromosome, od_matrix)
                onboard_cost = self.optimizer.calculate_onboard_cost(chromosome, od_matrix, zone_pattern, express_pattern)
                operation_cost = self.optimizer.calculate_operation_cost(chromosome, zone_pattern, express_pattern)
                # Add penalty for overcrowding
                total_capacity = num_trains * self.params.train_capacity * self.params.max_occupancy_ratio
                total_demand = sum(current_rates) * interval_minutes
                overcrowd_penalty = 0
                if total_demand > total_capacity:
                    overcrowd_penalty = self.params.c1 * (total_demand - total_capacity) * 2  # Heavy penalty
                # Add penalty for excessive headway
                headway_penalty = 0
                if headway > 15:
                    headway_penalty = self.params.c1 * sum(current_rates) * (headway - 15) * 2
                best_cost = waiting_cost + onboard_cost + operation_cost + overcrowd_penalty + headway_penalty
                history = []
                result = {
                    'time': current_time,
                    'real_time': current_time + base_time,
                    'time_str': self.format_time(current_time + base_time),
                    'headway': headway,
                    'cost': best_cost,
                    'schedule': schedule_for_plot,
                    'convergence': history,
                    'zone_pattern': zone_pattern,
                    'express_pattern': express_pattern,
                    'total_trains': num_trains,
                    'cost_breakdown': self.calculate_cost_breakdown(best_cost)
                }
            else:
                zone_pattern, express_pattern = self.optimizer.calculate_bangladesh_patterns(current_rates)
                od_matrix = self.metro_system.generate_dhaka_od_matrix(current_rates)

                best_solution, best_cost, history = self.optimizer.run_genetic_algorithm(
                    od_matrix, zone_pattern, express_pattern
                )

                interval_minutes = self.params.change_detection_interval
                num_trains = int(np.ceil(interval_minutes / best_solution.headway))
                if best_solution.schedule:
                    schedule_for_plot = [best_solution.schedule[i % len(best_solution.schedule)] for i in range(num_trains)]
                else:
                    schedule_for_plot = ['01'] * num_trains
                result = {
                    'time': current_time,
                    'real_time': current_time + base_time,
                    'time_str': self.format_time(current_time + base_time),
                    'headway': best_solution.headway,
                    'cost': best_cost,
                    'schedule': schedule_for_plot,
                    'convergence': history,
                    'zone_pattern': zone_pattern,
                    'express_pattern': express_pattern,
                    'total_trains': num_trains,
                    'cost_breakdown': self.calculate_cost_breakdown(best_cost)
                }
            self.results_history.append(result)

            # Update plots
            self.root.after(0, lambda: self.update_all_plots(result))

            current_time += self.params.change_detection_interval

            # Dynamic simulation loop
            while current_time <= self.params.total_time and self.simulation_running:
                real_time = current_time + base_time
                new_rates = self.metro_system.generate_dhaka_demand_pattern(real_time)

                # Update time display
                self.root.after(0, lambda t=real_time: self.time_var.set(f"System Time: {self.format_time(t)}"))

                # Check 20% threshold rule
                time_since_last_opt = current_time - last_optimization_time
                significant_change = self.detect_demand_change(current_rates, new_rates,
                                                               self.params.threshold_percent_change)

                # --- Use CSV headway if available ---
                csv_headway = self.metro_system.get_csv_headway(real_time, direction=0)
                if csv_headway:
                    headway = csv_headway
                    interval_minutes = current_time - last_optimization_time if last_optimization_time != 0 else self.params.change_detection_interval
                    num_trains = int(np.ceil(interval_minutes / headway))
                    schedule_for_plot = ['01'] * num_trains
                    zone_pattern, express_pattern = self.optimizer.calculate_bangladesh_patterns(new_rates)
                    od_matrix = self.metro_system.generate_dhaka_od_matrix(new_rates)
                    # Calculate cost using the same formulas as the optimizer
                    chromosome = BangladeshMetroChromosome(self.params, headway, schedule_for_plot)
                    waiting_cost = self.optimizer.calculate_waiting_cost(chromosome, od_matrix)
                    onboard_cost = self.optimizer.calculate_onboard_cost(chromosome, od_matrix, zone_pattern, express_pattern)
                    operation_cost = self.optimizer.calculate_operation_cost(chromosome, zone_pattern, express_pattern)
                    # Add penalty for overcrowding
                    total_capacity = num_trains * self.params.train_capacity * self.params.max_occupancy_ratio
                    total_demand = sum(new_rates) * interval_minutes
                    overcrowd_penalty = 0
                    if total_demand > total_capacity:
                        overcrowd_penalty = self.params.c1 * (total_demand - total_capacity) * 2  # Heavy penalty
                    # Add penalty for excessive headway
                    headway_penalty = 0
                    if headway > 15:
                        headway_penalty = self.params.c1 * sum(new_rates) * (headway - 15) * 2
                    best_cost = waiting_cost + onboard_cost + operation_cost + overcrowd_penalty + headway_penalty
                    history = []
                    result = {
                        'time': current_time,
                        'real_time': real_time,
                        'time_str': self.format_time(real_time),
                        'headway': headway,
                        'cost': best_cost,
                        'schedule': schedule_for_plot,
                        'convergence': history,
                        'zone_pattern': zone_pattern,
                        'express_pattern': express_pattern,
                        'total_trains': num_trains,
                        'cost_breakdown': self.calculate_cost_breakdown(best_cost)
                    }
                    self.results_history.append(result)
                    self.root.after(0, lambda r=result: self.update_all_plots(r))
                    last_optimization_time = current_time
                elif significant_change and time_since_last_opt >= self.params.change_detection_interval:
                    self.progress_var.set(f"🔄 Re-optimizing - {self.format_time(real_time)} (Demand change > 20%)")

                    current_rates = new_rates
                    zone_pattern, express_pattern = self.optimizer.calculate_bangladesh_patterns(current_rates)
                    od_matrix = self.metro_system.generate_dhaka_od_matrix(current_rates)

                    best_solution, best_cost, history = self.optimizer.run_genetic_algorithm(
                        od_matrix, zone_pattern, express_pattern
                    )

                    interval_minutes = current_time - last_optimization_time if last_optimization_time != 0 else self.params.change_detection_interval
                    num_trains = int(np.ceil(interval_minutes / best_solution.headway))
                    if best_solution.schedule:
                        schedule_for_plot = [best_solution.schedule[i % len(best_solution.schedule)] for i in range(num_trains)]
                    else:
                        schedule_for_plot = ['01'] * num_trains
                    result = {
                        'time': current_time,
                        'real_time': real_time,
                        'time_str': self.format_time(real_time),
                        'headway': best_solution.headway,
                        'cost': best_cost,
                        'schedule': schedule_for_plot,
                        'convergence': history,
                        'zone_pattern': zone_pattern,
                        'express_pattern': express_pattern,
                        'total_trains': num_trains,
                        'cost_breakdown': self.calculate_cost_breakdown(best_cost)
                    }
                    self.results_history.append(result)
                    self.root.after(0, lambda r=result: self.update_all_plots(r))
                    last_optimization_time = current_time
                else:
                    self.progress_var.set(f"✅ No significant change - {self.format_time(real_time)}")

                current_time += self.params.change_detection_interval

            if self.simulation_running:
                self.progress_var.set("🎉 Bangladesh Metro optimization completed successfully!")
                self.root.after(0, self.update_performance_analysis)

        except Exception as e:
            self.progress_var.set(f"❌ Error: {str(e)}")
            messagebox.showerror("Simulation Error", f"An error occurred: {str(e)}")
        finally:
            self.root.after(0, lambda: self.stop_simulation())

    def format_time(self, minutes_from_midnight):
        """Format time from minutes to HH:MM AM/PM"""
        hours = (minutes_from_midnight // 60) % 24
        mins = minutes_from_midnight % 60
        period = "AM" if hours < 12 else "PM"
        display_hour = hours if hours <= 12 else hours - 12
        if display_hour == 0:
            display_hour = 12
        return f"{display_hour:02d}:{mins:02d} {period}"

    def calculate_cost_breakdown(self, total_cost):
        """Calculate cost breakdown for analysis"""
        # Approximate breakdown based on typical BRT systems
        return {
            'waiting_cost': total_cost * 0.35,
            'onboard_cost': total_cost * 0.40,
            'operation_cost': total_cost * 0.25
        }

    def detect_demand_change(self, prev_rates, curr_rates, threshold):
        """Detect significant changes in demand (20% threshold rule)"""
        total_change = 0
        valid_comparisons = 0

        for prev, curr in zip(prev_rates, curr_rates):
            if prev > 0.001:  # Avoid division by very small numbers
                change_ratio = abs(curr - prev) / prev
                total_change += change_ratio
                valid_comparisons += 1

        if valid_comparisons == 0:
            return False

        average_change = total_change / valid_comparisons
        return average_change > threshold

    def update_all_plots(self, result):
        """Update all plots with new results"""
        self.plot_metro_schedule(result)
        self.plot_convergence(result['convergence'])
        if len(self.results_history) > 1:
            self.update_performance_analysis()

    def plot_metro_schedule(self, result):
        """Plot the optimized Bangladesh Metro schedule"""
        self.schedule_fig.clear()
        ax = self.schedule_fig.add_subplot(111)

        if not result['schedule']:
            ax.text(0.5, 0.5, "No schedule to display", ha='center', va='center', transform=ax.transAxes)
            self.schedule_canvas.draw()
            return

        # Bangladesh flag colors
        colors = ['#006A4E', '#F42A41', '#FFD700'] * (len(result['schedule']) // 3 + 1)
        schedule_types = {'01': 'All Stations', '10': 'Zone Service', '11': 'Express Service'}

        station_names = self.params.station_names[:self.params.num_stops]

        for i, sched_type in enumerate(result['schedule']):
            if sched_type == '01':  # All stations
                pattern = [1] * self.params.num_stops
            elif sched_type == '10':  # Zone
                pattern = result['zone_pattern']
            else:  # Express
                pattern = result['express_pattern']

            stops = [idx for idx, stop in enumerate(pattern) if stop == 1]

            if stops:
                y_val = i * 0.5  # Reduce vertical space between trains
                ax.plot(stops, [y_val] * len(stops), 'o-', linewidth=4,
                        markersize=10, color=colors[i % len(colors)],
                        label=f'Train {i + 1} ({schedule_types.get(sched_type, "Unknown")})',
                        alpha=0.8)

        yticks = [i * 0.5 for i in range(len(result['schedule']))]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"Train {i + 1}" for i in range(len(result['schedule']))])
        ax.set_xticks(range(len(station_names)))
        ax.set_xticklabels([name.replace(' ', '\n') for name in station_names],
                           rotation=45, ha='right', fontsize=8)
        ax.set_xlabel("Metro Stations", fontsize=12, fontweight='bold')
        ax.set_ylabel("Train Services", fontsize=12, fontweight='bold')
        ax.set_title(f"Bangladesh Metro Rail Schedule - {result['time_str']}\n"
                     f"Headway: {result['headway']:.1f} min | Total Trains: {result['total_trains']} | "
                     f"Total Cost: {result['cost']:.0f} BDT/hour",
                     fontsize=14, fontweight='bold', color='#006A4E')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

        # Add Bangladesh flag colors to background
        ax.set_facecolor('#f8f8f8')

        self.schedule_fig.tight_layout()
        self.schedule_canvas.draw()

    def plot_convergence(self, history):
        """Plot GA convergence with Bangladesh styling"""
        self.convergence_fig.clear()
        ax = self.convergence_fig.add_subplot(111)

        ax.plot(history, color='#006A4E', linewidth=3, alpha=0.8)
        ax.fill_between(range(len(history)), history, alpha=0.3, color='#006A4E')

        ax.set_xlabel("Generation", fontsize=12, fontweight='bold')
        ax.set_ylabel("Total System Cost (BDT/hour)", fontsize=12, fontweight='bold')
        ax.set_title("Genetic Algorithm Optimization Convergence\nBangladesh Metro Rail System",
                     fontsize=14, fontweight='bold', color='#006A4E')
        ax.grid(True, alpha=0.4)
        ax.set_facecolor('#f8f8f8')

        # Add improvement annotation
        if len(history) > 1:
            improvement = ((history[0] - history[-1]) / history[0]) * 100
            ax.annotate(f'Improvement: {improvement:.1f}%',
                        xy=(len(history) - 1, history[-1]),
                        xytext=(len(history) * 0.7, history[0] * 0.9),
                        arrowprops=dict(arrowstyle='->', color='#F42A41', lw=2),
                        fontsize=10, fontweight='bold', color='#F42A41')

        self.convergence_fig.tight_layout()
        self.convergence_canvas.draw()

    def update_performance_analysis(self):
        """Update comprehensive performance analysis"""
        if len(self.results_history) < 2:
            return

        self.performance_fig.clear()

        # Create subplots
        gs = self.performance_fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = self.performance_fig.add_subplot(gs[0, 0])
        ax2 = self.performance_fig.add_subplot(gs[0, 1])
        ax3 = self.performance_fig.add_subplot(gs[1, 0])
        ax4 = self.performance_fig.add_subplot(gs[1, 1])

        times = [r['real_time'] for r in self.results_history]
        time_labels = [self.format_time(t) for t in times]
        headways = [r['headway'] for r in self.results_history]
        costs = [r['cost'] for r in self.results_history]
        train_counts = [r['total_trains'] for r in self.results_history]

        # Headway evolution
        ax1.plot(times, headways, 'o-', color='#006A4E', linewidth=3, markersize=8)
        ax1.set_xlabel("Time of Day")
        ax1.set_ylabel("Headway (minutes)")
        ax1.set_title("Service Frequency Evolution", fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(times[::max(1, len(times) // 5)])
        ax1.set_xticklabels([self.format_time(t) for t in times[::max(1, len(times) // 5)]], rotation=45)

        # Cost evolution
        ax2.plot(times, costs, 'o-', color='#F42A41', linewidth=3, markersize=8)
        ax2.set_xlabel("Time of Day")
        ax2.set_ylabel("Total Cost (BDT/hour)")
        ax2.set_title("System Cost Evolution", fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(times[::max(1, len(times) // 5)])
        ax2.set_xticklabels([self.format_time(t) for t in times[::max(1, len(times) // 5)]], rotation=45)

        # Fleet size
        ax3.bar(range(len(train_counts)), train_counts, color='#FFD700', alpha=0.8, edgecolor='#006A4E')
        ax3.set_xlabel("Optimization Round")
        ax3.set_ylabel("Number of Trains")
        ax3.set_title("Fleet Size Requirements", fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # Cost distribution
        ax4.hist(costs, bins=max(3, len(costs) // 3), color='#006A4E', alpha=0.7, edgecolor='black')
        ax4.set_xlabel("Total Cost (BDT/hour)")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Cost Distribution", fontweight='bold')
        ax4.grid(True, alpha=0.3)

        self.performance_fig.suptitle("Bangladesh Metro Rail - System Performance Analysis",
                                      fontsize=16, fontweight='bold', color='#006A4E')

        self.performance_canvas.draw()

        # Update cost analysis
        self.update_cost_analysis()

    def update_cost_analysis(self):
        """Update detailed cost analysis"""
        if not self.results_history:
            return

        self.cost_fig.clear()

        # Create cost breakdown analysis
        gs = self.cost_fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = self.cost_fig.add_subplot(gs[0, 0])
        ax2 = self.cost_fig.add_subplot(gs[0, 1])
        ax3 = self.cost_fig.add_subplot(gs[1, :])

        # Latest cost breakdown pie chart
        latest_result = self.results_history[-1]
        breakdown = latest_result['cost_breakdown']

        labels = ['Passenger Waiting', 'On-board Travel', 'Vehicle Operation']
        sizes = [breakdown['waiting_cost'], breakdown['onboard_cost'], breakdown['operation_cost']]
        colors = ['#F42A41', '#FFD700', '#006A4E']

        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f"Cost Breakdown - {latest_result['time_str']}", fontweight='bold')

        # Cost savings over time
        if len(self.results_history) > 1:
            costs = [r['cost'] for r in self.results_history]
            savings = [(costs[0] - cost) / costs[0] * 100 for cost in costs]
            times = [r['real_time'] for r in self.results_history]

            ax2.plot(times, savings, 'o-', color='#006A4E', linewidth=3, markersize=6)
            ax2.set_xlabel("Time of Day")
            ax2.set_ylabel("Cost Savings (%)")
            ax2.set_title("Cumulative Cost Optimization", fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(times[::max(1, len(times) // 4)])
            ax2.set_xticklabels([self.format_time(t) for t in times[::max(1, len(times) // 4)]], rotation=45)

        # Detailed cost evolution
        waiting_costs = [r['cost_breakdown']['waiting_cost'] for r in self.results_history]
        onboard_costs = [r['cost_breakdown']['onboard_cost'] for r in self.results_history]
        operation_costs = [r['cost_breakdown']['operation_cost'] for r in self.results_history]
        times = range(len(self.results_history))

        ax3.plot(times, waiting_costs, 'o-', label='Waiting Cost', color='#F42A41', linewidth=2)
        ax3.plot(times, onboard_costs, 's-', label='On-board Cost', color='#FFD700', linewidth=2)
        ax3.plot(times, operation_costs, '^-', label='Operation Cost', color='#006A4E', linewidth=2)

        ax3.set_xlabel("Optimization Round")
        ax3.set_ylabel("Cost (BDT/hour)")
        ax3.set_title("Detailed Cost Component Evolution", fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        self.cost_fig.suptitle("Bangladesh Metro Rail - Cost Analysis",
                               fontsize=16, fontweight='bold', color='#006A4E')

        self.cost_canvas.draw()

    def save_results(self):
        """Save simulation results with Bangladesh Metro formatting"""
        if not self.results_history:
            messagebox.showwarning("No Results", "No optimization results to save!")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bangladesh_metro_optimization_{timestamp}.json"

            # Prepare comprehensive results
            export_data = {
                'system_info': {
                    'title': 'Bangladesh Metro Rail Scheduling Optimization',
                    'generation_time': datetime.now().isoformat(),
                    'total_optimizations': len(self.results_history),
                    'system_parameters': {
                        'num_stops': self.params.num_stops,
                        'station_names': self.params.station_names,
                        'operation_hours': self.params.total_time / 60,
                        'threshold_change': self.params.threshold_percent_change
                    }
                },
                'optimization_results': self.results_history,
                'summary_statistics': {
                    'average_headway': np.mean([r['headway'] for r in self.results_history]),
                    'average_cost': np.mean([r['cost'] for r in self.results_history]),
                    'total_cost_savings': (self.results_history[0]['cost'] -
                                           self.results_history[-1]['cost']) if len(self.results_history) > 1 else 0,
                    'average_fleet_size': np.mean([r['total_trains'] for r in self.results_history])
                }
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            messagebox.showinfo("Results Saved",
                                f"Bangladesh Metro optimization results saved successfully!\n\n"
                                f"File: {filename}\n"
                                f"Total optimizations: {len(self.results_history)}\n"
                                f"Data includes: System parameters, optimization results, "
                                f"cost analysis, and performance statistics.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving results: {str(e)}")


# Bangladesh Metro System Classes
class BangladeshMetroSystem:
    """Bangladesh Metro Rail system with Dhaka-specific demand patterns or CSV input"""

    def __init__(self, params: BangladeshMetroParameters):
        self.params = params
        self.num_stops = params.num_stops
        self.run_times = params.run_times
        self.dwell_time = params.dwell_time
        self.acc_dec_time = params.acc_dec_time
        self.station_names = params.station_names
        self.csv_demand_data = None
        self.csv_station_order = None
        self.load_csv_demand()

    def load_csv_demand(self):
        """Load demand data from metro_weighted_station_passengers.csv if available"""
        csv_path = 'metro_weighted_station_passengers.csv'
        if not os.path.exists(csv_path):
            return
        self.csv_demand_data = []
        self.csv_station_order = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse time to minutes from midnight
                start_h, start_m = map(int, row['Start Time'].split(':'))
                end_h, end_m = map(int, row['End Time'].split(':'))
                start_min = start_h * 60 + start_m
                end_min = end_h * 60 + end_m
                # Store station order if not already
                if row['Station'] not in self.csv_station_order:
                    self.csv_station_order.append(row['Station'])
                self.csv_demand_data.append({
                    'direction': int(row['Direction']),
                    'start_min': start_min,
                    'end_min': end_min,
                    'station': row['Station'],
                    'ppm': float(row['Passengers per Minute']),
                    'headway': float(row.get('Headway', 0)) # Add headway to CSV data
                })

    def get_csv_demand(self, time_minutes, direction=0):
        """Return a list of demand (ppm) for all stations at the given time and direction from CSV."""
        if not self.csv_demand_data:
            return None
        # Find all intervals that include this time and direction
        station_ppm = {s:0.0 for s in self.csv_station_order}
        for entry in self.csv_demand_data:
            if entry['direction'] == direction and entry['start_min'] <= time_minutes <= entry['end_min']:
                station_ppm[entry['station']] = entry['ppm']
        # If no match, fallback to closest previous interval for each station
        for s in self.csv_station_order:
            if station_ppm[s] == 0.0:
                # Find the latest interval before time_minutes
                best = None
                for entry in self.csv_demand_data:
                    if entry['direction'] == direction and entry['station'] == s and entry['start_min'] <= time_minutes:
                        if best is None or entry['start_min'] > best['start_min']:
                            best = entry
                if best:
                    station_ppm[s] = best['ppm']
        # Return in the order of self.station_names
        return [station_ppm.get(name, 0.0) for name in self.station_names]

    def get_csv_headway(self, time_minutes, direction=0):
        """Return the headway (in minutes) for the current interval and direction from CSV, or None if not found."""
        if not self.csv_demand_data:
            return None
        headways = []
        for entry in self.csv_demand_data:
            if entry['direction'] == direction and entry['start_min'] <= time_minutes <= entry['end_min']:
                headways.append(entry.get('headway', None) or entry.get('Headway', None) or None)
        # Use the most common headway if multiple, or the first
        headways = [float(h) for h in headways if h is not None]
        if headways:
            # Use the minimum headway (safest for service)
            return min(headways)
        return None

    def generate_dhaka_demand_pattern(self, time_minutes):
        """Generate passenger demand pattern from CSV if available, else fallback to default."""
        csv_rates = self.get_csv_demand(time_minutes, direction=0)
        if csv_rates and any(csv_rates):
            return csv_rates
        # fallback to original method
        base_rate = 150  # Higher base rate for Dhaka

        # Dhaka-specific peak patterns
        hour = (time_minutes // 60) % 24

        if 7 <= hour <= 10:  # Morning peak
            peak_multiplier = 4.0 + 1.0 * np.sin((hour - 7) / 3 * np.pi)
        elif 17 <= hour <= 20:  # Evening peak
            peak_multiplier = 3.5 + 0.8 * np.sin((hour - 17) / 3 * np.pi)
        elif 12 <= hour <= 14:  # Lunch time
            peak_multiplier = 2.0
        elif 22 <= hour <= 24 or 0 <= hour <= 5:  # Night time
            peak_multiplier = 0.3
        else:  # Regular hours
            peak_multiplier = 1.2

        arrival_rates = []
        for j in range(self.num_stops):
            station_name = self.station_names[j] if j < len(self.station_names) else f"Station_{j + 1}"

            # Station-specific factors based on Dhaka geography
            if station_name in ["Motijheel", "Farmgate", "Karwan Bazar", "Shahbagh"]:
                station_factor = 3.0  # Major business districts
            elif station_name in ["Uttara North", "Uttara Center", "Kamalapur"]:
                station_factor = 2.5  # Major terminals/residential
            elif station_name in ["Dhaka University", "Bangladesh Secretariat"]:
                station_factor = 2.2  # Educational/government
            elif "Mirpur" in station_name:
                station_factor = 2.0  # Residential areas
            else:
                station_factor = 1.0

            # Add day-of-week factor (assume weekday)
            weekday_factor = 1.0

            # Calculate final rate with randomness
            rate = max(30, int(np.random.normal(
                loc=base_rate * peak_multiplier * station_factor * weekday_factor,
                scale=base_rate * 0.3
            )))

            arrival_rates.append(rate / 60.0)  # Convert to passengers/min

        return arrival_rates

    def generate_dhaka_od_matrix(self, arrival_rates):
        """Generate OD matrix with Dhaka-specific travel patterns"""
        od_matrix = np.zeros((self.num_stops, self.num_stops))

        for j in range(self.num_stops):
            for k in range(self.num_stops):
                if j != k:
                    # Distance decay
                    distance = abs(j - k)
                    distance_factor = np.exp(-0.08 * distance)

                    # Dhaka-specific directional bias
                    direction_bias = 1.0

                    # Business district attractions
                    business_districts = [10, 11, 12]  # Farmgate, Karwan Bazar, Shahbagh indices
                    if k in business_districts:
                        direction_bias *= 1.8
                    if j in business_districts:
                        direction_bias *= 0.7  # Less outbound from business districts

                    # Residential to business flow
                    residential_areas = [0, 1, 2, 4, 5]  # Uttara, Mirpur areas
                    if j in residential_areas and k in business_districts:
                        direction_bias *= 1.5

                    od_matrix[j][k] = arrival_rates[k] * distance_factor * direction_bias * 0.3

        return od_matrix


class BangladeshMetroChromosome:
    """Chromosome class for Bangladesh Metro optimization"""

    def __init__(self, params: BangladeshMetroParameters, headway=None, schedule=None):
        self.params = params
        if headway is None:
            self.headway = random.uniform(params.headway_min, params.headway_max)
        else:
            self.headway = max(params.headway_min, min(params.headway_max, headway))

        if schedule is None:
            # Enforce minimum 6 trains per hour
            trains_per_hour = max(6, int(60 / self.headway))
            # Ensure a balanced mix of schedule types
            base_types = ['01', '10', '11']
            schedule_types = (base_types * ((trains_per_hour + 2) // 3))[:trains_per_hour]
            random.shuffle(schedule_types)
            self.schedule = schedule_types
        else:
            self.schedule = schedule

        self.ls = len(self.schedule)


class BangladeshMetroOptimizer:
    """Optimizer specifically designed for Bangladesh Metro Rail"""

    def __init__(self, metro_system: BangladeshMetroSystem, params: BangladeshMetroParameters):
        self.metro = metro_system
        self.params = params

    def calculate_bangladesh_patterns(self, arrival_rates):
        """Calculate service patterns optimized for Bangladesh Metro"""
        # Zone pattern: stations with above 70th percentile demand
        zone_threshold = np.percentile(arrival_rates, 70)
        zone_pattern = [1 if rate >= zone_threshold else 0 for rate in arrival_rates]

        # Express pattern: stations with above 85th percentile demand
        express_threshold = np.percentile(arrival_rates, 85)
        express_pattern = [1 if rate >= express_threshold else 0 for rate in arrival_rates]

        # Ensure major stations are always included
        major_stations = [0, self.metro.num_stops - 1]  # Terminals
        for idx in major_stations:
            if idx < len(zone_pattern):
                zone_pattern[idx] = 1
                express_pattern[idx] = 1

        # Ensure minimum connectivity
        if sum(zone_pattern) < 4:
            sorted_indices = sorted(range(len(arrival_rates)),
                                    key=lambda i: arrival_rates[i], reverse=True)
            for i in sorted_indices[:4]:
                zone_pattern[i] = 1

        if sum(express_pattern) < 3:
            sorted_indices = sorted(range(len(arrival_rates)),
                                    key=lambda i: arrival_rates[i], reverse=True)
            for i in sorted_indices[:3]:
                express_pattern[i] = 1

        return zone_pattern, express_pattern

    def get_stop_pattern(self, schedule_type, zone_pattern, express_pattern):
        """Get stop pattern for vehicle with given schedule type"""
        if schedule_type == '01':  # All stations
            return [1] * self.metro.num_stops
        elif schedule_type == '10':  # Zone service
            return zone_pattern
        elif schedule_type == '11':  # Express service
            return express_pattern
        return [1] * self.metro.num_stops

    def evaluate_objective_function(self, chromosome, od_matrix, zone_pattern, express_pattern):
        """Evaluate objective function with Bangladesh-specific considerations"""
        if not chromosome.schedule or chromosome.headway <= 0:
            return float('inf')

        f1 = self.calculate_waiting_cost(chromosome, od_matrix)
        f2 = self.calculate_onboard_cost(chromosome, od_matrix, zone_pattern, express_pattern)
        f3 = self.calculate_operation_cost(chromosome, zone_pattern, express_pattern)

        # Add capacity constraint penalty
        capacity_penalty = self.calculate_capacity_penalty(chromosome, od_matrix, zone_pattern, express_pattern)

        # Add penalty if not all stations are covered by the combination of all trains
        all_covered = [0] * self.metro.num_stops
        for schedule_type in chromosome.schedule:
            pattern = self.get_stop_pattern(schedule_type, zone_pattern, express_pattern)
            for i, stop in enumerate(pattern):
                if stop:
                    all_covered[i] = 1
        if not all(all_covered):
            # Large penalty if any station is not covered
            uncovered = all_covered.count(0)
            coverage_penalty = 1e6 * uncovered
        else:
            coverage_penalty = 0

        return f1 + f2 + f3 + capacity_penalty + coverage_penalty

    def calculate_capacity_penalty(self, chromosome, od_matrix, zone_pattern, express_pattern):
        """Calculate penalty for exceeding train capacity"""
        penalty = 0.0
        max_capacity = self.params.train_capacity * self.params.max_occupancy_ratio

        for schedule_type in chromosome.schedule:
            pattern = self.get_stop_pattern(schedule_type, zone_pattern, express_pattern)

            # Estimate maximum load
            max_load = 0
            for j in range(self.metro.num_stops - 1):
                if pattern[j] == 1:
                    load = sum(od_matrix[k][j] for k in range(j + 1, self.metro.num_stops)
                               if k < len(pattern) and pattern[k] == 1)
                    max_load = max(max_load, load * chromosome.headway)

            if max_load > max_capacity:
                # Exponential penalty for overcrowding
                overcrowd_ratio = max_load / max_capacity
                penalty += self.params.c1 * 100 * (overcrowd_ratio - 1) ** 2

        return penalty

    def calculate_waiting_cost(self, chromosome, od_matrix):
        """Calculate passenger waiting cost"""
        f1 = 0.0
        for j in range(self.metro.num_stops):
            rj = sum(od_matrix[k][j] for k in range(self.metro.num_stops) if k != j)
            # Average waiting time is half the headway
            f1 += self.params.c1 * rj * (chromosome.headway / 2.0)
        return f1

    def calculate_onboard_cost(self, chromosome, od_matrix, zone_pattern, express_pattern):
        """Calculate passenger on-board cost"""
        f2 = 0.0
        for schedule_type in chromosome.schedule:
            pattern = self.get_stop_pattern(schedule_type, zone_pattern, express_pattern)

            for j in range(self.metro.num_stops - 1):
                if j < len(pattern) and pattern[j] == 1:
                    # Estimate passenger load
                    load = sum(od_matrix[k][j] for k in range(j + 1, self.metro.num_stops)
                               if k < len(pattern) and pattern[k] == 1)

                    # Calculate travel time
                    travel_time = (self.metro.run_times[j] +
                                   self.metro.acc_dec_time +
                                   self.metro.dwell_time)

                    f2 += self.params.c2 * load * travel_time * chromosome.headway

        return f2

    def calculate_operation_cost(self, chromosome, zone_pattern, express_pattern):
        """Calculate vehicle operation cost"""
        f3 = 0.0
        for schedule_type in chromosome.schedule:
            pattern = self.get_stop_pattern(schedule_type, zone_pattern, express_pattern)

            for j in range(self.metro.num_stops - 1):
                if j < len(pattern):
                    operation_time = (self.metro.run_times[j] +
                                      self.metro.acc_dec_time * pattern[j] +
                                      self.metro.dwell_time * pattern[j])
                    f3 += self.params.c3 * operation_time

        return f3

    def run_genetic_algorithm(self, od_matrix, zone_pattern, express_pattern):
        """Run genetic algorithm optimization"""
        # Initialize population
        population = [BangladeshMetroChromosome(self.params) for _ in range(self.params.pop_size)]

        best_solution = None
        best_fitness = float('inf')
        fitness_history = []

        for generation in range(self.params.generations):
            # Evaluate fitness
            fitness_scores = []
            for chromosome in population:
                fitness = self.evaluate_objective_function(chromosome, od_matrix,
                                                           zone_pattern, express_pattern)
                fitness_scores.append(fitness)

            # Track best solution
            gen_best_idx = np.argmin(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]

            if gen_best_fitness < best_fitness:
                best_fitness = gen_best_fitness
                best_solution = BangladeshMetroChromosome(
                    self.params,
                    population[gen_best_idx].headway,
                    population[gen_best_idx].schedule.copy()
                )

            fitness_history.append(best_fitness)

            # Early termination if converged
            if generation > 50 and len(set(fitness_history[-10:])) == 1:
                break

            # Selection
            selected = self.tournament_selection(population, fitness_scores)

            # Create next generation
            new_population = []
            for i in range(0, self.params.pop_size, 2):
                parent1 = selected[i]
                parent2 = selected[min(i + 1, self.params.pop_size - 1)]

                if random.random() < self.params.crossover_rate:
                    child1 = self.crossover(parent1, parent2)
                    child2 = self.crossover(parent2, parent1)
                else:
                    child1 = BangladeshMetroChromosome(self.params, parent1.headway, parent1.schedule.copy())
                    child2 = BangladeshMetroChromosome(self.params, parent2.headway, parent2.schedule.copy())

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.params.pop_size]

        return best_solution, best_fitness, fitness_history

    def tournament_selection(self, population, fitness_scores, tournament_size=5):
        """Tournament selection with larger tournament size"""
        selected = []
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)),
                                               min(tournament_size, len(population)))
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx])
        return selected

    def crossover(self, parent1, parent2):
        """Enhanced crossover operation"""
        # Headway crossover with some randomness
        alpha = random.uniform(0.3, 0.7)
        new_headway = alpha * parent1.headway + (1 - alpha) * parent2.headway
        new_headway = max(self.params.headway_min, min(self.params.headway_max, new_headway))

        # Schedule crossover
        if parent1.schedule and parent2.schedule:
            # Two-point crossover
            min_len = min(len(parent1.schedule), len(parent2.schedule))
            max_len = max(len(parent1.schedule), len(parent2.schedule))

            if min_len > 2:
                point1 = random.randint(1, min_len - 1)
                point2 = random.randint(point1, min_len)
                new_schedule = (parent1.schedule[:point1] +
                                parent2.schedule[point1:point2] +
                                parent1.schedule[point2:])
            else:
                new_schedule = random.choice([parent1.schedule, parent2.schedule]).copy()

            # Potentially extend schedule
            if random.random() < 0.3 and len(new_schedule) < max_len:
                extension = random.choice([parent1.schedule, parent2.schedule])
                if len(extension) > len(new_schedule):
                    new_schedule.extend(extension[len(new_schedule):])
        else:
            new_schedule = (parent1.schedule if parent1.schedule else parent2.schedule).copy()

        return BangladeshMetroChromosome(self.params, new_headway, new_schedule)

    def mutate(self, chromosome):
        """Enhanced mutation operation"""
        # Headway mutation
        if random.random() < self.params.mutation_rate:
            mutation_strength = (self.params.headway_max - self.params.headway_min) * 0.15
            chromosome.headway += random.gauss(0, mutation_strength)
            chromosome.headway = max(self.params.headway_min,
                                     min(self.params.headway_max, chromosome.headway))

        # Schedule mutations
        if chromosome.schedule:
            # Point mutation
            if random.random() < self.params.mutation_rate:
                idx = random.randint(0, len(chromosome.schedule) - 1)
                chromosome.schedule[idx] = random.choice(['01', '10', '11'])

            # Insertion mutation
            if random.random() < self.params.mutation_rate * 0.5:
                new_gene = random.choice(['01', '10', '11'])
                insert_pos = random.randint(0, len(chromosome.schedule))
                chromosome.schedule.insert(insert_pos, new_gene)

            # Deletion mutation
            if len(chromosome.schedule) > 2 and random.random() < self.params.mutation_rate * 0.3:
                del_pos = random.randint(0, len(chromosome.schedule) - 1)
                chromosome.schedule.pop(del_pos)

        chromosome.ls = len(chromosome.schedule)
        return chromosome


def main():
    """Main function to run the Bangladesh Metro GUI application"""
    root = tk.Tk()
    app = BangladeshMetroGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()