from dynamic_headway_sim import (
    CorridorSimulator, RollingHorizonController,
    NOMINAL_PROFILES, SIM_REAL_DURATION, DECISION_EPOCH,
    COST_WAIT_PER_MIN, INTER_STATION_TIME, DWELL_BASE, S,
    COST_TRAIN_PER_HOUR, H_MIN, H_MAX
)

import threading, queue
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

STATIC_HEADWAY = 480.0  # 8 minutes

class LiveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic vs Static Headway Cost Comparison")

        # Info labels
        self.cost_dyn_var = tk.StringVar(value="Dynamic Cost: 0")
        self.cost_stat_var = tk.StringVar(value="Static Cost: 0")
        self.wait_dyn_var = tk.StringVar(value="Dynamic Avg Wait: 0")
        self.wait_stat_var = tk.StringVar(value="Static Avg Wait: 0")

        ttk.Label(root, textvariable=self.cost_dyn_var, font=("Arial", 12)).pack()
        ttk.Label(root, textvariable=self.cost_stat_var).pack()
        ttk.Label(root, textvariable=self.wait_dyn_var).pack()
        ttk.Label(root, textvariable=self.wait_stat_var).pack()

        # Figure with 2 axes: left = headway plan, right = cost curve
        self.fig, (self.ax_head, self.ax_cost) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        ttk.Button(root, text="Start Simulation", command=self.start_sim).pack(pady=5)

        # Data holders
        self.data_q = queue.Queue()
        self.time_points = []
        self.cost_dyn_list = []
        self.cost_stat_list = []
        self.running = False

    def start_sim(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.sim_loop, daemon=True).start()
            self.root.after(500, self.update_ui)

    def sim_loop(self):
        # Dynamic simulation
        dyn = CorridorSimulator(sim_duration=SIM_REAL_DURATION, arrival_profile=NOMINAL_PROFILES)
        dyn.start_arrival_processes(0)
        controller = RollingHorizonController()
        dyn.schedule_trains_from_origin([0.0])

        # Static simulation
        stat = CorridorSimulator(sim_duration=SIM_REAL_DURATION, arrival_profile=NOMINAL_PROFILES)
        stat.start_arrival_processes(0)
        static_times = [i for i in range(0, int(SIM_REAL_DURATION), int(STATIC_HEADWAY))]
        stat.schedule_trains_from_origin(static_times)

        now = 0.0

        while now < SIM_REAL_DURATION:
            # Run GA to decide dynamic headway
            seed_state = controller.observe_state(dyn)
            forecast = controller.forecast(now)
            best, fit, _ = controller.run_ga(seed_state, forecast)
            next_hw = max(H_MIN, min(H_MAX, best[0]))
            dyn.schedule_trains_from_origin([next_hw])

            # Step forward in time
            next_t = min(now + DECISION_EPOCH, SIM_REAL_DURATION)
            dyn.env.run(until=next_t)
            stat.env.run(until=next_t)
            now = next_t

            # Calculate total cost function
            def total_cost(m):
                wait_min = m['avg_wait_min'] * m['total_boardings']
                wait_cost = COST_WAIT_PER_MIN * wait_min
                route_time = (S - 1) * INTER_STATION_TIME + S * DWELL_BASE
                op_cost = m['trains_used'] * (route_time / 3600.0) * COST_TRAIN_PER_HOUR
                return wait_cost + op_cost

            md = dyn.get_metrics()
            ms = stat.get_metrics()
            cost_d = total_cost(md)
            cost_s = total_cost(ms)

            self.time_points.append(now/60)
            self.cost_dyn_list.append(cost_d)
            self.cost_stat_list.append(cost_s)

            # Push to UI queue
            self.data_q.put({
                'headways': best,
                'cost_d': cost_d,
                'cost_s': cost_s,
                'wait_d': md['avg_wait_min'],
                'wait_s': ms['avg_wait_min']
            })

        self.running = False

    def update_ui(self):
        while not self.data_q.empty():
            d = self.data_q.get()

            self.cost_dyn_var.set(f"Dynamic Cost: {d['cost_d']:.1f}")
            self.cost_stat_var.set(f"Static Cost: {d['cost_s']:.1f}")
            self.wait_dyn_var.set(f"Dynamic Avg Wait: {d['wait_d']:.2f} min")
            self.wait_stat_var.set(f"Static Avg Wait: {d['wait_s']:.2f} min")

            # Left plot: GA best headway plan
            self.ax_head.cla()
            self.ax_head.set_title("Current GA Headway Plan")
            self.ax_head.set_xlabel("Train index")
            self.ax_head.set_ylabel("Headway (min)")
            self.ax_head.plot(range(len(d['headways'])), [h/60 for h in d['headways']], '-o')

            # Right plot: live cost comparison
            self.ax_cost.cla()
            self.ax_cost.set_title("Dynamic vs Static Cost")
            self.ax_cost.set_xlabel("Time (min)")
            self.ax_cost.set_ylabel("Total Cost")
            self.ax_cost.plot(self.time_points, self.cost_dyn_list, 'r-', label="Dynamic")
            self.ax_cost.plot(self.time_points, self.cost_stat_list, 'b-', label="Static")
            self.ax_cost.legend()

            self.canvas.draw()

        if self.running:
            self.root.after(1000, self.update_ui)

if __name__ == "__main__":
    root = tk.Tk()
    LiveApp(root)
    root.mainloop()
