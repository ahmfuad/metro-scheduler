"""
dynamic_headway_sim.py

Prototype: SimPy macroscopic simulator + rolling-horizon GA controller for dynamic headway.
Author: ChatGPT (generated)
Date: 2025-09-14

Notes:
- Simplified macroscopic model (queue per station, Poisson arrivals, FCFS boarding).
- GA is custom, lightweight (no external GA lib).
- Evaluator for GA runs independent SimPy environments (fast forward) seeded with observed queues.
- This is a prototype for experimentation and extension.
"""

import simpy
import random
import math
import statistics
import time
import copy
from collections import deque
import numpy as np
import pandas as pd
from datetime import datetime

# Optional plotting
try:
    import matplotlib.pyplot as plt
    PLOTTING = True
except Exception:
    PLOTTING = False

RND = random.Random(42)
NP_RND = np.random.default_rng(42)


# ----------------------
# Simulation parameters
# ----------------------
SIM_REAL_DURATION =  60 * 60  # simulate 60 minutes (real-time simulation)
DECISION_EPOCH = 180         # seconds (controller runs every 3 minutes)
PLANNING_HORIZON = 15 * 60   # seconds (plan next 15 minutes)
DELTA = 60                   # discretization for forecast / GA genes (1-minute bins)
GENE_LEN = PLANNING_HORIZON // DELTA

S = 5                       # Number of stations in corridor
INTER_STATION_TIME = 120    # seconds between stations
TRAIN_CAPACITY = 2184
DOORS = 4
DWELL_BASE = 5.0            # minimum dwell seconds
BOARD_TIME_PER_PASS = 1.2   # seconds
ALIGHT_TIME_PER_PASS = 0.9
STATION_DESIGN_CAP = 1800   # station capacity (pax)
STATION_ACCESS_THRESHOLD = 0.7

COST_WAIT_PER_MIN = 3.5     # arbitrary monetary unit per passenger-minute
COST_TRAIN_PER_HOUR = 146 * 60   # operator cost per train-hour
H_MIN = 5 * 60                  # minimum headway allowed (seconds)
H_MAX = 15*60               # maximum headway allowed (seconds)

# GA hyperparams
POP_SIZE = 50
GENERATIONS = 30
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.25
MUTATION_STD = 20.0  # seconds for Gaussian mutation

# Fast-sim time limit per GA fitness (to avoid very long runs)
FAST_SIM_TIME_LIMIT = 2.0  # seconds (a soft cap; not enforced strictly here — keep evaluator fast)


# ----------------------
# Demand generator / sensors
# ----------------------
# For prototype, we synthesize a time-varying arrival rate per station (passengers/sec).
# Later you will replace this with camera+RFID fused estimates.

def make_nominal_rate_profile(total_seconds):
    """
    Returns lambda_profile: shape (S, total_seconds) of arrival rates (pax/sec)
    Simple Gaussian-shaped morning peak per station with different amplitudes.
    """
    t = np.arange(total_seconds)
    profiles = np.zeros((S, total_seconds))
    # parameters per station (amplitude, mean_time, sigma)
    params = [
        (0.08, 20*60, 40*60),
        (0.10, 22*60, 40*60),
        (0.14, 24*60, 35*60),
        (0.07, 18*60, 50*60),
        (0.05, 26*60, 45*60)
    ]
    for u in range(S):
        a, mu, sigma = params[u]
        profiles[u] = a * np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    return profiles  # pax/sec


NOMINAL_PROFILES = make_nominal_rate_profile(SIM_REAL_DURATION)
'''

def time_to_seconds(tstr):
    # Expects 'HH:MM'
    h, m = map(int, tstr.split(':'))
    return h*3600 + m*60

def load_passenger_profile(csv_path, sim_duration_sec, num_stations):
    df = pd.read_csv(csv_path)
    # Prepare 2D array [station, time_sec]
    profile = np.zeros((num_stations, sim_duration_sec))
    
    for _, row in df.iterrows():
        st = int(row['Station'])
        start = time_to_seconds(row['Start Time'])
        end = time_to_seconds(row['End Time'])
        lam_per_sec = float(row['Passengers per Minute']) / 60.0
        
        # Clip within simulation duration
        start = max(0, min(sim_duration_sec-1, start))
        end = max(0, min(sim_duration_sec, end))
        
        profile[st, start:end] = lam_per_sec
    
    return profile

csv_profile = load_passenger_profile(
    '/home/awsaf/Desktop/Projects/Metro Scheduler/metro-scheduler/PCKN/metro_weighted_station_passengers.csv',
    SIM_REAL_DURATION,
    S
)
NOMINAL_PROFILES = csv_profile

'''

# ----------------------
# SimPy model (macroscopic)
# ----------------------
class StationState:
    def __init__(self, idx):
        self.idx = idx
        self.platform_queue = deque()  # arrival times of passengers inside platform
        self.outside_queue = deque()   # arrival times waiting outside
        self.in_station_count = 0      # number currently inside (for capacity checks)
        # stats
        self.wait_times = []


class TrainObj:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.onboard = 0


class CorridorSimulator:
    """
    A SimPy-based macroscopic simulator that can be used in two modes:
    - 'real' mode: runs the actual system where the controller dispatches trains live
    - 'fast-eval' mode: used by GA to simulate candidate headway plans forward from a seed state
    """
    def __init__(self, sim_duration, seed_state=None, arrival_profile=None):
        """
        sim_duration: seconds to simulate
        seed_state: optional list of initial StationState-like dicts: {'platform': n, 'outside': m}
        arrival_profile: array shape (S, sim_duration) of pax/sec arrival rates
        """
        self.env = simpy.Environment()
        self.sim_duration = sim_duration
        self.stations = [StationState(i) for i in range(S)]
        self.trains = []
        self.arrival_profile = arrival_profile  # if None, will use global NOMINAL_PROFILES truncated/shifted
        if seed_state:
            # seed initial waiting passengers (we only seed counts, not per-passenger timestamps)
            for u, s in enumerate(seed_state):
                platform_n = int(s.get('platform', 0))
                outside_n = int(s.get('outside', 0))
                now = 0.0
                for _ in range(platform_n):
                    self.stations[u].platform_queue.append(now)
                for _ in range(outside_n):
                    self.stations[u].outside_queue.append(now)
                self.stations[u].in_station_count = platform_n
        # bookkeeping
        self.total_boardings = 0
        self.total_alightings = 0
        self.trains_used = 0
        self.dwell_time_total = 0.0

    def start_arrival_processes(self, offset=0):
        """Start poisson arrival processes for each station. offset in seconds into given profile."""
        for u in range(S):
            lam_profile = None
            if self.arrival_profile is not None:
                # arrival_profile expected shape (S, sim_duration)
                lam_profile = self.arrival_profile[u]
            else:
                # fallback: slice global nominal
                lam_profile = NOMINAL_PROFILES[u, offset:offset + self.sim_duration]
            self.env.process(self._arrival_proc(u, lam_profile))

    def _arrival_proc(self, u, lam_profile):
        """Generate arrivals for station u during simulation time using non-homogeneous Poisson via thinning.
           lam_profile: array of pax/sec with length >= sim_duration
        """
        t = 0.0
        while t < self.sim_duration:
            # set safe upper bound for thinning
            lam_max = max(lam_profile[min(int(t), len(lam_profile)-1): min(int(t)+60, len(lam_profile))])  # local max
            if lam_max <= 0:
                # idle skip
                yield self.env.timeout(1.0)
                t += 1.0
                continue
            # propose inter-arrival from exponential(lam_max)
            u_exp = RND.random()
            inter = -math.log(u_exp) / lam_max
            yield self.env.timeout(inter)
            t += inter
            if t >= self.sim_duration:
                break
            # accept with probability lam(t)/lam_max
            lam_t = lam_profile[min(int(t), len(lam_profile)-1)]
            if RND.random() <= lam_t / lam_max:
                # arrival occurs at station u at time env.now
                self._record_arrival(u, self.env.now)

    def _record_arrival(self, u, tstamp):
        """Record arrival as outside or platform depending on capacity allowance (seeded state uses in_station_count)."""
        st = self.stations[u]
        # Determine safe capacity gamma_u (design minus max alight assumptions). For prototype, use STATION_DESIGN_CAP.
        gamma_u = STATION_DESIGN_CAP
        # threshold theta * gamma
        if st.in_station_count < STATION_ACCESS_THRESHOLD * gamma_u:
            # allow access -> platform queue
            st.platform_queue.append(tstamp)
            st.in_station_count += 1
        else:
            st.outside_queue.append(tstamp)

    def schedule_trains_from_origin(self, origin_times):
        """
        origin_times: list of times (seconds from now=0) at which a new train departs origin (station 0) in this simulator.
        We create a process per train that moves along stations, handles alight/board and records dwell.
        """
        for i, start in enumerate(origin_times):
            # create train at simulated time 'start'
            self.env.process(self._train_process(start, f"Train-{i+1}"))

    def _train_process(self, start_time, name):
        yield self.env.timeout(start_time)
        self.trains_used += 1
        onboard = 0
        # run along stations
        for u in range(S):
            arrive_time = self.env.now
            # alighting - simple fraction model: assume some fraction alight (not tracked owners)
            alight = int(0.05 * onboard)  # toy model
            onboard -= alight
            self.total_alightings += alight
            # boarding from platform queue up to capacity
            st = self.stations[u]
            space = TRAIN_CAPACITY - onboard
            boarded = 0
            # first move some from outside to platform if platform has space under threshold
            # For simplicity, we allow access from outside while station is undersaturated
            while st.outside_queue and st.in_station_count < STATION_ACCESS_THRESHOLD * STATION_DESIGN_CAP:
                t_arr = st.outside_queue.popleft()
                st.platform_queue.append(t_arr)
                st.in_station_count += 1

            while st.platform_queue and space > 0:
                pax_arrival_time = st.platform_queue.popleft()
                wait_time = self.env.now - pax_arrival_time
                st.wait_times.append(wait_time)
                boarded += 1
                onboard += 1
                space -= 1
                self.total_boardings += 1
            # compute dwell as parallel-door model
            dwell_secs = max(DWELL_BASE, (boarded * BOARD_TIME_PER_PASS + alight * ALIGHT_TIME_PER_PASS) / DOORS)
            self.dwell_time_total += dwell_secs
            yield self.env.timeout(dwell_secs)
            # travel to next station (unless last)
            if u < S - 1:
                yield self.env.timeout(INTER_STATION_TIME)
        # train done (end of run). For this fast sim, no recovery & reuse is modeled.

    def run(self, until=None):
        if until is None:
            until = self.sim_duration
        self.start_arrival_processes()
        self.env.run(until=until)

    def get_metrics(self):
        # Aggregate wait times from all stations
        all_waits = []
        for st in self.stations:
            all_waits.extend(st.wait_times)
        avg_wait_sec = statistics.mean(all_waits) if all_waits else 0.0
        avg_wait_min = avg_wait_sec / 60.0
        return {
            'avg_wait_min': avg_wait_min,
            'trains_used': self.trains_used,
            'total_boardings': self.total_boardings,
            'dwell_total_sec': self.dwell_time_total
        }
    
    # ----------------------
    # Static headway simulation
    # ----------------------
    def run_static_simulation(headway_seconds, sim_duration=SIM_REAL_DURATION):
        """
        Run a simple corridor sim with fixed headway trains and return cost + metrics.
        """
        sim = CorridorSimulator(sim_duration=sim_duration, seed_state=None, arrival_profile=NOMINAL_PROFILES)
        sim.start_arrival_processes(offset=0)

        # schedule trains evenly spaced
        origin_times = list(np.arange(0, sim_duration, headway_seconds))
        sim.schedule_trains_from_origin(origin_times)

        sim.run(until=sim_duration)
        metrics = sim.get_metrics()

        # cost same as GA fitness style
        if metrics['total_boardings'] > 0:
            total_wait_min = metrics['avg_wait_min'] * metrics['total_boardings']
        else:
            total_wait_min = 0.0

        wait_cost = COST_WAIT_PER_MIN * total_wait_min
        approx_route_time = (S - 1) * INTER_STATION_TIME + (S * DWELL_BASE)
        train_hours = metrics['trains_used'] * (approx_route_time / 3600.0)
        op_cost = train_hours * COST_TRAIN_PER_HOUR

        total_cost = wait_cost + op_cost
        return total_cost, metrics



# ----------------------
# GA implementation
# ----------------------
def random_headway_gene():
    """Return a headway vector (seconds) of length GENE_LEN as initial candidate.
       We generate values between H_MIN and min(H_MAX, some scale)."""
    # seed near an even-headway at 3 minutes as baseline
    base = 180.0
    return [max(H_MIN, min(H_MAX, RND.gauss(base, 30.0))) for _ in range(GENE_LEN)]


def pop_init(pop_size):
    pop = []
    for _ in range(pop_size):
        pop.append(random_headway_gene())
    return pop


def crossover_blend(a, b, alpha=0.5):
    """Arithmetic blend crossover for two parents a and b (lists)."""
    child1 = []
    child2 = []
    for x, y in zip(a, b):
        if RND.random() < 0.5:
            # linear combination
            r = RND.random()
            c1 = r * x + (1 - r) * y
            c2 = (1 - r) * x + r * y
        else:
            c1, c2 = x, y
        child1.append(c1)
        child2.append(c2)
    return child1, child2


def mutate_gaussian(indiv, mut_rate=MUTATION_RATE, std=MUTATION_STD):
    """Mutate an individual in-place (Gaussian noise), clip by bounds."""
    for i in range(len(indiv)):
        if RND.random() < mut_rate:
            indiv[i] += RND.gauss(0, std)
            if indiv[i] < H_MIN:
                indiv[i] = H_MIN
            if indiv[i] > H_MAX:
                indiv[i] = H_MAX


def tournament_selection(pop, fitnesses, k=TOURNAMENT_K):
    n = len(pop)
    best = None
    best_fit = None
    for _ in range(k):
        i = RND.randrange(n)
        if best is None or fitnesses[i] < best_fit:
            best = pop[i]
            best_fit = fitnesses[i]
    return copy.deepcopy(best)


# ----------------------
# Fitness evaluator (uses fast SimPy forward sim)
# ----------------------
def build_origin_times_from_headways(headway_vector, horizon_seconds):
    """
    Given a headway vector (seconds per gene) of length GENE_LEN and delta time,
    produce origin departure times in [0, horizon_seconds).
    """
    # interpret each gene as a headway for the corresponding DELTA window
    times = []
    t = 0.0
    gene_idx = 0
    while t < horizon_seconds:
        h = headway_vector[min(gene_idx, len(headway_vector)-1)]
        # avoid zero or tiny
        h = max(H_MIN, min(H_MAX, float(h)))
        times.append(t)
        t += h
        gene_idx += 1
        if len(times) > 200:  # safety break
            break
    return times


def evaluate_headway_candidate(headway_vector, seed_state, forecast_profile):
    """
    Evaluate a candidate headway plan by running a fast sim for PLANNING_HORIZON seconds.
    - seed_state: list of dicts per station {'platform': n, 'outside': m}
    - forecast_profile: array shape (S, PLANNING_HORIZON) with pax/sec arrival rates for evaluation window
    Returns: scalar fitness (lower better), diagnostics dict
    """
    # build origin times from headway vector
    origin_times = build_origin_times_from_headways(headway_vector, PLANNING_HORIZON)
    # create fast simulator with seeded state and forecast profile
    sim = CorridorSimulator(sim_duration=PLANNING_HORIZON, seed_state=seed_state, arrival_profile=forecast_profile)
    sim.schedule_trains_from_origin(origin_times)
    sim.run(until=PLANNING_HORIZON)
    metrics = sim.get_metrics()

    # compute costs: waiting cost + operational cost (trains used * running hours fraction)
    total_wait_min = metrics['avg_wait_min'] * sum([sim.total_boardings if sim.total_boardings else 1])  # simple scalarization
    # more realistic approach: boarding count * avg_wait
    if metrics['total_boardings'] > 0:
        total_wait_min = metrics['avg_wait_min'] * metrics['total_boardings']
    else:
        total_wait_min = 0.0

    wait_cost = COST_WAIT_PER_MIN * total_wait_min
    # estimate operator train-hours: each train in planning horizon runs approx route_time (S-1)*inter + sum(dwell)
    approx_route_time = (S - 1) * INTER_STATION_TIME + (S * DWELL_BASE)
    train_hours = metrics['trains_used'] * (approx_route_time / 3600.0)
    op_cost = train_hours * COST_TRAIN_PER_HOUR

    # penalize platform overcrowding (rough): if any seed_state platform + forecast arrivals exceed station capacity, penalize
    crowd_pen = 0.0
    for u in range(S):
        seeded_platform = seed_state[u].get('platform', 0)
        # estimate arrivals in horizon
        arrival_est = int(np.sum(forecast_profile[u]) * 1.0) if forecast_profile is not None else 0
        if seeded_platform + arrival_est > STATION_DESIGN_CAP:
            crowd_pen += 1000.0 * (seeded_platform + arrival_est - STATION_DESIGN_CAP)

    # stability penalty: deviation from baseline headway (optional)
    baseline = 180.0
    stab_pen = sum(abs(h - baseline) for h in headway_vector) * 0.01

    fitness = wait_cost + op_cost + crowd_pen + stab_pen

    diag = {
        'wait_cost': wait_cost,
        'op_cost': op_cost,
        'trains_used': metrics['trains_used'],
        'avg_wait_min': metrics['avg_wait_min'],
        'boardings': metrics['total_boardings'],
        'dwell_sec': metrics['dwell_total_sec']
    }
    return fitness, diag


# ----------------------
# Controller / rolling horizon
# ----------------------
class RollingHorizonController:
    def __init__(self, decision_epoch=DECISION_EPOCH, planning_horizon=PLANNING_HORIZON, delta=DELTA):
        self.decision_epoch = decision_epoch
        self.planning_horizon = planning_horizon
        self.delta = delta
        self.prev_best = None
        self.log = []

    def observe_state(self, real_simulator_env):
        """
        From the running 'real' sim (CorridorSimulator instance), extract observed platform/outside queue sizes per station.
        We'll return a seed_state: list of dicts per station {platform, outside}
        """
        seed_state = []
        for st in real_simulator_env.stations:
            seed_state.append({
                'platform': len(st.platform_queue),
                'outside': len(st.outside_queue)
            })
        return seed_state

    def forecast(self, now_sim_sec):
        """
        Simple forecast: extract nominal profile slice for next planning horizon.
        Replace this with ML-based forecast using camera+RFID.
        Returns array shape (S, planning_horizon) of pax/sec
        """
        horizon = int(self.planning_horizon)
        forecast = np.zeros((S, horizon))
        # use NOMINAL_PROFILES (station x time) offset by now
        global NOMINAL_PROFILES
        T_total = NOMINAL_PROFILES.shape[1]
        for u in range(S):
            start = int(now_sim_sec)
            end = min(start + horizon, T_total)
            slice_len = end - start
            if slice_len > 0:
                forecast[u, :slice_len] = NOMINAL_PROFILES[u, start:end]
            # if the horizon extends beyond available nominal profile, leave zeros or repeat last value
        return forecast

    def run_ga(self, seed_state, forecast_profile, time_budget_seconds=10.0):
        """
        Run a GA to optimize headway_vector over planning horizon. Returns best individual and diag.
        For simplicity we run a small fixed-generation GA.
        """
        pop = pop_init(POP_SIZE)
        fitnesses = [None] * len(pop)
        # evaluate initial population
        for i, indiv in enumerate(pop):
            fit, diag = evaluate_headway_candidate(indiv, seed_state, forecast_profile)
            fitnesses[i] = fit

        for gen in range(GENERATIONS):
            new_pop = []
            new_fits = []
            while len(new_pop) < POP_SIZE:
                # selection
                p1 = tournament_selection(pop, fitnesses)
                p2 = tournament_selection(pop, fitnesses)
                # crossover
                if RND.random() < CROSSOVER_RATE:
                    c1, c2 = crossover_blend(p1, p2)
                else:
                    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                # mutate
                mutate_gaussian(c1)
                mutate_gaussian(c2)
                # evaluate
                f1, _ = evaluate_headway_candidate(c1, seed_state, forecast_profile)
                f2, _ = evaluate_headway_candidate(c2, seed_state, forecast_profile)
                new_pop.append(c1); new_fits.append(f1)
                if len(new_pop) < POP_SIZE:
                    new_pop.append(c2); new_fits.append(f2)
            pop = new_pop
            fitnesses = new_fits
            # optional: print progress
            if gen % 5 == 0:
                best_idx = int(np.argmin(fitnesses))
                print(f"[GA] gen {gen} best fit {fitnesses[best_idx]:.1f}")
        # return best
        best_idx = int(np.argmin(fitnesses))
        best = pop[best_idx]
        best_fit = fitnesses[best_idx]
        best_diag = None
        # recompute diag for best
        _, best_diag = evaluate_headway_candidate(best, seed_state, forecast_profile)
        return best, best_fit, best_diag

    def choose_and_apply(self, real_simulator, now_sim_sec):
        """
        Observe, forecast, optimize, and return the 'dispatch decision' to apply to real_simulator.
        For prototype: interpret plan as successive origin departures; we will dispatch the first train at t=0 (i.e., immediately)
        if the first headway is <= decision_epoch, otherwise we wait.
        We'll apply only one origin train at a time (rolling). This is simplistic but demonstrates the loop.
        """
        seed_state = self.observe_state(real_simulator)
        forecast_profile = self.forecast(now_sim_sec)
        print(f"[Controller] Observed platform queues: {[len(st.platform_queue) for st in real_simulator.stations]}")
        print("[Controller] Running GA...")
        t0 = time.time()
        best, fit, diag = self.run_ga(seed_state, forecast_profile)
        tspent = time.time() - t0
        print(f"[Controller] GA done in {tspent:.1f}s fit={fit:.1f} diag={diag}")
        # Decide first dispatch: create origin times using best headway vector but only looking at first departure
        # If station has many waiting passengers, dispatch immediately; otherwise obey h0
        h0 = max(H_MIN, min(H_MAX, best[0]))
        # heuristics: dispatch immediately if any platform queue > threshold (say 50 pax) else schedule next after h0
        immediate = any(len(st.platform_queue) > 50 for st in real_simulator.stations)
        if immediate:
            dispatch_time = 0.0
        else:
            dispatch_time = h0
        # apply to real_simulator: we create a train process that departs origin after dispatch_time
        origin_time = dispatch_time
        print(f"[Controller] Applying dispatch at t+{origin_time:.0f}s")
        real_simulator.schedule_trains_from_origin([origin_time])
        # log
        self.log.append({
            'now': now_sim_sec,
            'fit': fit,
            'diag': diag,
            'h0': h0,
            'dispatch_delay': dispatch_time
        })
        # shift prev_best
        self.prev_best = best
        return dispatch_time


# ----------------------
# Top-level real-time simulation loop
# ----------------------
def run_real_time_simulation():
    # Create the real simulator for full SIM_REAL_DURATION

    '''
    csv_profile = load_passenger_profile(
        '/home/awsaf/Desktop/Projects/Metro Scheduler/metro-scheduler/PCKN/metro_weighted_station_passengers.csv',
        SIM_REAL_DURATION,
        S
    )

    real_sim = CorridorSimulator(sim_duration=SIM_REAL_DURATION,
                                seed_state=None,
                                arrival_profile=csv_profile)

    '''
    real_sim = CorridorSimulator(sim_duration=SIM_REAL_DURATION,
                                seed_state=None,
                                arrival_profile=NOMINAL_PROFILES)
    # Start the continuous arrival processes for the 'real' sim
    real_sim.start_arrival_processes(offset=0)
    controller = RollingHorizonController()
    # We'll simulate in steps: advance sim by DECISION_EPOCH, but before each advance, controller decides & schedules next train.
    now = 0.0
    # For initial warm-up, dispatch one train at t=0
    real_sim.schedule_trains_from_origin([0.0])
    next_control_time = 0.0
    last_print = time.time()
    # We'll drive the real_sim env manually in increments
    while now < SIM_REAL_DURATION:
        # at control epoch, run controller and then step env until next control
        if now >= next_control_time - 1e-6:
            print(f"\n[Main] Controller epoch at sim t={int(now)}s")
            # controller will schedule at least one train (immediately or after delay)
            controller.choose_and_apply(real_sim, now)
            next_control_time = now + DECISION_EPOCH
        # step sim to next control or to end
        step_until = min(next_control_time, SIM_REAL_DURATION)
        real_sim.env.run(until=step_until)
        now = step_until
        # print small status occasionally
        if time.time() - last_print > 5.0:
            print(f"[Main] sim t={int(now)}s")
            last_print = time.time()

    # At end, gather metrics
    metrics = real_sim.get_metrics()
    print("\n=== REAL SIM RESULTS ===")
    print(metrics)
    # print controller log summary
    print("\nController log entries:", len(controller.log))
    for entry in controller.log[:5]:
        print(entry)
    return real_sim, controller


if __name__ == "__main__":
    start_time = time.time()

    '''
    csv_profile = load_passenger_profile(
        '/home/awsaf/Desktop/Projects/Metro Scheduler/metro-scheduler/PCKN/metro_weighted_station_passengers.csv',
        SIM_REAL_DURATION,
        S
    )
    real_sim = CorridorSimulator(sim_duration=SIM_REAL_DURATION,
                                seed_state=None,
                                arrival_profile=csv_profile)
    '''


    real_sim, controller = run_real_time_simulation()

    dyn_metrics = real_sim.get_metrics()
    print(f"\nDynamic total wall time: {time.time() - start_time:.1f}s")

    # run static at 8 minutes (480s)
    print("\nRunning static 8-min headway sim...")
    static_cost, static_metrics = CorridorSimulator.run_static_simulation(480)

    # compute dynamic cost
    # (approx, same formula as evaluate_headway_candidate)
    if dyn_metrics['total_boardings'] > 0:
        dyn_wait = dyn_metrics['avg_wait_min'] * dyn_metrics['total_boardings']
    else:
        dyn_wait = 0.0
    dyn_wait_cost = COST_WAIT_PER_MIN * dyn_wait
    approx_route_time = (S - 1) * INTER_STATION_TIME + (S * DWELL_BASE)
    dyn_train_hours = dyn_metrics['trains_used'] * (approx_route_time / 3600.0)
    dyn_op_cost = dyn_train_hours * COST_TRAIN_PER_HOUR
    dyn_cost = dyn_wait_cost + dyn_op_cost

    print("\n=== COMPARISON ===")
    print(f"Static cost = {static_cost:.1f}, Dynamic cost = {dyn_cost:.1f}")

    if PLOTTING:
        labels = ['Static (8 min)', 'Dynamic (GA)']
        costs = [static_cost, dyn_cost]
        waits = [static_metrics['avg_wait_min'], dyn_metrics['avg_wait_min']]
        trains = [static_metrics['trains_used'], dyn_metrics['trains_used']]

        plt.figure(figsize=(10,4))
        plt.subplot(1,3,1)
        plt.bar(labels, costs)
        plt.ylabel("Total cost")
        plt.subplot(1,3,2)
        plt.bar(labels, waits)
        plt.ylabel("Avg wait (min)")
        plt.subplot(1,3,3)
        plt.bar(labels, trains)
        plt.ylabel("Trains used")
        plt.tight_layout()
        plt.show()
