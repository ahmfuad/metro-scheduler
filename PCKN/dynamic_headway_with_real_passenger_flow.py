"""
dynamic_headway_sim.py

Prototype: SimPy macroscopic simulator + rolling-horizon GA controller for dynamic headway.
Author: ChatGPT (generated), adapted for Dhaka MRT Line 6 OD data
Date: 2025-09-14

Notes:
- Simplified macroscopic model (queue per station, Poisson arrivals, FCFS boarding).
- GA is custom, lightweight (no external GA lib).
- Evaluator for GA runs independent SimPy environments (fast forward) seeded with observed queues.
- This is a prototype for experimentation and extension.
- Adapted for 16 stations, full operating day simulation, OD-based forward flows, destination-based alighting.
- Static schedule is piece-wise as provided.
- Station capacities set based on paper example.
- Demand scaled to daily forward flows from OD, using Gaussian profiles.
- Fixed ValueError in _arrival_proc by adding explicit length checks instead of using 'or' which triggers bool on numpy array.
- Added real-time updates: print current metrics and update matplotlib plot at each decision epoch.
- Embedded real-time plot in Tkinter window for better responsiveness.
- Added static headway cost graph for comparison.
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

INF = 10000000000

# Optional plotting
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import tkinter as tk
    PLOTTING = True
except Exception:
    PLOTTING = False

RND = random.Random(42)
NP_RND = np.random.default_rng(42)


# ----------------------
# Simulation parameters
# ----------------------
BASE_HOUR = 7
BASE_MIN = 10
BASE_SEC = BASE_HOUR * 3600 + BASE_MIN * 60
SIM_REAL_DURATION = 14 * 3600  # simulate ~14 hours (operating day)
DECISION_EPOCH = 300           # seconds (controller runs every 5 minutes)
PLANNING_HORIZON = 30 * 60     # seconds (plan next 30 minutes)
DELTA = 120                    # discretization for forecast / GA genes (2-minute bins)
GENE_LEN = PLANNING_HORIZON // DELTA

S = 16                         # Number of stations in corridor
INTER_STATION_TIME = 120       # seconds between stations
TRAIN_CAPACITY = 2184
DOORS = 4
DWELL_BASE = 5.0               # minimum dwell seconds
BOARD_TIME_PER_PASS = 1.2      # seconds
ALIGHT_TIME_PER_PASS = 0.9
STATION_DESIGN_CAP = 4000      # gamma_u from paper (CS - LA_max)
STATION_ACCESS_THRESHOLD = 0.9 # theta_u from paper

COST_WAIT_PER_MIN = 3.5        # arbitrary monetary unit per passenger-minute
COST_TRAIN_PER_HOUR = 146 * 60 # operator cost per train-hour
H_MIN = 5 * 60                 # minimum headway allowed (seconds)
H_MAX = 15 * 60                # maximum headway allowed (seconds)

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
# OD matrix construction
# ----------------------
DAYS_IN_PERIOD = 31.0
stations = ["S01UN", "S02UC", "S03US", "S04PL", "S05M11", "S06M10", "S07KP", "S08SP", "S09AG", "S10BS", "S11FG", "S12KB", "S13SB", "S14DU", "S15BS", "S16MJ"]
OD = np.zeros((S, S))
row_lists = [
    [2614, 3037, 53646, 73547, 151650, 31849, 38966, 87927, 17074, 51458, 71943, 54814, 59100, 124996, 129478],
    [2969, 436, 20366, 18739, 43402, 6491, 9154, 19617, 5563, 16584, 19693, 10565, 13516, 20292, 25264],
    [2916, 509, 9962, 11964, 17116, 4310, 4096, 6435, 1668, 4692, 5576, 2407, 3716, 4793, 5312],
    [57564, 20518, 9289, 4016, 40037, 30485, 28453, 64101, 13027, 47142, 65538, 33534, 38943, 71497, 61751],
    [83913, 21074, 12003, 4380, 12204, 28951, 46666, 57070, 13099, 55348, 66701, 52291, 46920, 87489, 80082],
    [143726, 38428, 12369, 29628, 8758, 9286, 28499, 67369, 23705, 92755, 127172, 76593, 93890, 182115, 183857],
    [37329, 7941, 3883, 30302, 25984, 11170, 1686, 15182, 10365, 33712, 45013, 29968, 26111, 52877, 42318],
    [43241, 9830, 4268, 28862, 42404, 35812, 1695, 8196, 10738, 56666, 72141, 40657, 40681, 66749, 64595],
    [86770, 18388, 5063, 55044, 48258, 76663, 13452, 8121, 3816, 27329, 73252, 50620, 61135, 134467, 132613],
    [19011, 6681, 1556, 15426, 15161, 35349, 11831, 11520, 3694, 4343, 15101, 20197, 26216, 26852, 40084],
    [50574, 15974, 3916, 41363, 47465, 98554, 30103, 54173, 30598, 4863, 17310, 38713, 73446, 98291, 155142],
    [69580, 19413, 4578, 59207, 58015, 137840, 37952, 62915, 69492, 13947, 11557, 19873, 53785, 119543, 140129],
    [58401, 11123, 2151, 31138, 47247, 86685, 27723, 37785, 54448, 23193, 36606, 22141, 4757, 31337, 102505],
    [61059, 14102, 2985, 37057, 42198, 101944, 23023, 38020, 62311, 26032, 65659, 53077, 4360, 11158, 60329],
    [106143, 18859, 4016, 55950, 68090, 177188, 42021, 55432, 127587, 27037, 80565, 113438, 26901, 12013, 18701],
    [143459, 29421, 5132, 62738, 82818, 215612, 40877, 68244, 145448, 40804, 144861, 146226, 92737, 57968, 16106]
]

for i in range(S):
    k = 0
    for v in range(S):
        if v == i:
            OD[i, v] = 0
        else:
            OD[i, v] = row_lists[i][k]
            k += 1

# Compute daily forward demand per station
daily_forward = [0.0] * S
for u in range(S):
    sum_f = 0.0
    for v in range(u + 1, S):
        sum_f += OD[u, v]
    daily_forward[u] = sum_f / DAYS_IN_PERIOD

# Precompute forward probs for each u (probs for v = u+1 to S-1)
forward_probs = [None] * S
for u in range(S):
    sum_p = 0.0
    for v in range(u + 1, S):
        sum_p += OD[u, v]
    if sum_p > 0:
        p = []
        for v in range(u + 1, S):
            p.append(OD[u, v] / sum_p)
        forward_probs[u] = p
    else:
        forward_probs[u] = []

# ----------------------
# Demand generator / sensors
# ----------------------
# For prototype, we synthesize a time-varying arrival rate per station (pax/sec).
# Scaled to match daily forward OD totals.
def make_nominal_rate_profile(total_seconds):
    """
    Returns lambda_profile: shape (S, total_seconds) of arrival rates (pax/sec)
    Simple Gaussian-shaped morning peak per station with different amplitudes.
    """
    t = np.arange(total_seconds)
    profiles = np.zeros((S, total_seconds))
    # parameters per station (amplitude, mean_time, sigma)
    params = []
    for i in range(S):
        a = 0.05 + 0.005 * i  # vary amplitude
        mu = 2*3600 + i * 600  # spread peaks
        sigma = 2*3600
        params.append((a, mu, sigma))
    for u in range(S):
        a, mu, sigma = params[u]
        profiles[u] = a * np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    # Add evening peak
    for u in range(S):
        a, mu, sigma = params[u]
        mu_even = mu + 8*3600  # ~8 hours later
        profiles[u] += a * 0.8 * np.exp(-0.5 * ((t - mu_even) / sigma) ** 2)
    return profiles  # pax/sec

NOMINAL_PROFILES = make_nominal_rate_profile(SIM_REAL_DURATION)
# Scale to match daily forward
for u in range(S):
    sum_p = np.sum(NOMINAL_PROFILES[u])
    if sum_p > 0:
        NOMINAL_PROFILES[u] *= daily_forward[u] / sum_p

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
    def __init__(self, sim_duration, seed_state=None, arrival_profile=None):
        self.env = simpy.Environment()
        self.sim_duration = sim_duration
        self.stations = [StationState(i) for i in range(S)]
        self.trains_used = 0
        self.total_boardings = 0
        self.total_alightings = 0
        self.dwell_time_total = 0.0
        self.arrival_profile = arrival_profile if arrival_profile is not None else NOMINAL_PROFILES
        if seed_state:
            for u, s in enumerate(seed_state):
                platform_n = int(s.get('platform', 0))
                outside_n = int(s.get('outside', 0))
                now = 0.0
                for _ in range(platform_n):
                    self.stations[u].platform_queue.append(now)
                for _ in range(outside_n):
                    self.stations[u].outside_queue.append(now)
                self.stations[u].in_station_count = platform_n

    def start_arrival_processes(self, offset=0):
        for u in range(S):
            lam_profile = self.arrival_profile[u, offset:offset + self.sim_duration]
            self.env.process(self._arrival_proc(u, lam_profile))

    def _arrival_proc(self, u, lam_profile):
        """Generate arrivals for station u during simulation time using non-homogeneous Poisson via thinning.
           lam_profile: array of pax/sec with length >= sim_duration
        """
        t = 0.0
        while t < self.sim_duration:
            # set safe upper bound for thinning
            if len(lam_profile) == 0:
                lam_max = 0.0
            else:
                start_idx = max(0, min(int(t), len(lam_profile) - 1))
                end_idx = min(int(t) + 60, len(lam_profile))
                if end_idx > start_idx:
                    lam_max = np.max(lam_profile[start_idx:end_idx])
                else:
                    lam_max = 0.0
            if lam_max <= 0:
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
            if len(lam_profile) == 0:
                lam_t = 0.0
            else:
                lam_t = lam_profile[min(int(t), len(lam_profile) - 1)]
            if RND.random() <= lam_t / lam_max:
                # arrival occurs at station u at time env.now
                self._record_arrival(u, self.env.now)

    def _record_arrival(self, u, tstamp):
        st = self.stations[u]
        gamma_u = STATION_DESIGN_CAP
        if st.in_station_count < STATION_ACCESS_THRESHOLD * gamma_u:
            st.platform_queue.append(tstamp)
            st.in_station_count += 1
        else:
            st.outside_queue.append(tstamp)

    def schedule_trains_from_origin(self, origin_times):
        for i, start in enumerate(origin_times):
            self.env.process(self._train_process(start, f"Train-{i+1}"))

    def _train_process(self, start_time, name):
        yield self.env.timeout(start_time)
        self.trains_used += 1
        onboard = 0
        alight_at = [0] * S
        for u in range(S):
            if u > 0:
                yield self.env.timeout(INTER_STATION_TIME)
            arrive_time = self.env.now
            alight = alight_at[u]
            onboard -= alight
            self.total_alightings += alight
            alight_at[u] = 0
            st = self.stations[u]
            # move from outside to platform
            while st.outside_queue and st.in_station_count < STATION_ACCESS_THRESHOLD * STATION_DESIGN_CAP:
                t_arr = st.outside_queue.popleft()
                st.platform_queue.append(t_arr)
                st.in_station_count += 1
            space = TRAIN_CAPACITY - onboard
            boarded = 0
            while st.platform_queue and space > 0:
                pax_arrival_time = st.platform_queue.popleft()
                wait_time = self.env.now - pax_arrival_time
                st.wait_times.append(wait_time)
                boarded += 1
                onboard += 1
                space -= 1
                self.total_boardings += 1
            # distribute boarded passengers to destinations
            if boarded > 0 and u < S - 1:
                probs = forward_probs[u]
                if len(probs) > 0:
                    counts = NP_RND.multinomial(boarded, probs)
                    for k, count in enumerate(counts):
                        v = u + 1 + k
                        alight_at[v] += count
            # compute dwell
            dwell_secs = max(DWELL_BASE, (boarded * BOARD_TIME_PER_PASS + alight * ALIGHT_TIME_PER_PASS) / DOORS)
            self.dwell_time_total += dwell_secs
            yield self.env.timeout(dwell_secs)
        # train done (end of run). For this fast sim, no recovery & reuse is modeled.

    def run(self, until=None):
        if until is None:
            until = self.sim_duration
        self.start_arrival_processes()
        self.env.run(until=until)

    def get_metrics(self):
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

    @classmethod
    def run_static_with_schedule(cls, schedule, sim_duration=SIM_REAL_DURATION):
        """
        Run a corridor sim with piece-wise headway trains and return cost + metrics.
        """
        sim = cls(sim_duration=sim_duration)
        # generate origin_times from piece-wise schedule
        origin_times = []
        current = 0
        base_sec = BASE_HOUR * 3600 + BASE_MIN * 60
        for start_str, end_str, hw_min in schedule:
            start_h, start_m = map(int, start_str.split(':'))
            end_h, end_m = map(int, end_str.split(':'))
            start_abs = start_h * 3600 + start_m * 60
            end_abs = end_h * 3600 + end_m * 60
            start_sec = max(0, start_abs - base_sec)
            end_sec = min(sim_duration, end_abs - base_sec)
            current = max(current, start_sec)
            while current <= end_sec:
                origin_times.append(current)
                current += hw_min * 60
        sim.schedule_trains_from_origin(origin_times)
        sim.run(until=sim_duration)
        metrics = sim.get_metrics()
        # cost
        total_wait_min = metrics['avg_wait_min'] * metrics['total_boardings'] if metrics['total_boardings'] > 0 else 0.0
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
    total_wait_min = metrics['avg_wait_min'] * metrics['total_boardings'] if metrics['total_boardings'] > 0 else 0.0
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
    real_sim = CorridorSimulator(sim_duration=SIM_REAL_DURATION)
    real_sim.start_arrival_processes(offset=0)
    controller = RollingHorizonController()
    # Initial dispatch at t=0
    real_sim.schedule_trains_from_origin([0.0])
    next_control_time = 0.0
    now = 0.0
    last_print = time.time()

    # For real-time plotting with Tkinter
    root = None
    canvas = None
    fig = None
    ax = None
    if PLOTTING:
        root = tk.Tk()
        root.title("Real-time Simulation Metrics")
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack()
        times = []
        dyn_costs = []
        avg_waits = []
        trains_used_list = []
        # Precompute static costs at intervals
        static_schedule = [
            ('7:10', '7:30', 10),
            ('7:31', '11:36', 8),
            ('11:37', '14:36', 10),
            ('14:37', '20:20', 8),
            ('20:21', '21:00', 10),
        ]
        static_costs = []
        for t in range(0, SIM_REAL_DURATION + 1, DECISION_EPOCH):
            temp_sim = CorridorSimulator(sim_duration=t)
            temp_sim.start_arrival_processes(offset=0)
            temp_sim.schedule_trains_from_origin([0.0])  # Initial train
            # Ensure until is greater than current simulation time
            run_until = max(1, t)  # Start from 1 second if t=0
            temp_sim.env.run(until=run_until)
            total_wait_min = temp_sim.get_metrics()['avg_wait_min'] * temp_sim.get_metrics()['total_boardings'] if temp_sim.get_metrics()['total_boardings'] > 0 else 0.0
            wait_cost = COST_WAIT_PER_MIN * total_wait_min
            approx_route_time = (S - 1) * INTER_STATION_TIME + (S * DWELL_BASE)
            train_hours = temp_sim.get_metrics()['trains_used'] * (approx_route_time / 3600.0)
            op_cost = train_hours * COST_TRAIN_PER_HOUR
            static_costs.append(wait_cost + op_cost)
    else:
        times = []
        dyn_costs = []
        avg_waits = []
        trains_used_list = []
        static_costs = []

    while now < SIM_REAL_DURATION:
        if now >= next_control_time - 1e-6:
            print(f"\n[Main] Controller epoch at sim t={int(now)}s")
            controller.choose_and_apply(real_sim, now)
            next_control_time = now + DECISION_EPOCH
        step_until = min(next_control_time, SIM_REAL_DURATION)
        real_sim.env.run(until=step_until)
        now = step_until
        if time.time() - last_print > 5.0:
            print(f"[Main] sim t={int(now)}s")
            last_print = time.time()

        # Real-time update after each step
        current_metrics = real_sim.get_metrics()
        current_wait_min = current_metrics['avg_wait_min'] * current_metrics['total_boardings'] if current_metrics['total_boardings'] > 0 else 0.0
        current_wait_cost = COST_WAIT_PER_MIN * current_wait_min
        approx_route_time = (S - 1) * INTER_STATION_TIME + (S * DWELL_BASE)
        current_train_hours = current_metrics['trains_used'] * (approx_route_time / 3600.0)
        current_op_cost = current_train_hours * COST_TRAIN_PER_HOUR
        current_total_cost = current_wait_cost + current_op_cost

        print(f"[Real-time] at sim t={now}s ({now/3600:.2f} hours), total_cost={current_total_cost:.1f}, wait_cost={current_wait_cost:.1f}, op_cost={current_op_cost:.1f}, avg_wait={current_metrics['avg_wait_min']:.2f} min, trains_used={current_metrics['trains_used']}, boardings={current_metrics['total_boardings']}")

        times.append(now / 3600)
        dyn_costs.append(current_total_cost)
        avg_waits.append(current_metrics['avg_wait_min'])
        trains_used_list.append(current_metrics['trains_used'])

        if PLOTTING:
            ax.clear()
            ax.plot(times, dyn_costs, label='Dynamic Total Cost')
            ax.plot(times, static_costs[:len(times)], label='Static Total Cost', linestyle='--')
            ax.plot(times, avg_waits, label='Avg Wait (min)')
            ax.plot(times, trains_used_list, label='Trains Used')
            ax.set_xlabel('Simulation Hours')
            ax.set_ylabel('Metrics')
            ax.legend()
            canvas.draw()
            root.update()

    metrics = real_sim.get_metrics()
    print("\n=== REAL SIM RESULTS ===")
    print(metrics)
    print("\nController log (headway in each epoch):")
    for entry in controller.log:
        epoch_time = entry['now']
        h0_sec = entry['h0']
        dispatch_delay = entry['dispatch_delay']
        time_str = str(datetime.timedelta(seconds=epoch_time))
        print(f"Time: {time_str}, Headway: {h0_sec / 60:.1f} min, Dispatch delay: {dispatch_delay / 60:.1f} min")

    if PLOTTING:
        root.mainloop()

    return real_sim, controller


if __name__ == "__main__":
    start_time = time.time()
    real_sim, controller = run_real_time_simulation()
    dyn_metrics = real_sim.get_metrics()
    print(f"\nDynamic total wall time: {time.time() - start_time:.1f}s")

    # Static schedule
    schedule = [
        ('7:10', '7:30', 10),
        ('7:31', '11:36', 8),
        ('11:37', '14:36', 10),
        ('14:37', '20:20', 8),
        ('20:21', '21:00', 10),
    ]
    print("\nRunning static piece-wise headway sim...")
    static_cost, static_metrics = CorridorSimulator.run_static_with_schedule(schedule)

    # Compute dynamic cost
    dyn_wait_min = dyn_metrics['avg_wait_min'] * dyn_metrics['total_boardings'] if dyn_metrics['total_boardings'] > 0 else 0.0
    dyn_wait_cost = COST_WAIT_PER_MIN * dyn_wait_min
    approx_route_time = (S - 1) * INTER_STATION_TIME + (S * DWELL_BASE)
    dyn_train_hours = dyn_metrics['trains_used'] * (approx_route_time / 3600.0)
    dyn_op_cost = dyn_train_hours * COST_TRAIN_PER_HOUR
    dyn_cost = dyn_wait_cost + dyn_op_cost

    print("\n=== COMPARISON ===")
    print(f"Static cost = {static_cost:.1f}, Dynamic cost = {dyn_cost:.1f}")

    if PLOTTING:
        labels = ['Static (piece-wise)', 'Dynamic (GA)']
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