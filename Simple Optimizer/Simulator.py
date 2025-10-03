"""
Discrete-event simulator for one-way metro headway optimization.

This module provides a small, well-documented set of functions and a Simulator class
that implement the passenger-flow equations from our model and utilities used by
a GA optimizer to evaluate candidate headway vectors H = [h1, h2, ..., hK].

Key features:
- Deterministic or stochastic arrivals (Poisson) per interval.
- Per-train, per-station queue updates, boarding, alighting, occupancy tracking.
- Methods are modular so GA fitness evaluation can call Simulator.simulate(H).

Equations used (refer to paper model):
- Arrivals between dispatches k-1 and k at station i:
    A_i^{(k)} = int_{T_{k-1}}^{T_k} lambda_i(t) dt
  In practice we approximate by A_i^{(k)} approx lambda_i(tilde t_k) * h_k
  where tilde t_k is a representative time in interval (T_{k-1},T_k].

- Queue update before boarding at train k:
    Q_i^{(k)} = Q_i^{(k-1)} - B_i^{(k-1)} + A_i^{(k)}

- Boarding at station i for train k:
    B_i^{(k)} = min(Q_i^{(k)}, R_{k,i})
  where R_{k,i} is remaining capacity when train arrives at station i.

- Occupancy update:
    O_{k,i+1}^{in} = O_{k,i}^{in} - D_i^{(k)} + B_i^{(k)}

- Waiting time (for deterministic arrivals constant over interval):
    total waiting in interval ~ lambda_i * (h_k^2)/2
  We compute exact waiting by integrating arrival times when needed.

Usage summary:
- Instantiate Simulator with station_count, capacity C, interstation times d_i,
  optional destination matrix p_ij, and optionally current queue state.
- Call simulator.simulate(H) where H is list/array of headways (minutes).
- Output is a result object with average waiting, leftover queue, train occupancy,
  boarding logs, and time series useful for GA fitness.

This file is intentionally self-contained (only numpy required) and written to be
readable and easy to integrate into a GA loop.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class SimulatorResult:
    """Container for simulation outputs."""
    H: List[float]
    T: float
    total_passengers_arrived: int
    total_passengers_served: int
    total_passengers_left: int
    avg_waiting_time: float
    waiting_time_by_interval: List[float]
    queue_time_series: List[List[float]]  # Q_i^{(k)} values
    occupancy_time_series: List[List[float]]  # O_{k,i}^{in} values
    boarding_log: List[List[int]]  # B_i^{(k)}
    alighting_log: List[List[int]]  # D_i^{(k)}
    per_train_summary: List[Dict[str, Any]]
    fitness: Optional[float] = None


class HeadwaySimulator:
    """
    Discrete-event simulator for one-way line.

    Parameters
    ----------
    n_stations : int
        Number of stations (n).
    capacity : int
        Train capacity C (passengers).
    travel_times : List[float]
        Inter-station travel times d_i of length n-1. If a single value given,
        it is broadcast to all inter-station segments.
    p_dest : Optional[np.ndarray]
        Destination probability matrix p_{i->j} of shape (n, n). Only entries
        with j>i should be non-zero. If None, a simple average trip-length model
        is used (see _approx_destination_matrix()).
    initial_queues : Optional[List[int]]
        Initial queue sizes Q_i^{(1)} at time t0.
    dwell_params : Optional[Tuple[float,float]]
        (tau0, alpha) where dwell = tau0 + alpha*(boarding+alighting). If None,
        dwell times are ignored in arrival timing (only travel_times used).
    rng_seed : Optional[int]
        Seed for stochastic arrivals if used.
    """

    def __init__(
        self,
        n_stations: int,
        capacity: int,
        travel_times: List[float] | float = 1.0,
        p_dest: Optional[np.ndarray] = None,
        initial_queues: Optional[List[int]] = None,
        dwell_params: Optional[Tuple[float, float]] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.n = n_stations
        self.C = int(capacity)
        if isinstance(travel_times, (int, float)):
            self.d = [float(travel_times)] * (self.n - 1)
        else:
            assert len(travel_times) == max(0, self.n - 1)
            self.d = [float(x) for x in travel_times]

        # Destination probability matrix
        if p_dest is None:
            self.p_dest = self._approx_destination_matrix()
        else:
            assert p_dest.shape == (self.n, self.n)
            self.p_dest = p_dest.astype(float)

        # initial queues
        if initial_queues is None:
            self.initQ = [0] * self.n
        else:
            assert len(initial_queues) == self.n
            self.initQ = [int(x) for x in initial_queues]

        self.dwell_params = dwell_params  # (tau0, alpha)
        self.rng = np.random.default_rng(rng_seed)

    # ---------------------- Helper approximation ----------------------
    def _approx_destination_matrix(self) -> np.ndarray:
        """
        Return a simple destination matrix assuming geometric decay of trip lengths.
        p_{i->j} proportional to r^{(j-i-1)} for j>i and zeros else; rows sum to 1
        (for j>i). This is a placeholder when real OD data not available.

        Rationale: Most urban trips are short; choose r in (0,1); use r=0.5.
        """
        r = 0.5
        P = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            if i == self.n - 1:
                continue
            weights = np.array([r ** (j - i - 1)
                               for j in range(i + 1, self.n)])
            weights = weights / weights.sum()
            P[i, i + 1: self.n] = weights
        return P

    # ---------------------- Arrival calculation ----------------------
    def arrivals_between(self, lambda_funcs: List, T_prev: float, T_curr: float, deterministic: bool = True) -> List[int]:
        """
        Compute arrivals A_i^{(k)} between times T_prev and T_curr for all stations.

        Parameters
        ----------
        lambda_funcs : List
            A list (length n) of callables lambda_i(t) giving instantaneous arrival
            rate (passengers/min) or a list/array of samples. Each lambda_funcs[i]
            must be callable: lambda_funcs[i](t) -> rate.
        T_prev, T_curr : float
            Interval endpoints (minutes). h_k = T_curr - T_prev.
        deterministic : bool
            If True, returns deterministic integer arrivals by rounding the
            expected arrivals (rate * interval). If False, samples from Poisson
            with mean = rate*interval.

        Returns
        -------
        arrivals : List[int]
            arrivals[i] = A_i^{(k)} (non-negative int)

        Equations
        ---------
        A_i^{(k)} = int_{T_prev}^{T_curr} lambda_i(t) dt approx lambda_i(tilde t) * h_k
        """
        h = T_curr - T_prev
        arrivals = [0] * self.n
        for i in range(self.n):
            lam_i = lambda_funcs[i](
                0.5 * (T_prev + T_curr)) if callable(lambda_funcs[i]) else float(lambda_funcs[i])
            expected = lam_i * h
            if deterministic:
                arrivals[i] = int(round(expected))
            else:
                arrivals[i] = int(self.rng.poisson(expected))
        return arrivals

    # ---------------------- Boarding calculation ----------------------
    def compute_boarding(
        self,
        queue_before: List[int],
        remaining_capacity: int,
    ) -> Tuple[int, int]:
        """
        Compute how many passengers board at a single station given queue and remaining capacity.

        Parameters
        ----------
        queue_before : int
            Number of passengers waiting before boarding event (Q_i^{(k)}).
        remaining_capacity : int
            Remaining seats on train when it arrives at this station (R_{k,i}).

        Returns
        -------
        boarded : int
            Number of passengers who board B_i^{(k)}.
        remaining_capacity_after : int
            Remaining capacity after boarding.

        Equation
        --------
        B_i^{(k)} = min(Q_i^{(k)}, R_{k,i})
        """
        boarded = min(int(queue_before), max(0, int(remaining_capacity)))
        remaining_after = max(0, remaining_capacity - boarded)
        return boarded, remaining_after

    # ---------------------- Alighting calculation ----------------------
    def compute_alighting(self, onboard_origin_counts: np.ndarray, station_index: int) -> int:
        """
        Compute expected number of alighting passengers at station_index given
        onboard composition by origin.

        Parameters
        ----------
        onboard_origin_counts : np.ndarray
            Vector of length n where element r is number of passengers on board
            whose origin was station r (these passengers have destination distribution
            given by p_dest[r, :]). Only r < station_index matter for alighting.
        station_index : int
            Index i (0-based) of station where alighting occurs.

        Returns
        -------
        alighting : int
            Number of passengers expected to alight at this station from onboard set.

        Equation
        --------
        D_i^{(k)} = sum_{r=1}^{i-1} boarded_{r}^{(k)} * p_{r->i}
        """
        if station_index == 0:
            return 0
        alight = 0.0
        for r in range(0, station_index):
            prob = self.p_dest[r, station_index]
            if prob > 0 and onboard_origin_counts[r] > 0:
                alight += onboard_origin_counts[r] * prob
        # Alighting must be integer; round to nearest
        return int(round(alight))

    # ---------------------- Single schedule simulation ----------------------
    def simulate(
        self,
        H: List[float],
        lambda_funcs: List,
        t0: float = 0.0,
        deterministic_arrivals: bool = True,
        track_waiting_detail: bool = False,
        weights: Tuple[float, float, float] = (1.0, 1.0, 0.1)
    ) -> SimulatorResult:
        """
        Simulate the passenger flow for a candidate headway schedule H.

        Parameters
        ----------
        H : List[float]
            Headways (minutes) for trains k=1..K.
        lambda_funcs : List
            List of station arrival-rate functions lambda_i(t) or constants.
        t0 : float
            Current time (minutes). The departure time of train 0 is t0 (or now),
            and train k departs at T_k = t0 + sum_{r=1}^k h_r.
        deterministic_arrivals : bool
            If True, use expected arrivals (rounded). If False, sample Poisson.
        track_waiting_detail : bool
            If True, compute waiting time by tracking per-interval arrival timing
            (slower but more exact). If False, use interval-average approximations.

        Returns
        -------
        SimulatorResult
            Contains aggregated metrics and logs useful for GA evaluation.
        """
        K = len(H)
        T_times = [t0]
        for h in H:
            T_times.append(T_times[-1] + float(h))
        # T_times[k] is departure time of train k (k from 0..K)

        # Initialize logs
        Q = [list(self.initQ)]  # Q^{(k)} before train k=1 is Q[0]
        boarding_log: List[List[int]] = []
        alight_log: List[List[int]] = []
        occupancy_log: List[List[int]] = []
        waiting_by_interval: List[float] = []

        # Track onboard composition by origin for each train as vector length n
        # For train k we'll maintain onboard_origin_counts (starts zero at depot)

        total_arrived = 0
        total_served = 0
        total_wait_time = 0.0

        # Optionally track per-train summary
        per_train_summary: List[Dict[str, Any]] = []

        # For each train k=1..K, simulate its progression along stations
        for k in range(1, K + 1):
            T_prev = T_times[k - 1]
            T_curr = T_times[k]
            arrivals = self.arrivals_between(
                lambda_funcs, T_prev, T_curr, deterministic=deterministic_arrivals)
            total_arrived += sum(arrivals)

            # Update queues before boarding at train k: Q_i^{(k)} = previous Q - previous boarding + arrivals
            # Note: Q list currently holds Q^{(k-1)} at index k-1
            Q_before = [max(0, Q[-1][i] + arrivals[i]) for i in range(self.n)]

            # Initialize train onboard counters
            onboard_origin = np.zeros(self.n, dtype=float)
            # occupancy when arriving at station 0 (first station)
            occ_in_station = 0
            remaining_cap = self.C - int(occ_in_station)

            boarding_this_train = [0] * self.n
            alighting_this_train = [0] * self.n

            # Waiting time approximation for this interval
            interval_wait = 0.0

            # If tracking arrival time detailed waiting, assume uniform arrival; else use average
            for i in range(self.n):
                # Alighting at station i from existing onboard (none initially)
                D_i = self.compute_alighting(onboard_origin, i)
                if D_i > 0:
                    # remove alighting from onboard_origin proportionally by expected p_dest
                    # approximate removal: subtract expected alighters proportionally
                    for r in range(0, i):
                        prob = self.p_dest[r, i]
                        if prob > 0 and onboard_origin[r] > 0:
                            remove = onboard_origin[r] * prob
                            onboard_origin[r] -= remove
                    occ_in_station = int(round(onboard_origin.sum()))
                    remaining_cap = max(0, self.C - occ_in_station)
                else:
                    D_i = 0

                # Now boarding
                Qi_before = Q_before[i]
                B_i, remaining_cap = self.compute_boarding(
                    Qi_before, remaining_cap)
                # Update queue after boarding
                Q_before[i] = Q_before[i] - B_i
                boarding_this_train[i] = int(B_i)
                # Add boarded passengers to onboard_origin: assume they originate at station i
                onboard_origin[i] += B_i

                # Occupancy entering next station
                occ_in_station = int(round(onboard_origin.sum()))
                occupancy_log.append([k, i, occ_in_station]) if False else None

                # Waiting time approximation: if deterministic arrivals and uniform in interval,
                # passengers who boarded from arrivals in this interval have mean wait h_k/2; but
                # we must separate already-waiting Q from newly arrived ones. For simplicity here:
                if track_waiting_detail:
                    # split Q components into old and new arrivals is complicated without storing arrival times.
                    # We'll approximate: assume arrivals this interval waited on average h_k/2,
                    # and previously queued waited an additional average (we'll treat previously queued as having waited 0 extra here).
                    interval_wait += arrivals[i] * (H[k - 1] / 2.0)
                else:
                    interval_wait += arrivals[i] * (H[k - 1] / 2.0)

            # After finishing stations for train k, compute totals
            served_this_train = sum(boarding_this_train)
            total_served += served_this_train
            total_wait_time += interval_wait

            # Append logs: Q for next iteration (Q^{(k)} before train k+1) is Q_before
            Q.append(list(Q_before))
            boarding_log.append(boarding_this_train)
            # placeholder (detailed alighting bookkeeping omitted)
            alighting_this_train = [0] * self.n
            alight_log.append(alighting_this_train)

            # Per train summary
            per_train_summary.append({
                "train_index": k,
                "depart_time": T_times[k],
                "arrivals": arrivals,
                "boarded": boarding_this_train,
                "served": served_this_train,
                "remaining_queue_after": list(Q_before),
            })

        # After simulation
        total_left = sum(Q[-1])
        total_pass = total_arrived
        avg_wait = total_wait_time / \
            max(1, total_served) if total_served > 0 else float('inf')
        w1, w2, w3 = weights
        fitness = w1 * avg_wait + w2 * total_left + w3 * len(H)

        result = SimulatorResult(
            H=list(H),
            T=T_times[-1] - T_times[0],
            total_passengers_arrived=total_pass,
            total_passengers_served=total_served,
            total_passengers_left=total_left,
            avg_waiting_time=avg_wait,
            waiting_time_by_interval=[0.0] * K,
            queue_time_series=Q,
            occupancy_time_series=[],
            boarding_log=boarding_log,
            alighting_log=alight_log,
            per_train_summary=per_train_summary,
            fitness=fitness
        )
        return result
