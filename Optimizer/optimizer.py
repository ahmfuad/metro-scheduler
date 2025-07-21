import random
import numpy as np

# === CONSTANTS FROM THE PAPER ===
C1 = 0.4  # yuan/min (passenger waiting cost)
C2 = 0.2  # yuan/min (on-board cost)
C3 = 0.4  # yuan/min (vehicle operating cost)
T0 = 1    # min (dwelling time)
c = 0.67  # min (accel/decel time)
speed = 26  # km/h

# === ROUTE DATA FROM THE PAPER ===
distances_m = [800, 1000, 750, 800, 700, 500, 900, 650, 750, 600, 800]
running_times = [d / 1000 / speed * 60 for d in distances_m]
num_stops = 12

# === SCHEDULE FORM MATRIX (Table 1) ===
# Each sublist: [normal, zone, express] for each stop
stop_schedule_matrix = [
    [1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0],
    [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0],
    [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1]
]

# === OD MATRIX (Passenger demand from stop j to stop k) ===
# For simplicity, assume symmetric fixed rate for demonstration
rjk = np.zeros((num_stops, num_stops))
for j in range(num_stops):
    for k in range(j + 1, num_stops):
        rjk[j][k] = 2  # 2 passengers/min from j to k

# === GENETIC ALGORITHM ===
class BRTGAFullModel:
    def __init__(self, population_size=20, generations=100, crossover_rate=0.8, mutation_rate=0.005):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.headway_range = (5, 20)
        self.T = 60  # 1 hour studied period

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            headway = random.randint(*self.headway_range)
            freq = int(self.T / headway)
            schedule = [random.randint(1, 3) for _ in range(freq)]  # 1: normal, 2: zone, 3: express
            population.append((headway, schedule))
        return population

    def get_stop_mask(self, sched_type):
        idx = sched_type - 1
        return [row[idx] for row in stop_schedule_matrix]

    def simulate_bus_run(self, headway, schedule):
        freq = len(schedule)
        Li = np.zeros((freq, num_stops))  # onboard load
        Ai = np.zeros((freq, num_stops))  # alighting
        Bi = np.zeros((freq, num_stops))  # boarding
        Si = np.zeros((freq, num_stops))  # skipped passengers

        Wij = np.zeros((freq, num_stops, num_stops))  # OD load in vehicle

        skipped_wait = np.zeros((num_stops, num_stops))  # carryover wait

        for i in range(freq):
            sched_type = schedule[i]
            stop_mask = self.get_stop_mask(sched_type)

            for j in range(num_stops):
                if not stop_mask[j]:
                    continue

                # Board all waiting passengers from j to any downstream stop
                for k in range(j + 1, num_stops):
                    if stop_mask[k]:
                        new_passengers = rjk[j][k] * headway + skipped_wait[j][k]
                        Wij[i][j][k] = new_passengers
                        Bi[i][j] += new_passengers
                        skipped_wait[j][k] = 0
                    else:
                        skipped_wait[j][k] += rjk[j][k] * headway

                # Alighting
                if j > 0:
                    for src in range(j):
                        alight = Wij[i][src][j]
                        Ai[i][j] += alight
                        Wij[i][src][j] = 0

                # Update onboard
                if j > 0:
                    Li[i][j] = Li[i][j - 1] + Bi[i][j] - Ai[i][j]
                else:
                    Li[i][j] = Bi[i][j] - Ai[i][j]

                # Save skipped passenger count
                Si[i][j] = sum(skipped_wait[j][k] for k in range(j + 1, num_stops))

        return Li, Ai, Bi, Si

    def fitness(self, individual):
        headway, schedule = individual
        freq = len(schedule)

        # simulate passenger flow
        Li, Ai, Bi, Si = self.simulate_bus_run(headway, schedule)

        f1 = f2 = f3 = 0

        for i in range(freq):
            sched_type = schedule[i]
            stop_mask = self.get_stop_mask(sched_type)

            for j in range(num_stops):
                tj = running_times[j - 1] if j > 0 else 0
                stops_here = stop_mask[j]

                # --- f1: Passenger waiting cost ---
                rj = sum(rjk[j][k] for k in range(j + 1, num_stops))  # total arrival at j
                f1 += C1 * (rj * headway / 2 + Si[i][j] * headway)

                # --- f2: On-board cost ---
                f2 += C2 * (Li[i][j] * (tj + c) + (Li[i][j] - Ai[i][j]) * T0)

                # --- f3: Vehicle operation cost ---
                f3 += C3 * (tj + c + T0 * stops_here)

        return -(f1 + f2 + f3)

    def select(self, population, fitnesses):
        total = sum(fitnesses)
        probs = [f / total for f in fitnesses]
        return random.choices(population, weights=probs, k=2)

    def crossover(self, parent1, parent2):
        h1, s1 = parent1
        h2, s2 = parent2
        point = random.randint(1, min(len(s1), len(s2)) - 1)
        return (h1, s1[:point] + s2[point:]), (h2, s2[:point] + s1[point:])

    def mutate(self, individual):
        h, schedule = individual
        for i in range(len(schedule)):
            if random.random() < self.mutation_rate:
                schedule[i] = random.randint(1, 3)
        return (h, schedule)

    def run(self):
        population = self.initialize_population()
        for _ in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in population]
            new_population = []

            while len(new_population) < self.population_size:
                parent1, parent2 = self.select(population, fitnesses)
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

        best = max(population, key=self.fitness)
        return best, -self.fitness(best)
