import numpy as np
import random
from Simulator import HeadwaySimulator


class GeneticAlgorithm:
    """Simple GA for headway optimization."""

    def __init__(self, sim, lambdas, alpha=1.0, beta=0.1, gamma=0.5,
                 pop_size=20, generations=50,
                 headway_min=3, headway_max=10, num_trains=5,
                 mutation_rate=0.2, crossover_rate=0.7):
        """
        Simple GA for headway optimization.

        Parameters
        ----------
        sim : HeadwaySimulator
            Simulator object.
        lambdas : list
            Arrival rates at stations.
        alpha, beta : float
            Weights for waiting time vs operating cost.
        pop_size : int
            Number of candidate solutions per generation.
        generations : int
            Number of generations to evolve.
        headway_min, headway_max : float
            Allowed headway range in minutes.
        num_trains : int
            Number of trains in the schedule.
        mutation_rate, crossover_rate : float
            GA parameters.
        """
        self.sim = sim
        self.lambdas = lambdas
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pop_size = pop_size
        self.generations = generations
        self.headway_min = headway_min
        self.headway_max = headway_max
        self.num_trains = num_trains
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def random_solution(self):
        """Generate a random headway schedule H."""
        return [random.randint(self.headway_min, self.headway_max)
                for _ in range(self.num_trains)]

    def result(self, H):
        """Evaluate fitness J(H). Lower is better."""
        result = self.sim.simulate(
            H, self.lambdas, weights=(self.alpha, self.beta, self.gamma))
        return result

    def select_parent(self, population, fitnesses):
        """Tournament selection."""
        i, j = random.sample(range(len(population)), 2)
        return population[i] if fitnesses[i] < fitnesses[j] else population[j]

    def crossover(self, parent1, parent2):
        """Single-point crossover."""
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]
        point = random.randint(1, self.num_trains - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(self, H):
        """Mutate by randomly tweaking one headway."""
        if random.random() < self.mutation_rate:
            idx = random.randint(0, self.num_trains - 1)
            H[idx] = random.randint(self.headway_min, self.headway_max)
        return H

    def run(self):
        """Run GA optimization and return best schedule found."""
        # Initialize population
        population = [self.random_solution() for _ in range(self.pop_size)]

        best_H, best_fitness = None, float("inf")
        best_result = None
        fitness_history = []

        for gen in range(self.generations):

            results = [self.result(H) for H in population]

            fitnesses = [res.fitness for res in results]

            # Track best
            for H, f in zip(population, results):
                if f.fitness < best_fitness:
                    best_H, best_fitness, best_result = H[:], f.fitness, f

            print(f"Gen {gen+1}: Best J(H)={best_fitness:.2f}, H={best_H}")

            if best_result is not None:
                '''
                print(f"Gen {gen+1}: Best J(H)={best_result.fitness:.2f}, "
                    f"Avg Wait={best_result.avg_waiting_time:.2f}, "
                    f"Leftover={best_result.total_passengers_left}, "
                    f"Served={best_result.total_passengers_served}, "
                    f"H={best_H}")
                '''
                fitness_history.append(best_fitness)
            # else:
            #     print(f"Gen {gen+1}: No valid result yet.")

            # Next generation
            new_population = []
            while len(new_population) < self.pop_size:
                p1 = self.select_parent(population, fitnesses)
                p2 = self.select_parent(population, fitnesses)
                c1, c2 = self.crossover(p1, p2)
                new_population.append(self.mutate(c1))
                if len(new_population) < self.pop_size:
                    new_population.append(self.mutate(c2))
            population = new_population

        return best_H, best_fitness, fitness_history, best_result
