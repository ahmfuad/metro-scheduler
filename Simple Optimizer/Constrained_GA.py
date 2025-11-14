import numpy as np
import random
from Simulator import HeadwaySimulator

class ConstrainedGeneticAlgorithm:
    """
    Genetic Algorithm for headway optimization that internally enforces
    the constraint sum(H) <= 60.
    """

    def __init__(self, sim, lambdas, alpha=1.0, beta=0.1, gamma=0.5,
                 pop_size=20, generations=50,
                 headway_min=3, headway_max=10, num_trains=5,
                 mutation_rate=0.2, crossover_rate=0.7, max_total_headway=60):
        """
        Parameters are the same as the original GA, with the addition of:
        
        max_total_headway : int
            The maximum allowed sum of headways in a schedule (e.g., 60 minutes).
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
        self.max_total_headway = max_total_headway

    def _generate_constrained_solution(self):
        """
        Generate a random headway schedule H that satisfies sum(H) <= max_total_headway.
        It keeps trying until a valid solution is found.
        """
        while True:
            H = [random.randint(self.headway_min, self.headway_max)
                 for _ in range(self.num_trains)]
            if sum(H) <= self.max_total_headway:
                return H

    def result(self, H):
        """Evaluate fitness J(H). Lower is better."""
        # The constraint is now handled by the generation, crossover, and mutation
        # so we don't need to check it here.
        result = self.sim.simulate(
            H, self.lambdas, weights=(self.alpha, self.beta, self.gamma))
        return result

    def select_parent(self, population, fitnesses):
        """Tournament selection."""
        i, j = random.sample(range(len(population)), 2)
        return population[i] if fitnesses[i] < fitnesses[j] else population[j]

    def crossover(self, parent1, parent2):
        """
        Single-point crossover that attempts to produce valid children.
        If children are invalid, it retries with different parents or returns originals.
        """
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]
        
        # Try a few times to produce a valid crossover
        for _ in range(5): # 5 attempts
            point = random.randint(1, self.num_trains - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            if sum(child1) <= self.max_total_headway and sum(child2) <= self.max_total_headway:
                return child1, child2
        
        # If it fails after several attempts, just return the parents to maintain population validity
        return parent1[:], parent2[:]

    def mutate(self, H):
        """
        Mutate by randomly tweaking one headway, ensuring the constraint is not violated.
        """
        if random.random() < self.mutation_rate:
            idx = random.randint(0, self.num_trains - 1)
            original_value = H[idx]
            
            # Try a few times to find a valid mutation
            for _ in range(5): # 5 attempts
                new_value = random.randint(self.headway_min, self.headway_max)
                H[idx] = new_value
                if sum(H) <= self.max_total_headway:
                    return H # Valid mutation found
            
            # If no valid mutation is found, revert to the original
            H[idx] = original_value
        return H

    def run(self):
        """Run GA optimization and return best schedule found."""
        # Initialize population with valid solutions
        population = [self._generate_constrained_solution() for _ in range(self.pop_size)]

        best_H, best_fitness = [], float("inf")
        best_result = None
        fitness_history = []

        for gen in range(self.generations):
            results = [self.result(H) for H in population]
            fitnesses = [res.fitness for res in results]

            # Track best
            for H, f in zip(population, results):
                if f.fitness < best_fitness:
                    best_H, best_fitness, best_result = H[:], f.fitness, f

            h_str = ", ".join([f"{x:.1f}" for x in best_H]) if best_H else ""
            print(f"Gen {gen+1}: Best J(H)={best_fitness:.2f}, H=[{h_str}]")
            fitness_history.append(best_fitness)

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
