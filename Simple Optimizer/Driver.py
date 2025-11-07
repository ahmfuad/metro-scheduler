import random
import numpy as np
import matplotlib.pyplot as plt
from Simulator import HeadwaySimulator
from GA import GeneticAlgorithm

# noise added
def constant_lambda(rate: float, noise: bool = True, noise_level: float = 25):
    """Return a function lambda(t) = rate."""

    return lambda t: float(rate) + (random.uniform(-noise_level, noise_level) if noise else 0)


def sinusoidal_lambda(base: float, amplitude: float, period: float):
    """Return a function lambda(t) = base + amplitude*sin(2*pi*t/period).

    Useful to model a time-varying demand over the horizon.
    """

    def f(t: float) -> float:
        return max(0.0, base + amplitude * np.sin(2.0 * np.pi * t / period))

    return f


def main():
    # --- Parameters ---
    n_stations = 16
    train_capacity = 1200
    # minutes between stations (Taken from https://dhaka-metrorail.com/)
    travel_times = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    p_matrix = np.array([
        [0.000, 0.003, 0.003, 0.056, 0.077, 0.159, 0.033, 0.041,
            0.092, 0.018, 0.054, 0.076, 0.058, 0.062, 0.131, 0.136],
        [0.000, 0.000, 0.002, 0.089, 0.082, 0.189, 0.028, 0.040,
            0.085, 0.024, 0.072, 0.086, 0.046, 0.059, 0.088, 0.110],
        [0.000, 0.000, 0.000, 0.121, 0.146, 0.209, 0.053, 0.050,
            0.078, 0.020, 0.057, 0.068, 0.029, 0.045, 0.058, 0.065],
        [0.000, 0.000, 0.000, 0.000, 0.008, 0.080, 0.061, 0.057,
            0.129, 0.026, 0.095, 0.131, 0.067, 0.078, 0.143, 0.124],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.022, 0.053, 0.085,
            0.104, 0.024, 0.101, 0.122, 0.096, 0.086, 0.160, 0.146],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.010, 0.032,
            0.076, 0.027, 0.105, 0.144, 0.087, 0.106, 0.206, 0.208],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.007,
            0.059, 0.040, 0.131, 0.175, 0.117, 0.102, 0.206, 0.165],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.023, 0.030, 0.157, 0.200, 0.113, 0.113, 0.185, 0.179],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.008, 0.057, 0.152, 0.105, 0.127, 0.278, 0.274],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.033, 0.114, 0.152, 0.197, 0.202, 0.302],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.045, 0.101, 0.192, 0.257, 0.405],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.060, 0.161, 0.359, 0.420],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.034, 0.226, 0.740],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.156, 0.844],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
    ])

    # total service minutes in JULY 2025 - 25,080 minutes (service times are taken from https://dhaka-metrorail.com/)
    # constant arrival rates (passengers/minute)
    lambdas = [
        constant_lambda(37.96248006),
        constant_lambda(9.276355662),
        constant_lambda(3.407974482),
        constant_lambda(23.36104466),
        constant_lambda(26.64238437),
        constant_lambda(44.58333333),
        constant_lambda(14.90594099),
        constant_lambda(20.9942185),
        constant_lambda(31.69820574),
        constant_lambda(10.08859649),
        constant_lambda(30.32236842),
        constant_lambda(35.00103668),
        constant_lambda(23.01594896),
        constant_lambda(24.05558214),
        constant_lambda(37.23847687),
        constant_lambda(51.53313397)
    ]

    # initialize simulator
    sim = HeadwaySimulator(
        n_stations=n_stations,
        capacity=train_capacity,
        travel_times=travel_times,
        p_dest=p_matrix
    )

    # initialize GA
    ga = GeneticAlgorithm(
        sim=sim,
        lambdas=lambdas,
        alpha=3.5,
        beta=0.25,
        gamma=50.0,
        pop_size=25,
        generations=30,
        headway_min=3,
        headway_max=10,
        num_trains=8,
        mutation_rate=0.15,
        crossover_rate=0.6
    )

    # run GA
    best_H, best_J, fitness_history, result = ga.run()

    print("\n=== Final Result ===")
    print("Best headway schedule:", best_H)
    print("Best fitness J(H):", best_J)
    print("Best avg_waiting_time:", result.avg_waiting_time)

# ------------------ Test for static schedule ------------------

    H = [5, 5, 5, 5, 5, 5, 5, 5]

    static_result = sim.simulate(H, lambdas, weights=(ga.alpha, ga.beta, ga.gamma))

    print("Static: Fitness J(H):", static_result.fitness)
    print("Static: Average waiting time:", static_result.avg_waiting_time)
    print("Static: Served:", static_result.total_passengers_served)
    print("Static: Leftover:", static_result.total_passengers_left)

# ---------------------------- Plot ----------------------------

    fig = plt.subplot()

    plt.plot(range(1, len(fitness_history) + 1), fitness_history,
             label=f'Best Fitness/Generation = {best_J:.2f}', marker='o', ms=2.5, mfc='w', linewidth=0.5)

    plt.axhline(y=static_result.fitness, color='red', linestyle='--',
                label=f'Target Fitness = {static_result.fitness:.2f}', linewidth=0.5)

    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')

    plt.legend(fontsize='small', labelcolor='black',
               loc='upper right', frameon=False)

    for axis in ['top', 'bottom', 'left', 'right']:
        fig.spines[axis].set_linewidth(0.5)

    plt.tick_params(axis='both', labelsize=8, width=0.5)

    plt.grid(False)
    plt.show()


if __name__ == "__main__":
    main()
