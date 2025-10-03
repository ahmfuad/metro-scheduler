import numpy as np
import matplotlib.pyplot as plt
from Simulator import HeadwaySimulator
from GA import GeneticAlgorithm


def constant_lambda(rate: float):
    """Return a function lambda(t) = rate."""

    return lambda t: float(rate)


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
    train_capacity = 500
    # minutes between stations (Taken from https://dhaka-metrorail.com/)
    travel_times = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    p_matrix = np.array([
        [0.000, 0.003, 0.003, 0.056, 0.077, 0.159, 0.033, 0.041,
            0.092, 0.018, 0.054, 0.076, 0.058, 0.062, 0.131, 0.136],
        [0.013, 0.000, 0.002, 0.088, 0.081, 0.187, 0.028, 0.039,
            0.084, 0.024, 0.071, 0.085, 0.045, 0.058, 0.087, 0.109],
        [0.034, 0.006, 0.000, 0.117, 0.140, 0.200, 0.050, 0.048,
            0.075, 0.020, 0.055, 0.065, 0.028, 0.043, 0.056, 0.062],
        [0.098, 0.035, 0.016, 0.000, 0.007, 0.068, 0.052, 0.049,
            0.109, 0.022, 0.080, 0.112, 0.057, 0.066, 0.122, 0.105],
        [0.126, 0.032, 0.018, 0.007, 0.000, 0.018, 0.043, 0.070,
            0.085, 0.020, 0.083, 0.100, 0.078, 0.070, 0.131, 0.120],
        [0.129, 0.034, 0.011, 0.026, 0.008, 0.000, 0.008, 0.025,
            0.060, 0.021, 0.083, 0.114, 0.068, 0.084, 0.163, 0.164],
        [0.100, 0.021, 0.010, 0.081, 0.070, 0.030, 0.000, 0.005,
            0.041, 0.028, 0.090, 0.120, 0.080, 0.070, 0.141, 0.113],
        [0.082, 0.019, 0.008, 0.055, 0.081, 0.068, 0.003, 0.000,
            0.016, 0.020, 0.108, 0.137, 0.077, 0.077, 0.127, 0.123],
        [0.109, 0.023, 0.006, 0.069, 0.061, 0.096, 0.017, 0.010,
            0.000, 0.005, 0.034, 0.092, 0.064, 0.077, 0.169, 0.167],
        [0.075, 0.026, 0.006, 0.061, 0.060, 0.140, 0.047, 0.046,
            0.015, 0.000, 0.017, 0.060, 0.080, 0.104, 0.106, 0.158],
        [0.067, 0.021, 0.005, 0.054, 0.062, 0.130, 0.040, 0.071,
            0.040, 0.006, 0.000, 0.023, 0.051, 0.097, 0.129, 0.204],
        [0.079, 0.022, 0.005, 0.067, 0.066, 0.157, 0.043, 0.072,
            0.079, 0.016, 0.013, 0.000, 0.023, 0.061, 0.136, 0.160],
        [0.101, 0.019, 0.004, 0.054, 0.082, 0.150, 0.048, 0.065,
            0.094, 0.040, 0.063, 0.038, 0.000, 0.008, 0.054, 0.178],
        [0.101, 0.023, 0.005, 0.061, 0.070, 0.169, 0.038, 0.063,
            0.103, 0.043, 0.109, 0.088, 0.007, 0.000, 0.018, 0.100],
        [0.114, 0.020, 0.004, 0.060, 0.073, 0.190, 0.045, 0.059,
            0.137, 0.029, 0.086, 0.121, 0.029, 0.013, 0.000, 0.020],
        [0.111, 0.023, 0.004, 0.049, 0.064, 0.167, 0.032, 0.053,
            0.113, 0.032, 0.112, 0.113, 0.072, 0.045, 0.012, 0.000]
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
        alpha=0.5,
        beta=0.25,
        gamma=20.0,
        pop_size=25,
        generations=80,
        headway_min=3,
        headway_max=15,
        num_trains=10,
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

    H = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

    static_result = sim.simulate(H, lambdas, weights=(1.0, 0.1, 2.5))

    print("Static: Fitness J(H):", static_result.fitness)
    print("Static: Average waiting time:", static_result.avg_waiting_time)
    print("Static: Served:", static_result.total_passengers_served)
    print("Static: Leftover:", static_result.total_passengers_left)

# ---------------------------- Plot ----------------------------

    fig = plt.subplot()

    plt.plot(range(1, len(fitness_history) + 1), fitness_history,
             label='Best Fitness/Generation', marker='o', ms=2.5, mfc='w', linewidth=0.5)

    plt.axhline(y=static_result.fitness, color='red', linestyle='--',
                label=f'Target Fitness = {static_result.fitness:.2f}', linewidth=0.5)

    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')

    plt.legend(fontsize='small', labelcolor='black',
               loc='lower left', frameon=False)

    for axis in ['top', 'bottom', 'left', 'right']:
        fig.spines[axis].set_linewidth(0.5)

    plt.tick_params(axis='both', labelsize=8, width=0.5)

    plt.grid(False)
    plt.show()


if __name__ == "__main__":
    main()
