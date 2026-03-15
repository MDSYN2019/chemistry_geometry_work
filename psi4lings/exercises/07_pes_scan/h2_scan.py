"""Exercise 07: create a simple H2 bond scan."""
import psi4


def run() -> list[tuple[float, float]]:
    distances = [0.5, 0.7, 0.9, 1.1, 1.3]
    results: list[tuple[float, float]] = []

    # TODO: loop over distances, build H2 geometry, and compute RHF/STO-3G energy
    # TODO: append (distance, energy) to results

    return results


if __name__ == "__main__":
    print(run())
