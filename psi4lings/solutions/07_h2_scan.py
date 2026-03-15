"""Solution 07."""
import psi4


def run() -> list[tuple[float, float]]:
    distances = [0.5, 0.7, 0.9, 1.1, 1.3]
    results: list[tuple[float, float]] = []

    psi4.set_options({"basis": "sto-3g", "reference": "rhf"})
    for distance in distances:
        mol = psi4.geometry(
            f"""
            0 1
            H
            H 1 {distance}
            symmetry c1
            """
        )
        energy = float(psi4.energy("scf", molecule=mol))
        results.append((distance, energy))

    return results


if __name__ == "__main__":
    print(run())
