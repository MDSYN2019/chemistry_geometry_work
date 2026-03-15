"""Exercise 04: optimize water geometry."""
import psi4


def run() -> float:
    mol = psi4.geometry(
        """
        0 1
        O
        H 1 1.10
        H 1 1.10 2 120.0
        symmetry c1
        """
    )

    # TODO: set basis/reference and run geometry optimization with psi4.optimize
    final_energy = 0.0
    return final_energy


if __name__ == "__main__":
    print(run())
