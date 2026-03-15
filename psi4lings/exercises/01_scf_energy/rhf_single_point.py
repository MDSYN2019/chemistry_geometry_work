"""Exercise 01: run an RHF single-point energy on water."""
import psi4


def run() -> float:
    mol = psi4.geometry(
        """
        0 1
        O
        H 1 0.958
        H 1 0.958 2 104.5
        symmetry c1
        """
    )

    # TODO: set options for RHF/STO-3G and a modest SCF convergence
    psi4.set_options({})

    # TODO: compute and return the SCF energy as a float
    energy = 0.0
    return energy


if __name__ == "__main__":
    print(run())
