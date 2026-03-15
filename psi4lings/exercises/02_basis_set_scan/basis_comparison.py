"""Exercise 02: compare SCF energies across basis sets."""
import psi4


BASIS_SETS = ["sto-3g", "6-31g", "cc-pvdz"]


def run() -> dict[str, float]:
    mol = psi4.geometry(
        """
        0 1
        O
        H 1 0.958
        H 1 0.958 2 104.5
        symmetry c1
        """
    )

    energies: dict[str, float] = {}
    for basis in BASIS_SETS:
        # TODO: set the basis/reference and compute SCF energy
        energies[basis] = 0.0

    return energies


if __name__ == "__main__":
    print(run())
