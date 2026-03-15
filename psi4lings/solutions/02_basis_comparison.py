"""Solution 02."""
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
        psi4.set_options({"basis": basis, "reference": "rhf"})
        energies[basis] = float(psi4.energy("scf", molecule=mol))
    return energies


if __name__ == "__main__":
    print(run())
