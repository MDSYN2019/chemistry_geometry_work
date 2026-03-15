"""Solution 01."""
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
    psi4.set_options({"basis": "sto-3g", "reference": "rhf", "e_convergence": 1e-8})
    energy = psi4.energy("scf", molecule=mol)
    return float(energy)


if __name__ == "__main__":
    print(run())
