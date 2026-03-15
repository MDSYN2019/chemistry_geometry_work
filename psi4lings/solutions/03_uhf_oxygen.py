"""Solution 03."""
import psi4


def run() -> float:
    mol = psi4.geometry(
        """
        0 3
        O
        O 1 1.21
        symmetry c1
        """
    )
    psi4.set_options({"basis": "6-31g", "reference": "uhf", "d_convergence": 1e-8})
    return float(psi4.energy("scf", molecule=mol))


if __name__ == "__main__":
    print(run())
