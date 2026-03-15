"""Solution 04."""
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
    psi4.set_options({"basis": "6-31g", "reference": "rhf", "g_convergence": "gau_tight"})
    final_energy = psi4.optimize("scf", molecule=mol)
    return float(final_energy)


if __name__ == "__main__":
    print(run())
