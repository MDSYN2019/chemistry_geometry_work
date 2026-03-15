"""Exercise 05: run a harmonic frequency analysis."""
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

    # TODO: run psi4.frequency("scf", return_wfn=True)
    # TODO: read and return the first harmonic frequency from the wavefunction
    first_mode_cm1 = 0.0
    return first_mode_cm1


if __name__ == "__main__":
    print(run())
