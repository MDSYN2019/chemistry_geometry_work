"""Exercise 03: open-shell O2 using UHF."""
import psi4


def run() -> float:
    # Triplet O2 ground state
    mol = psi4.geometry(
        """
        0 3
        O
        O 1 1.21
        symmetry c1
        """
    )

    # TODO: run UHF/6-31G single-point and return the SCF energy
    psi4.set_options({})
    return 0.0


if __name__ == "__main__":
    print(run())
