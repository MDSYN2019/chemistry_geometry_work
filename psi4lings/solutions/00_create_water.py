"""Solution 00."""
import psi4


def run() -> psi4.core.Molecule:
    mol = psi4.geometry(
        """
        0 1
        O
        H 1 0.958
        H 1 0.958 2 104.5
        symmetry c1
        no_reorient
        no_com
        """
    )
    return mol


if __name__ == "__main__":
    print(run().natom())
