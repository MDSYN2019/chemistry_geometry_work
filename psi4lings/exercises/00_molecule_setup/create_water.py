"""Exercise 00: build a Psi4 molecule for water."""
import psi4


def run() -> psi4.core.Molecule:
    # TODO: create a neutral singlet water molecule in C1 symmetry
    # Hint: use psi4.geometry with an XYZ-style block
    mol = psi4.geometry("""
    O
    H 1 1.0
    H 1 1.0 2 104.5
    """)

    # TODO: ensure center-of-mass and orientation are fixed
    return mol


if __name__ == "__main__":
    print(run().natom())
