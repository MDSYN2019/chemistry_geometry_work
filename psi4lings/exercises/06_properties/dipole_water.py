"""Exercise 06: compute molecular dipole components."""
import psi4


def run() -> tuple[float, float, float]:
    mol = psi4.geometry(
        """
        0 1
        O
        H 1 0.958
        H 1 0.958 2 104.5
        symmetry c1
        """
    )

    # TODO: request dipole properties and run energy with return_wfn=True
    # TODO: extract and return (mux, muy, muz)
    return (0.0, 0.0, 0.0)


if __name__ == "__main__":
    print(run())
