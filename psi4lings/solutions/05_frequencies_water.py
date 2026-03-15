"""Solution 05."""
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
    psi4.set_options({"basis": "sto-3g", "reference": "rhf"})
    _, wfn = psi4.frequency("scf", molecule=mol, return_wfn=True)
    frequencies = wfn.frequencies().to_array()
    positive_freqs = [f for f in frequencies if f > 1.0]
    return float(positive_freqs[0])


if __name__ == "__main__":
    print(run())
