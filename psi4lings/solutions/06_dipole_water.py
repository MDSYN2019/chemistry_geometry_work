"""Solution 06."""
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
    psi4.set_options({"basis": "6-31g", "reference": "rhf"})
    _, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
    psi4.oeprop(wfn, "DIPOLE")
    mux = float(wfn.variable("SCF DIPOLE X"))
    muy = float(wfn.variable("SCF DIPOLE Y"))
    muz = float(wfn.variable("SCF DIPOLE Z"))
    return (mux, muy, muz)


if __name__ == "__main__":
    print(run())
