"""
Hartree Fock
------------

SCF routine based on the paper

"""
import logging

Z = 2

def two_energy_expectation_value(Z: int) -> None:
    """
    Basis set expansion
    """
    eps_1 = -Z ** 2/2
    eps_2 = -Z ** 2/8
    I_1111 = 5/8 * Z
    I_1112 = 2**12*np.sqrt(2)/27/7**4 * Z
    I_1122 = 16/9**3*Z
    I_1212 = 17/3**4*Z
    I_1222 = 2**9*npsqrt(2)/27/5**5 * Z
    I_2222 = 77/2**9 * Z
    pass

def hatree_fock(c_1, c_2): 
    F_11 = eps_1 + I_1111*c_1** 2 
