// new.h
#ifndef GEOMETRY_FUNCTION
#define GEOMETRY_FUNCTION

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <cmath>
#include <tuple>
#include <unordered_map>
// variant 
#include <variant>

#include "geometry_functions.hpp"

void read_coordinates(std::string &structure,
                      std::vector<std::vector<double>> &VecInput);

void read_coordinates_hamiltonian(std::string &structure, std::vector<std::vector<double> >& vec_input);

void output_lines(std::vector<std::vector<double> >& VecInput);

std::tuple<double, double, double> center_of_mass(std::vector<std::vector<double>> &VecInput);

double Rij(double &Xi, double &Xj, double &Yi, double &Yj, double &Zi,
           double &Zj);

double bond_angles(double &Xi, double &Xj, double &Xk, double &Yi, double &Yj,
                   double &Yk, double &Zi, double &Zj, double &Zk);

double out_of_plane_angle(double &Xi, double &Xj, double &Xk, double &Xl,
                       double &Yi, double &Yj, double &Yk, double &Yl,
                       double &Zi, double &Zj, double &Zk, double &Zl);

double torsion_angle(double &Xi, double &Xj, double &Xk, double &Xl, double &Yi,
                     double &Yj, double &Yk, double &Yl, double &Zi, double &Zj,
                     double &Zk, double &Zl);

std::tuple<double, double, double> centerOfMass(std::vector<std::vector<double> >& VecInput);
Eigen::Vector3d moments_of_inertia(std::vector<std::vector<double> >& VecInput);
Eigen::MatrixXd reshape_vector(std::vector<std::vector<double> >& input, int rows, int column);
Eigen::MatrixXd weigh_matrix(Eigen::MatrixXd& hessian_eigen_matrix, std::vector<std::vector<double>> geometry_input, std::unordered_map<int, int>& atomic_masses, int rows, int column);
Eigen::MatrixXd compute_core_hamiltonian(std::vector<std::vector<double> >& ke, std::vector<std::vector<double> >& nuclear_attraction_integral);
Eigen::MatrixXd orthogonalize_basis_set(std::vector<std::vector<double> >& overlap_matrix);
Eigen::MatrixXd inital_density_matrix(Eigen::MatrixXd &orthogonalized_basis_set_s, Eigen::MatrixXd &core_hamiltonian);

#endif
